from __future__ import annotations

import os
import json
import pickle
import logging
import json_fix
import warnings

import rustworkx as rx

from pathlib import Path
from itertools import chain
from collections import namedtuple
from functools import lru_cache, reduce
from typing import Any, Iterable, Optional
from more_itertools import batched, collapse

from territories.partitions import TerritorialUnit, Partition, Node
from territories.exceptions import MissingTreeException, MissingTreeCache, NotOnTreeError, EmptyTerritoryError

try:
    from pydantic_core import CoreSchema
    from pydantic import GetJsonSchemaHandler, GetCoreSchemaHandler
    from pydantic_core import core_schema
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False



logger = logging.getLogger(__name__)

CHECKSUM = "current version is A2FH9"

class Territory:
    """Class to represent territories.

    A Territory object can be any combination of entities, such as municipalities, countries, county, länders, states, etc, as long as it belongs to the DAG of entities.
    The package guarantee that the representation of a territory will always be efficient.
    For instance, if I create a `Territory` object with all regions from a country, it will simplify it to only the country object.
    """
    tree: Optional[rx.PyDiGraph] = None
    root_index: Optional[int] = None
    name_to_id: dict[str, int] = {}

    @staticmethod
    def to_part(node: Node) -> TerritorialUnit:
        atomic = False
        match (node.level, node.id):
            case "ARR", _:
                partition = Partition.ARR
                atomic = True
            case "COM", comm_id:
                partition = Partition.COM
                try:
                    atomic = comm_id.split(':')[1][:2] not in ('69', '75', '13')
                except Exception:
                    pass
            case "DEP", _:
                partition = Partition.DEP
            case "REG", _:
                partition = Partition.REG
            case "CNTRY", _:
                partition = Partition.CNTRY
            case p:
                raise TypeError(f"Unknow terrotory type {p}")


        return TerritorialUnit(
            name=node.label,
            atomic=atomic,
            tu_id=node.id,
            level=partition,
            postal_code=getattr(node, "postal_code", None),
            inhabitants=getattr(node, "inhabitants", None)
        )


    @classmethod
    def reset(cls):
        # if tree is a reference to a foreign object
        # I do not want to assign None to it
        # so first I destroy the reference
        del cls.tree
        cls.tree = None
        cls.root_index = None
        cls.name_to_id = {}


    @classmethod
    def load_tree_from_bytes(cls, data: bytes):
        cls.reset()
        try:
            checksum, cls.name_to_id, cls.tree = pickle.loads(data)
        except ValueError:
            raise Exception("The data is not a valid territorial tree.")
        if checksum != CHECKSUM:
            raise Exception("The data is not a valid territorial tree.")
        assert isinstance(cls.tree, rx.PyDiGraph)
        cls.root_index = next(i for i in cls.tree.node_indices() if cls.tree.in_degree(i) == 0)


    @classmethod
    def load_tree(cls, filepath: Optional[str] = None):
        """Attempt to load the territorial tree from a file.

        If no file is provided, it will look on the API_CACHE_DIR env. variable.

        Args:
            filepath (Optional[str], optional): Path to a file. Defaults to None.

        Raises:
            MissingTreeCache: If no file is provided and the env. variable is missing


        File that can be loaded are the ones created by `Territory.save_tree()`
        """
        cls.reset()
        path = None
        if filepath is None:
            cache_dir = os.environ.get("API_CACHE_DIR") or os.environ.get("CACHE_DIR")
            if cache_dir is None:
                raise MissingTreeCache("No filepath is specified and you have no API_CACHE_DIR or CACHE_DIR env. variable")
            path = Path(cache_dir, "territorial_tree_state.pickle")
        if isinstance(filepath, (str, Path, os.PathLike)):
            path = filepath
        if not path: # wrong behavior, filepath should be any representation of a filepath
            raise TypeError("filepath has to ba a string")
        try:
            with open(path, "rb") as file:
                try:
                    checksum, cls.name_to_id, cls.tree = pickle.load(file)
                except ValueError:
                    raise Exception("The file is not a valid territorial tree cache. Delete it and try again.")
                if checksum != CHECKSUM:
                    raise Exception("The file is not a valid territorial tree cache. Delete it and try again.")
        except FileNotFoundError:
            raise MissingTreeCache(f"Tree object was not found at {path}")

        assert isinstance(cls.tree, rx.PyDiGraph)
        cls.root_index = next(i for i in cls.tree.node_indices() if cls.tree.in_degree(i) == 0)
        names: list[str] = [cls.tree.get_node_data(i).tu_id for i in cls.tree.node_indices()]

        for name in names:
            assert name == cls.hash(name).tu_id


    @classmethod
    def save_tree(cls, filepath: Optional[str] = None, return_bytes: bool = False) -> Optional[bytes]:
        """Save the territorial tree and the hashmap to a file.

        If no file is provided, it will look for the API_CACHE_DIR env. variable to create a new one.

        Args:
            filepath (Optional[str], optional): File path to save the tree state to. Defaults to None.
            return_bytes (bool, optional): Return the bytes of the saved file. Defaults to False.
        """
        path = None
        if filepath is None and not return_bytes:
            try:
                path = Path(os.environ["API_CACHE_DIR"], "territorial_tree_state.pickle")
            except KeyError:
                logger.warning("failed to save the tree in cache directory. Please set the env variable API_CACHE_DIR")
                return
        if isinstance(filepath, str):
            path = filepath
        if path:
            with open(path, "wb") as file:
                pickle.dump((CHECKSUM, cls.name_to_id, cls.tree), file)
        if return_bytes:
            return pickle.dumps((CHECKSUM, cls.name_to_id, cls.tree))
        if path is None and not return_bytes:
            raise Exception("You must provide a filepath or set the API_CACHE_DIR env. variable, or use return_bytes=True")


    @classmethod
    def build_tree(cls, data_stream: Iterable[Node], save_tree = True, filepath: Optional[str] = None):
        """Build the territorial tree from a stream of objects.
        You can use the built-in territories.partitions.Node object, but any object with attributes **id**, **parent_id**, **level** and **label** will work.

        The id attribute will be assigned as **es_code** attribute in TerritorialUnit nodes.

        Args:
            data_stream (Iterable[Node]): An iterable of objects to add on the tree.
            save_tree (bool, optional): Save to disk the constructed tree. Defaults to True.
            filepath (Optional[str], optional): File path to save the tree state to. If not provided, API_CACHE_DIR env. var. will be used. Defaults to None.
        """
        cls.reset()

        tree = rx.PyDiGraph()
        mapper = {}
        orphans: list[OrphanNode] = []
        batch_size = 1024
        OrphanNode = namedtuple('OrphanNode', ('id', 'parent_id', 'label', 'level', 'tree_id'))
        for batch in batched(data_stream, batch_size):
            entities_indices = tree.add_nodes_from(tuple(cls.to_part(node) for node in batch))

            for node, tree_idx in zip(batch, entities_indices):
                if not tree.get_node_data(tree_idx).atomic:
                    mapper[node.id] = tree_idx
                object.__setattr__(tree.get_node_data(tree_idx), 'tree_id', tree_idx)

            edges = []
            for node, tree_idx in zip(batch, entities_indices):
                if node.parent_id in mapper:
                    edges.append((mapper[node.parent_id], tree_idx, None))
                else:
                    if node.parent_id: # do not append root node to orphans
                        # object.__setattr__(node, 'tree_id', tree_idx)
                        # orphans.append(node)

                        # this is a lot more expensive than updating the node object
                        # but we have no guarantee that it is mutable (can be a tuple)
                        orphan = OrphanNode(
                            id=node.id,
                            parent_id=node.parent_id,
                            label=node.label,
                            level=node.level,
                            tree_id=tree_idx
                            )
                        orphans.append(orphan)
            tree.add_edges_from(edges)

        edges = tuple((mapper[orphan.parent_id], orphan.tree_id, None) for orphan in orphans if orphan.parent_id in mapper)
        tree.add_edges_from(edges)

        last_orphans = tuple(orphan for orphan in orphans if orphan.parent_id not in mapper)
        if last_orphans:
            logger.warning(f"{len(last_orphans)} elements where not added to the tree because they have no parents : {last_orphans}")

        cls.name_to_id = {tree.get_node_data(i).tu_id : i for i in tree.node_indices()}
        cls.tree = tree
        cls.root_index = next(i for i in tree.node_indices() if tree.in_degree(i) == 0)

        if save_tree:
            cls.save_tree(filepath=filepath)


    @classmethod
    def assign_tree(cls, tree: rx.PyDiGraph):
        """DEPRECATED. Do not use this method. Its only purpose is for quick and easy tests.

        Directly assign a tree to the class.

        Args:
            tree (rx.PyDiGraph): Tree of `territories.TerritorialUnit` objects
        """
        cls.reset()

        elements: list[TerritorialUnit] = [tree.get_node_data(i) for i in tree.node_indices()]
        for i, e in enumerate(elements):
            object.__setattr__(e, 'tree_id', i)
            object.__setattr__(e, 'tu_id', e.name)

        cls.name_to_id = {tree.get_node_data(i).tu_id : i for i in tree.node_indices()}
        cls.tree = tree
        cls.root_index = next(i for i in tree.node_indices() if tree.in_degree(i) == 0)


    @classmethod
    def successors(cls, tu: TerritorialUnit) -> list[TerritorialUnit]:
        """Returns the successors of a territorial unit in the territorial tree

        Args:
            tu (TerritorialUnit | str): A TerritorialUnit or a string. If a strign, it must ba a unique id for a TerritorialUnit (like **DEP:69**).

        Raises:
            NotOnTreeError: Raise an exception if the id is not on the territorial tree.

        Returns:
            list[TerritorialUnit]: list of TerritorialUnit objects that are children of the given TerritorialUnit.
        """
        assert isinstance(tu, TerritorialUnit)
        assert tu.tree_id is not None
        assert isinstance(cls.tree, rx.PyDiGraph)
        return cls.tree.successors(tu.tree_id)


    @staticmethod
    @lru_cache(maxsize=256)
    def _ancestors(tree: rx.PyDiGraph, node: int) -> set[int]:
        return rx.ancestors(tree, node)


    @staticmethod
    def contains(a, b, tree):
        return (a == b) or (a in rx.ancestors(tree, b))


    @classmethod
    def minimize(cls, node: int, items: set[int]) -> set[int]:
        """Make sure the representation of a Territory is always minimal.
        """
        # if len(items) == 0:
        #     return set()
        # if node in items:
        #     return {node}
        # children = set(cls.tree.successor_indices(node))
        # if children.issubset(items):
        #     return {node}
        # gen = (cls.minimize(child, tuple(item for item in items if cls.contains(child, item, cls.tree))) for child in children)
        # union = set.union(*gen)
        # if union == children:
        #     return {node}
        # return union
        assert isinstance(cls.tree, rx.PyDiGraph)


        # best method
        if not items:
            return set()
        if node in items:
            return {node}
        children = set(cls.tree.successor_indices(node))
        valids = children & items # O(min(children, items))
        if len(valids) == len(children):
            return {node}
        for valid in valids:
            items.remove(valid)
        for child in children:
            if child not in valids and cls.tree.out_degree(child):
                if any(node in cls._ancestors(cls.tree, item) for item in items):
                    valids.update(cls.minimize(child, items))
        if valids == children:
            return {node}
        return valids




    @classmethod
    def union(cls, *others: Iterable[Territory | TerritorialUnit]) -> Territory:
        """Returns the union of given elements as a new Territory object

        Args:
            Any number of `territories.Territory` or `territories.TerritorialUnit` objects

        Returns:
            Territory: A new Territory object containing all elements
        """
        return reduce(lambda x, y: x + y, iter(others))


    @classmethod
    def intersection(cls, *others: Iterable[Territory | TerritorialUnit]) -> Territory:
        """Returns the intersection of given elements as a new Territory object

        Args:
            Any number of `territories.Territory` or `territories.TerritorialUnit` objects

        Returns:
            Territory: A new Territory object contained by all elements
        """
        return reduce(lambda x, y: x & y, iter(others))


    @classmethod
    def LCA(cls, *others: Iterable[Territory | TerritorialUnit]) -> Optional[TerritorialUnit]:
        """Return the lowest common ancestor of the given territorial units.
        If Territory objects are given, it will use their corresponding territorial units.

        Details of this algorithm [here](https://networkx.org/nx-guides/content/algorithms/lca/LCA.html).

        Returns:
            TerritorialUnit | None: A TerritorialUnit object being the lowest common ancestor of every given territorial units. If no common ancestor is found, it will return None.
        """
        if not others:
            raise EmptyTerritoryError("An empty territory has no ancestors")
        assert isinstance(cls.tree, rx.PyDiGraph)
        # not necessary, maybe better performance for small territories
        # if len(others) == 1:
        #     node = others[0]
        #     if isinstance(node, TerritorialUnit) and node.tree_id:
        #         return cls.tree.predecessors(node.tree_id).pop()
        #     if isinstance(node, Territory) and len(node.territorial_units) == 1:
        #         tree_id = next(n.tree_id for n in node.territorial_units)
        #         return cls.tree.predecessors(tree_id).pop()
        others = set.union(*({e} if isinstance(e, TerritorialUnit) else e.territorial_units for e in others))
        common_ancestors = set.intersection(*(rx.ancestors(cls.tree, e.tree_id) for e in others))
        match len(common_ancestors):
            case 0:
                return None
                # raise EmptyTerritoryError("No common ancestor found")
            case 1:
                return cls.tree.get_node_data(common_ancestors.pop())
            case _:
                ancestor = next(iter(common_ancestors))
        # search the lowest node of the tree in common ancestors
        while True:
            successor = None
            for child in cls.tree.successor_indices(ancestor):
                if child in common_ancestors:
                    successor = child
                    break
            if successor is None:
                return cls.tree.get_node_data(ancestor)
            ancestor = successor


    @classmethod
    def get_parent(cls, other: TerritorialUnit) -> Optional[TerritorialUnit]:
        """Return the parent of the given territorial unit.

        Args:
            other (TerritorialUnit): A TerritorialUnit object

        Returns:
            Optional[TerritorialUnit]: A TerritorialUnit object being the parent of the given territorial unit. None if the territorial unit has no parent.
        """
        try:
            assert isinstance(cls.tree, rx.PyDiGraph)
            return cls.tree.predecessors(other.tree_id).pop()
        except IndexError:
            return None


    @classmethod
    def get_parents(cls, *others: Iterable[Territory | TerritorialUnit]) -> Territory:
        """Return the parent of the given territorial unit.
        If Territory objects are given, it will use their corresponding territorial units.

        Returns:
            Territory | None: A Territory object being the parent of the given territorial unit. The territory will be empty if the territorial unit has no parent.
        """
        if not others:
            raise EmptyTerritoryError("An empty territory has no parent")
        assert isinstance(cls.tree, rx.PyDiGraph)
        parent_tus = collapse(cls.tree.predecessors(node.tree_id) for node in collapse(others))
        return Territory(*parent_tus)


    @classmethod
    def all_ancestors(cls, *others: Iterable[Territory | TerritorialUnit]) -> set[TerritorialUnit]:
        """Return a set of all ancestors of every territorial unit or territory.
        If Territory objects are given, it will use their corresponding territorial units.

        Returns:
            set[TerritorialUnit]: The union of all ancestors of every territorial units given as input.
        """
        if not others:
            raise EmptyTerritoryError("An empty territory has no ancestors")
        assert isinstance(cls.tree, rx.PyDiGraph)
        others = set.union(*({e} if isinstance(e, TerritorialUnit) else e.territorial_units for e in others))
        ancestors  = set.union(*(rx.ancestors(cls.tree, e.tree_id) for e in others))
        return {cls.tree.get_node_data(i) for i in ancestors}


    @classmethod
    def all_descendants(cls, *others: Iterable[Territory | TerritorialUnit]) -> set[TerritorialUnit]:
        """Return a set of all descendants of every territorial unit ro territory.
        If Territory objects are given, it will use their corresponding territorial units.

        Returns:
            set[TerritorialUnit]: The union of all descendants of every territorial units given as input.
        """
        if not others:
            raise EmptyTerritoryError("An empty territory has no ancestors")
        assert isinstance(cls.tree, rx.PyDiGraph)
        others = set.union(*({e} if isinstance(e, TerritorialUnit) else e.territorial_units for e in others))
        ancestors  = set.union(*(rx.descendants(cls.tree, e.tree_id) for e in others))
        return {cls.tree.get_node_data(i) for i in ancestors}


    @classmethod
    def _sub(cls, a: TerritorialUnit, b: TerritorialUnit) -> set[TerritorialUnit]:
        if a == b:
            return set()
        assert isinstance(cls.tree, rx.PyDiGraph)
        if a.tree_id in rx.ancestors(cls.tree, b.tree_id):
            children = cls.tree.successors(a.tree_id)
            return set.union(*(cls._sub(child, b) for child in children))
        return {a}


    @classmethod
    def _and(cls, a: TerritorialUnit, b: TerritorialUnit) -> set[TerritorialUnit]:
        if a == b:
            return {a}
        if a.tree_id in rx.ancestors(cls.tree, b.tree_id): # if a in b
            return {b}
        if b.tree_id in rx.ancestors(cls.tree, a.tree_id): # if b in a
            return {a}
        return set()


    @classmethod
    def hash(cls, name: str) -> TerritorialUnit:
        """Return the tree indice of an object given its name

        Args:
            name (str): Name of the object. Currently the ElasticSearch name (like COM:2894)

        Returns:
            TerritorialUnit: Object on the tree (`tree.get_node_data(i)`)
        """
        if not isinstance(name, str):
            raise Exception(f"tu_ids are string, you provided a {type(name).__name__} : {name}")
        try:
            node_id = cls.name_to_id[name]
            return cls.tree.get_node_data(node_id)
        except KeyError:
            raise NotOnTreeError(name)


    @classmethod
    def from_name(cls, tu_id: str) -> TerritorialUnit:
        """Return a TerritorialUnit object from its unique id, like **COM:2894** or **DEP:69** 😏.

        ⚠️ YOU SHOULD PROBABLY NOT USE THIS METHOD, USE `from_names()` METHOD INSTEAD !
        This method returns a TerritorialUnit object, which is not directly linked to the territorial tree.

        Raises:
            NotOnTreeError: Raise an exception if the id is not on the territorial tree.

        Returns:
            Territory: TerritorialUnit object.

        exemple :
        ```python
        Territory.from_name('COM:01044')
        >>> Douvres
        ```
        """
        return cls.hash(tu_id)


    @classmethod
    def from_tu_ids(cls, *args: Iterable[str | Iterable[str]]) -> Territory:
        """Create a new Territory object from tu_ids
        Currently names are ElasticSearch code, like **COM:2894** or **DEP:69** 😏.
        Raises:
            NotOnTreeError: Raise an exception if one  or more names are not an ElasticSearch code on the territorial tree.

        Returns:
            Territory: Territory object with territories associated with the given names.

        exemple :
        ```python
        Territory.from_tu_ids(['COM:01044', 'COM:01149'])
        >>> Douvres|Billiat
        Territory.from_tu_ids('DEP:69')
        >>> Rhône
        ```
        """
        if cls.tree is None:
                raise MissingTreeException('Tree is not initialized. Initialize it with Territory.build_tree()')
        if not args:
            raise TypeError("`from_tu_ids()` needs at least one arguments")
        if isinstance(args[0], (list, tuple, set, dict)):
            if len(args) > 1:
                raise TypeError("`from_tu_ids()` needs only one iterable of tu_ids")
            tu_ids = iter(args[0])
        else:
            tu_ids = iter(args)      
        try:
            entities_idxs = {cls.hash(name) for name in tu_ids}
            return Territory(*entities_idxs)
        except NotOnTreeError as e:
            wrong_elements = {e}
            for name in tu_ids:
                try:
                    cls.hash(name)
                except NotOnTreeError:
                    wrong_elements.add(name)
            verb = "where" if len(wrong_elements) > 1 else "was"
            wrong_elements = ', '.join(str(e) for e in wrong_elements)
            raise NotOnTreeError(f"{wrong_elements} {verb} not found in the territorial tree")


    @classmethod
    def from_names(cls, *args: Iterable[str]) -> Territory:
        """Create a new Territory object from names
        Currently names are ElasticSearch code, like **COM:2894** or **DEP:69** 😏.
        Raises:
            NotOnTreeError: Raise an exception if one  or more names are not an ElasticSearch code on the territorial tree.

        Returns:
            Territory: Territory object with territories associated with the given names.

        soon to be deprecated, use `from_tu_ids()` instead

        exemple :
        ```python
        Territory.from_names('COM:01044', 'COM:01149')
        >>> Douvres|Billiat
        ```
        """
        warnings.warn("This method is deprecated, use from_tu_ids() instead", UserWarning)
        if cls.tree is None:
            raise MissingTreeException('Tree is not initialized. Initialize it with Territory.build_tree()')
        entities_idxs = (cls.hash(name) for name in args)
        try:
            return Territory(*entities_idxs)
        except (NotOnTreeError):
            wrong_elements = set()
            for name in args:
                try:
                    cls.hash(name)
                except NotOnTreeError:
                    wrong_elements.add(name)
            verb = "where" if len(wrong_elements) > 1 else "was"
            wrong_elements = ', '.join(str(e) for e in wrong_elements)
            raise NotOnTreeError(f"{wrong_elements} {verb} not found in the territorial tree")


    def __init__(self, *args: Iterable[TerritorialUnit]) -> None:
        """Create a Territory instance.

        A Territory is composed of one or several TerritorialUnit, that represents elements on the territorial tree.
        All territories instances share a reference to the territorial tree.

        Raises:
            MissingTreeException: You can't build Territory instances if the territorial tree has not been initialized.
        """
        if self.tree is None:
            raise MissingTreeException('Tree is not initialized. Initialize it with Territory.build_tree()')
        territorial_units = set(args)
        if territorial_units:
            entities_idxs = {e.tree_id for e in territorial_units}
            #  guarantee the Territory is always represented in minimal form
            # territories are immutable
            self.territorial_units: frozenset[TerritorialUnit] = frozenset(self.tree.get_node_data(i) for i in self.minimize(self.root_index, entities_idxs))
        else:
            self.territorial_units: set[TerritorialUnit] = frozenset()


    @property
    def type(self) -> Partition:
        if self.is_empty():
            return Partition.EMPTY
        return min(self).level


    @property
    def tu_ids(self) -> list[str]:
        """Return the tu_ids of every territorial units in the territory.

        Returns:
            list[str]: List of tu_ids
        """
        return [e.tu_id for e in self.territorial_units]


    def __iter__(self):
        return iter(self.territorial_units)


    def __len__(self) -> int:
        return len(self.territorial_units)


    def __eq__(self, other: Territory | TerritorialUnit) -> bool:
        # should also check for equality of ids
        # since some entities share the same territory but are not equal
        # ex : Parlement and ADEME both occupy France, yet are not the same entities
        if isinstance(other, TerritorialUnit):
            return self.territorial_units == {other}
        return self.territorial_units == other.territorial_units


    def __bool__(self):
        return len(self.territorial_units) != 0


    def __add__(self, other: Territory) -> Territory:
        return Territory(
            *(self.territorial_units | other.territorial_units)
        )


    def is_contained(self, other: Territory) -> bool:
        if self == other:
            return True
        for entity in self.territorial_units:
            ancestors = rx.ancestors(self.tree, entity.tree_id) | {entity.tree_id}
            if not any(other_entity.tree_id in ancestors for other_entity in other.territorial_units):
                return False
        return True


    def __contains__(self, other: Territory | TerritorialUnit) -> bool:
        if isinstance(other, TerritorialUnit):
            ancestors = rx.ancestors(self.tree, other.tree_id) | {other.tree_id}
            return any(child.tree_id in ancestors for child in self.territorial_units)
        return other.is_contained(self)


    def is_disjoint(self, other: Territory) -> bool:
        pass


    def __or__(self, other: Territory | TerritorialUnit) -> Territory:
        if not self.territorial_units:
            entities = tuple()
        else:
            entities = self.territorial_units
        if isinstance(other, TerritorialUnit):
            return Territory(*chain(entities, [other]))
        if other.territorial_units is not None:
            return Territory(*chain(entities, other.territorial_units))
        return self


    def __and__(self, other: Territory | TerritorialUnit) -> Territory:
        if isinstance(other, TerritorialUnit):
            return  Territory(*chain(*(self._and(child, other) for child in self.territorial_units)))
        if (not other.territorial_units) or (not self.territorial_units):
            return Territory()
        if self in other:
            return self

        return Territory.union(*(self & child for child in other.territorial_units))


    def __sub__(self, other: Territory | TerritorialUnit) -> Territory:
        if isinstance(other, TerritorialUnit):
            return Territory(*chain(*(self._sub(child, other) for child in self.territorial_units)))
        if (not other.territorial_units) or (not self.territorial_units):
            return self
        if self in other:
            return Territory()

        return Territory.intersection(*(self - child for child in other.territorial_units))


    def __json__(self):
        return self.territorial_units


    def __repr__(self) -> str:
        if self.territorial_units:
            return '|'.join(str(e) for e in sorted(self.territorial_units, reverse=True))
        return 'ø'


    if HAS_PYDANTIC:

        # this method should raise specific Pydantic Exceptions
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            _source_type: Any,
            _handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            """This is used by Pydantic to generate the schema for the Territory class.
            It needs to handle all possible ways to create a Territory object.
            """
            return core_schema.union_schema([
                # Handle Territory instances directly
                core_schema.is_instance_schema(Territory),
                # Handle strings (parse them as territory IDs)
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls._try_parse)
                ]),
                # Handle lists/iterables of strings (parse as multiple territory IDs)
                core_schema.chain_schema([
                    core_schema.list_schema(core_schema.str_schema()),
                    core_schema.no_info_plain_validator_function(
                        lambda v: cls.from_tu_ids(v)
                    )
                ]),
                # Handle lists/iterables of TerritorialUnit instances
                core_schema.chain_schema([
                    core_schema.list_schema(core_schema.is_instance_schema(TerritorialUnit)),
                    core_schema.no_info_plain_validator_function(
                        lambda v: cls(*v)
                    )
                ])
            ])

        # this crash with some validators. Needs to be tested
        def __get_pydantic_json_schema__(
            self,
            _schema_generator: GetJsonSchemaHandler,
            _field_schema: Any
        ) -> dict[str, Any]:
            return {
                'type': 'string',
                'description': 'Territory represented as string (format: ["COM:12345", "DEP:69"] or multiple IDs separated by "|")'
            }

        # needs to implement this for backward compatibility. Or not. Probbly not
        # @classmethod
        # def __get_validators__(cls) -> Iterator[Callable[..., Any]]:
        #     yield cls.validate


        # @classmethod
        # def validate(cls, value: Any) -> Territory:
        #     if isinstance(value, str):
        #         return cls.try_parse(value)
        #     if isinstance(value, cls):
        #         return value
        #     if isinstance(value, Iterable):
        #         return cls.from_tu_ids(*value)
        #     return cls.from_tu_ids(value)


    @classmethod
    def _try_parse(cls, input_string: str) -> Territory:
        try:
            tus = json.loads(input_string)
        except json.JSONDecodeError:
            tus = input_string.split('|')
        return cls.from_tu_ids(tus)


    def lowest_common_ancestor(self) -> Optional[TerritorialUnit]:
        """Return the lowest common ancestor of the territorial units of this territory.

        Details of this algorithm [here](https://networkx.org/nx-guides/content/algorithms/lca/LCA.html).
        Returns:
            TerritorialUnit| None: A TerritorialUnit object being the lowest common ancestor of every territorial units of the territory. None if the territory has no parents.
        """
        return self.LCA(*self.territorial_units)


    def ancestors(self, include_itself: bool = False) -> set[TerritorialUnit]:
        """Return a set of all ancestors of every territorial unit of this territory.

        Args:
            include_itself (bool, optional): Wether to include or not the node in its ancestors. Defaults to False.

        Returns:
            set[TerritorialUnit]: The union of all ancestors of every territorial unit of the territory.
        """
        if not self.territorial_units:
            raise EmptyTerritoryError("An empty territory has no ancestors")
        ancestors = set.union(*(rx.ancestors(self.tree, e.tree_id) for e in self.territorial_units))
        res = {self.tree.get_node_data(i) for i in ancestors}
        if include_itself:
            res = res | self.territorial_units
        return res


    def descendants(self, include_itself: bool = False) -> set[TerritorialUnit]:
        """Return a set of all descendants of every territorial unit of this territory.

        Args:
            include_itself (bool, optional): Wether to include or not the node in its descendants. Defaults to False.

        Returns:
            set[TerritorialUnit]: The union of all descendants of every territorial unit of the territory.
        """
        if not self.territorial_units:
            raise EmptyTerritoryError("An empty territory has no descendants")
        descendants = set.union(*(rx.descendants(self.tree, e.tree_id) for e in self.territorial_units))
        res = {self.tree.get_node_data(i) for i in descendants}
        if include_itself:
            res = res | self.territorial_units
        return res


    def parents(self) -> Territory:
        return self.get_parents(self)


    def is_empty(self) -> bool:
        """
        Returns:
            bool: Return wether the territory object is an empty territory
        """
        return len(self.territorial_units) == 0
