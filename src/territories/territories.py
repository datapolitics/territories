from __future__ import annotations

import os
import pickle
import logging

import rustworkx as rx

from pathlib import Path
from itertools import chain
from functools import reduce
from more_itertools import batched
from typing import Iterable, Optional, Callable
from perfect_hash import generate_hash, IntSaltHash

from territories.partitions import TerritorialUnit, Partition, Node
from territories.exceptions import MissingTreeException, MissingTreeCache, NotOnTreeError

logger = logging.getLogger(__name__)

class Territory:
    """Class to represent territories.

    A Territory object can be any combination of entities, such as municipalities, countries, county, länders, states, etc, as long as it belongs to the DAG of entities.
    The package guarantee that the representation of a territory will always be efficient.
    For instance, if I create a `Territory` object with all regions from a country, it will simplify it to only the country object.
    """
    tree: Optional[rx.DiGraph] = None
    root_index: Optional[int] = None
    perfect_hash_fct: Optional[Callable[[str], int]] = None
    perfect_hash_params: Optional[tuple] = None


    @staticmethod
    def compute_hash_function(names: list[str]) -> tuple:
        # create perfect hash table
        f1, f2, G = generate_hash(names, Hash=IntSaltHash)
        G = tuple(G) # tuples have slightly faster access time
        NG = len(G)
        NS = len(f1.salt)
        S1 = tuple(f1.salt)
        S2 = tuple(f2.salt)

        return (G, NG, NS, S1, S2)


    @staticmethod
    def create_hash_function(params: tuple) -> Callable[[str], int]:
        G, NG, NS, S1, S2 = params

        def hash_f(key, T):
            return sum(T[i % NS] * ord(c) for i, c in enumerate(key)) % NG

        def perfect_hash(key):
            return (G[hash_f(key, S1)] + G[hash_f(key, S2)]) % NG

        return perfect_hash


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
            case _:
                partition = None


        return TerritorialUnit(
            name=node.label,
            atomic=atomic,
            es_code=node.id,
            partition_type=partition,
            postal_code=getattr(node, "postal_code", None)
        )


    @classmethod
    def reset(cls):
        # if tree is a reference to a foreign object
        # I do not want to assign None to it
        # so first I destroy the reference
        del cls.tree
        cls.tree = None
        cls.perfect_hash_fct = None
        cls.perfect_hash_params = None
        cls.root_index = None


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

        if filepath is None:
            cache_dir = os.environ.get("API_CACHE_DIR")
            if cache_dir is None:
                raise MissingTreeCache(f"No filepath is specified and you have no API_CACHE_DIR env. variable")
            path = Path(cache_dir, "territorial_tree_state.pickle")
        if isinstance(filepath, str):
            path = filepath
        try:
            with open(path, "rb") as file:
                cls.perfect_hash_params, cls.tree = pickle.load(file)
                cls.perfect_hash_fct = cls.create_hash_function(cls.perfect_hash_params)
        except FileNotFoundError:
            raise MissingTreeCache(f"Tree object was not found at {path}")

        cls.root_index = next(i for i in cls.tree.node_indices() if cls.tree.in_degree(i) == 0)  
        names: list[TerritorialUnit] = [cls.tree.get_node_data(i).es_code for i in cls.tree.node_indices()]

        for name in names:
            i = cls.hash(name)
            assert name == cls.tree.get_node_data(i).es_code


    @classmethod
    def save_tree(cls, filepath: Optional[str] = None):
        """Save the territorial tree and the perfect hash function to a file. 

        If no file is provided, it will look for the API_CACHE_DIR env. variable to create a new one.

        Args:
            filepath (Optional[str], optional): File path to save the tree state to. Defaults to None.
        """
        if filepath is None:
            try:
                path = Path(os.environ["API_CACHE_DIR"], "territorial_tree_state.pickle")
            except KeyError:
                logger.warning("failed to save the tree in cache directory. Please set the env variable API_CACHE_DIR")
                return
        if isinstance(filepath, str):
            path = filepath
        with open(path, "wb") as file:
            pickle.dump((cls.perfect_hash_params, cls.tree), file)
       

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
        orphans: list[Node] = []
        batch_size = 1024
        for batch in batched(data_stream, batch_size):
            entities_indices = tree.add_nodes_from(tuple(cls.to_part(node) for node in batch))

            for node, tree_idx in zip(batch, entities_indices):
                if node.level != Partition.COM: # communes don't have any children
                    mapper[node.id] = tree_idx
                object.__setattr__(tree.get_node_data(tree_idx), 'tree_id', tree_idx)

            edges = []
            for node, tree_idx in zip(batch, entities_indices):
                if node.parent_id in mapper:
                    edges.append((mapper[node.parent_id], tree_idx, None))
                else:
                    if node.parent_id: # do not append root node to orphans
                        object.__setattr__(node, 'tree_id', tree_idx)
                        orphans.append(node)
            tree.add_edges_from(edges)


        edges = tuple((mapper[orphan.parent_id], orphan.tree_id, None) for orphan in orphans if orphan.parent_id in mapper)
        tree.add_edges_from(edges)

        orphans = tuple(orphan for orphan in orphans if orphan.parent_id not in mapper)
        if orphans:
            logger.warning(f"{len(orphans)} elements where not added to the tree because they have no parents : {orphans}")


        names = [tree.get_node_data(i).es_code for i in tree.node_indices()]
        logger.info(f"There are {len(names)} elements in the tree")

        cls.perfect_hash_params = cls.compute_hash_function(names)
        perfect_hash = cls.create_hash_function(cls.perfect_hash_params)

        for name in names:
            i = perfect_hash(name)
            assert name == tree.get_node_data(i).es_code

        cls.tree = tree
        cls.perfect_hash_fct = perfect_hash  
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

        names = [e.name for e in elements]
        cls.perfect_hash_params = cls.compute_hash_function(names)
        perfect_hash = cls.create_hash_function(cls.perfect_hash_params)

        for name in names:
            i = perfect_hash(name)
            assert name == tree.get_node_data(i).name

        cls.tree = tree
        cls.perfect_hash_fct = perfect_hash        
        cls.root_index = next(i for i in tree.node_indices() if tree.in_degree(i) == 0)


    @classmethod
    def hash(cls, name: str) -> int:
        """Return the tree indice of an object given its name

        Args:
            name (str): Name of the object. Currently the ElasticSearch name (like COM:2894)

        Returns:
            int: Indice of the object on the tree (`tree.get_node_data(i)`)
        """
        return cls.perfect_hash_fct(name)


    @staticmethod
    def contains(a: int, b: int, tree: rx.PyDiGraph) -> bool:
        return (a == b) or (a in rx.ancestors(tree, b)) # b in a


    @classmethod
    def minimize(cls, node: int, items: Iterable[int]) -> set[int]:
        """Make sure the representation of a Territory is always minimal.
        """
        if len(items) == 0:
            return set()
        
        if node in items:
            return {node}
        
        children = set(cls.tree.successor_indices(node))

        if children == set(items):
            return {node}

        gen = (cls.minimize(child, tuple(item for item in items if cls.contains(child, item, cls.tree))) for child in children)
        union =  set.union(*gen)

        # no sure I need this
        if union == children:
            return {node}
        
        return union
    

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
    def LCA(cls, *others: Iterable[Territory | TerritorialUnit]) -> Territory:
        """Return the lowest common ancestor of the given territorial units.
        If Territory objects are given, it will use their corresponding territorial units.

        Details of this algorithm [here](https://networkx.org/nx-guides/content/algorithms/lca/LCA.html).
        Returns:
            Territory: A territory associated with a single territorial unit
        """
        others = set.union(*({e} if isinstance(e, TerritorialUnit) else e.entities for e in others))
        common_ancestors = set.intersection(*(rx.ancestors(cls.tree, e.tree_id) for e in others))
        match len(common_ancestors):
            case 0:
                return Territory()
            case 1:
                return Territory(cls.tree.get_node_data(common_ancestors.pop()))
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
                return Territory(cls.tree.get_node_data(ancestor))
            ancestor = successor


    @classmethod
    def all_ancestors(cls, *others: Iterable[Territory | TerritorialUnit]) -> set[TerritorialUnit]:
        """Return a set of all ancestors of every territorial unit of this territory.
        If Territory objects are given, it will use their corresponding territorial units.

        Returns:
            set[TerritorialUnit]: The union of all ancestors of every territorial unit given as input.
        """
        others = set.union(*({e} if isinstance(e, TerritorialUnit) else e.entities for e in others))
        ancestors  = set.union(*(rx.ancestors(cls.tree, e.tree_id) for e in others))
        return {cls.tree.get_node_data(i) for i in ancestors}


    @classmethod
    def _sub(cls, a: TerritorialUnit, b: TerritorialUnit) -> set[TerritorialUnit]:
        if a == b:
            return set()
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
    def from_name(cls, *args: Iterable[str]) -> Territory:
        """Create a new Territory object from names
        Currently names are ElasticSearch code, like **COM:2894** or **DEP:69** 😏.
        Raises:
            NotOnTreeError: Raise an exception if one  or more names are not an ElasticSearch code on the territorial tree.

        Returns:
            Territory: Territory object with territories associated with the given names.

        exemple :
        ```python
        Territory.from_name('COM:01044', 'COM:01149')
        >>> Douvres|Billiat
        ```
        """
        entities_idxs = (cls.hash(name) for name in args)
        try:
            return Territory(*(cls.tree.get_node_data(i) for i in entities_idxs))
        except (OverflowError, IndexError):
            raise NotOnTreeError("One or several elements where not found in territorial tree")


    def __init__(self, *args: Iterable[TerritorialUnit]) -> None:
        """Create a Territory instance.

        A Territory is composed of one or several TerritorialUnit, that represents elements on the territorial tree.
        All territories instances share a reference to the territorial tree.

        Raises:
            MissingTreeException: You can't build Territory instances if the territorial tree has not been initialized.
        """
        if self.tree is None:
            raise MissingTreeException('Tree is not initialized. Initialize it with Territory.build_tree()')
        entities = set(args)
        if entities:
            entities_idxs = [e.tree_id for e in entities]
            #  guarantee the Territory is always represented in minimal form
            self.entities = {self.tree.get_node_data(i) for i in self.minimize(self.root_index, entities_idxs)}
        else:
            self.entities = set()


    def __eq__(self, value: Territory) -> bool:
        # should also check for equality of ids
        # since some entities share the same territory but are not equal
        # ex : Parlement and ADEME both occupy France, yet are not the same entities
        return self.entities == value.entities


    def __add__(self, other: Territory) -> Territory:
        return Territory(
            *(self.entities | other.entities)
        )
    

    def is_contained(self, other: Territory) -> bool:
        if self == other:
            return True
        for entity in self.entities:
            ancestors = rx.ancestors(self.tree, entity.tree_id) | {entity.tree_id}
            if not any(other_entity.tree_id in ancestors for other_entity in other.entities):
                return False
        return True
    

    def __contains__(self, other: Territory | TerritorialUnit) -> bool:
        if isinstance(other, TerritorialUnit):
            ancestors = rx.ancestors(self.tree, other.tree_id) | {other.tree_id}
            return any(child.tree_id in ancestors for child in self.entities)
        return other.is_contained(self)
    

    def is_disjoint(self, other: Territory) -> bool:
        pass


    def __or__(self, other: Territory | TerritorialUnit) -> Territory:
        if not self.entities:
            entities = tuple()
        else:
            entities = self.entities
        if isinstance(other, TerritorialUnit):
            return Territory(*chain(entities, [other]))
        if other.entities is not None:
            return Territory(*chain(entities, other.entities))
        return self
    


    def __and__(self, other: Territory | TerritorialUnit) -> Territory:
        if isinstance(other, TerritorialUnit):
            return  Territory(*chain(*(self._and(child, other) for child in self.entities)))
        if (not other.entities) or (not self.entities):
            return Territory()
        if self in other:
            return self

        return Territory.union(*(self & child for child in other.entities))
     

    def __sub__(self, other: Territory | TerritorialUnit) -> Territory:
        if isinstance(other, TerritorialUnit):
            return Territory(*chain(*(self._sub(child, other) for child in self.entities)))
        if (not other.entities) or (not self.entities):
            return self
        if self in other:
            return Territory()

        return Territory.intersection(*(self - child for child in other.entities))


    def __repr__(self) -> str:
        if self.entities:
            return '|'.join(str(e) for e in self.entities)
        return '{}'


    def to_es_filter(self) -> list[str]:
        """Return the filter list to append to an ElasticSearch query to filter by this territory. 

        Returns:
            list[str]: Something like `[{"term" : {"tu_zone" : "DEP:69"}}, {"term" : {"tu_zone" : "COM:75023"}}]`
        """
        return [{"term" : {"tu_zone" : e.es_code}} for e in self.entities]
    

    def lowest_common_ancestor(self) -> Territory:
        """Return the lowest common ancestor of the territorial units of this territory.

        Details of this algorithm [here](https://networkx.org/nx-guides/content/algorithms/lca/LCA.html).
        Returns:
            Territory: A territory associated with a single territorial unit
        """
        return self.LCA(*self.entities)


    def ancestors(self, include_itself: bool = False) -> set[TerritorialUnit]:
        """Return a set of all ancestors of every territorial unit of this territory.

        Args:
            include_itself (bool, optional): Wether to include or not the node in its ancestors. Defaults to False.

        Returns:
            set[TerritorialUnit]: The union of all ancestors of every territorial unit of the territory.
        """
        ancestors = set.union(*(rx.ancestors(self.tree, e.tree_id) for e in self.entities))
        res = {self.tree.get_node_data(i) for i in ancestors}
        if include_itself:
            res = res | self.entities
        return res
