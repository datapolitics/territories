from __future__ import annotations
import pickle

import rustworkx as rx


from typing import Iterable, Optional, Callable
from perfect_hash import generate_hash, Format
from more_itertools import batched
from functools import reduce
from itertools import chain

from territories.partitions import Part, Partition, Node


class Territory:
    tree: Optional[rx.DiGraph] = None
    root_index: Optional[int] = None
    perfect_hash_fct: Optional[Callable[[str], int]] = None


    @staticmethod
    def create_hash_function(names: list[str]) -> Callable[[str], int]:
        # create perfect hash table
        f1, f2, G = generate_hash(names)
        G = tuple(G) # lookup is faster on a tuple

        fmt = Format()
        NG = len(G)
        NS = len(f1.salt)
        S1 = fmt(f1.salt)
        S2 = fmt(f2.salt)

        def hash_f(key, T):
            return sum(ord(T[i % NS]) * ord(c) for i, c in enumerate(key)) % NG

        def perfect_hash(key):
            return (G[hash_f(key, S1)] + G[hash_f(key, S2)]) % NG

        return perfect_hash


    @staticmethod
    def to_part(node: Node) -> Part:
        match node.level:
            case "COM":
                partition = Partition.COMMUNE
            case "DEP":
                partition = Partition.DEP
            case "REG":
                partition = Partition.REGION
            case "CNTRY":
                partition = Partition.COUNTRY
            case _:
                partition = None

        return Part(
            name=node.label,
            es_code=node.id,
            partition_type=partition
        )


    @classmethod
    def reset(cls):
        # if tree is a reference to a foreign object
        # I do not want to assign None to it
        # so first I destroy the reference
        del cls.tree
        cls.tree = None
        cls.perfect_hash_fct = None
        cls.root_index = None


    @classmethod
    def load_tree(cls, file: pickle._ReadableFileobj):
        cls.reset()
        cls.tree = pickle.load(file)
        cls.root_index = next(i for i in cls.tree.node_indices() if cls.tree.in_degree(i) == 0)


    @classmethod
    def build_tree(cls, data_stream: Iterable[Node]):
        cls.reset()
        
        tree = rx.PyDiGraph()
        mapper = {}
        orphans: list[Part] = []
        batch_size = 1024
        for batch in batched(data_stream, batch_size):
            entities_indices = tree.add_nodes_from(tuple(cls.to_part(node) for node in batch))

            for node, tree_idx in zip(batch, entities_indices):
                if node.level != Partition.COMMUNE: # communes don't have any children
                    mapper[node.id] = tree_idx
                object.__setattr__(tree.get_node_data(tree_idx), 'tree_id', tree_idx)

            edges = []
            for node, tree_id in zip(batch, entities_indices):
                if node.parent_id is None:
                    pass # root node
                if node.parent_id in mapper:
                    edges.append((mapper[node.parent_id], tree_id, None))
                else:
                    object.__setattr__(node, 'tree_id', tree_id)
                    orphans.append(node)
            tree.add_edges_from(edges)


        edges = tuple((mapper[node.parent_id], orphan.tree_id, None) for orphan in orphans if orphan.parent_id in mapper)
        tree.add_edges_from(edges)

        orphans = tuple(orphan for orphan in orphans if orphan.parent_id not in mapper)
        print(f"{len(orphans)} elements where not added to the tree because they have no parents")


        names: list[Part] = [tree.get_node_data(i).es_code for i in tree.node_indices()]
        print(f"there are {len(names)} element in the tree")

        # perfect_hash = cls.create_hash_function(names)

        # for name in names:
        #     i = perfect_hash(name)
        #     assert name == tree.get_node_data(i).es_code

        cls.tree = tree
        # cls.perfect_hash_fct = perfect_hash  
        cls.root_index = next(i for i in tree.node_indices() if tree.in_degree(i) == 0)


    @classmethod
    def assign_tree(cls, tree):
        cls.reset()

        elements: list[Part] = [tree.get_node_data(i) for i in tree.node_indices()]
        for i, e in enumerate(elements):
            object.__setattr__(e, 'tree_id', i)

        names = [e.name for e in elements]
        perfect_hash = cls.create_hash_function(names)
        for name in names:
            i = perfect_hash(name)
            assert name == tree.get_node_data(i).name

        cls.tree = tree
        cls.perfect_hash_fct = perfect_hash        
        cls.root_index = next(i for i in tree.node_indices() if tree.in_degree(i) == 0)


    @classmethod
    def hash(cls, name: str) -> int:
        return cls.perfect_hash_fct(name)


    @staticmethod
    def contains(a: int, b: int, tree: rx.PyDiGraph) -> bool:
        return (a == b) or (a in rx.ancestors(tree, b))


    @classmethod
    def minimize(cls, node: int, items: Iterable[int]) -> set[int]:
        """evaluate complexity of this method

        Args:
            node (Entity): _description_
            items (Iterable[Entity]): _description_

        Returns:
            set[Entity]: _description_
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

        if union == children:
            return {node}
        return union
    

    @classmethod
    def union(csl, *others):
        return reduce(lambda x, y: x + y, iter(others))


    @classmethod
    def intersection(csl, *others):
        return reduce(lambda x, y: x & y, iter(others))


    @classmethod
    def _sub(cls, a: Part, b: Part) -> set[Part]:
        if a == b:
            return set()
        if a.tree_id in rx.ancestors(cls.tree, b.tree_id):
            children = cls.tree.successors(a.tree_id)
            return set.union(*(cls._sub(child, b) for child in children))
        return {a}
    

    @classmethod
    def _and(cls, a: Part, b: Part) -> set[Part]:
        if a == b:
            return {a}
        if a.tree_id in rx.ancestors(cls.tree, b.tree_id): # if a in b
            return {b}
        if b.tree_id in rx.ancestors(cls.tree, a.tree_id): # if b in a
            return {a}
        return set()


    @classmethod
    def from_name(cls, *args: Iterable[str]):
        entities_idxs = [cls.hash(name) for name in args]
        entities = {cls.tree.get_node_data(i) for i in cls.minimize(cls.root_index, entities_idxs)} # useless
        return Territory(*entities)
    

    def __init__(self, *args: Iterable[Part]) -> None:
        if self.tree is None:
            raise Exception('Tree is not initialized')
        entities = set(args)
        if entities:
            entities_idxs = [self.hash(e.name) for e in entities]
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
        print(self, other)
        if self == other:
            return True
        for entity in self.entities:
            ancestors = rx.ancestors(self.tree, entity.tree_id) | {entity.tree_id}
            if not any(other_entity.tree_id in ancestors for other_entity in other.entities):
                return False
        return True
    

    def __contains__(self, other: Territory | Part) -> bool:
        if isinstance(other, Part):
            ancestors = rx.ancestors(self.tree, other.tree_id) | {other.tree_id}
            return any(child.tree_id in ancestors for child in self.entities)
        return other.is_contained(self)
    

    def is_disjoint(self, other: Territory) -> bool:
        pass


    def __or__(self, other: Territory | Part) -> Territory:
        if not self.entities:
            entities = tuple()
        else:
            entities = self.entities
        if isinstance(other, Part):
            return Territory(*chain(entities, [other]))
        if other.entities is not None:
            return Territory(*chain(entities, other.entities))
        return self
    


    def __and__(self, other: Territory | Part) -> Territory:
        if isinstance(other, Part):
            return  Territory(*chain(*(self._and(child, other) for child in self.entities)))
        if (not other.entities) or (not self.entities):
            return Territory()
        if self in other:
            return self

        return Territory.union(*(self & child for child in other.entities))
     

    def __sub__(self, other: Territory | Part) -> Territory:
        if isinstance(other, Part):
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
        return [{"term" : {"tu_zone" : e.es_code}} for e in self.entities]