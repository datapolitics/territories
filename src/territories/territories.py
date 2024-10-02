from __future__ import annotations

import networkx as nx

from typing import Iterable, Optional
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from enum import Enum


class Partition(Enum):
    COMMUNE = 0
    EPCI = 1
    DEP = 2
    REGION = 4
    PAYS = 5


@dataclass(frozen=True)
class Part:
    name: str
    atomic: bool = True
    partition_type: Partition = Partition.COMMUNE
    geo_bound: None = None
    es_code: Optional[str] = None

    def __repr__(self) -> str:
        return self.name

    def contains(self, other, tree: nx.DiGraph) -> bool:
        return (self == other) or (self in nx.ancestors(tree, other))

    

class Territory:
    tree: Optional[nx.DiGraph] = None

    @classmethod
    def assign_tree(cls, tree):
        cls.tree = tree


    @classmethod
    def minimize(cls, node: Part, items: Iterable[Part]) -> set[Part]:
        """evaluate complexity of this method

        Args:
            tree (nx.DiGraph): _description_
            node (Entity): _description_
            items (Iterable[Entity]): _description_

        Returns:
            set[Entity]: _description_
        """
        if len(items) == 0:
            return set()
        if node in items:
            return {node}
        # if len(items) == 1:
            # return {items[0]}
        children = set(cls.tree.successors(node))
        if children == set(items):
            return {node}

        gen = (cls.minimize(child, tuple(item for item in items if child.contains(item, cls.tree))) for child in children)
        # print(type(iter(gen)))
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
        if a in nx.ancestors(cls.tree, b):
            return set.union(*(cls._sub(child, b) for child in cls.tree.successors(a)))
        return {a}
    

    @classmethod
    def _and(cls, a: Part, b: Part) -> set[Part]:
        if a == b:
            return {a}
        # if a in b
        if a in nx.ancestors(cls.tree, b):
            return {b}
        # if b in a
        if b in nx.ancestors(cls.tree, a):
            return {a}
        return set()


    def __init__(self, *args: Iterable[Part]) -> None:
        if self.tree is None:
            raise Exception('Tree is not initialized')
        entities = set(args)
        if entities:
            root = next(n for n, d in self.tree.in_degree() if d==0)
            #  guarantee the Territory is always represented in minimal form
            self.entities = self.minimize(root, entities)
        else:
            self.entities = set()


    def __eq__(self, value: Territory) -> bool:
        # should also check for equality if ids
        # since some entities share the same territory but are not equal
        # ex : Parlement and ADEME both occupy France, yet are not the same entities
        return self.entities == value.entities


    def __add__(self, other: Territory) -> Territory:
        return Territory(
            *(self.entities | other.entities)
        )
    

    def is_contained(self, other: Territory | Part) -> bool:
        for entity in self.entities:
            parents = nx.ancestors(self.tree, entity) | {entity}
            if isinstance(other, Part):
                if other not in parents:
                    return False
            else:
                if not any(other_entity in parents for other_entity in other.entities):
                    return False
        return True
    

    def __contains__(self, other: Territory | Part) -> bool:
        if isinstance(other, Part):
            other = Territory(other)
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
            return Part()
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


    def to_es(self) -> list[str]:
        return [e.es_code for e in self.entities]