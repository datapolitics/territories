from __future__ import annotations

import networkx as nx

from dataclasses import dataclass, field
from typing import Iterable, Optional
from functools import reduce
from itertools import chain
from enum import Enum

class Type(Enum):
    COMMUNE = 0
    EPCI = 1
    DEP = 2
    REGION = 4
    PAYS = 5

@dataclass(frozen=True)
class Entity:
    name: str
    atomic: bool = True
    type: Type = Type.COMMUNE
    geo_bound: None = None
    es_code: Optional[str] = None

    def __repr__(self) -> str:
        return self.name

    def contains(self, other, tree: nx.DiGraph) -> bool:
        return (self == other) or (self in nx.ancestors(tree, other))
    
        
    def __and__(self, other, tree):
        if other is None:
            return None
        if self in other:
            return self
        if self.is_disjoint(other):
            return None
        return reduce(lambda x, y: x | y, [other & child for child in self.entities])
    


class Territory:
    tree: Optional[nx.DiGraph] = None

    @classmethod
    def assign_tree(cls, tree):
        cls.tree = tree


    @classmethod
    def minimize(cls, node: Entity, items: Iterable[Entity]) -> set[Entity]:
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
    

    def empty(cls):
        return 


    def __init__(
            self,
            # tree: nx.DiGraph,
            entities: Optional[Iterable[Entity]] = None
            ) -> None:
        if self.tree is None:
            raise Exception('Tree is not initialized')
        if entities:
            entities = set(entities)
            root = next(n for n, d in self.tree.in_degree() if d==0)
            #  guarantee the Territory is always represented in minimal form
            self.entities = self.minimize(root, entities)
        else:
            self.entities = None
        # self.tree = tree


    def __eq__(self, value: Territory) -> bool:
        return self.entities == value.entities
    

    def __add__(self, other: Territory) -> Territory:
        return Territory(
            self.entities | other.entities
        )


    def __sub__(self, other: Territory | Entity) -> Territory:
        # if self in other
        # return {}

        # if self is disjoint
        # return self
        # incorrect
        if isinstance(other, Entity):
            other = Territory((other, ))
        if other.entities is None:
            return self
        for entity in other.entities:
            if entity == self:
                return Territory()
            if entity in self:
                # need to expand self here
                return Territory(entities=(child - entity for child in self.entities))
        return Territory(
            self.entities - other.entities
        )
    

    def is_contained(self, other: Territory | Entity) -> bool:
        for entity in self.entities:
            parents = nx.ancestors(self.tree, entity) | {entity}
            # print(parents)
            if isinstance(other, Entity):
                if other not in parents:
                    return False
            else:
                if not any(other_entity in parents for other_entity in other.entities):
                    return False
        return True
    

    def __contains__(self, other: Territory | Entity) -> bool:
        if isinstance(other, Entity):
            other = Territory((other, ))
        return other.is_contained(self)
    

    def is_disjoint(self, other: Territory) -> bool:
        pass


    def __or__(self, other: Territory | Entity) -> Territory:
        if self.entities is None:
            entities = tuple()
        else:
            entities = self.entities
        if isinstance(other, Entity):
            return Territory(chain(entities, [other]))
        if other.entities is not None:
            return Territory(chain(entities, other.entities))
        return self
    

    def __and__(self, other: Territory) -> Territory:
        # intersection is hard ?
        # cases
        # entity is contained in other -> return {entity}
        # entity is disjoint from other -> return {}
        # none of the above :=> return a & other for a in self.entities
        if self.entities is None:
            return self
        if isinstance(other, Entity):
            # print("instanciate")
            # here goes into infinite recursion 
            other = Territory(entities={other})
        # print(len(other.entities))
        if self in other:
            return self
        if self.is_disjoint(other):
            return Territory()
        if len(self.entities) == 1:
            if self.is_contained(other):
                return self
            else:
                return Territory()
        return reduce(lambda x, y: x | y, [other & child for child in self.entities])
        # return reduce(lambda x, y: x | y, [child for child in self.entities])


    def __repr__(self) -> str:
        return '|'.join(str(e) for e in self.entities)