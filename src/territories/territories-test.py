from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import date, datetime
from dataclasses import dataclass
from territories.doc_types import Type

from uuid import uuid4
from enum import Enum
from functools import reduce


class GeoBounds:

    def __init__(self, x_upper, x_lower, y_upper, y_lower) -> None:
        self.x_upper = x_upper
        self.x_lower = x_lower
        self.y_upper = y_upper
        self.y_lower = y_lower

    def connected(self, other) -> bool:
        x_invalid = (self.x_upper < other.x_lower) or (self.x_lower > other.x_upper)
        y_invalid = (self.y_upper < other.y_lower) or (self.y_lower > other.y_upper)
        return not(x_invalid or y_invalid)
    
    def __add__(self, other):
        return GeoBounds(
            x_upper=max(self.x_upper, other.x_upper),
            x_lower=min(self.x_lower, other.x_lower),
            y_upper=max(self.y_upper, other.y_upper),
            y_lower=min(self.y_lower, other.y_lower),
        )

    @classmethod
    def union(csl, *others):
        add = lambda x, y: x + y
        return reduce(add, others)




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
    """Assume there is a partition of any territory into atomic parts
    """

    def __init__(self) -> None:
        self.name = None
        self.id
        self.is_atomic
        self.n_children
        self.space_limits = GeoBounds
        self.parents: list[Territory] = []
        self.children: list[Territory] = []

    @classmethod
    def from_children(cls, children, name: Optional[str] = None):
        return cls(
            name,
            uuid4().int,
            False,
            1 + sum(child.n_children for child in children),
            GeoBounds,
            [],
            [child]
        )

    def maybe_connected_fast(self, other):
        return self.space_limits.connected(other.space_limits)

    def intersect(self, other):
        # for those method to work fast
        # we need a way to tell very quicly if two territories are connected or disjoint

        if self.maybe_connected_fast(self, other):
            for child in self.children:
                if self.intersect(child, other):
                    return True
        return False

    def intersection(self, *others):
        pass

    def __in__(self, other):
        if self.id == other.id:
            return True
        if self.parents is None:
            return False # this element is the universe
        for parent in self.parents:
            if parent in other:
                return True
        return False
    
    def lca(self, other):
        pass

    def __eq__(self, other):
        return self.id == other.id

    def __add__(self, other):
        # check if LCA is their union
        pass


    def union(self, other):
        return self + other

    def as_ES_filter(self):
        pass

if __name__ == "__main__":

    # atoms: level 0
    marseille = Territory()
    lyon = Territory()
    paris = Territory()
    bordeaux = Territory()
    nantes = Territory()
    lille = Territory()

    # level 1
    rhone_dep = Territory()
    metropole = Territory()
    paca = Territory.from_child(marseille)

    france = Territory()


    assert lyon in france
    assert lyon in rhone_dep
    assert lyon in metropole
    assert lyon.intersect(marseille) is False
    assert lyon in marseille is False

    # assert 
