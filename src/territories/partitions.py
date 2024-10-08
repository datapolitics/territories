import rustworkx as rx

from enum import Enum
from typing import Optional
from functools import reduce
from dataclasses import dataclass, field


class Partition(Enum):
    COM = 0
    DEP = 1
    REG = 2
    CNTRY = 3

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    level: str
    parent_id: Optional[str] = None
    postal_code: Optional[str] = None
    tree_id: Optional[int] = None



@dataclass(frozen=True)
class TerritorialUnit:
    """A known territory, such as a city, a departement or a region.
    """
    name: str
    atomic: bool = True
    partition_type: Partition = Partition.COM
    es_code: Optional[str] = None
    postal_code: Optional[str] = None
    tree_id: Optional[int] = field(default=None, compare=False)


    def __repr__(self) -> str:
        return self.name


    def contains(self, other, tree: rx.PyDiGraph) -> bool:
        assert self.tree_id
        assert other.tree_id
        return (self == other) or (self.tree_id in rx.ancestors(tree, other.tree_id))
    
        
    def __and__(self, other):
        if other is None:
            return None
        if self in other:
            return self
        if self.is_disjoint(other):
            return None
        return reduce(lambda x, y: x | y, [other & child for child in self.entities])