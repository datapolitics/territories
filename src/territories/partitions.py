from __future__ import annotations

import json_fix

import rustworkx as rx

from enum import Enum
from typing import Optional
from dataclasses import dataclass, field, asdict


class Partition(Enum):
    ARR = 0
    COM = 1
    DEP = 2
    REG = 3
    CNTRY = 4
    UE = 5

    def __str__(self) -> str:
        return self.name
    
    def __json__(self):
        return self.name

    def __lt__(self, other) -> bool:
        if isinstance(other, Partition):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other) -> bool:
        if isinstance(other, Partition):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, Partition):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other) -> bool:
        if isinstance(other, Partition):
            return self.value >= other.value
        return NotImplemented



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
    tu_id: str
    atomic: bool = True
    partition_type: Partition = Partition.COM
    postal_code: Optional[str] = None
    tree_id: Optional[int] = field(default=None, compare=False)


    def __repr__(self) -> str:
        return self.name


    def __lt__(self, other: TerritorialUnit) -> bool:
        if self.partition_type.value == other.partition_type.value:
            return self.name >= other.name
        return self.partition_type.value >= other.partition_type.value


    def contains(self, other, tree: rx.PyDiGraph) -> bool:
        assert self.tree_id
        assert other.tree_id
        return (self == other) or (self.tree_id in rx.ancestors(tree, other.tree_id))
    

    def to_dict(self):
        return asdict(self)
    

    def __json__(self):
        dict_repr = asdict(self)
        dict_repr.pop('tree_id')
        return dict_repr