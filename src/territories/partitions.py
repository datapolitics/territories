from __future__ import annotations

import json_fix

import rustworkx as rx

from enum import Enum
from typing import Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field, asdict


class Partition(Enum):
    """Represents the different levels of territorial units.
    They can be ordered by their respective scales.

    I am not conviced this is a good idea to have an empty partition.
    Maybe we should force the user to check if the territory is empty before extracting it's level.
    """
    EMPTY = 0 # maybe this is not a good idea
    ARR = 1
    COM = 2
    DEP = 3
    REG = 4
    CNTRY = 5
    UE = 6

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


@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...
    
    @property
    def label(self) -> str: ...
    
    @property
    def level(self) -> str: ...
    
    @property
    def parent_id(self) -> Optional[str]: ...
    
    # @property
    # def postal_code(self) -> Optional[str]: ...
    
    # @property
    # def inhabitants(self) -> Optional[int]: ...

    



@dataclass(frozen=True)
class TerritorialUnit:
    """A known territory, such as a city, a departement or a region.
    """
    name: str
    tu_id: str
    atomic: bool = True
    level: Partition = Partition.COM
    postal_code: Optional[str] = None
    inhabitants: Optional[int] = None
    tree_id: Optional[int] = field(default=None, compare=False)


    def __repr__(self) -> str:
        return self.name


    def __lt__(self, other: TerritorialUnit) -> bool:
        if self.level.value == other.level.value:
            return self.name >= other.name
        return self.level.value >= other.level.value


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