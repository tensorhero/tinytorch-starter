"""Slice descriptor for NDArray indexing.

Analogous to Python's built-in slice, but explicit for tinynum's
NDArray.slice() method.
"""

from __future__ import annotations
from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class Slice:
    """Describes a slice range for one axis: [start, stop) with step."""

    start: int
    stop: int
    step: int = 1

    @staticmethod
    def of(start: int, stop: int, step: int = 1) -> "Slice":
        """Creates a slice [start, stop) with the given step."""
        return Slice(start, stop, step)

    @staticmethod
    def all() -> "Slice":
        """Selects all elements along the axis (equivalent to ':')."""
        return Slice(0, sys.maxsize, 1)
