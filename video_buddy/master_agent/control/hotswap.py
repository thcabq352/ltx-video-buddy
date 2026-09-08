"""Keep the last two variants warm for sub-second swaps on 16GB."""

from __future__ import annotations

from collections import OrderedDict


class VariantCache:
    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._warm: OrderedDict[str, str] = OrderedDict()

    def touch(self, variant: str, path: str = "") -> list[str]:
        key = (variant or "").strip().lower()
        if key in self._warm:
            self._warm.move_to_end(key)
        self._warm[key] = path
        while len(self._warm) > self.capacity:
            self._warm.popitem(last=False)
        return self.warm()

    def warm(self) -> list[str]:
        return list(self._warm.keys())

    def evicted(self, variant: str) -> bool:
        return (variant or "").strip().lower() not in self._warm
