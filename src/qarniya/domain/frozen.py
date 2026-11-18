"""Dictionnaire en lecture seule ET transportable entre processus.

`types.MappingProxyType` est en lecture seule mais ne peut pas être copié par `pickle` ; or PyTorch
copie les données vers ses processus de chargement (indispensable sous Windows). D'où cette classe.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class FrozenMapping(Mapping[K, V], Generic[K, V]):
    __slots__ = ("_data",)

    def __init__(self, data: Mapping[K, V] | None = None) -> None:
        self._data: dict[K, V] = dict(data or {})

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __hash__(self) -> int:
        return hash(frozenset(self._data.items()))
