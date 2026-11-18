"""Registre des adaptateurs : le YAML donne un nom (`adapter: csv_images`), on trouve la classe.

Ajouter un adaptateur = écrire une classe décorée par `@register_adapter("nom")`.
Aucun `if/elif` à modifier ailleurs (principe ouvert/fermé).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from qarniya.data.adapters.base import DatasetAdapter
from qarniya.data.config import ConfigError, DatasetConfig

AdapterT = TypeVar("AdapterT", bound=type[DatasetAdapter])

_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(name: str) -> Callable[[AdapterT], AdapterT]:
    def decorator(cls: AdapterT) -> AdapterT:
        if name in _REGISTRY:
            raise ValueError(f"Adaptateur déjà enregistré : {name}")
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def build_adapter(config: DatasetConfig) -> DatasetAdapter:
    try:
        adapter_cls = _REGISTRY[config.adapter]
    except KeyError:
        raise ConfigError(
            f"Adaptateur inconnu '{config.adapter}'. Disponibles : {available_adapters()}"
        ) from None
    return adapter_cls(config)
