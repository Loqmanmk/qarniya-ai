"""Registre des modèles : la configuration donne un nom, le registre construit le réseau.

Ajouter une architecture = une fonction décorée par `@register_model("nom")`.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import Field
from torch import nn

from qarniya.config_utils import ConfigError, FrozenModel
from qarniya.models.multimap import MultiMapClassifier


class ModelSpec(FrozenModel):
    """Description d'un modèle, sauvegardée avec ses poids pour pouvoir le reconstruire."""

    name: str = "multimap"
    backbone: str = "resnet18"
    pretrained: bool = True
    hidden_dim: int = Field(default=128, ge=8)
    dropout: float = Field(default=0.3, ge=0, lt=1)


ModelBuilder = Callable[[ModelSpec], nn.Module]
_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Modèle déjà enregistré : {name}")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(spec: ModelSpec) -> nn.Module:
    try:
        builder = _REGISTRY[spec.name]
    except KeyError:
        raise ConfigError(
            f"Modèle inconnu '{spec.name}'. Disponibles : {available_models()}"
        ) from None
    try:
        return builder(spec)
    except ValueError as exc:  # ex. backbone inconnu
        raise ConfigError(str(exc)) from exc


@register_model("multimap")
def _build_multimap(spec: ModelSpec) -> nn.Module:
    return MultiMapClassifier(
        backbone=spec.backbone,
        pretrained=spec.pretrained,
        hidden_dim=spec.hidden_dim,
        dropout=spec.dropout,
    )
