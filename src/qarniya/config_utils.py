"""Outils communs à toutes les configurations YAML du projet (données, modèle, entraînement)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from qarniya.domain.exceptions import QarniyaError

ModelT = TypeVar("ModelT", bound=BaseModel)


class ConfigError(QarniyaError):
    """Fichier de configuration absent ou invalide."""


class FrozenModel(BaseModel):
    """Base des configurations : immuable, et toute clé inconnue est une erreur
    (une faute de frappe dans le YAML ne doit jamais être ignorée en silence)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def load_yaml_model(cls: type[ModelT], path: str | Path) -> ModelT:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return cls.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Configuration invalide ({path}) :\n{exc}") from exc
