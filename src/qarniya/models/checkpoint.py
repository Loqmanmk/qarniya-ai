"""Sauvegarde et rechargement d'un modèle entraîné.

Un fichier .pt contient les poids ET tout ce qu'il faut pour s'en servir correctement :
architecture, préparation des images, seuils, version du logiciel, commit Git, métriques.
→ Un modèle n'est jamais utilisé avec une mauvaise préparation d'image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from qarniya import __version__
from qarniya.domain.exceptions import QarniyaError
from qarniya.models.preprocessing import PreprocessingSpec
from qarniya.models.registry import ModelSpec, build_model

FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    model_spec: ModelSpec
    preprocessing: PreprocessingSpec
    metadata: dict[str, Any]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_spec: ModelSpec,
    preprocessing: PreprocessingSpec,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "qarniya_version": __version__,
            "model_spec": model_spec.model_dump(),
            "preprocessing": preprocessing.model_dump(),
            "metadata": metadata or {},
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> LoadedModel:
    if not path.is_file():
        raise QarniyaError(f"Modèle introuvable : {path}")
    # weights_only=True : on refuse d'exécuter du code caché dans le fichier (sécurité)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != FORMAT_VERSION:
        raise QarniyaError(f"Format de modèle non supporté : {payload.get('format_version')}")

    # les poids viennent du fichier : inutile de retélécharger les poids ImageNet
    spec = ModelSpec.model_validate(payload["model_spec"] | {"pretrained": False})
    model = build_model(spec)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return LoadedModel(
        model=model,
        model_spec=ModelSpec.model_validate(payload["model_spec"]),
        preprocessing=PreprocessingSpec.model_validate(payload["preprocessing"]),
        metadata=dict(payload["metadata"]),
    )
