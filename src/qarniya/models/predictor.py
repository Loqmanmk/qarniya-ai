"""Prédiction : un modèle sauvegardé + des examens → probabilités de kératocône.

Utilisé par l'évaluation (étape 5) et par les services / l'API (étape 7) : une seule implémentation,
avec la préparation d'images enregistrée DANS le modèle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from qarniya.domain import CornealExam
from qarniya.models.checkpoint import LoadedModel, load_checkpoint
from qarniya.models.preprocessing import Preprocessor


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _InferenceDataset(Dataset[torch.Tensor]):
    def __init__(self, exams: Sequence[CornealExam], preprocessor: Preprocessor) -> None:
        self.exams = list(exams)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.exams)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.preprocessor.to_tensor(self.exams[index])


class Predictor:
    def __init__(
        self,
        loaded: LoadedModel,
        device: torch.device | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
    ) -> None:
        self.loaded = loaded
        self.device = device or default_device()
        self.model = loaded.model.to(self.device).eval()
        self.preprocessor = Preprocessor(loaded.preprocessing)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def from_checkpoint(cls, path: Path, device: torch.device | None = None) -> Predictor:
        device = device or default_device()
        return cls(load_checkpoint(path, device), device)

    @torch.no_grad()
    def _run(
        self, exams: Sequence[CornealExam], step: Callable[[torch.Tensor], torch.Tensor]
    ) -> np.ndarray:
        loader = DataLoader(
            _InferenceDataset(exams, self.preprocessor),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return torch.cat([step(x.to(self.device)).float().cpu() for x in loader]).numpy()

    def predict(self, exams: Sequence[CornealExam]) -> np.ndarray:
        """Probabilité de kératocône (0 à 1) pour chaque examen, dans le même ordre."""
        if not exams:
            return np.array([], dtype=float)
        return self._run(exams, lambda x: torch.sigmoid(self.model(x))).astype(float)

    def _embed_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        embed_fn = getattr(self.model, "embed", None)
        if not callable(embed_fn):
            raise TypeError("ce modèle ne fournit pas d'empreinte (`embed`)")
        return cast(Callable[[torch.Tensor], torch.Tensor], embed_fn)

    def embed(self, exams: Sequence[CornealExam]) -> np.ndarray:
        """Empreintes (n, d) des examens, pour le détecteur d'images inhabituelles."""
        return self._run(exams, self._embed_fn())

    @torch.no_grad()
    def embed_tensor(self, x: torch.Tensor) -> np.ndarray:
        """Empreinte d'un tenseur déjà préparé (1, 4, 3, H, W)."""
        result: np.ndarray = self._embed_fn()(x.to(self.device)).float().cpu().numpy()
        return result
