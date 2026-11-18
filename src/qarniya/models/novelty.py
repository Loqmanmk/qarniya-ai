"""Détecteur d'images INHABITUELLES : « je ne sais pas » plutôt qu'un score au hasard.

Un réseau de neurones donne TOUJOURS un score, même pour une image qui ne ressemble à rien de ce
qu'il a appris (autre appareil, autre échelle de couleurs, mauvaise carte…). Ce score n'a alors
aucune valeur. On détecte ces cas ainsi :

1. chaque œil d'entraînement a une « empreinte » (nombres calculés par le réseau avant décision) ;
2. pour un nouvel œil, on mesure la distance aux k empreintes d'entraînement les plus proches ;
3. si elle dépasse celle de 99 % des yeux de VALIDATION (des yeux normaux, mais jamais vus),
   l'image est déclarée inhabituelle.

Sauvegarde en .npz (tableaux de nombres uniquement, `allow_pickle=False` : aucun code exécutable).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qarniya.domain.exceptions import QarniyaError


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized: np.ndarray = x / np.maximum(norms, 1e-12)
    return normalized


@dataclass(frozen=True)
class NoveltyDetector:
    reference: np.ndarray  # empreintes d'entraînement normalisées (n, d)
    threshold: float  # distance au-delà de laquelle une image est inhabituelle
    k: int = 5

    @classmethod
    def fit(
        cls,
        train_embeddings: np.ndarray,
        calibration_embeddings: np.ndarray,
        k: int = 5,
        quantile: float = 0.99,
    ) -> NoveltyDetector:
        if len(train_embeddings) < k:
            raise ValueError(f"au moins {k} examens d'entraînement sont nécessaires")
        if not 0 < quantile < 1:
            raise ValueError("quantile doit être dans ]0, 1[")
        detector = cls(_normalize(train_embeddings), threshold=np.inf, k=k)
        distances = detector.distances(calibration_embeddings)
        return cls(detector.reference, float(np.quantile(distances, quantile)), k)

    def distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Distance cosinus moyenne aux k plus proches voisins d'entraînement (0 = identique)."""
        similarity = _normalize(embeddings) @ self.reference.T  # (m, n)
        k = min(self.k, similarity.shape[1])
        nearest = np.sort(similarity, axis=1)[:, -k:]
        result: np.ndarray = 1.0 - nearest.mean(axis=1)
        return result

    def is_unusual(self, embeddings: np.ndarray) -> np.ndarray:
        result: np.ndarray = self.distances(embeddings) > self.threshold
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            reference=self.reference.astype(np.float16),  # 2x plus léger, précision suffisante
            threshold=np.array(self.threshold),
            k=np.array(self.k),
        )

    @classmethod
    def load(cls, path: Path) -> NoveltyDetector:
        if not path.is_file():
            raise QarniyaError(f"Détecteur introuvable : {path} (lancez `qarniya fit-novelty`)")
        with np.load(path, allow_pickle=False) as data:
            return cls(
                reference=data["reference"].astype(np.float32),
                threshold=float(data["threshold"]),
                k=int(data["k"]),
            )
