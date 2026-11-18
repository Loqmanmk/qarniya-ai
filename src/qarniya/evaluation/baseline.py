"""Modèle de référence (baseline) : une régression logistique sur les CHIFFRES de l'appareil
(Kmax, épaisseur, astigmatisme…), sans regarder les images.

Question à laquelle il répond : « les images apportent-elles quelque chose de plus que les chiffres
que le médecin a déjà ? » Si notre réseau ne fait pas mieux, il ne sert à rien.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from qarniya.domain import CornealExam

DEFAULT_FEATURES = (
    "kmax",
    "pachymetry_thinnest",
    "pachymetry_central",
    "astigmatism",
    "asphericity_anterior",
    "asphericity_posterior",
)


class TabularBaseline:
    def __init__(self, features: Sequence[str] = DEFAULT_FEATURES) -> None:
        self.features = tuple(features)
        # valeur manquante → médiane de l'entraînement ; mise à l'échelle ; régression logistique
        self._pipeline: Pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

    def _matrix(self, exams: Sequence[CornealExam]) -> np.ndarray:
        return np.array(
            [[e.clinical.get(f, np.nan) for f in self.features] for e in exams], dtype=float
        )

    def available_features(self, exams: Sequence[CornealExam]) -> tuple[str, ...]:
        return tuple(f for f in self.features if any(f in e.clinical for e in exams))

    def fit(self, exams: Sequence[CornealExam], targets: Sequence[int]) -> TabularBaseline:
        self.features = self.available_features(exams)
        if not self.features:
            raise ValueError("aucune variable clinique disponible pour la baseline")
        self._pipeline.fit(self._matrix(exams), np.asarray(targets, dtype=int))
        return self

    def predict(self, exams: Sequence[CornealExam]) -> np.ndarray:
        probabilities: np.ndarray = self._pipeline.predict_proba(self._matrix(exams))[:, 1]
        return probabilities
