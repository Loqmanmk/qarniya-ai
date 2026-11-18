"""Deuxième avis « chiffres » : Kmax et épaisseur minimale, deux valeurs que le médecin a déjà.

Régression logistique entraînée sur le jeu d'ENTRAÎNEMENT, sauvegardée en JSON (des nombres
uniquement : rien d'exécutable, lisible par un humain, versionnable).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from qarniya.domain import CornealExam
from qarniya.domain.exceptions import QarniyaError

CLINICAL_FILE = "clinical_opinion.json"
FEATURES = ("kmax", "pachymetry_thinnest")
LABELS = {"kmax": "Kmax (D)", "pachymetry_thinnest": "épaisseur minimale (µm)"}
PLAUSIBLE = {"kmax": (30.0, 90.0), "pachymetry_thinnest": (150.0, 800.0)}  # garde-fou de saisie


@dataclass(frozen=True)
class ClinicalOpinionModel:
    features: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    @classmethod
    def fit(cls, exams: Sequence[CornealExam], targets: Sequence[int]) -> ClinicalOpinionModel:
        rows = [(e, t) for e, t in zip(exams, targets, strict=True) if _has_all(e)]
        if len({t for _, t in rows}) < 2:
            raise QarniyaError("pas assez d'examens avec Kmax et épaisseur pour l'avis chiffres")
        x = np.array([[e.clinical[f] for f in FEATURES] for e, _ in rows], dtype=float)
        y = np.array([t for _, t in rows], dtype=int)
        means, scales = x.mean(axis=0), x.std(axis=0)
        scales[scales == 0] = 1.0
        model = LogisticRegression(max_iter=1000).fit((x - means) / scales, y)
        return cls(
            features=FEATURES,
            means=tuple(float(v) for v in means),
            scales=tuple(float(v) for v in scales),
            coefficients=tuple(float(v) for v in model.coef_[0]),
            intercept=float(model.intercept_[0]),
        )

    def probability(self, values: dict[str, float]) -> float:
        missing = [f for f in self.features if f not in values]
        if missing:
            raise QarniyaError(f"valeurs manquantes pour l'avis chiffres : {missing}")
        validate_clinical_values(values)
        z = self.intercept + sum(
            c * (values[f] - m) / s
            for f, m, s, c in zip(
                self.features, self.means, self.scales, self.coefficients, strict=True
            )
        )
        return 1.0 / (1.0 + math.exp(-z))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ClinicalOpinionModel:
        if not path.is_file():
            raise QarniyaError(f"Avis chiffres introuvable : {path} (lancez `qarniya prepare`)")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            features=tuple(raw["features"]),
            means=tuple(raw["means"]),
            scales=tuple(raw["scales"]),
            coefficients=tuple(raw["coefficients"]),
            intercept=float(raw["intercept"]),
        )


def validate_clinical_values(values: dict[str, float]) -> None:
    """Refuse les saisies impossibles (faute de frappe : 4.5 au lieu de 45, µm au lieu de mm…)."""
    for name, value in values.items():
        low, high = PLAUSIBLE.get(name, (-math.inf, math.inf))
        if not low <= value <= high:
            raise QarniyaError(
                f"{LABELS.get(name, name)} = {value} : hors des valeurs plausibles "
                f"[{low:g} – {high:g}]"
            )


def _has_all(exam: CornealExam) -> bool:
    return all(f in exam.clinical for f in FEATURES)
