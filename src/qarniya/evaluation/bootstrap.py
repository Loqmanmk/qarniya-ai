"""Intervalles de confiance par bootstrap, AU NIVEAU PATIENT.

Un chiffre seul (« 96 % de sensibilité ») cache l'incertitude : sur 90 malades, un seul œil de plus
ou de moins change le résultat d'1 point. Le bootstrap rejoue l'évaluation sur des milliers de
tirages du jeu de test et donne une fourchette honnête : « 96 % [91 – 99] ».

On tire des PATIENTS (avec leurs deux yeux), pas des yeux : les deux yeux d'un patient ne sont pas
indépendants, les tirer séparément donnerait une fourchette faussement étroite.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qarniya.evaluation.metrics import compute_binary_metrics


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high}


def bootstrap_confidence_intervals(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    groups: Sequence[str],
    threshold: float,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    group_ids = np.asarray(groups)
    if not len(truth) == len(prob) == len(group_ids):
        raise ValueError("y_true, y_prob et groups doivent avoir la même taille")

    unique_groups = np.unique(group_ids)
    index_by_group = {g: np.flatnonzero(group_ids == g) for g in unique_groups}
    rng = np.random.default_rng(seed)

    samples: dict[str, list[float]] = {"auc": [], "sensitivity": [], "specificity": []}
    for _ in range(n_resamples):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in drawn])
        m = compute_binary_metrics(truth[idx].tolist(), prob[idx].tolist(), threshold)
        for name in samples:
            value = getattr(m, name)
            if not math.isnan(value):  # un tirage sans malade n'a pas de sensibilité
                samples[name].append(value)

    alpha = (1 - level) / 2
    return {
        name: ConfidenceInterval(
            float(np.quantile(values, alpha)) if values else math.nan,
            float(np.quantile(values, 1 - alpha)) if values else math.nan,
        )
        for name, values in samples.items()
    }
