"""Métriques médicales d'une classification binaire (malade / non malade)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float  # qualité du classement, indépendante du seuil (0,5 = hasard, 1 = parfait)
    sensitivity: float  # part des malades détectés
    specificity: float  # part des sains reconnus comme sains
    accuracy: float
    threshold: float
    tp: int  # vrais positifs : malades détectés
    tn: int  # vrais négatifs : sains reconnus
    fp: int  # faux positifs : fausses alertes
    fn: int  # faux négatifs : malades MANQUÉS (l'erreur la plus grave)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def compute_binary_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> BinaryMetrics:
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    if truth.shape != prob.shape:
        raise ValueError("y_true et y_prob doivent avoir la même taille")
    pred = (prob >= threshold).astype(int)

    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    # AUC indéfinie si une seule classe est présente
    auc = float(roc_auc_score(truth, prob)) if len(set(truth.tolist())) == 2 else math.nan

    return BinaryMetrics(
        auc=auc,
        sensitivity=_ratio(tp, tp + fn),
        specificity=_ratio(tn, tn + fp),
        accuracy=_ratio(tp + tn, len(truth)),
        threshold=threshold,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )
