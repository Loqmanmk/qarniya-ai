"""Choix du seuil de décision, sur la VALIDATION uniquement, puis appliqué tel quel au test.

Règle médicale de dépistage : d'abord garantir la sensibilité visée (ne pas manquer de kératocône),
ensuite maximiser la spécificité (limiter les fausses alertes).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def threshold_for_sensitivity(
    y_true: Sequence[int], y_prob: Sequence[float], target_sensitivity: float = 0.95
) -> float:
    """Seuil qui atteint `target_sensitivity` avec la meilleure spécificité possible.

    Les seuils candidats sont les MILIEUX entre scores consécutifs : le seuil retenu tombe au
    milieu d'un « trou » entre deux scores, ce qui le rend moins fragile sur de nouveaux yeux.
    À spécificité égale, on garde le seuil le plus bas (le plus prudent pour le dépistage).
    """
    if not 0 < target_sensitivity <= 1:
        raise ValueError("target_sensitivity doit être dans ]0, 1]")
    truth = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_prob, dtype=float)
    n_pos, n_neg = int(truth.sum()), int((truth == 0).sum())
    if n_pos == 0:
        raise ValueError("aucun cas positif : impossible de régler la sensibilité")

    scores = np.unique(prob)
    candidates = np.concatenate(([0.0], (scores[:-1] + scores[1:]) / 2, [np.nextafter(1.0, 2)]))

    best: tuple[float, float, float] | None = None  # (spécificité, sensibilité, -seuil)
    for threshold in candidates:
        pred = prob >= threshold
        sensitivity = (pred & (truth == 1)).sum() / n_pos
        if sensitivity < target_sensitivity:
            continue
        specificity = (~pred & (truth == 0)).sum() / n_neg if n_neg else 1.0
        key = (float(specificity), float(sensitivity), -float(threshold))
        if best is None or key > best:
            best = key
    if best is None:  # impossible : le seuil 0 donne toujours 100 % de sensibilité
        raise RuntimeError("aucun seuil candidat n'atteint la sensibilité visée")
    return -best[2]
