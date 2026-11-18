"""Fusion de plusieurs avis : le « vote moyen » (soft voting).

Pourquoi la moyenne et pas un modèle appris (stacking) ?
- sur la validation, le modèle images sépare parfaitement les classes : un modèle appris dessus
  donnerait tout le poids aux images et n'apprendrait rien des chiffres ;
- la moyenne n'a AUCUN paramètre appris : impossible de la sur-ajuster. Seul le seuil est choisi,
  sur la validation, comme pour les autres modèles.

Lecture : deux experts complémentaires. Quand l'un est sûr de lui et l'autre hésite, le plus sûr
l'emporte ; quand ils se contredisent franchement, le score tombe au milieu → zone « à surveiller ».
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def soft_vote(
    probabilities: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Moyenne (éventuellement pondérée) de probabilités de même taille."""
    if not probabilities:
        raise ValueError("au moins une série de probabilités est nécessaire")
    stacked = np.vstack([np.asarray(p, dtype=float) for p in probabilities])
    if weights is None:
        weights = [1.0] * len(probabilities)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(probabilities),) or np.any(w < 0) or w.sum() == 0:
        raise ValueError("un poids positif par série de probabilités")
    fused: np.ndarray = (w[:, None] * stacked).sum(axis=0) / w.sum()
    return fused
