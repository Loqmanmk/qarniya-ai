"""Grad-CAM : OÙ le modèle a-t-il regardé pour décider ?

Principe : on mesure à quel point chaque zone de la dernière couche convolutive a fait MONTER
le score de kératocône (gradient × activation). Les zones qui ont pesé apparaissent en couleur.

En plus de la carte de chaleur, on indique la PART de chacune des 4 cartes dans la décision :
« la carte d'élévation postérieure a compté pour 45 % ». C'est ce que le médecin veut savoir.

Une carte de chaleur est une aide à la lecture, PAS une preuve : elle montre ce qui a influencé
le modèle, pas ce qui est médicalement vrai.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from qarniya.domain import EyeSide, MapType


@dataclass(frozen=True)
class Explanation:
    probability: float
    # valeurs 0–1, même échelle pour les 4 cartes, dans le sens de l'image d'origine
    heatmaps: dict[MapType, np.ndarray]
    map_importance: dict[MapType, float]  # part de chaque carte dans la décision (somme = 1)

    @property
    def main_map(self) -> MapType:
        return max(self.map_importance, key=lambda m: self.map_importance[m])


class GradCam:
    def __init__(self, model: nn.Module) -> None:
        layer = getattr(model, "explain_layer", None)
        if not isinstance(layer, nn.Module):
            raise TypeError("le modèle doit exposer `explain_layer` (sa dernière couche conv.)")
        self.model = model
        self.layer = layer

    def explain(self, x: torch.Tensor, eye: EyeSide, mirrored_left_eye: bool = True) -> Explanation:
        """x : (1, 4, 3, H, W) préparé par le Preprocessor (œil gauche en miroir)."""
        if x.shape[0] != 1:
            raise ValueError("Grad-CAM s'applique à un seul œil à la fois")
        captured: list[torch.Tensor] = []
        handle = self.layer.register_forward_hook(lambda _m, _i, out: captured.append(out))
        self.model.eval()
        try:
            with torch.enable_grad():
                logit = self.model(x).sum()
                activations = captured[0]  # (4 cartes, canaux, h, w)
                (gradients,) = torch.autograd.grad(logit, activations)
        finally:
            handle.remove()

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # importance de chaque canal
        cams = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()  # (4, h, w)
        total = float(cams.sum())
        importance = cams.reshape(len(cams), -1).sum(axis=1) / total if total > 0 else None
        cams = cams / cams.max() if cams.max() > 0 else cams  # même échelle pour les 4 cartes

        if eye is EyeSide.LEFT and mirrored_left_eye:
            cams = cams[:, :, ::-1]  # on remet la chaleur dans le sens de l'image d'origine

        maps = list(MapType)
        return Explanation(
            probability=float(torch.sigmoid(logit.detach()).item()),
            heatmaps={m: np.ascontiguousarray(cams[i]) for i, m in enumerate(maps)},
            map_importance={
                m: float(importance[i]) if importance is not None else 1 / len(maps)
                for i, m in enumerate(maps)
            },
        )
