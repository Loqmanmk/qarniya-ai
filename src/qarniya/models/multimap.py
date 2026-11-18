"""Réseau « multi-cartes » : un même extracteur d'images lit chacune des 4 cartes,
puis une petite tête fusionne les 4 descripteurs en un score de risque.

Pourquoi un extracteur PARTAGÉ (mêmes poids pour les 4 cartes) ?
4 fois moins de paramètres → moins de surapprentissage avec ~900 yeux d'entraînement.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models

# fonction(pretrained) -> (tronc convolutif, nombre de canaux en sortie)
BackboneFactory = Callable[[bool], tuple[nn.Module, int]]


def _resnet(builder: Callable[..., nn.Module], weights: object, dim: int) -> BackboneFactory:
    def make(pretrained: bool) -> tuple[nn.Module, int]:
        net = builder(weights=weights if pretrained else None)
        trunk = nn.Sequential(*list(net.children())[:-2])  # sans pooling final ni couche fc
        return trunk, dim

    return make


def _efficientnet_b0(pretrained: bool) -> tuple[nn.Module, int]:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    return models.efficientnet_b0(weights=weights).features, 1280


BACKBONES: dict[str, BackboneFactory] = {
    "resnet18": _resnet(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": _resnet(models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
    "efficientnet_b0": _efficientnet_b0,
}


class MultiMapClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        n_maps: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Backbone inconnu '{backbone}'. Disponibles : {sorted(BACKBONES)}")
        self.features, feature_dim = BACKBONES[backbone](pretrained)
        self.n_maps = n_maps
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim * n_maps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def explain_layer(self) -> nn.Module:
        """Dernière couche convolutive : utilisée par Grad-CAM pour la carte de chaleur."""
        return self.features

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Empreinte d'un œil : (lot, 4 x canaux) nombres résumant les 4 cartes, avant la décision.
        Sert à repérer les images inhabituelles (cf. models/novelty.py)."""
        batch, n_maps, channels, height, width = x.shape
        if n_maps != self.n_maps:
            raise ValueError(f"{self.n_maps} cartes attendues, reçu {n_maps}")
        feats = self.features(x.reshape(batch * n_maps, channels, height, width))
        pooled: torch.Tensor = self.pool(feats).reshape(batch, -1)
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (lot, 4 cartes, 3, H, W) → logits (lot,). Probabilité = sigmoid(logit)."""
        logits: torch.Tensor = self.head(self.embed(x)).squeeze(1)
        return logits
