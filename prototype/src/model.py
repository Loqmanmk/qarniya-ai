"""Modèle : un même ResNet18 (pré-entraîné ImageNet) lit chacune des 4 cartes,
puis on fusionne les 4 descripteurs pour donner un score de risque."""
import torch
import torch.nn as nn
from torchvision import models


class QarniyaNet(nn.Module):
    def __init__(self, n_maps=4, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        # on garde tout sauf la dernière couche (fc)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # -> (B, 512, h, w)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.n_maps = n_maps
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * n_maps, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # x : (B, 4, 3, H, W)
        b, m, c, h, w = x.shape
        f = self.features(x.view(b * m, c, h, w))       # (B*4, 512, h', w')
        f = self.pool(f).view(b, m * 512)               # (B, 2048)
        return self.head(f).squeeze(1)                  # logits (B,)
