"""Préparation des images, PARTAGÉE entre entraînement et analyse.

Bug classique en IA médicale : préparer les images différemment à l'entraînement et en production
(taille, normalisation, orientation…) → le modèle se trompe sans que personne ne s'en rende compte.
Ici, une seule classe fait ce travail, et ses réglages sont sauvegardés DANS le modèle.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageOps
from pydantic import Field
from torchvision import transforms

from qarniya.config_utils import FrozenModel
from qarniya.domain import CornealExam, EyeSide

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreprocessingSpec(FrozenModel):
    image_size: int = Field(default=256, ge=32, le=1024)
    # œil gauche retourné en miroir : nasal/temporal du même côté que l'œil droit
    mirror_left_eye: bool = True


class Preprocessor:
    def __init__(self, spec: PreprocessingSpec) -> None:
        self.spec = spec
        size = (spec.image_size, spec.image_size)
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self._eval = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), normalize])
        # Augmentation légère et SANS changement de couleur : la couleur est l'information médicale
        self._train = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def load_images(self, exam: CornealExam) -> list[Image.Image]:
        """Les 4 cartes, dans l'ordre fixe du domaine, orientées de façon homogène."""
        images = []
        for path in exam.maps.ordered():
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            if self.spec.mirror_left_eye and exam.eye is EyeSide.LEFT:
                rgb = ImageOps.mirror(rgb)
            images.append(rgb)
        return images

    def to_tensor(self, exam: CornealExam, augment: bool = False) -> torch.Tensor:
        """Tenseur (4 cartes, 3 canaux, H, W) prêt pour le modèle."""
        transform = self._train if augment else self._eval
        return torch.stack([transform(img) for img in self.load_images(exam)])
