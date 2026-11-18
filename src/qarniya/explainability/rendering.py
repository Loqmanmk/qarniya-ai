"""Superposition de la carte de chaleur sur les cartes de topographie.

Choix visuels :
- la carte d'origine est passée en GRIS : ses couleurs (l'échelle de l'appareil) ne se mélangent
  pas avec la chaleur ;
- la chaleur est d'UNE seule teinte (orange) dont l'opacité suit l'intensité : pas d'arc-en-ciel,
  lisible aussi par les daltoniens ; transparent là où le modèle n'a pas regardé.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from qarniya.domain import MapType

HEAT_COLOR = np.array([235, 104, 52], dtype=np.float32)  # orange de la palette validée
MAX_OPACITY = 0.75
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

MAP_LABELS = {
    MapType.AXIAL: "Courbure axiale",
    MapType.ANTERIOR_ELEVATION: "Élévation antérieure",
    MapType.POSTERIOR_ELEVATION: "Élévation postérieure",
    MapType.PACHYMETRY: "Pachymétrie",
}


def overlay(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Image RGB (H, W, 3) : carte en gris + chaleur orange."""
    base = np.asarray(image.convert("L").convert("RGB"), dtype=np.float32)
    heat_img = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(
        (base.shape[1], base.shape[0]), Image.Resampling.BILINEAR
    )
    alpha = (np.asarray(heat_img, dtype=np.float32) / 255 * MAX_OPACITY)[..., None]
    blended: np.ndarray = (base * (1 - alpha) + HEAT_COLOR * alpha).astype(np.uint8)
    return blended


def save_explanation_figure(
    images: Sequence[Image.Image],
    heatmaps: dict[MapType, np.ndarray],
    importance: dict[MapType, float],
    title: str,
    path: Path,
) -> None:
    """4 panneaux : chaque carte avec sa chaleur et sa part dans la décision."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=SURFACE)
    for ax, map_type, image in zip(axes, MapType, images, strict=True):
        ax.imshow(overlay(image, heatmaps[map_type]))
        ax.set_title(
            f"{MAP_LABELS[map_type]}\npart dans la décision : {importance[map_type]:.0%}",
            color=TEXT_PRIMARY,
            fontsize=10,
        )
        ax.axis("off")
    wrapped = "\n".join(textwrap.wrap(title, width=150))  # titre long : sur plusieurs lignes
    fig.suptitle(wrapped, color=TEXT_PRIMARY, fontsize=11, x=0.01, ha="left")
    fig.text(
        0.01, 0.01,
        "Orange = zones qui ont fait monter le score. Aide à la lecture, pas une preuve médicale.",
        color=TEXT_SECONDARY, fontsize=9,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
