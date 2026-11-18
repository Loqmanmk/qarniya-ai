"""Figures pour le rapport et les slides (PNG, fond clair).

Palette validée (daltonisme compris) : Qarniya = bleu, baseline = orange, matrice = un seul bleu
du clair au foncé. Texte en encre neutre, jamais dans la couleur d'une série.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : fonctionne sur serveur et en CI
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

from qarniya.evaluation.metrics import BinaryMetrics

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # 1 : bleu, 2 : orange, 3 : aqua (validés ensemble)
SEQUENTIAL = LinearSegmentedColormap.from_list("qarniya_blue", ["#eef4fc", "#2a78d6", "#123a6b"])


@dataclass(frozen=True)
class RocSeries:
    name: str
    y_true: Sequence[int]
    y_prob: Sequence[float]
    metrics: BinaryMetrics  # au seuil retenu : point de fonctionnement affiché


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_roc(series: Sequence[RocSeries], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color=TEXT_SECONDARY, linewidth=1, label="Hasard")
    for color, s in zip(SERIES, series, strict=False):
        fpr, tpr, _ = roc_curve(s.y_true, s.y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{s.name} (AUC {s.metrics.auc:.3f})")
        point = (1 - s.metrics.specificity, s.metrics.sensitivity)
        ax.scatter(*point, s=70, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(
            f"{s.name} : sens. {s.metrics.sensitivity:.0%}, spéc. {s.metrics.specificity:.0%}",
            point,
            xytext=(12, -14 - 16 * series.index(s)),
            textcoords="offset points",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
    ax.set(xlim=(-0.01, 1), ylim=(0, 1.01))
    ax.set_xlabel("1 − spécificité (fausses alertes)", color=TEXT_PRIMARY)
    ax.set_ylabel("Sensibilité (kératocônes détectés)", color=TEXT_PRIMARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_confusion(metrics: BinaryMetrics, path: Path, title: str) -> None:
    cells = np.array([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
    names = np.array([["Sains reconnus", "Fausses alertes"], ["KC MANQUÉS", "KC détectés"]])
    fig, ax = plt.subplots(figsize=(4.6, 4), facecolor=SURFACE)
    ax.imshow(cells, cmap=SEQUENTIAL, vmin=0, vmax=max(int(cells.max()), 1))
    for (row, col), value in np.ndenumerate(cells):
        dark = value > cells.max() * 0.55
        ax.text(
            col, row, f"{value}\n{names[row, col]}", ha="center", va="center", fontsize=10,
            color="white" if dark else TEXT_PRIMARY, fontweight="bold" if row != col else "normal",
        )  # fmt: skip
    ax.set_xticks([0, 1], ["Prédit sain", "Prédit KC"], color=TEXT_PRIMARY)
    ax.set_yticks([0, 1], ["Réellement sain", "Réellement KC"], color=TEXT_PRIMARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
