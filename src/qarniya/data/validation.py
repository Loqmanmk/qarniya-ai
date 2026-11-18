"""Contrôle qualité des images avant analyse (user story US3 : refuser une entrée douteuse).

Un outil médical doit savoir dire « je ne peux pas analyser cette image » plutôt que
de donner un score au hasard.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from qarniya.domain import CornealExam, MapType


@dataclass(frozen=True)
class QualityIssue:
    map_type: MapType
    message: str


@dataclass(frozen=True)
class QualityReport:
    exam_id: str
    issues: tuple[QualityIssue, ...]

    @property
    def is_acceptable(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class ImageQualityChecker:
    min_side: int = 128  # en dessous, la carte est illisible
    max_aspect_ratio: float = 1.5  # les cartes sont à peu près carrées
    min_std: float = 4.0  # image (presque) unie = vide ou noire

    def check_image(self, path: Path) -> list[str]:
        try:
            with Image.open(path) as img:
                img.load()
                rgb = img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return ["fichier illisible ou absent"]

        problems: list[str] = []
        width, height = rgb.size
        if min(width, height) < self.min_side:
            problems.append(f"résolution trop faible ({width}x{height})")
        if max(width, height) / max(min(width, height), 1) > self.max_aspect_ratio:
            problems.append(f"proportions inattendues ({width}x{height})")
        if float(np.asarray(rgb, dtype=np.float32).std()) < self.min_std:
            problems.append("image vide ou uniforme")
        return problems

    def check_exam(self, exam: CornealExam) -> QualityReport:
        issues = [
            QualityIssue(map_type, message)
            for map_type, path in exam.maps.paths.items()
            for message in self.check_image(path)
        ]
        return QualityReport(exam.exam_id, tuple(issues))
