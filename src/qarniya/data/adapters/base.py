"""Interface commune à tous les adaptateurs de datasets.

Chaque source de données (CornOrb, export du médecin, dossier d'images…) est lue par un adaptateur
qui la traduit en objets du domaine (`CornealExam`). Le reste du logiciel ne voit jamais
le format d'origine : c'est le patron de conception « Adaptateur ».
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

from qarniya.data.config import DatasetConfig
from qarniya.domain import CornealExam


@dataclass(frozen=True)
class SkippedRecord:
    """Un enregistrement écarté, et pourquoi (traçabilité : on ne jette rien en silence)."""

    identifier: str
    reason: str


@dataclass(frozen=True)
class LoadResult:
    exams: tuple[CornealExam, ...]
    skipped: tuple[SkippedRecord, ...] = ()
    # découpage fourni par la source (exam_id -> "train" / "val" / "test"), si elle en a un
    predefined_splits: Mapping[str, str] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Lit une source de données et produit des `CornealExam`."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> LoadResult:
        """Lit toute la source. Une ligne invalide ne lève pas d'erreur : elle va dans `skipped`."""
