"""Entités métier : un examen de topographie et ses 4 cartes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from qarniya.domain.enums import Diagnosis, EyeSide, MapType
from qarniya.domain.exceptions import InvalidExamError, MissingMapError
from qarniya.domain.frozen import FrozenMapping


@dataclass(frozen=True)
class CornealMaps:
    """Les 4 images d'un œil. Garantit à la construction qu'aucune carte ne manque."""

    paths: Mapping[MapType, Path]

    def __post_init__(self) -> None:
        missing = [m.value for m in MapType if m not in self.paths]
        if missing:
            raise MissingMapError(f"Cartes manquantes : {', '.join(missing)}")
        # copie en lecture seule : l'objet ne peut plus être modifié après création
        object.__setattr__(self, "paths", FrozenMapping(self.paths))

    def ordered(self) -> list[Path]:
        """Les chemins toujours dans le même ordre (celui de MapType), comme l'attend le modèle."""
        return [self.paths[m] for m in MapType]


@dataclass(frozen=True)
class CornealExam:
    """Un examen de topographie d'un œil, à une date donnée."""

    exam_id: str
    patient_code: str  # code anonyme, jamais le nom (loi 09-08)
    eye: EyeSide
    maps: CornealMaps
    exam_date: date | None = None
    device: str | None = None  # ex. "Orbscan 3", "Pentacam"
    diagnosis: Diagnosis | None = None  # connu pour l'entraînement, inconnu en usage réel
    clinical: Mapping[str, float] = field(default_factory=dict)  # Kmax, épaisseur minimale…

    def __post_init__(self) -> None:
        if not self.exam_id.strip():
            raise InvalidExamError("exam_id ne peut pas être vide")
        if not self.patient_code.strip():
            raise InvalidExamError("patient_code ne peut pas être vide")
        object.__setattr__(self, "clinical", FrozenMapping(self.clinical))

    @property
    def is_labelled(self) -> bool:
        return self.diagnosis is not None
