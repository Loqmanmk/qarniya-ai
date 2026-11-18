"""Couche domaine : le vocabulaire métier du projet (œil, examen, carte, risque).

Règle d'architecture : ce paquet ne dépend d'AUCUNE bibliothèque technique
(ni torch, ni pandas, ni FastAPI). Il ne contient que de la logique métier pure.
"""

from qarniya.domain.entities import CornealExam, CornealMaps
from qarniya.domain.enums import Diagnosis, EyeSide, MapType, RiskLevel
from qarniya.domain.exceptions import InvalidExamError, MissingMapError, QarniyaError
from qarniya.domain.risk import (
    OpinionComparison,
    ProgressionAssessment,
    RiskAssessment,
    RiskThresholds,
)

__all__ = [
    "CornealExam",
    "CornealMaps",
    "Diagnosis",
    "EyeSide",
    "InvalidExamError",
    "MapType",
    "MissingMapError",
    "OpinionComparison",
    "ProgressionAssessment",
    "QarniyaError",
    "RiskAssessment",
    "RiskLevel",
    "RiskThresholds",
]
