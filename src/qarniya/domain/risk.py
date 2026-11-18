"""Règles métier du risque : de la probabilité du modèle au niveau affiché au médecin,
et comparaison de deux examens pour détecter une progression."""

from __future__ import annotations

from dataclasses import dataclass

from qarniya.domain.enums import RiskLevel
from qarniya.domain.exceptions import QarniyaError


def _check_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise QarniyaError(f"{name} doit être entre 0 et 1, reçu {value}")


@dataclass(frozen=True)
class RiskThresholds:
    """Seuils de décision, configurables : un centre LASIK voudra des seuils plus bas (prudence)."""

    monitor: float = 0.3  # à partir de là : « à surveiller »
    high: float = 0.7  # à partir de là : « élevé »

    def __post_init__(self) -> None:
        _check_probability(self.monitor, "monitor")
        _check_probability(self.high, "high")
        if self.monitor >= self.high:
            raise QarniyaError("le seuil 'monitor' doit être strictement inférieur au seuil 'high'")

    def classify(self, probability: float) -> RiskLevel:
        _check_probability(probability, "probability")
        if probability >= self.high:
            return RiskLevel.HIGH
        if probability >= self.monitor:
            return RiskLevel.MONITOR
        return RiskLevel.LOW


@dataclass(frozen=True)
class RiskAssessment:
    """Résultat de l'analyse d'un examen."""

    exam_id: str
    probability: float  # probabilité de kératocône estimée par le modèle (0 à 1)
    level: RiskLevel

    def __post_init__(self) -> None:
        _check_probability(self.probability, "probability")

    @classmethod
    def from_probability(
        cls, exam_id: str, probability: float, thresholds: RiskThresholds | None = None
    ) -> RiskAssessment:
        level = (thresholds or RiskThresholds()).classify(probability)
        return cls(exam_id=exam_id, probability=probability, level=level)


@dataclass(frozen=True)
class ProgressionAssessment:
    """Comparaison de deux examens du même œil (ancien puis récent)."""

    before: RiskAssessment
    after: RiskAssessment
    significant_change: float = 0.15  # variation de score jugée significative

    @property
    def delta(self) -> float:
        return self.after.probability - self.before.probability

    @property
    def is_progressing(self) -> bool:
        return self.delta >= self.significant_change

    @property
    def is_improving(self) -> bool:
        return self.delta <= -self.significant_change


@dataclass(frozen=True)
class OpinionComparison:
    """Deux avis indépendants sur le même œil : les images (réseau) et les chiffres (Kmax…).

    Règle métier (étape 5) : quand les deux avis se contredisent franchement (l'un « faible »,
    l'autre « élevé »), le niveau final est au minimum « à surveiller » : le médecin doit regarder.
    """

    images: RiskAssessment
    numbers: RiskAssessment

    @property
    def disagree(self) -> bool:
        return {self.images.level, self.numbers.level} == {RiskLevel.LOW, RiskLevel.HIGH}

    @property
    def final_level(self) -> RiskLevel:
        if self.disagree:
            return RiskLevel.MONITOR
        return self.images.level  # sinon l'avis des images (meilleur résultat au test) prime
