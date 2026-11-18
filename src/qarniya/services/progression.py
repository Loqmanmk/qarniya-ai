"""Suivi de progression (user story US2) : deux examens du même œil, à deux dates.

C'est la question clinique qui décide d'un cross-linking : « est-ce que ça s'aggrave ? ».
"""

from __future__ import annotations

from dataclasses import dataclass

from qarniya.domain import CornealExam, InvalidExamError, ProgressionAssessment
from qarniya.services.analysis import AnalysisReport, AnalysisService


@dataclass(frozen=True)
class ProgressionReport:
    before: AnalysisReport
    after: AnalysisReport

    @property
    def progression(self) -> ProgressionAssessment | None:
        """Comparaison chiffrée, seulement si les DEUX analyses sont fiables."""
        if not (self.before.is_reliable and self.after.is_reliable):
            return None
        if self.before.assessment is None or self.after.assessment is None:
            return None
        return ProgressionAssessment(self.before.assessment, self.after.assessment)

    def verdict(self) -> str:
        p = self.progression
        if p is None:
            return "Comparaison impossible : au moins un des deux examens n'est pas fiable."
        change = f"{p.before.probability:.0%} → {p.after.probability:.0%}"
        if p.is_progressing:
            return f"Score en hausse nette ({change}) : possible progression, à contrôler."
        if p.is_improving:
            return f"Score en baisse ({change})."
        return f"Score stable ({change})."


class ProgressionService:
    def __init__(self, analysis: AnalysisService) -> None:
        self.analysis = analysis

    def compare(self, before: CornealExam, after: CornealExam) -> ProgressionReport:
        if before.eye is not after.eye:
            raise InvalidExamError("les deux examens doivent concerner le même œil")
        return ProgressionReport(self.analysis.analyse(before), self.analysis.analyse(after))
