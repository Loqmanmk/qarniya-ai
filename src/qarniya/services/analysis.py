"""Service d'analyse d'un œil : le cas d'usage principal (user stories US1 et US3).

Enchaînement, avec trois garde-fous AVANT de donner un score :
1. contrôle qualité des 4 images      → refus si une carte est illisible, vide, grise…
2. score de risque + carte de chaleur  → probabilité, niveau, part de chaque carte
3. détecteur d'images inhabituelles   → « score non fiable » si l'œil ne ressemble pas aux
                                          yeux d'entraînement (autre appareil, autre échelle…)

Ce service est utilisé tel quel par la ligne de commande, l'API et l'interface web.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qarniya.data import ImageQualityChecker
from qarniya.domain import (
    CornealExam,
    OpinionComparison,
    RiskAssessment,
    RiskLevel,
    RiskThresholds,
)
from qarniya.explainability import MAP_LABELS, Explanation, GradCam, save_explanation_figure
from qarniya.models import NoveltyDetector, Predictor
from qarniya.services.clinical import CLINICAL_FILE, FEATURES, ClinicalOpinionModel

NOVELTY_FILE = "novelty.npz"


class AnalysisStatus(str, Enum):
    ANALYSED = "analysé"
    UNUSUAL = "image inhabituelle"  # score calculé mais NON fiable
    REJECTED = "refusé"  # aucun score : entrée inexploitable


@dataclass(frozen=True)
class AnalysisReport:
    exam: CornealExam
    status: AnalysisStatus
    reasons: tuple[str, ...] = ()
    assessment: RiskAssessment | None = None
    explanation: Explanation | None = None
    novelty_distance: float | None = None
    numbers: RiskAssessment | None = None  # 2e avis : Kmax + épaisseur, si le médecin les a saisis

    @property
    def is_reliable(self) -> bool:
        return self.status is AnalysisStatus.ANALYSED

    @property
    def comparison(self) -> OpinionComparison | None:
        if self.assessment is None or self.numbers is None:
            return None
        return OpinionComparison(self.assessment, self.numbers)

    @property
    def final_level(self) -> RiskLevel | None:
        """Niveau affiché : celui des images, relevé à « à surveiller » en cas de désaccord."""
        if self.comparison is not None:
            return self.comparison.final_level
        return self.assessment.level if self.assessment else None

    def summary(self, with_warning: bool = True) -> str:
        """Phrase lisible par le médecin. `with_warning=False` quand l'avertissement est
        déjà affiché à part (bandeau de l'interface web)."""
        if self.status is AnalysisStatus.REJECTED:
            return "Analyse impossible : " + " ; ".join(self.reasons)
        if self.assessment is None or self.explanation is None:
            return "Analyse incomplète"
        level = self.final_level or self.assessment.level
        text = f"Risque {level.label} (images : {self.assessment.probability:.0%})."
        if self.assessment.level is not RiskLevel.LOW:
            # la chaleur montre ce qui fait MONTER le score : sans risque, elle n'a pas de sens
            main = MAP_LABELS[self.explanation.main_map].lower()
            text += f" Signe surtout vu sur la carte « {main} »."
        if self.numbers is not None:
            text += (
                f" Avis des chiffres : {self.numbers.level.label} ({self.numbers.probability:.0%})."
            )
            if self.comparison is not None and self.comparison.disagree:
                text += " Désaccord images / chiffres : à examiner."
        if self.status is AnalysisStatus.UNUSUAL and with_warning:
            text = "⚠ SCORE NON FIABLE : " + " ; ".join(self.reasons) + ". " + text
        return text


class AnalysisService:
    def __init__(
        self,
        predictor: Predictor,
        thresholds: RiskThresholds | None = None,
        quality: ImageQualityChecker | None = None,
        novelty: NoveltyDetector | None = None,
        clinical: ClinicalOpinionModel | None = None,
    ) -> None:
        self.predictor = predictor
        self.thresholds = thresholds or RiskThresholds()
        self.quality = quality or ImageQualityChecker()
        self.novelty = novelty
        self.clinical = clinical
        self._gradcam = GradCam(predictor.model)

    @classmethod
    def from_run(cls, run_dir: Path, device: torch.device | None = None) -> AnalysisService:
        """Charge le modèle et, s'ils existent, le détecteur d'images inhabituelles et l'avis
        des chiffres (créés par `qarniya prepare`)."""
        novelty_path, clinical_path = run_dir / NOVELTY_FILE, run_dir / CLINICAL_FILE
        return cls(
            Predictor.from_checkpoint(run_dir / "model.pt", device),
            novelty=NoveltyDetector.load(novelty_path) if novelty_path.is_file() else None,
            clinical=ClinicalOpinionModel.load(clinical_path) if clinical_path.is_file() else None,
        )

    def analyse(self, exam: CornealExam) -> AnalysisReport:
        quality = self.quality.check_exam(exam)
        if not quality.is_acceptable:
            reasons = tuple(f"{MAP_LABELS[i.map_type]} : {i.message}" for i in quality.issues)
            return AnalysisReport(exam, AnalysisStatus.REJECTED, reasons)

        preprocessing = self.predictor.loaded.preprocessing
        x = self.predictor.preprocessor.to_tensor(exam).unsqueeze(0).to(self.predictor.device)
        explanation = self._gradcam.explain(x, exam.eye, preprocessing.mirror_left_eye)
        assessment = RiskAssessment.from_probability(
            exam.exam_id, explanation.probability, self.thresholds
        )

        status, reasons, distance = AnalysisStatus.ANALYSED, (), None
        if self.novelty is not None:
            distance = float(self.novelty.distances(self.predictor.embed_tensor(x))[0])
            if distance > self.novelty.threshold:
                status = AnalysisStatus.UNUSUAL
                reasons = (
                    "cet œil ne ressemble pas aux yeux d'entraînement "
                    "(autre appareil ou autre échelle de couleurs ?)",
                )
        numbers = None
        if self.clinical is not None and all(f in exam.clinical for f in FEATURES):
            numbers = RiskAssessment.from_probability(
                exam.exam_id, self.clinical.probability(dict(exam.clinical)), self.thresholds
            )
        return AnalysisReport(exam, status, reasons, assessment, explanation, distance, numbers)

    def save_figure(self, report: AnalysisReport, path: Path) -> None:
        if report.explanation is None or report.assessment is None:
            raise ValueError("pas de carte de chaleur pour un examen refusé")
        images = []
        for image_path in report.exam.maps.ordered():
            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))
        exam = report.exam
        title = f"{exam.patient_code} {exam.eye.value} — {report.summary()}"
        save_explanation_figure(
            images, report.explanation.heatmaps, report.explanation.map_importance, title, path
        )


def triage_order(report: AnalysisReport) -> tuple[int, float]:
    """Ordre de la liste de dépistage : d'abord ce qui demande l'attention du médecin."""
    if report.status is not AnalysisStatus.ANALYSED or report.assessment is None:
        return (0, 0.0)  # refusés et inhabituels : à revoir en premier
    level_rank = {RiskLevel.HIGH: 1, RiskLevel.MONITOR: 2, RiskLevel.LOW: 3}
    level = report.final_level or report.assessment.level
    return (level_rank[level], -report.assessment.probability)


def fit_novelty_detector(
    predictor: Predictor,
    train: list[CornealExam],
    calibration: list[CornealExam],
    quantile: float = 0.99,
) -> tuple[NoveltyDetector, np.ndarray]:
    """Construit le détecteur ; rend aussi les distances de calibration (pour contrôle)."""
    detector = NoveltyDetector.fit(
        predictor.embed(train), predictor.embed(calibration), quantile=quantile
    )
    return detector, detector.distances(predictor.embed(calibration))
