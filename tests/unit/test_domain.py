"""Tests de la couche domaine. Chaque test suit le schéma AAA : Arrange (préparer),
Act (agir), Assert (vérifier)."""

from pathlib import Path

import pytest

from qarniya.domain import (
    CornealExam,
    CornealMaps,
    EyeSide,
    InvalidExamError,
    MapType,
    MissingMapError,
    ProgressionAssessment,
    QarniyaError,
    RiskAssessment,
    RiskLevel,
    RiskThresholds,
)


def make_maps() -> CornealMaps:
    return CornealMaps({m: Path(f"{m.value}.png") for m in MapType})


# ---------- CornealMaps ----------
def test_maps_are_returned_in_fixed_order() -> None:
    maps = make_maps()
    assert [p.stem for p in maps.ordered()] == ["axial", "anterior", "posterior", "pachymetry"]


def test_missing_map_is_rejected() -> None:
    incomplete = {MapType.AXIAL: Path("a.png"), MapType.PACHYMETRY: Path("p.png")}
    with pytest.raises(MissingMapError, match="anterior"):
        CornealMaps(incomplete)


def test_maps_cannot_be_modified_after_creation() -> None:
    maps = make_maps()
    with pytest.raises(TypeError):
        maps.paths[MapType.AXIAL] = Path("other.png")  # type: ignore[index]


# ---------- CornealExam ----------
def test_exam_requires_patient_code() -> None:
    with pytest.raises(InvalidExamError):
        CornealExam(exam_id="e1", patient_code="  ", eye=EyeSide.RIGHT, maps=make_maps())


def test_unlabelled_exam() -> None:
    exam = CornealExam(exam_id="e1", patient_code="P017", eye=EyeSide.LEFT, maps=make_maps())
    assert not exam.is_labelled


# ---------- RiskThresholds ----------
@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, RiskLevel.LOW),
        (0.29, RiskLevel.LOW),
        (0.30, RiskLevel.MONITOR),
        (0.69, RiskLevel.MONITOR),
        (0.70, RiskLevel.HIGH),
        (1.0, RiskLevel.HIGH),
    ],
)
def test_default_thresholds(probability: float, expected: RiskLevel) -> None:
    assert RiskThresholds().classify(probability) is expected


def test_custom_thresholds_for_lasik_center() -> None:
    cautious = RiskThresholds(monitor=0.1, high=0.4)
    assert cautious.classify(0.2) is RiskLevel.MONITOR


@pytest.mark.parametrize(("monitor", "high"), [(0.7, 0.3), (0.5, 0.5), (-0.1, 0.5), (0.2, 1.5)])
def test_invalid_thresholds_are_rejected(monitor: float, high: float) -> None:
    with pytest.raises(QarniyaError):
        RiskThresholds(monitor=monitor, high=high)


def test_probability_out_of_range_is_rejected() -> None:
    with pytest.raises(QarniyaError):
        RiskAssessment(exam_id="e1", probability=1.2, level=RiskLevel.HIGH)


# ---------- Progression ----------
def test_progression_detected_when_score_rises() -> None:
    before = RiskAssessment.from_probability("old", 0.20)
    after = RiskAssessment.from_probability("new", 0.55)
    progression = ProgressionAssessment(before, after)
    assert progression.delta == pytest.approx(0.35)
    assert progression.is_progressing
    assert not progression.is_improving


def test_small_change_is_considered_stable() -> None:
    progression = ProgressionAssessment(
        RiskAssessment.from_probability("old", 0.50),
        RiskAssessment.from_probability("new", 0.55),
    )
    assert not progression.is_progressing
    assert not progression.is_improving
