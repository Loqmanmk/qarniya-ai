"""Seuil, fourchettes de confiance, baseline et figures : les chiffres présentés au jury."""

from pathlib import Path

import numpy as np
import pytest

from qarniya.domain import CornealExam, CornealMaps, EyeSide, MapType
from qarniya.evaluation import (
    bootstrap_confidence_intervals,
    compute_binary_metrics,
    soft_vote,
    threshold_for_sensitivity,
)
from qarniya.evaluation.baseline import TabularBaseline
from qarniya.evaluation.figures import RocSeries, plot_confusion, plot_roc


# ---------- seuil ----------
def test_threshold_reaches_target_sensitivity_with_best_specificity() -> None:
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.95, 0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.05]
    # 75 % suffit : on peut laisser passer le malade à 0,3 et éviter la fausse alerte à 0,6
    t75 = threshold_for_sensitivity(y_true, y_prob, 0.75)
    assert compute_binary_metrics(y_true, y_prob, t75).specificity == 1.0
    # 100 % exigé : il faut descendre sous 0,3, au prix d'une fausse alerte
    t100 = threshold_for_sensitivity(y_true, y_prob, 1.0)
    m = compute_binary_metrics(y_true, y_prob, t100)
    assert m.sensitivity == 1.0 and m.fp == 1


def test_threshold_falls_in_the_middle_of_the_gap() -> None:
    # séparation parfaite : le seuil doit tomber entre le plus haut sain (0,2) et le plus bas malade
    t = threshold_for_sensitivity([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9], 0.95)
    assert t == pytest.approx(0.4)


def test_threshold_without_positive_case_is_rejected() -> None:
    with pytest.raises(ValueError, match="positif"):
        threshold_for_sensitivity([0, 0], [0.1, 0.2])


# ---------- bootstrap ----------
def _synthetic(
    n_patients: int = 60, correlated_eyes: bool = False
) -> tuple[list[int], list[float], list[str]]:
    rng = np.random.default_rng(0)
    y, p, g = [], [], []
    for i in range(n_patients):
        label = i % 3 == 0
        patient_score = rng.normal(0.7 if label else 0.3, 0.2)
        for _ in range(2):  # deux yeux par patient
            score = patient_score if correlated_eyes else rng.normal(0.7 if label else 0.3, 0.2)
            y.append(int(label))
            p.append(float(np.clip(score, 0, 1)))
            g.append(f"P{i}")
    return y, p, g


def test_bootstrap_interval_contains_point_estimate_and_is_reproducible() -> None:
    y, p, g = _synthetic()
    point = compute_binary_metrics(y, p, 0.5)
    ci = bootstrap_confidence_intervals(y, p, g, 0.5, n_resamples=300, seed=1)
    for name in ("auc", "sensitivity", "specificity"):
        assert ci[name].low <= getattr(point, name) <= ci[name].high
        assert ci[name].low < ci[name].high
    assert ci == bootstrap_confidence_intervals(y, p, g, 0.5, n_resamples=300, seed=1)


def test_bootstrap_by_patient_is_wider_than_by_eye() -> None:
    """Deux yeux d'un patient se ressemblent : les tirer à part sous-estime l'incertitude."""
    y, p, g = _synthetic(correlated_eyes=True)
    by_patient = bootstrap_confidence_intervals(y, p, g, 0.5, n_resamples=500)
    by_eye = bootstrap_confidence_intervals(y, p, [str(i) for i in range(len(y))], 0.5, 500)
    width = lambda ci: ci["sensitivity"].high - ci["sensitivity"].low  # noqa: E731
    assert width(by_patient) > width(by_eye)


# ---------- fusion ----------
def test_soft_vote_is_the_average_of_opinions() -> None:
    images = np.array([0.99, 0.02, 0.90])
    numbers = np.array([0.01, 0.98, 0.80])
    assert soft_vote([images, numbers]) == pytest.approx([0.50, 0.50, 0.85])
    assert soft_vote([images, numbers], weights=[3, 1]) == pytest.approx([0.745, 0.26, 0.875])


def test_soft_vote_rejects_bad_weights() -> None:
    with pytest.raises(ValueError):
        soft_vote([np.array([0.5])], weights=[0.0])
    with pytest.raises(ValueError):
        soft_vote([])


# ---------- baseline ----------
def _exam(i: int, kmax: float | None) -> CornealExam:
    maps = CornealMaps({m: Path(f"{i}{m.value}.png") for m in MapType})
    clinical = {} if kmax is None else {"kmax": kmax}
    return CornealExam(f"e{i}", f"P{i}", EyeSide.RIGHT, maps, clinical=clinical)


def test_baseline_learns_from_kmax_and_tolerates_missing_values() -> None:
    exams = [_exam(i, 43.0 + i * 0.1) for i in range(20)] + [_exam(i, 52.0) for i in range(20, 40)]
    targets = [0] * 20 + [1] * 20
    baseline = TabularBaseline().fit(exams, targets)
    assert baseline.features == ("kmax",)  # seules les variables présentes sont utilisées
    probs = baseline.predict([_exam(98, 42.5), _exam(99, 58.0), _exam(100, None)])
    assert probs[0] < 0.5 < probs[1]
    assert 0.0 <= probs[2] <= 1.0  # valeur manquante remplacée par la médiane


def test_baseline_without_clinical_data_is_rejected() -> None:
    with pytest.raises(ValueError, match="clinique"):
        TabularBaseline().fit([_exam(1, None), _exam(2, None)], [0, 1])


# ---------- figures ----------
def test_figures_are_written(tmp_path: Path) -> None:
    y, p, _ = _synthetic(20)
    m = compute_binary_metrics(y, p, 0.5)
    plot_roc([RocSeries("Modèle", y, p, m)], tmp_path / "roc.png", "ROC")
    plot_confusion(m, tmp_path / "cm.png", "Matrice")
    assert (tmp_path / "roc.png").stat().st_size > 10_000
    assert (tmp_path / "cm.png").stat().st_size > 10_000
