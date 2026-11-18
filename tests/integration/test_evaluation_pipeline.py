"""De bout en bout : entraînement → évaluation sur le test → rapport, et protection du test."""

import json
from pathlib import Path

import pytest

from qarniya.domain.exceptions import QarniyaError
from qarniya.evaluation.runner import evaluate_run

pytestmark = pytest.mark.integration


def test_evaluation_writes_full_report(trained_run: Path) -> None:
    result = evaluate_run(trained_run, n_bootstrap=50)

    out = trained_run / "evaluation"
    for name in ("metrics.json", "report.md", "predictions_test.csv", "errors.csv",
                 "roc_test.png", "confusion_test.png"):  # fmt: skip
        assert (out / name).is_file(), name

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_test_eyes"] == result.n_test == 4  # patients 10 et 11, deux yeux chacun
    assert metrics["protocol"]["threshold_chosen_on"] == "validation"
    assert metrics["protocol"]["bootstrap_unit"] == "patient"
    # images, chiffres seuls (Kmax), puis fusion des deux : même protocole pour les trois
    assert [e.key for e in result.evaluations] == ["images", "chiffres", "fusion"]
    assert set(metrics["models"]) == {"images", "chiffres", "fusion"}
    assert (out / "confusion_fusion_test.png").is_file()
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Fusion images + chiffres" in report and "aucun paramètre appris" in report


def test_test_set_cannot_be_evaluated_twice_by_accident(trained_run: Path) -> None:
    evaluate_run(trained_run, n_bootstrap=20)
    with pytest.raises(QarniyaError, match="déjà été évalué"):
        evaluate_run(trained_run, n_bootstrap=20)
    evaluate_run(trained_run, n_bootstrap=20, force=True)  # possible, mais explicitement
