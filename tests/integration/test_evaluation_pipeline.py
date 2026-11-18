"""De bout en bout : entraînement → évaluation sur le test → rapport, et protection du test."""

import json
from pathlib import Path

import pytest

from qarniya.domain.exceptions import QarniyaError
from qarniya.evaluation.runner import evaluate_run
from qarniya.training import ExperimentConfig, run_experiment

pytestmark = pytest.mark.integration


@pytest.fixture
def trained_run(experiment_yaml: Path) -> Path:
    return run_experiment(ExperimentConfig.from_yaml(experiment_yaml), epochs=1).run_dir


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
    assert result.baseline is None  # le mini-dataset n'a pas de Kmax : pas de baseline
    assert "Qarniya" in (out / "report.md").read_text(encoding="utf-8")


def test_test_set_cannot_be_evaluated_twice_by_accident(trained_run: Path) -> None:
    evaluate_run(trained_run, n_bootstrap=20)
    with pytest.raises(QarniyaError, match="déjà été évalué"):
        evaluate_run(trained_run, n_bootstrap=20)
    evaluate_run(trained_run, n_bootstrap=20, force=True)  # possible, mais explicitement
