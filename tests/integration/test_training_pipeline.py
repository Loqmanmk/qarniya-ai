"""De bout en bout : YAML d'expérience → entraînement → dossier de résultats → modèle rechargeable.

Mini-dataset synthétique, CPU, petites images : quelques secondes, tourne aussi en CI.
"""

import json
from pathlib import Path

import pytest
import torch

from qarniya.models import Preprocessor, load_checkpoint
from qarniya.training import ExperimentConfig, run_experiment

pytestmark = pytest.mark.integration


def test_experiment_produces_a_complete_reproducible_run(experiment_yaml: Path) -> None:
    config = ExperimentConfig.from_yaml(experiment_yaml)
    result = run_experiment(config)

    run = result.run_dir
    expected = ("model.pt", "history.csv", "summary.json", "experiment.yaml", "dataset.yaml")
    for name in expected:
        assert (run / name).is_file(), name

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    # patients 0-3 : validation (8 yeux) · 4-9 : entraînement (12 yeux) · 10-11 : test, jamais lu
    assert summary["n_train"] == 12 and summary["n_val"] == 8
    assert summary["best_epoch"] == result.best.epoch
    assert len(result.history) == 2
    assert any((run / "tensorboard").glob("events.out.tfevents.*"))

    # le modèle sauvegardé se recharge avec SA préparation d'images et fonctionne
    loaded = load_checkpoint(result.checkpoint)
    assert loaded.preprocessing.image_size == 64
    assert loaded.metadata["experiment"] == "test_run"
    assert loaded.metadata["positive_labels"] == ["keratoconus", "suspect"]
    x = torch.randn(1, 4, 3, 64, 64)
    with torch.no_grad():
        assert loaded.model(x).shape == (1,)
    assert isinstance(Preprocessor(loaded.preprocessing), Preprocessor)


def test_same_seed_gives_same_training(experiment_yaml: Path) -> None:
    config = ExperimentConfig.from_yaml(experiment_yaml)
    first = run_experiment(config, epochs=1)
    second = run_experiment(config, epochs=1)
    assert first.history[0].train_loss == pytest.approx(second.history[0].train_loss, rel=1e-5)


def test_project_experiment_configs_are_valid() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in (root / "configs" / "experiments").glob("*.yaml"):
        ExperimentConfig.from_yaml(path)
