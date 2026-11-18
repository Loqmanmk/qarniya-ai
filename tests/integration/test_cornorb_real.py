"""Vérifie le vrai dataset CornOrb s'il est présent (ignoré en CI, sans données)."""

from pathlib import Path

import pytest

from qarniya.data import DatasetConfig, build_adapter, split_dataset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "data" / "cornorb.yaml"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (ROOT / "data" / "cornorb" / "clinical_data_and_labels.csv").exists(),
        reason="données CornOrb non téléchargées",
    ),
]


def test_cornorb_loads_with_expected_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)  # les chemins du YAML sont relatifs à la racine du dépôt
    config = DatasetConfig.from_yaml(CONFIG)
    result = build_adapter(config).load()
    splits = split_dataset(result, config.split)

    assert len(result.exams) == 1361
    assert len(result.skipped) == 93  # uniquement les doublons ambigus
    assert (len(splits.train), len(splits.val), len(splits.test)) == (872, 222, 267)
    assert splits.leaked_patients() == set()
