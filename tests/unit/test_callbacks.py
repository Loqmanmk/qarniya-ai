"""Les callbacks écrivent l'historique et les courbes au fil de l'entraînement."""

import csv
from pathlib import Path

from qarniya.evaluation import compute_binary_metrics
from qarniya.training import CsvHistoryLogger, TensorBoardLogger
from qarniya.training.records import EpochRecord


def record(epoch: int, val_loss: float) -> EpochRecord:
    metrics = compute_binary_metrics([0, 1, 1, 0], [0.2, 0.8, 0.6, 0.4])
    return EpochRecord(epoch, 0.5, val_loss, metrics, 3e-4, improved=epoch == 1)


def test_csv_history_is_written_after_each_epoch(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    logger = CsvHistoryLogger(path)

    logger.on_epoch_end(record(1, 0.4))
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1  # déjà lisible après la 1re époque

    logger.on_epoch_end(record(2, 0.3))
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert float(rows[1]["val_loss"]) == 0.3
    assert "val_sensitivity" in rows[0]


def test_tensorboard_writes_event_files(tmp_path: Path) -> None:
    logger = TensorBoardLogger(tmp_path / "tb")
    logger.on_epoch_end(record(1, 0.4))
    logger.on_train_end([])
    assert any((tmp_path / "tb").glob("events.out.tfevents.*"))
