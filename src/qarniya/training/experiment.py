"""Une expérience = une configuration → un dossier de résultats complet et reproductible.

outputs/runs/<date>_<nom>/
├── experiment.yaml   copie exacte de la configuration utilisée
├── dataset.yaml      copie de la configuration des données
├── model.pt          meilleur modèle (poids + préparation + métadonnées)
├── history.csv       métriques à chaque époque (écrit au fil de l'eau)
├── tensorboard/      courbes en direct : tensorboard --logdir outputs/runs
└── summary.json      résumé : meilleure époque, métriques de validation, commit Git…

Le jeu de TEST n'est jamais utilisé ici : il est réservé à l'évaluation finale (étape 5).
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from qarniya import __version__
from qarniya.data import DatasetConfig, DatasetSplits, build_adapter, split_dataset
from qarniya.domain import CornealExam
from qarniya.models import Preprocessor, build_model, save_checkpoint
from qarniya.training.callbacks import CsvHistoryLogger, TensorBoardLogger, TrainingCallback
from qarniya.training.config import Device, ExperimentConfig
from qarniya.training.dataset import ExamDataset
from qarniya.training.records import EpochRecord
from qarniya.training.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    best: EpochRecord
    history: tuple[EpochRecord, ...]

    @property
    def checkpoint(self) -> Path:
        return self.run_dir / "model.pt"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: Device) -> torch.device:
    if device is Device.CUDA and not torch.cuda.is_available():
        raise RuntimeError("device: cuda demandé mais aucun GPU n'est disponible")
    if device is Device.CPU or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def git_commit() -> str | None:
    """Commit Git du code utilisé (traçabilité). None si Git est absent : jamais bloquant."""
    git = shutil.which("git")  # chemin complet : on n'exécute pas un « git » inconnu du PATH
    if git is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603  (commande fixe, aucune donnée externe)
            [git, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def load_splits(config: ExperimentConfig, limit: int | None = None) -> DatasetSplits:
    data_config = DatasetConfig.from_yaml(config.data_config)
    result = build_adapter(data_config).load()
    splits = split_dataset(result, data_config.split)
    if limit is None:
        return splits
    rng = random.Random(config.training.seed)  # noqa: S311  (échantillonnage, pas de la sécurité)

    def sample(exams: tuple[CornealExam, ...], n: int) -> tuple[CornealExam, ...]:
        return tuple(rng.sample(exams, min(n, len(exams))))

    return DatasetSplits(sample(splits.train, limit), sample(splits.val, max(limit // 4, 2)), ())


def run_experiment(
    config: ExperimentConfig, epochs: int | None = None, limit: int | None = None
) -> ExperimentResult:
    """Lance une expérience complète. `epochs` et `limit` servent aux essais rapides."""
    if epochs is not None:
        config = config.model_copy(
            update={"training": config.training.model_copy(update={"epochs": epochs})}
        )
    params = config.training
    seed_everything(params.seed)
    device = resolve_device(params.device)
    logger.info("Appareil de calcul : %s", device)

    splits = load_splits(config, limit)
    logger.info("Entraînement : %d yeux | validation : %d yeux", len(splits.train), len(splits.val))

    preprocessor = Preprocessor(config.preprocessing)
    train_ds = ExamDataset(splits.train, preprocessor, params.positive_labels, params.augment)
    val_ds = ExamDataset(splits.val, preprocessor, params.positive_labels)
    loader_options: dict[str, Any] = {
        "batch_size": params.batch_size,
        "num_workers": params.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": params.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)

    n_pos = sum(train_ds.target(i) for i in range(len(train_ds)))
    pos_weight = (len(train_ds) - n_pos) / max(n_pos, 1)

    run_dir = _create_run_dir(config)
    model = build_model(config.model)
    trainer = Trainer(model, params, device, pos_weight)
    metadata_base = {
        "experiment": config.name,
        "positive_labels": [d.value for d in params.positive_labels],
        "git_commit": git_commit(),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    callbacks: list[TrainingCallback] = [
        BestModelCheckpoint(run_dir / "model.pt", model, config, metadata_base),
        CsvHistoryLogger(run_dir / "history.csv"),
        TensorBoardLogger(run_dir / "tensorboard"),
    ]
    history = trainer.fit(train_loader, val_loader, callbacks)
    best = min(history, key=lambda r: r.val_loss)
    _write_summary(run_dir, config, best, len(history), splits, metadata_base)
    logger.info(
        "Meilleure époque : %d (val_AUC=%.3f). Résultats : %s", best.epoch, best.val.auc, run_dir
    )
    return ExperimentResult(run_dir, best, tuple(history))


class BestModelCheckpoint(TrainingCallback):
    """Sauvegarde le modèle chaque fois que la validation s'améliore."""

    def __init__(
        self, path: Path, model: torch.nn.Module, config: ExperimentConfig, metadata: dict[str, Any]
    ) -> None:
        self.path, self.model, self.config, self.metadata = path, model, config, metadata

    def on_epoch_end(self, record: EpochRecord) -> None:
        if record.improved:
            save_checkpoint(
                self.path,
                self.model,
                self.config.model,
                self.config.preprocessing,
                self.metadata | {"epoch": record.epoch, "val_metrics": record.val.to_dict()},
            )


def _create_run_dir(config: ExperimentConfig) -> Path:
    run_dir = config.output_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "experiment.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copy(config.data_config, run_dir / "dataset.yaml")
    return run_dir


def _write_summary(
    run_dir: Path,
    config: ExperimentConfig,
    best: EpochRecord,
    epochs_run: int,
    splits: DatasetSplits,
    metadata: dict[str, Any],
) -> None:
    summary = {
        "qarniya_version": __version__,
        **metadata,
        "best_epoch": best.epoch,
        "epochs_run": epochs_run,
        "epochs_max": config.training.epochs,
        "val_loss": best.val_loss,
        "val_metrics": best.val.to_dict(),
        "n_train": len(splits.train),
        "n_val": len(splits.val),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
