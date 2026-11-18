"""Callbacks : actions déclenchées à chaque époque, sans toucher à la boucle d'entraînement.

Sauvegarder le meilleur modèle, écrire l'historique, alimenter TensorBoard… sont des callbacks.
En ajouter un (ex. MLflow, Weights & Biases) ne change pas `Trainer` (principe ouvert/fermé).
"""

from __future__ import annotations

import csv
import importlib
import logging
import math
from pathlib import Path
from typing import Any

from qarniya.training.records import EpochRecord

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base des callbacks : chaque méthode est facultative (ne rien faire par défaut)."""

    def on_epoch_end(self, record: EpochRecord) -> None:
        """Appelé après chaque époque (entraînement + validation)."""

    def on_train_end(self, history: list[EpochRecord]) -> None:
        """Appelé une fois à la fin."""


class CsvHistoryLogger(TrainingCallback):
    """Ajoute une ligne à history.csv à CHAQUE époque : lisible pendant l'entraînement."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_epoch_end(self, record: EpochRecord) -> None:
        row = record.to_row()
        is_new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)


class TensorBoardLogger(TrainingCallback):
    """Courbes en direct dans TensorBoard :  tensorboard --logdir outputs/runs"""

    def __init__(self, log_dir: Path) -> None:
        self._writer: Any = None
        try:
            # import dynamique : la dépendance reste facultative (et non typée)
            summary_writer: Any = importlib.import_module("torch.utils.tensorboard").SummaryWriter
        except ImportError:  # tensorboard non installé : on continue sans courbes
            logger.warning("TensorBoard non installé : pas de courbes (pip install tensorboard)")
            return
        self._writer = summary_writer(log_dir=str(log_dir))

    def on_epoch_end(self, record: EpochRecord) -> None:
        if self._writer is None:
            return
        scalars = {
            "1_perte/entrainement": record.train_loss,
            "1_perte/validation": record.val_loss,
            "2_validation/AUC": record.val.auc,
            "2_validation/sensibilite": record.val.sensitivity,
            "2_validation/specificite": record.val.specificity,
            "2_validation/kc_manques": record.val.fn,
            "3_optimisation/pas_apprentissage": record.learning_rate,
        }
        for tag, value in scalars.items():
            if not (isinstance(value, float) and math.isnan(value)):
                self._writer.add_scalar(tag, value, record.epoch)
        self._writer.flush()  # visible immédiatement dans le tableau de bord

    def on_train_end(self, history: list[EpochRecord]) -> None:
        if self._writer is not None:
            self._writer.close()
