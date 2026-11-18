"""Boucle d'entraînement générique : elle ne connaît ni le dataset, ni l'architecture.

Elle reçoit un modèle, des chargeurs de données et des paramètres, et rend un historique.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from qarniya.evaluation import compute_binary_metrics
from qarniya.training.callbacks import TrainingCallback
from qarniya.training.config import TrainingParams
from qarniya.training.records import EpochRecord

logger = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Predictions:
    targets: np.ndarray
    probabilities: np.ndarray
    loss: float


class Trainer:
    def __init__(
        self, model: nn.Module, params: TrainingParams, device: torch.device, pos_weight: float
    ) -> None:
        self.model = model.to(device)
        self.params = params
        self.device = device
        self.use_amp = params.mixed_precision and device.type == "cuda"
        # pos_weight compense le déséquilibre : il y a moins de malades que de sains
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=params.epochs
        )
        self.scaler = GradScaler(enabled=self.use_amp)

    def fit(
        self,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch],
        callbacks: Sequence[TrainingCallback] = (),
    ) -> list[EpochRecord]:
        """Entraîne et prévient les callbacks à chaque époque. Rend l'historique complet."""
        history: list[EpochRecord] = []
        best_loss, epochs_without_progress = math.inf, 0

        for epoch in range(1, self.params.epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            preds = self.predict(val_loader)
            improved = preds.loss < best_loss
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=preds.loss,
                val=compute_binary_metrics(preds.targets.tolist(), preds.probabilities.tolist()),
                # float() : selon la version de PyTorch, get_last_lr() rend un float ou un Tensor
                learning_rate=float(self.scheduler.get_last_lr()[0]),
                improved=improved,
            )
            self.scheduler.step()
            history.append(record)
            logger.info(
                "époque %d/%d  perte=%.4f  val_perte=%.4f  val_AUC=%.3f  sens=%.3f  spéc=%.3f%s",
                epoch, self.params.epochs, train_loss, preds.loss, record.val.auc,
                record.val.sensitivity, record.val.specificity, "  ★" if improved else "",
            )  # fmt: skip

            for callback in callbacks:
                callback.on_epoch_end(record)

            if improved:
                best_loss, epochs_without_progress = preds.loss, 0
            else:
                epochs_without_progress += 1
                if epochs_without_progress >= self.params.early_stopping_patience:
                    logger.info(
                        "Arrêt anticipé : pas d'amélioration depuis %d époques",
                        epochs_without_progress,
                    )
                    break
        for callback in callbacks:
            callback.on_train_end(history)
        return history

    def _train_one_epoch(self, loader: Iterable[Batch]) -> float:
        self.model.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss = self.loss_fn(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item() * len(y)
            count += len(y)
        return total / max(count, 1)

    @torch.no_grad()
    def predict(self, loader: Iterable[Batch]) -> Predictions:
        self.model.eval()
        targets, probs, total = [], [], 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(x).float()
            total += self.loss_fn(logits, y).item() * len(y)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy())
        y_true = np.concatenate(targets).astype(int) if targets else np.array([], dtype=int)
        y_prob = np.concatenate(probs) if probs else np.array([])
        return Predictions(y_true, y_prob, total / max(len(y_true), 1))
