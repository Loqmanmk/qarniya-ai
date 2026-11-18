"""Ce qu'on retient de chaque époque d'entraînement."""

from __future__ import annotations

from dataclasses import dataclass

from qarniya.evaluation import BinaryMetrics


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val: BinaryMetrics
    learning_rate: float
    improved: bool  # nouveau meilleur modèle (perte de validation la plus basse)

    def to_row(self) -> dict[str, float | int | bool]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rate": self.learning_rate,
            "improved": self.improved,
            **{f"val_{k}": v for k, v in self.val.to_dict().items()},
        }
