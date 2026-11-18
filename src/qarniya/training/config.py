"""Configuration d'une expérience : QUELLES données, QUEL modèle, COMMENT l'entraîner."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field

from qarniya.config_utils import FrozenModel, load_yaml_model
from qarniya.domain import Diagnosis
from qarniya.models import ModelSpec, PreprocessingSpec


class Device(str, Enum):
    AUTO = "auto"  # GPU si disponible, sinon CPU
    CUDA = "cuda"
    CPU = "cpu"


class TrainingParams(FrozenModel):
    epochs: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    # arrêt si la perte de validation stagne N époques de suite (évite le surapprentissage)
    early_stopping_patience: int = Field(default=5, ge=1)
    augment: bool = True
    mixed_precision: bool = True  # calcul en demi-précision sur GPU : ~2x plus rapide
    num_workers: int = Field(default=2, ge=0)
    device: Device = Device.AUTO
    seed: int = 42
    # diagnostics considérés « positifs » : en dépistage, un suspect doit être signalé
    positive_labels: tuple[Diagnosis, ...] = (Diagnosis.KERATOCONUS, Diagnosis.SUSPECT)


class ExperimentConfig(FrozenModel):
    name: str
    data_config: Path
    model: ModelSpec = Field(default_factory=ModelSpec)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    training: TrainingParams = Field(default_factory=TrainingParams)
    output_dir: Path = Path("outputs/runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        return load_yaml_model(cls, path)
