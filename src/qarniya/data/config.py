"""Description d'un dataset par configuration (fichier YAML validé par pydantic).

Changer de dataset = écrire un nouveau YAML. Aucune ligne de code à modifier.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator

from qarniya.config_utils import ConfigError, FrozenModel, load_yaml_model
from qarniya.domain import Diagnosis, EyeSide, MapType

PLACEHOLDERS = ("{patient}", "{eye}", "{map}")

__all__ = ["ColumnMapping", "ConfigError", "DatasetConfig", "SplitConfig", "SplitStrategy"]


class ColumnMapping(FrozenModel):
    """Nom des colonnes du CSV source qui correspondent à nos notions métier."""

    patient: str
    eye: str
    label: str | None = None
    exam_date: str | None = None


class SplitStrategy(str, Enum):
    COLUMN = "column"  # le dataset fournit déjà son découpage (ex. CornOrb)
    PATIENT = "patient"  # découpage aléatoire par patient (ex. données du médecin)


class SplitConfig(FrozenModel):
    """Comment découper en entraînement / validation / test."""

    strategy: SplitStrategy = SplitStrategy.PATIENT
    # stratégie COLUMN
    column: str | None = None
    train_value: str = "train"
    test_value: str = "test"
    val_fold_column: str | None = None
    val_fold: str | int | None = None  # comparé sous forme de texte au contenu du CSV
    # stratégie PATIENT
    test_size: float = Field(default=0.2, gt=0, lt=1)
    val_size: float = Field(default=0.15, gt=0, lt=1)
    seed: int = 42

    @model_validator(mode="after")
    def _check_column_strategy(self) -> SplitConfig:
        if self.strategy is SplitStrategy.COLUMN and not self.column:
            raise ValueError("la stratégie 'column' exige le champ 'column'")
        if (self.val_fold_column is None) != (self.val_fold is None):
            raise ValueError("'val_fold_column' et 'val_fold' vont ensemble")
        return self


class DatasetConfig(FrozenModel):
    """Tout ce qu'il faut savoir pour lire un dataset de topographie cornéenne."""

    name: str
    adapter: str = "csv_images"
    root: Path
    csv: Path | None = None
    # chemin relatif à `root`, avec {patient}, {eye} et {map}
    image_pattern: str
    columns: ColumnMapping | None = None
    # valeur trouvée dans les données -> notre valeur (ex. "OG" -> OS)
    eye_values: dict[str, EyeSide]
    map_tokens: dict[MapType, str]
    label_values: dict[str, Diagnosis] = Field(default_factory=dict)
    # nom métier -> nom de colonne (ex. kmax -> kmax_value_D)
    clinical_columns: dict[str, str] = Field(default_factory=dict)
    device: str | None = None
    # corrections connues de codes patient (ex. Excel transforme "3E132" en "3E+132")
    patient_code_fixes: dict[str, str] = Field(default_factory=dict)
    drop_ambiguous_duplicates: bool = True
    split: SplitConfig = Field(default_factory=SplitConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> DatasetConfig:
        missing_maps = [m.value for m in MapType if m not in self.map_tokens]
        if missing_maps:
            raise ValueError(f"map_tokens doit définir les 4 cartes, manque : {missing_maps}")
        missing_ph = [p for p in PLACEHOLDERS if p not in self.image_pattern]
        if missing_ph:
            raise ValueError(f"image_pattern doit contenir {missing_ph}")
        if self.columns and self.columns.label and not self.label_values:
            raise ValueError("une colonne 'label' est définie : 'label_values' est obligatoire")
        return self

    @property
    def csv_path(self) -> Path | None:
        return self.root / self.csv if self.csv else None

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        return load_yaml_model(cls, path)
