"""Couche données : lecture des datasets via des adaptateurs, découpage et contrôle qualité.

Point d'entrée habituel :
    config = DatasetConfig.from_yaml("configs/data/cornorb.yaml")
    result = build_adapter(config).load()
    splits = split_dataset(result, config.split)
"""

from qarniya.data.adapters import LoadResult, SkippedRecord, available_adapters, build_adapter
from qarniya.data.config import ConfigError, DatasetConfig
from qarniya.data.splitting import DatasetSplits, Split, split_dataset
from qarniya.data.validation import ImageQualityChecker, QualityReport

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "DatasetSplits",
    "ImageQualityChecker",
    "LoadResult",
    "QualityReport",
    "SkippedRecord",
    "Split",
    "available_adapters",
    "build_adapter",
    "split_dataset",
]
