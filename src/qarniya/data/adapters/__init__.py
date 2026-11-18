"""Adaptateurs de datasets : chaque source (CornOrb, dossier, CSV…) implémente la même interface.

Importer ce paquet enregistre automatiquement tous les adaptateurs disponibles.
"""

from qarniya.data.adapters import csv_images, image_folder  # noqa: F401  (enregistrement)
from qarniya.data.adapters.base import DatasetAdapter, LoadResult, SkippedRecord
from qarniya.data.adapters.registry import available_adapters, build_adapter, register_adapter

__all__ = [
    "DatasetAdapter",
    "LoadResult",
    "SkippedRecord",
    "available_adapters",
    "build_adapter",
    "register_adapter",
]
