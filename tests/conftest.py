"""Fixtures partagées : de faux mini-datasets créés à la volée dans un dossier temporaire.

Ainsi les tests tournent partout (PC, GitHub CI) sans télécharger les vraies données.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

CORNORB_MAPS = ("Axial", "Anterior", "Posterior", "Pachymetry")
FRENCH_MAPS = ("axiale", "anterieure", "posterieure", "pachymetrie")
MAP_KEYS = ("axial", "anterior", "posterior", "pachymetry")


def write_map(path: Path, size: int = 160, seed: int = 0) -> None:
    """Petite image colorée non uniforme (passe le contrôle qualité)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path)


def write_yaml(path: Path, content: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(content, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def cornorb_like(tmp_path: Path) -> Callable[..., Path]:
    """Fabrique un dataset au format CornOrb et renvoie le chemin de son YAML."""

    def make(
        rows: list[dict[str, str]],
        missing_images: tuple[str, ...] = (),
        extra_config: dict[str, Any] | None = None,
    ) -> Path:
        root = tmp_path / "cornorb"
        header = ["patient_code", "eye", "label", "kmax_value_D", "split_80_20", "fold"]
        lines = [",".join(header)] + [",".join(r.get(h, "") for h in header) for r in rows]
        root.mkdir(parents=True, exist_ok=True)
        (root / "data.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

        for i, r in enumerate(rows):
            folder_id = f"{r['patient_code']}_{r['eye']}"
            if folder_id in missing_images:
                continue
            for m in CORNORB_MAPS:
                patient, eye = r["patient_code"], r["eye"]
                write_map(root / "imgs" / patient / eye / f"{patient}_{eye}_{m}.png", seed=i)

        config: dict[str, Any] = {
            "name": "test_cornorb",
            "adapter": "csv_images",
            "root": str(root),
            "csv": "data.csv",
            "image_pattern": "imgs/{patient}/{eye}/{patient}_{eye}_{map}.png",
            "columns": {"patient": "patient_code", "eye": "eye", "label": "label"},
            "eye_values": {"OD": "OD", "OS": "OS"},
            "map_tokens": dict(zip(MAP_KEYS, CORNORB_MAPS, strict=True)),
            "label_values": {"0": "normal", "1": "keratoconus"},
            "clinical_columns": {"kmax": "kmax_value_D"},
            "split": {
                "strategy": "column",
                "column": "split_80_20",
                "val_fold_column": "fold",
                "val_fold": 1,
            },
        }
        config.update(extra_config or {})
        return write_yaml(tmp_path / "cornorb.yaml", config)

    return make


@pytest.fixture
def doctor_folder(tmp_path: Path) -> Callable[[list[str]], Path]:
    """Fabrique un dossier d'images au format du médecin (P017_OG_axiale.png) et renvoie le YAML."""

    def make(file_names: list[str]) -> Path:
        root = tmp_path / "medecin"
        for i, name in enumerate(file_names):
            write_map(root / name, seed=i)
        root.mkdir(parents=True, exist_ok=True)
        return write_yaml(
            tmp_path / "medecin.yaml",
            {
                "name": "medecin",
                "adapter": "image_folder",
                "root": str(root),
                "image_pattern": "{patient}_{eye}_{map}.png",
                "eye_values": {"OD": "OD", "OG": "OS"},
                "map_tokens": dict(zip(MAP_KEYS, FRENCH_MAPS, strict=True)),
            },
        )

    return make


def doctor_eye_files(patient: str, eye: str) -> list[str]:
    return [f"{patient}_{eye}_{m}.png" for m in FRENCH_MAPS]


# ---------- mini-expérience d'entraînement (partagée par entraînement et évaluation) ----------
def synthetic_rows() -> list[dict[str, str]]:
    rows = []
    for i in range(12):
        split, fold = ("train", "1") if i < 4 else ("train", "2")
        if i >= 10:
            split, fold = "test", ""
        for eye in ("OD", "OS"):
            kmax = 43.0 + 8.0 * (i % 2) + 0.1 * i  # Kmax plus élevé chez les malades
            rows.append(
                {"patient_code": f"P{i:02d}", "eye": eye, "label": str(i % 2),
                 "kmax_value_D": f"{kmax:.1f}", "split_80_20": split, "fold": fold}
            )  # fmt: skip
    return rows


@pytest.fixture
def experiment_yaml(cornorb_like: Callable[..., Path], tmp_path: Path) -> Path:
    data_yaml = cornorb_like(synthetic_rows())
    return write_yaml(
        tmp_path / "experiment.yaml",
        {
            "name": "test_run",
            "data_config": str(data_yaml),
            "model": {"backbone": "resnet18", "pretrained": False, "hidden_dim": 16},
            "preprocessing": {"image_size": 64},
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "num_workers": 0,
                "device": "cpu",
                "mixed_precision": False,
            },
            "output_dir": str(tmp_path / "runs"),
        },
    )


@pytest.fixture
def trained_run(experiment_yaml: Path) -> Path:
    """Un mini-modèle entraîné 1 époque (CPU, quelques secondes) et son dossier de résultats."""
    from qarniya.training import ExperimentConfig, run_experiment

    return run_experiment(ExperimentConfig.from_yaml(experiment_yaml), epochs=1).run_dir
