"""La configuration doit refuser clairement tout YAML incohérent."""

from pathlib import Path
from typing import Any

import pytest

from qarniya.data import ConfigError, DatasetConfig
from qarniya.domain import EyeSide, MapType
from tests.conftest import write_yaml

VALID: dict[str, Any] = {
    "name": "x",
    "root": "data/x",
    "image_pattern": "{patient}_{eye}_{map}.png",
    "eye_values": {"OD": "OD", "OG": "OS"},
    "map_tokens": {"axial": "a", "anterior": "b", "posterior": "c", "pachymetry": "d"},
}


def test_valid_config_is_parsed(tmp_path: Path) -> None:
    config = DatasetConfig.from_yaml(write_yaml(tmp_path / "c.yaml", VALID))
    assert config.eye_values["OG"] is EyeSide.LEFT
    assert config.map_tokens[MapType.PACHYMETRY] == "d"


def test_missing_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="introuvable"):
        DatasetConfig.from_yaml(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"map_tokens": {"axial": "a"}}, "map_tokens"),
        ({"image_pattern": "{patient}.png"}, "image_pattern"),
        ({"eye_values": {"OD": "droit"}}, "eye_values"),
        ({"unknown_field": 1}, "unknown_field"),  # faute de frappe détectée
        ({"columns": {"patient": "p", "eye": "e", "label": "l"}}, "label_values"),
        ({"split": {"strategy": "column"}}, "column"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    path = write_yaml(tmp_path / "c.yaml", VALID | change)
    with pytest.raises(ConfigError, match=message):
        DatasetConfig.from_yaml(path)


@pytest.mark.parametrize("name", ["cornorb.yaml", "medecin.yaml", "nouveaux_patients.yaml"])
def test_project_configs_are_valid(name: str) -> None:
    """Les YAML livrés avec le projet doivent toujours rester valides."""
    root = Path(__file__).resolve().parents[2]
    DatasetConfig.from_yaml(root / "configs" / "data" / name)
