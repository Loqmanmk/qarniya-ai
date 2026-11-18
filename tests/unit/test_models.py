"""Modèles, préparation des images et sauvegarde. Tout tourne sur CPU, sans poids téléchargés."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qarniya.config_utils import ConfigError
from qarniya.domain import CornealExam, CornealMaps, EyeSide, MapType
from qarniya.models import (
    ModelSpec,
    PreprocessingSpec,
    Preprocessor,
    available_models,
    build_model,
    load_checkpoint,
    save_checkpoint,
)
from tests.conftest import write_map

SMALL = PreprocessingSpec(image_size=64)


def make_exam(tmp_path: Path, eye: EyeSide) -> CornealExam:
    paths = {}
    for i, m in enumerate(MapType):
        paths[m] = tmp_path / f"{m.value}.png"
        write_map(paths[m], size=96, seed=i)
    return CornealExam("e1", "P1", eye, CornealMaps(paths))


# ---------- préparation ----------
def test_preprocessor_outputs_four_normalised_maps(tmp_path: Path) -> None:
    x = Preprocessor(SMALL).to_tensor(make_exam(tmp_path, EyeSide.RIGHT))
    assert x.shape == (4, 3, 64, 64)
    assert x.dtype == torch.float32


def test_left_eye_is_mirrored(tmp_path: Path) -> None:
    right = Preprocessor(SMALL).to_tensor(make_exam(tmp_path, EyeSide.RIGHT))
    left = Preprocessor(SMALL).to_tensor(make_exam(tmp_path, EyeSide.LEFT))
    assert torch.allclose(left, torch.flip(right, dims=[-1]), atol=1e-5)


def test_mirroring_can_be_disabled(tmp_path: Path) -> None:
    spec = PreprocessingSpec(image_size=64, mirror_left_eye=False)
    right = Preprocessor(spec).to_tensor(make_exam(tmp_path, EyeSide.RIGHT))
    left = Preprocessor(spec).to_tensor(make_exam(tmp_path, EyeSide.LEFT))
    assert torch.equal(left, right)


def test_augmentation_changes_the_image_but_not_its_shape(tmp_path: Path) -> None:
    torch.manual_seed(0)
    exam = make_exam(tmp_path, EyeSide.RIGHT)
    plain = Preprocessor(SMALL).to_tensor(exam)
    augmented = Preprocessor(SMALL).to_tensor(exam, augment=True)
    assert augmented.shape == plain.shape
    assert not torch.equal(augmented, plain)


# ---------- modèles ----------
@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0"])
def test_model_returns_one_logit_per_eye(backbone: str) -> None:
    model = build_model(ModelSpec(backbone=backbone, pretrained=False)).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 4, 3, 64, 64))
    assert logits.shape == (2,)


def test_wrong_number_of_maps_is_rejected() -> None:
    model = build_model(ModelSpec(pretrained=False))
    with pytest.raises(ValueError, match="4 cartes"):
        model(torch.randn(1, 3, 3, 64, 64))


def test_unknown_model_or_backbone_gives_clear_error() -> None:
    assert "multimap" in available_models()
    with pytest.raises(ConfigError, match="Modèle inconnu"):
        build_model(ModelSpec(name="transformer_magique"))
    with pytest.raises(ConfigError, match="Backbone inconnu"):
        build_model(ModelSpec(backbone="vgg_inexistant", pretrained=False))


# ---------- sauvegarde ----------
def test_checkpoint_roundtrip_gives_identical_predictions(tmp_path: Path) -> None:
    spec = ModelSpec(pretrained=False, hidden_dim=16)
    model = build_model(spec).eval()
    x = torch.randn(2, 4, 3, 64, 64)
    with torch.no_grad():
        expected = model(x)

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, spec, SMALL, {"epoch": 3})
    loaded = load_checkpoint(path)

    with torch.no_grad():
        assert torch.allclose(loaded.model(x), expected, atol=1e-6)
    assert loaded.preprocessing == SMALL  # la préparation voyage AVEC le modèle
    assert loaded.model_spec == spec
    assert loaded.metadata["epoch"] == 3


def test_missing_checkpoint_gives_clear_error(tmp_path: Path) -> None:
    from qarniya.domain.exceptions import QarniyaError

    with pytest.raises(QarniyaError, match="introuvable"):
        load_checkpoint(tmp_path / "absent.pt")


def test_write_map_helper_produces_valid_image(tmp_path: Path) -> None:
    write_map(tmp_path / "x.png", size=50)
    assert np.asarray(Image.open(tmp_path / "x.png")).shape == (50, 50, 3)
