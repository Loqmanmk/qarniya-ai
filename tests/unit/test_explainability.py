"""Carte de chaleur et détecteur d'images inhabituelles."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from qarniya.domain import EyeSide, MapType
from qarniya.explainability import GradCam, overlay
from qarniya.models import ModelSpec, NoveltyDetector, build_model


@pytest.fixture(scope="module")
def tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return build_model(ModelSpec(pretrained=False, hidden_dim=16)).eval()


# ---------- Grad-CAM ----------
def test_gradcam_gives_one_normalised_heatmap_per_map(tiny_model: nn.Module) -> None:
    x = torch.randn(1, 4, 3, 64, 64)
    explanation = GradCam(tiny_model).explain(x, EyeSide.RIGHT)

    assert set(explanation.heatmaps) == set(MapType)
    all_heat = np.stack(list(explanation.heatmaps.values()))
    assert all_heat.min() >= 0 and all_heat.max() <= 1
    assert sum(explanation.map_importance.values()) == pytest.approx(1.0)
    with torch.no_grad():
        expected = torch.sigmoid(tiny_model(x)).item()
    assert explanation.probability == pytest.approx(expected, abs=1e-5)


def test_left_eye_heatmap_is_put_back_in_original_orientation(tiny_model: nn.Module) -> None:
    x = torch.randn(1, 4, 3, 64, 64)
    right = GradCam(tiny_model).explain(x, EyeSide.RIGHT)
    left = GradCam(tiny_model).explain(x, EyeSide.LEFT, mirrored_left_eye=True)
    for m in MapType:
        assert np.allclose(left.heatmaps[m], right.heatmaps[m][:, ::-1])


def test_gradcam_requires_an_explainable_model() -> None:
    with pytest.raises(TypeError, match="explain_layer"):
        GradCam(nn.Linear(2, 1))


def test_overlay_without_heat_is_the_grayscale_map(tmp_path: Path) -> None:
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (50, 50, 3), dtype=np.uint8))
    result = overlay(image, np.zeros((8, 8)))
    gray = np.asarray(image.convert("L").convert("RGB"))
    assert result.shape == (50, 50, 3)
    assert np.array_equal(result, gray)


# ---------- détecteur d'images inhabituelles ----------
def _cloud(n: int, center: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros(32)
    base[0] = center
    return base + rng.normal(0, 0.05, (n, 32)) + 1.0


def test_novelty_flags_far_points_and_about_one_percent_of_calibration() -> None:
    train, calibration = _cloud(300, 0.0, 1), _cloud(500, 0.0, 2)
    detector = NoveltyDetector.fit(train, calibration, quantile=0.99)
    assert detector.is_unusual(calibration).mean() == pytest.approx(0.01, abs=0.01)

    usual, unusual = _cloud(5, 0.0, 3), _cloud(5, 0.0, 4)
    unusual[:, 1:] *= -1  # empreinte tournée : rien à voir avec l'entraînement
    assert not detector.is_unusual(usual).any()
    assert detector.is_unusual(unusual).all()


def test_novelty_detector_save_and_load(tmp_path: Path) -> None:
    detector = NoveltyDetector.fit(_cloud(50, 0.0, 1), _cloud(50, 0.0, 2))
    detector.save(tmp_path / "novelty.npz")
    loaded = NoveltyDetector.load(tmp_path / "novelty.npz")
    probe = _cloud(10, 0.0, 5)
    assert loaded.threshold == pytest.approx(detector.threshold)
    assert np.allclose(loaded.distances(probe), detector.distances(probe), atol=1e-3)


def test_novelty_needs_enough_reference_eyes() -> None:
    with pytest.raises(ValueError, match="au moins"):
        NoveltyDetector.fit(_cloud(3, 0.0, 1), _cloud(10, 0.0, 2), k=5)
