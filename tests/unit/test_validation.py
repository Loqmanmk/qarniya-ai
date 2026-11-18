"""Le contrôle qualité refuse les images inexploitables."""

from pathlib import Path

import numpy as np
from PIL import Image

from qarniya.data import ImageQualityChecker
from qarniya.domain import CornealExam, CornealMaps, EyeSide, MapType
from tests.conftest import write_map

checker = ImageQualityChecker()


def save(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path)
    return path


def test_good_image_passes(tmp_path: Path) -> None:
    write_map(tmp_path / "ok.png")
    assert checker.check_image(tmp_path / "ok.png") == []


def test_blank_image_is_rejected(tmp_path: Path) -> None:
    path = save(tmp_path / "black.png", np.zeros((200, 200, 3)))
    assert "image vide ou uniforme" in checker.check_image(path)


def test_tiny_image_is_rejected(tmp_path: Path) -> None:
    path = save(tmp_path / "tiny.png", np.random.default_rng(0).integers(0, 255, (40, 40, 3)))
    assert any("résolution" in p for p in checker.check_image(path))


def test_panoramic_image_is_rejected(tmp_path: Path) -> None:
    path = save(tmp_path / "wide.png", np.random.default_rng(0).integers(0, 255, (200, 600, 3)))
    assert any("proportions" in p for p in checker.check_image(path))


def test_corrupted_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"pas une image")
    assert checker.check_image(path) == ["fichier illisible ou absent"]


def test_exam_report_points_to_the_bad_map(tmp_path: Path) -> None:
    paths = {}
    for m in MapType:
        paths[m] = tmp_path / f"{m.value}.png"
        write_map(paths[m])
    save(paths[MapType.PACHYMETRY], np.zeros((200, 200, 3)))
    exam = CornealExam("e1", "P1", EyeSide.RIGHT, CornealMaps(paths))

    report = checker.check_exam(exam)
    assert not report.is_acceptable
    assert {i.map_type for i in report.issues} == {MapType.PACHYMETRY}  # seule la carte fautive


def test_grayscale_image_is_not_a_topography_map(tmp_path: Path) -> None:
    write_map(tmp_path / "color.png")
    Image.open(tmp_path / "color.png").convert("L").convert("RGB").save(tmp_path / "gray.png")
    assert any("sans couleurs" in p for p in checker.check_image(tmp_path / "gray.png"))
