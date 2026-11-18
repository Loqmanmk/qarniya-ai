"""API web de bout en bout (sans lancer de serveur : TestClient)."""

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from qarniya.api import create_app
from qarniya.api.app import MAX_UPLOAD_BYTES
from qarniya.data import DatasetConfig, build_adapter
from qarniya.services import AnalysisService

pytestmark = pytest.mark.integration
FIELDS = ("axial", "anterior", "posterior", "pachymetry")


@pytest.fixture
def client(trained_run: Path) -> Iterator[TestClient]:
    with TestClient(create_app(AnalysisService.from_run(trained_run))) as c:
        yield c


@pytest.fixture
def map_files(trained_run: Path) -> dict[str, bytes]:
    exam = build_adapter(DatasetConfig.from_yaml(trained_run / "dataset.yaml")).load().exams[0]
    return {f: p.read_bytes() for f, p in zip(FIELDS, exam.maps.ordered(), strict=True)}


def upload(maps: dict[str, bytes], prefix: str = "") -> dict[str, tuple[str, bytes, str]]:
    return {f"{prefix}{k}": (f"{k}.png", v, "image/png") for k, v in maps.items()}


def test_health_and_web_page(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["model"] == "test_run"
    page = client.get("/")
    assert page.status_code == 200 and "Qarniya AI" in page.text
    assert client.get("/static/app.js").status_code == 200


def test_analysis_returns_score_level_and_heatmaps(
    client: TestClient, map_files: dict[str, bytes]
) -> None:
    response = client.post("/api/analyses", data={"eye": "OD"}, files=upload(map_files))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "analysed"
    assert 0 <= body["probability"] <= 1
    assert body["final_level"] in {"faible", "a_surveiller", "eleve"}
    assert [m["map_type"] for m in body["maps"]] == list(FIELDS)
    assert sum(m["importance"] for m in body["maps"]) == pytest.approx(1.0)
    assert body["disclaimer"].startswith("Outil d'aide")


def test_security_headers_are_set(client: TestClient) -> None:
    headers = client.get("/api/health").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-frame-options"] == "DENY"
    assert headers["cache-control"] == "no-store"


def test_non_image_file_is_refused(client: TestClient, map_files: dict[str, bytes]) -> None:
    files = upload(map_files)
    files["axial"] = ("axial.png", b"<script>alert(1)</script>", "image/png")  # faux PNG
    response = client.post("/api/analyses", data={"eye": "OD"}, files=files)
    assert response.status_code == 415
    assert "Courbure axiale" in response.json()["detail"]


def test_unsupported_format_is_refused(client: TestClient, map_files: dict[str, bytes]) -> None:
    gif = io.BytesIO()
    Image.new("RGB", (200, 200), "red").save(gif, format="GIF")
    files = upload(map_files)
    files["posterior"] = ("posterior.gif", gif.getvalue(), "image/gif")
    response = client.post("/api/analyses", data={"eye": "OD"}, files=files)
    assert response.status_code == 415 and "GIF" in response.json()["detail"]


def test_too_large_file_is_refused(client: TestClient, map_files: dict[str, bytes]) -> None:
    files = upload(map_files)
    files["pachymetry"] = ("big.png", b"0" * (MAX_UPLOAD_BYTES + 1), "image/png")
    response = client.post("/api/analyses", data={"eye": "OD"}, files=files)
    assert response.status_code == 413


def test_missing_map_or_bad_eye_is_a_clear_422(
    client: TestClient, map_files: dict[str, bytes]
) -> None:
    files = upload(map_files)
    del files["axial"]
    assert client.post("/api/analyses", data={"eye": "OD"}, files=files).status_code == 422
    bad_eye = client.post("/api/analyses", data={"eye": "XX"}, files=upload(map_files))
    assert bad_eye.status_code == 422


def test_implausible_kmax_is_refused(client: TestClient, map_files: dict[str, bytes]) -> None:
    response = client.post(
        "/api/analyses", data={"eye": "OD", "kmax": "4.75"}, files=upload(map_files)
    )
    assert response.status_code == 422 and "plausibles" in response.json()["detail"]


def test_progression_compares_two_exams(client: TestClient, map_files: dict[str, bytes]) -> None:
    files = upload(map_files, "before_") | upload(map_files, "after_")
    response = client.post("/api/progressions", data={"eye": "OS"}, files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["comparable"] is True
    assert body["delta"] == pytest.approx(0.0, abs=1e-6)  # mêmes images : aucune évolution
    assert body["verdict"].startswith("Score stable")
