"""Deuxième avis « chiffres » : apprentissage, sauvegarde JSON, garde-fou de saisie."""

from pathlib import Path

import pytest

from qarniya.domain import CornealExam, CornealMaps, EyeSide, MapType, QarniyaError
from qarniya.services import ClinicalOpinionModel, validate_clinical_values


def exam(i: int, kmax: float, thinnest: float) -> CornealExam:
    maps = CornealMaps({m: Path(f"{i}{m.value}.png") for m in MapType})
    clinical = {"kmax": kmax, "pachymetry_thinnest": thinnest}
    return CornealExam(f"e{i}", f"P{i}", EyeSide.RIGHT, maps, clinical=clinical)


@pytest.fixture
def model() -> ClinicalOpinionModel:
    healthy = [exam(i, 43 + i * 0.05, 540 - i) for i in range(30)]
    sick = [exam(100 + i, 52 + i * 0.1, 450 - i) for i in range(30)]
    return ClinicalOpinionModel.fit(healthy + sick, [0] * 30 + [1] * 30)


def test_steep_and_thin_cornea_is_high_risk(model: ClinicalOpinionModel) -> None:
    assert model.probability({"kmax": 55.0, "pachymetry_thinnest": 440.0}) > 0.9
    assert model.probability({"kmax": 43.0, "pachymetry_thinnest": 545.0}) < 0.1


def test_json_roundtrip_gives_same_opinion(model: ClinicalOpinionModel, tmp_path: Path) -> None:
    model.save(tmp_path / "clinical.json")
    loaded = ClinicalOpinionModel.load(tmp_path / "clinical.json")
    values = {"kmax": 48.0, "pachymetry_thinnest": 500.0}
    assert loaded.probability(values) == pytest.approx(model.probability(values))


@pytest.mark.parametrize(
    "values",
    [{"kmax": 4.75}, {"kmax": 120.0}, {"pachymetry_thinnest": 0.48}, {"pachymetry_thinnest": 5000}],
)
def test_implausible_values_are_refused(values: dict[str, float]) -> None:
    with pytest.raises(QarniyaError, match="plausibles"):
        validate_clinical_values(values)


def test_missing_value_is_reported(model: ClinicalOpinionModel) -> None:
    with pytest.raises(QarniyaError, match="manquantes"):
        model.probability({"kmax": 47.0})
