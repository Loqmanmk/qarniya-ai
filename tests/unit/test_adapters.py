"""Les adaptateurs traduisent fidèlement une source en `CornealExam` et tracent chaque exclusion."""

from collections.abc import Callable
from pathlib import Path

import pytest

from qarniya.data import ConfigError, DatasetConfig, available_adapters, build_adapter
from qarniya.data.adapters import DatasetAdapter, LoadResult, register_adapter
from qarniya.domain import Diagnosis, EyeSide, MapType
from tests.conftest import doctor_eye_files


def row(patient: str, eye: str, label: str, kmax: str, split: str, fold: str) -> dict[str, str]:
    return {
        "patient_code": patient,
        "eye": eye,
        "label": label,
        "kmax_value_D": kmax,
        "split_80_20": split,
        "fold": fold,
    }


ROWS = [
    row("P1", "OD", "0", "43.1", "train", "2"),
    row("P1", "OS", "1", "52.0", "train", "2"),
    row("P2", "OD", "0", "", "test", ""),
]


def load(config_path: Path) -> LoadResult:
    return build_adapter(DatasetConfig.from_yaml(config_path)).load()


# ---------- csv_images ----------
def test_csv_adapter_builds_domain_exams(cornorb_like: Callable[..., Path]) -> None:
    result = load(cornorb_like(ROWS))
    assert len(result.exams) == 3
    exam = next(e for e in result.exams if e.patient_code == "P1" and e.eye is EyeSide.LEFT)
    assert exam.diagnosis is Diagnosis.KERATOCONUS
    assert exam.clinical["kmax"] == pytest.approx(52.0)
    assert exam.maps.paths[MapType.AXIAL].name == "P1_OS_Axial.png"


def test_empty_clinical_value_is_simply_absent(cornorb_like: Callable[..., Path]) -> None:
    result = load(cornorb_like(ROWS))
    exam = next(e for e in result.exams if e.patient_code == "P2")
    assert "kmax" not in exam.clinical


def test_missing_images_are_skipped_with_reason(cornorb_like: Callable[..., Path]) -> None:
    result = load(cornorb_like(ROWS, missing_images=("P2_OD",)))
    assert len(result.exams) == 2
    assert result.skipped[0].identifier == "P2_OD"
    assert "images manquantes" in result.skipped[0].reason


def test_ambiguous_duplicates_are_all_skipped(cornorb_like: Callable[..., Path]) -> None:
    duplicated = [*ROWS, {**ROWS[0], "kmax_value_D": "47.0"}]
    result = load(cornorb_like(duplicated))
    assert {e.exam_id for e in result.exams} == {"test_cornorb:P1_OS", "test_cornorb:P2_OD"}
    assert sum("double" in s.reason for s in result.skipped) == 2


def test_unknown_label_is_skipped(cornorb_like: Callable[..., Path]) -> None:
    result = load(cornorb_like([{**ROWS[0], "label": "9"}]))
    assert not result.exams
    assert "étiquette inconnue" in result.skipped[0].reason


def test_patient_code_fixes_are_applied(cornorb_like: Callable[..., Path]) -> None:
    # le CSV contient le code abîmé par Excel, les images sont rangées sous le bon code
    path = cornorb_like([{**ROWS[0], "patient_code": "3E132"}])
    csv = path.parent / "cornorb" / "data.csv"
    csv.write_text(csv.read_text(encoding="utf-8").replace("3E132", "3E+132"), encoding="utf-8")

    assert not load(path).exams  # sans correction : images introuvables

    fixed = cornorb_like(
        [{**ROWS[0], "patient_code": "3E132"}],
        extra_config={"patient_code_fixes": {"3E+132": "3E132"}},
    )
    csv.write_text(csv.read_text(encoding="utf-8").replace("3E132", "3E+132"), encoding="utf-8")
    assert [e.patient_code for e in load(fixed).exams] == ["3E132"]


def test_missing_csv_column_gives_clear_error(cornorb_like: Callable[..., Path]) -> None:
    path = cornorb_like(ROWS, extra_config={"clinical_columns": {"kmax": "colonne_absente"}})
    with pytest.raises(ConfigError, match="colonne_absente"):
        load(path)


def test_predefined_splits_are_read(cornorb_like: Callable[..., Path]) -> None:
    rows = [*ROWS, {**ROWS[0], "patient_code": "P3", "fold": "1"}]
    result = load(cornorb_like(rows))
    assert result.predefined_splits["test_cornorb:P3_OD"] == "val"
    assert result.predefined_splits["test_cornorb:P2_OD"] == "test"
    assert result.predefined_splits["test_cornorb:P1_OD"] == "train"


# ---------- image_folder ----------
def test_folder_adapter_reads_doctor_file_names(doctor_folder: Callable[[list[str]], Path]) -> None:
    files = doctor_eye_files("P017", "OG") + doctor_eye_files("P_018", "OD")
    result = load(doctor_folder(files))
    assert {(e.patient_code, e.eye) for e in result.exams} == {
        ("P017", EyeSide.LEFT),
        ("P_018", EyeSide.RIGHT),  # un « _ » dans le code patient ne pose pas de problème
    }
    assert all(not e.is_labelled for e in result.exams)


def test_folder_adapter_reports_incomplete_eye_and_unknown_files(
    doctor_folder: Callable[[list[str]], Path],
) -> None:
    files = [*doctor_eye_files("P017", "OD")[:3], "photo_vacances.png"]
    result = load(doctor_folder(files))
    assert not result.exams
    reasons = {s.identifier: s.reason for s in result.skipped}
    assert "pachymetry" in reasons["P017_OD"]
    assert reasons["photo_vacances.png"] == "nom de fichier non reconnu"


# ---------- registre ----------
def test_registry_lists_builtin_adapters() -> None:
    assert {"csv_images", "image_folder"} <= set(available_adapters())


def test_unknown_adapter_gives_clear_error(cornorb_like: Callable[..., Path]) -> None:
    path = cornorb_like(ROWS, extra_config={"adapter": "pentacam_xml"})
    with pytest.raises(ConfigError, match="Adaptateur inconnu"):
        load(path)


def test_new_adapter_can_be_plugged_without_touching_existing_code(
    cornorb_like: Callable[..., Path],
) -> None:
    """Démonstration du principe ouvert/fermé : on ajoute une classe, rien d'autre."""

    @register_adapter("empty_for_test")
    class EmptyAdapter(DatasetAdapter):
        def load(self) -> LoadResult:
            return LoadResult(exams=())

    path = cornorb_like(ROWS, extra_config={"adapter": "empty_for_test"})
    assert load(path).exams == ()
