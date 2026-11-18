"""Le découpage ne doit JAMAIS mettre un même patient dans deux ensembles."""

from pathlib import Path

import pytest

from qarniya.data import Split, split_dataset
from qarniya.data.adapters import LoadResult
from qarniya.data.config import SplitConfig, SplitStrategy
from qarniya.domain import CornealExam, CornealMaps, Diagnosis, EyeSide, MapType
from qarniya.domain.exceptions import QarniyaError


def exam(patient: str, eye: EyeSide, diagnosis: Diagnosis | None) -> CornealExam:
    maps = CornealMaps({m: Path(f"{patient}{m.value}.png") for m in MapType})
    return CornealExam(f"{patient}_{eye.value}", patient, eye, maps, diagnosis=diagnosis)


def cohort(n_patients: int = 60) -> tuple[CornealExam, ...]:
    exams = []
    for i in range(n_patients):
        diagnosis = Diagnosis.KERATOCONUS if i % 3 == 0 else Diagnosis.NORMAL
        exams += [exam(f"P{i:03d}", eye, diagnosis) for eye in EyeSide]
    return tuple(exams)


def test_patient_split_has_no_leakage_and_keeps_everything() -> None:
    exams = cohort()
    splits = split_dataset(LoadResult(exams), SplitConfig(strategy=SplitStrategy.PATIENT))
    assert splits.leaked_patients() == set()
    assert sum(len(splits.get(s)) for s in Split) == len(exams)


def test_patient_split_respects_proportions_and_stratification() -> None:
    splits = split_dataset(LoadResult(cohort(100)), SplitConfig(test_size=0.2, val_size=0.2))
    summary = splits.summary()
    assert summary["test"]["total"] == pytest.approx(40, abs=4)  # 20 % de 100 patients x 2 yeux
    # chaque ensemble contient des malades
    assert all(summary[s.value].get("keratoconus", 0) > 0 for s in Split)


def test_patient_split_is_reproducible() -> None:
    first = split_dataset(LoadResult(cohort()), SplitConfig(seed=7))
    second = split_dataset(LoadResult(cohort()), SplitConfig(seed=7))
    assert first == second


def test_patient_split_requires_labels() -> None:
    unlabelled = (exam("P1", EyeSide.RIGHT, None),)
    with pytest.raises(QarniyaError, match="étiquetés"):
        split_dataset(LoadResult(unlabelled), SplitConfig())


def test_column_split_follows_source() -> None:
    exams = cohort(3)
    predefined = {exams[0].exam_id: "test", exams[1].exam_id: "val"}
    config = SplitConfig(strategy=SplitStrategy.COLUMN, column="split")
    splits = split_dataset(LoadResult(exams, predefined_splits=predefined), config)
    assert splits.test == (exams[0],)
    assert splits.val == (exams[1],)
    assert splits.train == ()  # les autres n'ont pas de split connu : ignorés
