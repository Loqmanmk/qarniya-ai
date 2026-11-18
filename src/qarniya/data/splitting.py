"""Découpage entraînement / validation / test.

Règle d'or : **par patient**. Les deux yeux d'un même patient se ressemblent ; s'ils tombaient
l'un en entraînement et l'autre en test, le modèle serait évalué sur un patient « déjà vu »
et le score serait trop optimiste (fuite de données).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

from sklearn.model_selection import train_test_split

from qarniya.data.adapters.base import LoadResult
from qarniya.data.config import SplitConfig, SplitStrategy
from qarniya.domain import CornealExam, Diagnosis
from qarniya.domain.exceptions import QarniyaError


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[CornealExam, ...]
    val: tuple[CornealExam, ...]
    test: tuple[CornealExam, ...]

    def get(self, split: Split) -> tuple[CornealExam, ...]:
        return {Split.TRAIN: self.train, Split.VAL: self.val, Split.TEST: self.test}[split]

    def summary(self) -> dict[str, dict[str, int]]:
        """Nombre d'yeux par ensemble et par diagnostic (pour vérifier l'équilibre)."""
        summary: dict[str, dict[str, int]] = {}
        for s in Split:
            exams = self.get(s)
            summary[s.value] = dict(Counter(_label_name(e) for e in exams)) | {"total": len(exams)}
        return summary

    def leaked_patients(self) -> set[str]:
        """Patients présents dans plusieurs ensembles. Doit toujours être vide."""
        seen: dict[str, set[Split]] = defaultdict(set)
        for s in Split:
            for exam in self.get(s):
                seen[exam.patient_code].add(s)
        return {p for p, splits in seen.items() if len(splits) > 1}


def _label_name(exam: CornealExam) -> str:
    return exam.diagnosis.value if exam.diagnosis else "inconnu"


def split_dataset(result: LoadResult, config: SplitConfig) -> DatasetSplits:
    if config.strategy is SplitStrategy.COLUMN:
        return _split_from_source(result)
    return _split_by_patient(result.exams, config)


def _split_from_source(result: LoadResult) -> DatasetSplits:
    buckets: dict[str, list[CornealExam]] = defaultdict(list)
    for exam in result.exams:
        buckets[result.predefined_splits.get(exam.exam_id, "ignore")].append(exam)
    return DatasetSplits(tuple(buckets["train"]), tuple(buckets["val"]), tuple(buckets["test"]))


def _split_by_patient(exams: tuple[CornealExam, ...], config: SplitConfig) -> DatasetSplits:
    if any(not e.is_labelled for e in exams):
        raise QarniyaError("le découpage par patient exige des examens étiquetés")

    by_patient: dict[str, list[CornealExam]] = defaultdict(list)
    for exam in exams:
        by_patient[exam.patient_code].append(exam)
    patients = sorted(by_patient)
    # un patient est « malade » si au moins un de ses yeux l'est : sert à équilibrer les ensembles
    strata = [
        int(any(e.diagnosis is not Diagnosis.NORMAL for e in by_patient[p])) for p in patients
    ]

    rest, test = _stratified(patients, strata, config.test_size, config.seed)
    rest_strata = [strata[patients.index(p)] for p in rest]
    val_share = config.val_size / (1 - config.test_size)
    train, val = _stratified(rest, rest_strata, val_share, config.seed)

    def collect(group: list[str]) -> tuple[CornealExam, ...]:
        return tuple(e for p in sorted(group) for e in by_patient[p])

    return DatasetSplits(collect(train), collect(val), collect(test))


def _stratified(
    items: list[str], strata: list[int], size: float, seed: int
) -> tuple[list[str], list[str]]:
    # stratification impossible si une classe est trop rare : on retombe sur un tirage simple
    can_stratify = min(Counter(strata).values()) >= 2 and len(set(strata)) > 1
    first, second = train_test_split(
        items, test_size=size, random_state=seed, stratify=strata if can_stratify else None
    )
    return list(first), list(second)
