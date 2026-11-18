"""Pont entre les objets du domaine (`CornealExam`) et PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from qarniya.domain import CornealExam, Diagnosis
from qarniya.models import Preprocessor


class ExamDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        exams: Sequence[CornealExam],
        preprocessor: Preprocessor,
        positive_labels: Sequence[Diagnosis],
        augment: bool = False,
    ) -> None:
        if any(not e.is_labelled for e in exams):
            raise ValueError("l'entraînement exige des examens étiquetés")
        self.exams = list(exams)
        self.preprocessor = preprocessor
        self.positive = set(positive_labels)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.exams)

    def target(self, index: int) -> int:
        return int(self.exams[index].diagnosis in self.positive)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        exam = self.exams[index]
        x = self.preprocessor.to_tensor(exam, augment=self.augment)
        return x, torch.tensor(float(self.target(index)))
