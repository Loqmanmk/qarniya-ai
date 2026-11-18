"""Adaptateur « CSV + images » : un tableau (une ligne par œil) et 4 images par œil.

Couvre CornOrb et, par simple configuration, tout export du même genre (ex. le tableau du médecin).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from datetime import date

import pandas as pd

from qarniya.data.adapters.base import DatasetAdapter, LoadResult, SkippedRecord
from qarniya.data.adapters.paths import build_map_paths
from qarniya.data.adapters.registry import register_adapter
from qarniya.data.config import ColumnMapping, ConfigError, SplitStrategy
from qarniya.domain import CornealExam, CornealMaps, Diagnosis, EyeSide


@register_adapter("csv_images")
class CsvImagesAdapter(DatasetAdapter):
    def load(self) -> LoadResult:
        cfg = self.config
        if cfg.csv_path is None or cfg.columns is None:
            raise ConfigError("l'adaptateur 'csv_images' exige les champs 'csv' et 'columns'")
        if not cfg.csv_path.is_file():
            raise ConfigError(f"CSV introuvable : {cfg.csv_path}")

        table = pd.read_csv(cfg.csv_path, dtype=str, keep_default_na=False)
        cols = cfg.columns
        self._check_columns(table, cols)
        if cfg.patient_code_fixes:
            table[cols.patient] = table[cols.patient].replace(cfg.patient_code_fixes)

        ambiguous = self._ambiguous_keys(table, cols) if cfg.drop_ambiguous_duplicates else set()
        exams: list[CornealExam] = []
        skipped: list[SkippedRecord] = []
        splits: dict[str, str] = {}

        for row in table.to_dict(orient="records"):
            outcome = self._row_to_exam(row, cols, ambiguous)
            if isinstance(outcome, SkippedRecord):
                skipped.append(outcome)
                continue
            exams.append(outcome)
            split = self._predefined_split(row)
            if split:
                splits[outcome.exam_id] = split

        return LoadResult(tuple(exams), tuple(skipped), splits)

    # ---------- étapes ----------
    def _check_columns(self, table: pd.DataFrame, cols: ColumnMapping) -> None:
        wanted = [c for c in (cols.patient, cols.eye, cols.label, cols.exam_date) if c]
        wanted += list(self.config.clinical_columns.values())
        split = self.config.split
        if split.strategy is SplitStrategy.COLUMN:
            wanted += [c for c in (split.column, split.val_fold_column) if c]
        missing = [c for c in wanted if c not in table.columns]
        if missing:
            raise ConfigError(f"Colonnes absentes du CSV : {missing}")

    @staticmethod
    def _key(row: Mapping[str, str], cols: ColumnMapping) -> tuple[str, ...]:
        parts = [row[cols.patient].strip(), row[cols.eye].strip()]
        if cols.exam_date:
            parts.append(row[cols.exam_date].strip())
        return tuple(parts)

    def _ambiguous_keys(self, table: pd.DataFrame, cols: ColumnMapping) -> set[tuple[str, ...]]:
        """Clés (patient, œil[, date]) présentes plusieurs fois : on ne sait pas quelle ligne
        correspond aux images, on les écarte toutes."""
        counts = Counter(self._key(r, cols) for r in table.to_dict(orient="records"))
        return {k for k, n in counts.items() if n > 1}

    def _row_to_exam(
        self, row: Mapping[str, str], cols: ColumnMapping, ambiguous: set[tuple[str, ...]]
    ) -> CornealExam | SkippedRecord:
        cfg = self.config
        patient, eye_raw = row[cols.patient].strip(), row[cols.eye].strip()
        ident = f"{patient}_{eye_raw}"

        if self._key(row, cols) in ambiguous:
            return SkippedRecord(ident, "ligne en double avec des valeurs différentes")
        eye = cfg.eye_values.get(eye_raw)
        if eye is None:
            return SkippedRecord(ident, f"œil inconnu '{eye_raw}'")

        diagnosis: Diagnosis | None = None
        if cols.label:
            diagnosis = cfg.label_values.get(row[cols.label].strip())
            if diagnosis is None:
                return SkippedRecord(ident, f"étiquette inconnue '{row[cols.label]}'")

        paths = build_map_paths(cfg, patient, eye_raw)
        missing = [m.value for m, p in paths.items() if not p.is_file()]
        if missing:
            return SkippedRecord(ident, f"images manquantes : {', '.join(missing)}")

        exam_date = self._parse_date(row[cols.exam_date]) if cols.exam_date else None
        return CornealExam(
            exam_id=self._exam_id(patient, eye, exam_date),
            patient_code=patient,
            eye=eye,
            maps=CornealMaps(paths),
            exam_date=exam_date,
            device=cfg.device,
            diagnosis=diagnosis,
            clinical=self._clinical_values(row),
        )

    def _clinical_values(self, row: Mapping[str, str]) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, column in self.config.clinical_columns.items():
            try:
                value = float(row[column])
            except ValueError:
                continue  # valeur vide ou non numérique : simplement absente
            if not math.isnan(value):
                values[name] = value
        return values

    def _predefined_split(self, row: Mapping[str, str]) -> str | None:
        split = self.config.split
        if split.strategy is not SplitStrategy.COLUMN or split.column is None:
            return None
        raw = row[split.column].strip()
        if raw == split.test_value:
            return "test"
        if raw != split.train_value:
            return None
        if split.val_fold_column and row[split.val_fold_column].strip() == str(split.val_fold):
            return "val"
        return "train"

    @staticmethod
    def _parse_date(raw: str) -> date | None:
        try:
            return date.fromisoformat(raw.strip())
        except ValueError:
            return None

    def _exam_id(self, patient: str, eye: EyeSide, exam_date: date | None) -> str:
        suffix = f"_{exam_date.isoformat()}" if exam_date else ""
        return f"{self.config.name}:{patient}_{eye.value}{suffix}"
