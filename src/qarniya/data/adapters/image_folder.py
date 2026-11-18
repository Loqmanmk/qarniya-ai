"""Adaptateur « dossier d'images » : pas de tableau, tout est dans les noms de fichiers.

Exemple : P017_OG_axiale.png, P017_OG_anterieure.png… (format demandé au médecin).
Sert à analyser de nouveaux patients (sans étiquette) ou un export brut d'appareil.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from qarniya.data.adapters.base import DatasetAdapter, LoadResult, SkippedRecord
from qarniya.data.adapters.paths import pattern_to_regex
from qarniya.data.adapters.registry import register_adapter
from qarniya.data.config import ConfigError
from qarniya.domain import CornealExam, CornealMaps, EyeSide, MapType

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@register_adapter("image_folder")
class ImageFolderAdapter(DatasetAdapter):
    def load(self) -> LoadResult:
        cfg = self.config
        if not cfg.root.is_dir():
            raise ConfigError(f"Dossier introuvable : {cfg.root}")

        regex = pattern_to_regex(cfg)
        eye_by_raw = {k.lower(): v for k, v in cfg.eye_values.items()}
        map_by_token = {token.lower(): m for m, token in cfg.map_tokens.items()}

        groups: dict[tuple[str, EyeSide], dict[MapType, Path]] = defaultdict(dict)
        skipped: list[SkippedRecord] = []
        for path in sorted(p for p in cfg.root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            relative = path.relative_to(cfg.root).as_posix()
            match = regex.match(relative)
            if match is None:
                skipped.append(SkippedRecord(relative, "nom de fichier non reconnu"))
                continue
            eye = eye_by_raw[match["eye"].lower()]
            map_type = map_by_token[match["map"].lower()]
            groups[(match["patient"], eye)][map_type] = path

        exams: list[CornealExam] = []
        ordered = sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1].value))
        for (patient, eye), maps in ordered:
            ident = f"{patient}_{eye.value}"
            missing = [m.value for m in MapType if m not in maps]
            if missing:
                skipped.append(SkippedRecord(ident, f"images manquantes : {', '.join(missing)}"))
                continue
            exams.append(
                CornealExam(
                    exam_id=f"{cfg.name}:{ident}",
                    patient_code=patient,
                    eye=eye,
                    maps=CornealMaps(maps),
                    device=cfg.device,
                )
            )
        return LoadResult(tuple(exams), tuple(skipped))
