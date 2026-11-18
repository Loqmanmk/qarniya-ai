"""Format des réponses JSON de l'API (validé par pydantic, documenté automatiquement dans /docs)."""

from __future__ import annotations

from pydantic import BaseModel

DISCLAIMER = (
    "Outil d'aide au dépistage. Il ne pose pas de diagnostic : la décision reste celle de "
    "l'ophtalmologue."
)


class MapExplanationOut(BaseModel):
    map_type: str
    label: str
    importance: float  # part de la carte dans la décision (0–1)
    overlay_png: str  # image PNG en base64 : carte en gris + chaleur


class AnalysisOut(BaseModel):
    status: str  # analysed | unusual | rejected
    status_label: str
    summary: str
    reasons: list[str]
    probability: float | None = None
    level: str | None = None  # niveau des images
    final_level: str | None = None  # niveau affiché (relevé en cas de désaccord)
    final_level_label: str | None = None
    numbers_probability: float | None = None
    numbers_level_label: str | None = None
    disagreement: bool = False
    maps: list[MapExplanationOut] = []
    disclaimer: str = DISCLAIMER


class ProgressionOut(BaseModel):
    verdict: str
    comparable: bool
    is_progressing: bool | None = None
    delta: float | None = None
    before: AnalysisOut
    after: AnalysisOut
    disclaimer: str = DISCLAIMER


class HealthOut(BaseModel):
    status: str
    version: str
    model: str
    novelty_detector: bool
    numbers_opinion: bool
