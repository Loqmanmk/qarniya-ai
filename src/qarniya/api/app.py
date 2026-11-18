"""API REST de Qarniya AI + interface web du médecin.

    qarniya serve outputs/runs/<run>      → http://127.0.0.1:8000   (documentation : /docs)

Sécurité et confidentialité :
- fichiers limités en taille et en type (PNG/JPEG vérifiés en les décodant, pas sur leur nom) ;
- le nom de fichier envoyé n'est JAMAIS utilisé comme chemin (pas de traversée de dossiers) ;
- les images sont écrites dans un dossier temporaire supprimé à la fin de la requête :
  rien n'est conservé (loi 09-08) ; aucune image ni valeur n'est journalisée ;
- en-têtes de sécurité (CSP stricte, pas d'iframe, pas de cache) ;
- écoute en local par défaut (127.0.0.1).
"""

from __future__ import annotations

import base64
import io
import tempfile
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from qarniya import __version__
from qarniya.api.schemas import AnalysisOut, HealthOut, MapExplanationOut, ProgressionOut
from qarniya.domain import CornealExam, CornealMaps, EyeSide, MapType, QarniyaError
from qarniya.explainability import MAP_LABELS, overlay
from qarniya.services import (
    AnalysisReport,
    AnalysisService,
    AnalysisStatus,
    ProgressionService,
    validate_clinical_values,
)
from qarniya.services.report import build_report_pdf, validate_reference

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_FORMATS = {"PNG", "JPEG"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 Mo par carte
MAX_PIXELS = 4096 * 4096  # refuse les « bombes » de décompression
OVERLAY_SIZE = 360  # taille des images renvoyées (légères)
STATUS_CODES = {
    AnalysisStatus.ANALYSED: "analysed",
    AnalysisStatus.UNUSUAL: "unusual",
    AnalysisStatus.REJECTED: "rejected",
}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

Upload = Annotated[UploadFile, File()]


def create_app(analysis: AnalysisService) -> FastAPI:
    app = FastAPI(
        title="Qarniya AI",
        version=__version__,
        description="Aide au dépistage précoce du kératocône. Ne pose pas de diagnostic.",
    )
    progression = ProgressionService(analysis)
    lock = threading.Lock()  # un seul calcul à la fois : le modèle et Grad-CAM partagent un état

    @app.middleware("http")
    async def security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        if request.url.path in ("/docs", "/redoc", "/openapi.json"):
            return response  # la page Swagger charge ses propres scripts
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/api/health")
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            version=__version__,
            model=str(analysis.predictor.loaded.metadata.get("experiment", "inconnu")),
            novelty_detector=analysis.novelty is not None,
            numbers_opinion=analysis.clinical is not None,
        )

    @app.post("/api/analyses")
    def analyse_eye(
        eye: Annotated[EyeSide, Form()],
        axial: Upload,
        anterior: Upload,
        posterior: Upload,
        pachymetry: Upload,
        kmax: Annotated[float | None, Form()] = None,
        pachymetry_thinnest: Annotated[float | None, Form()] = None,
    ) -> AnalysisOut:
        """Analyse un œil : 4 cartes + (facultatif) Kmax et épaisseur minimale."""
        clinical = _clinical_values(kmax, pachymetry_thinnest)
        uploads = dict(zip(MapType, (axial, anterior, posterior, pachymetry), strict=True))
        with tempfile.TemporaryDirectory(prefix="qarniya_") as tmp:
            exam = _exam_from_uploads(Path(tmp), "examen", eye, uploads, clinical)
            with lock:
                return _to_out(analysis.analyse(exam))

    @app.post("/api/reports", response_class=Response)
    def report_pdf(
        eye: Annotated[EyeSide, Form()],
        axial: Upload,
        anterior: Upload,
        posterior: Upload,
        pachymetry: Upload,
        kmax: Annotated[float | None, Form()] = None,
        pachymetry_thinnest: Annotated[float | None, Form()] = None,
        reference: Annotated[str | None, Form(max_length=64)] = None,
    ) -> Response:
        """Même analyse, rendue sous forme de rapport PDF d'une page (rien n'est conservé)."""
        clinical = _clinical_values(kmax, pachymetry_thinnest)
        try:
            reference = validate_reference(reference)
        except QarniyaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        uploads = dict(zip(MapType, (axial, anterior, posterior, pachymetry), strict=True))
        model_name = str(analysis.predictor.loaded.metadata.get("experiment", "inconnu"))
        with tempfile.TemporaryDirectory(prefix="qarniya_") as tmp:
            exam = _exam_from_uploads(Path(tmp), "examen", eye, uploads, clinical)
            with lock:
                pdf = build_report_pdf(analysis.analyse(exam), model_name, reference)
        name = f"qarniya_{reference or 'rapport'}_{eye.value}.pdf"
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.post("/api/progressions")
    def compare_exams(
        eye: Annotated[EyeSide, Form()],
        before_axial: Upload,
        before_anterior: Upload,
        before_posterior: Upload,
        before_pachymetry: Upload,
        after_axial: Upload,
        after_anterior: Upload,
        after_posterior: Upload,
        after_pachymetry: Upload,
    ) -> ProgressionOut:
        """Compare deux examens du même œil (ancien, puis récent)."""
        before = (before_axial, before_anterior, before_posterior, before_pachymetry)
        after = (after_axial, after_anterior, after_posterior, after_pachymetry)
        with tempfile.TemporaryDirectory(prefix="qarniya_") as tmp:
            exams = [
                _exam_from_uploads(Path(tmp), name, eye, dict(zip(MapType, files, strict=True)), {})
                for name, files in (("ancien", before), ("recent", after))
            ]
            with lock:
                report = progression.compare(exams[0], exams[1])
                p = report.progression
                return ProgressionOut(
                    verdict=report.verdict(),
                    comparable=p is not None,
                    is_progressing=p.is_progressing if p else None,
                    delta=p.delta if p else None,
                    before=_to_out(report.before),
                    after=_to_out(report.after),
                )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


# ------------------------------------------------------------------ helpers
def _clinical_values(kmax: float | None, thinnest: float | None) -> dict[str, float]:
    values = {k: v for k, v in (("kmax", kmax), ("pachymetry_thinnest", thinnest)) if v is not None}
    try:
        validate_clinical_values(values)
    except QarniyaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return values


def _exam_from_uploads(
    folder: Path,
    name: str,
    eye: EyeSide,
    uploads: dict[MapType, UploadFile],
    clinical: dict[str, float],
) -> CornealExam:
    paths = {
        map_type: _save_upload(upload, folder / f"{name}_{map_type.value}.png", map_type)
        for map_type, upload in uploads.items()
    }
    return CornealExam(
        exam_id=f"api:{name}",
        patient_code="anonyme",  # l'API ne reçoit et ne garde aucune identité
        eye=eye,
        maps=CornealMaps(paths),
        clinical=clinical,
    )


def _save_upload(upload: UploadFile, destination: Path, map_type: MapType) -> Path:
    """Vérifie et enregistre une carte. Le nom de fichier fourni n'est jamais utilisé."""
    label = MAP_LABELS[map_type]
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"{label} : fichier trop lourd (max {MAX_UPLOAD_BYTES // 2**20} Mo)"
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            if probe.width * probe.height > MAX_PIXELS:
                raise HTTPException(413, f"{label} : image trop grande")
            probe.verify()  # contrôle d'intégrité du fichier
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(415, f"{label} : ce n'est pas une image PNG ou JPEG valide") from exc
    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(415, f"{label} : format {image_format} non accepté (PNG ou JPEG)")
    rgb.save(destination, format="PNG")  # ré-encodage : on ne garde aucune métadonnée d'origine
    return destination


def _png_base64(array_or_image: Image.Image) -> str:
    buffer = io.BytesIO()
    array_or_image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _to_out(report: AnalysisReport) -> AnalysisOut:
    maps: list[MapExplanationOut] = []
    if report.explanation is not None:
        for map_type, path in report.exam.maps.paths.items():
            with Image.open(path) as img:
                img.thumbnail((OVERLAY_SIZE, OVERLAY_SIZE))
                blended = overlay(img.convert("RGB"), report.explanation.heatmaps[map_type])
            maps.append(
                MapExplanationOut(
                    map_type=map_type.value,
                    label=MAP_LABELS[map_type],
                    importance=report.explanation.map_importance[map_type],
                    overlay_png=_png_base64(Image.fromarray(blended)),
                )
            )
    a, n, final = report.assessment, report.numbers, report.final_level
    return AnalysisOut(
        status=STATUS_CODES[report.status],
        status_label=report.status.value,
        summary=report.summary(with_warning=False),  # l'avertissement a son propre bandeau
        reasons=list(report.reasons),
        probability=a.probability if a else None,
        level=a.level.value if a else None,
        final_level=final.value if final else None,
        final_level_label=final.label if final else None,
        numbers_probability=n.probability if n else None,
        numbers_level_label=n.level.label if n else None,
        disagreement=bool(report.comparison and report.comparison.disagree),
        maps=maps,
    )
