"""Rapport PDF d'une analyse, à joindre au dossier patient (une page A4).

Confidentialité (loi 09-08) : le rapport ne contient AUCUNE identité. Le médecin peut seulement
ajouter une référence anonyme (ex. « P017 ») qui lui permet de le ranger dans son dossier.
"""

from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

from qarniya import __version__
from qarniya.domain import MapType, QarniyaError, RiskLevel
from qarniya.explainability import MAP_LABELS, overlay
from qarniya.services.analysis import AnalysisReport, AnalysisStatus

REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
A4 = (8.27, 11.69)  # pouces
INK, INK_2, RULE, WARN = "#0b0b0b", "#52514e", "#dddcd6", "#c2511f"
LEVEL_STYLE = {  # couleur de statut + icône + texte : jamais la couleur seule
    RiskLevel.LOW: ("#0ca30c", "✓"),
    RiskLevel.MONITOR: ("#b07800", "◐"),
    RiskLevel.HIGH: ("#d03b3b", "▲"),
}
DISCLAIMER = (
    "Outil d'aide au dépistage, non certifié comme dispositif médical. Il ne pose pas de "
    "diagnostic : la décision reste celle de l'ophtalmologue. Les zones colorées indiquent ce qui "
    "a fait monter le score ; ce n'est pas une preuve médicale."
)
SIGNATURE = (
    "Avis de l'ophtalmologue : ......................................    Signature : ............"
)


def validate_reference(reference: str | None) -> str | None:
    """Référence anonyme seulement : lettres, chiffres, « - » et « _ », 24 caractères maximum."""
    if reference is None or reference.strip() == "":
        return None
    reference = reference.strip()
    if not REFERENCE_PATTERN.fullmatch(reference):
        raise QarniyaError(
            "Référence invalide : utilisez un code anonyme (lettres, chiffres, - et _, 24 max), "
            "jamais le nom du patient."
        )
    return reference


def build_report_pdf(
    report: AnalysisReport, model_name: str, reference: str | None = None
) -> bytes:
    reference = validate_reference(reference)
    fig = plt.figure(figsize=A4)
    try:
        _draw(fig, report, model_name, reference)
        buffer = io.BytesIO()
        metadata = {
            "Title": "Qarniya AI : rapport d'analyse",
            "Creator": f"Qarniya AI {__version__}",
        }
        fig.savefig(buffer, format="pdf", metadata=metadata)
    finally:
        plt.close(fig)
    return buffer.getvalue()


# ------------------------------------------------------------------ dessin
def _text(fig: Figure, y: float, s: str, size: float = 10, color: str = INK, **kw: Any) -> None:
    fig.text(0.07, y, s, fontsize=size, color=color, va="top", **kw)


def _rule(fig: Figure, y: float) -> None:
    fig.add_artist(Line2D([0.07, 0.93], [y, y], color=RULE, linewidth=1))


def _wrap(s: str, width: int = 95) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


def _draw(fig: Figure, report: AnalysisReport, model_name: str, reference: str | None) -> None:
    exam = report.exam
    eye = "droit (OD)" if exam.eye.value == "OD" else "gauche (OS)"
    _text(fig, 0.965, "Qarniya AI", size=18, fontweight="bold")
    _text(fig, 0.935, "Rapport d'aide au dépistage du kératocône", size=11, color=INK_2)
    _text(
        fig,
        0.905,
        f"Date : {datetime.now():%d/%m/%Y %H:%M}    Œil : {eye}    Référence : {reference or '—'}",
        size=9.5,
    )
    _text(fig, 0.887, f"Modèle : {model_name} · Qarniya AI {__version__}", size=8.5, color=INK_2)
    _rule(fig, 0.872)

    if report.status is AnalysisStatus.REJECTED:
        _draw_rejected(fig, report)
    else:
        y = _draw_result(fig, report)
        _draw_maps(fig, report, y)
    _rule(fig, 0.095)
    _text(fig, 0.085, _wrap(DISCLAIMER, 115), size=7.5, color=INK_2)
    _text(fig, 0.032, SIGNATURE, size=9)


def _draw_rejected(fig: Figure, report: AnalysisReport) -> None:
    color = LEVEL_STYLE[RiskLevel.HIGH][0]
    _text(fig, 0.855, "✕ Analyse impossible", size=13, color=color, fontweight="bold")
    _text(fig, 0.825, _wrap(" ; ".join(report.reasons)), size=10)
    message = "Aucun score n'a été calculé. Merci de fournir des cartes lisibles."
    _text(fig, 0.765, message, size=10, color=INK_2)


def _draw_result(fig: Figure, report: AnalysisReport) -> float:
    """Niveau, score, explication, avis des chiffres. Rend la hauteur où placer les cartes."""
    y = 0.855
    unreliable = report.status is AnalysisStatus.UNUSUAL
    if unreliable:
        warning = "⚠ SCORE NON FIABLE : " + " ; ".join(report.reasons)
        _text(fig, y, _wrap(warning, 80), size=10, color=WARN, fontweight="bold")
        y -= 0.045
    level = report.final_level or RiskLevel.LOW
    color, icon = LEVEL_STYLE[level]
    probability = report.assessment.probability if report.assessment else 0.0
    _text(fig, y, f"{icon} {level.label.capitalize()}", size=16, color=color, fontweight="bold")
    suffix = " (non fiable)" if unreliable else ""
    _text(fig, y - 0.032, f"Score images : {probability:.0%}{suffix}", size=11)
    _text(fig, y - 0.058, _wrap(report.summary(with_warning=False)), size=10)
    y -= 0.115

    if report.numbers is not None:
        entered = {"Kmax": "kmax", "épaisseur min.": "pachymetry_thinnest"}
        values = ", ".join(
            f"{label} = {report.exam.clinical[key]:g}"
            for label, key in entered.items()
            if key in report.exam.clinical
        )
        line = f"Valeurs saisies : {values} → avis des chiffres {report.numbers.probability:.0%}"
        _text(fig, y, line, size=10, color=INK_2)
        if report.comparison is not None and report.comparison.disagree:
            monitor = LEVEL_STYLE[RiskLevel.MONITOR][0]
            note = "◐ Désaccord images / chiffres : niveau relevé à « à surveiller »."
            _text(fig, y - 0.02, note, size=10, color=monitor, fontweight="bold")
        y -= 0.05
    _rule(fig, y + 0.008)
    return y


def _draw_maps(fig: Figure, report: AnalysisReport, y: float) -> None:
    if report.explanation is None:
        return
    caption = "Zones qui ont fait monter le score (orange) et part de chaque carte :"
    _text(fig, y - 0.005, caption, size=9.5, color=INK_2)
    # grille 2 x 2 qui occupe la place restante entre le texte et le pied de page
    top, bottom, gap = y - 0.03, 0.115, 0.03
    height = (top - bottom - gap) / 2
    for index, map_type in enumerate(MapType):
        row, col = divmod(index, 2)
        rect = (0.07 + col * 0.45, top - (row + 1) * height - row * gap, 0.41, height - 0.02)
        ax = fig.add_axes(rect)
        with Image.open(report.exam.maps.paths[map_type]) as img:
            img.thumbnail((500, 500))
            ax.imshow(overlay(img.convert("RGB"), report.explanation.heatmaps[map_type]))
        share = report.explanation.map_importance[map_type]
        ax.set_title(f"{MAP_LABELS[map_type]} — {share:.0%} de la décision", fontsize=9, color=INK)
        ax.axis("off")
