"""Service d'analyse de bout en bout : score + chaleur, refus, image inhabituelle, tri, CLI."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from qarniya.cli import app
from qarniya.data import DatasetConfig, build_adapter
from qarniya.domain import CornealExam, CornealMaps, MapType, RiskLevel
from qarniya.models import NoveltyDetector
from qarniya.services import AnalysisService, AnalysisStatus, triage_order

pytestmark = pytest.mark.integration


def first_exam(trained_run: Path) -> CornealExam:
    config = DatasetConfig.from_yaml(trained_run / "dataset.yaml")
    return build_adapter(config).load().exams[0]


def test_accepted_exam_gets_score_level_and_heatmap(trained_run: Path, tmp_path: Path) -> None:
    service = AnalysisService.from_run(trained_run)
    report = service.analyse(first_exam(trained_run))

    assert report.status is AnalysisStatus.ANALYSED
    assert report.assessment is not None and report.explanation is not None
    assert 0 <= report.assessment.probability <= 1
    assert "Risque" in report.summary()
    service.save_figure(report, tmp_path / "figure.png")
    assert (tmp_path / "figure.png").stat().st_size > 10_000


def test_grey_map_is_refused_without_any_score(trained_run: Path, tmp_path: Path) -> None:
    exam = first_exam(trained_run)
    paths = dict(exam.maps.paths)
    grey = tmp_path / "grey.png"
    Image.open(paths[MapType.PACHYMETRY]).convert("L").convert("RGB").save(grey)
    paths[MapType.PACHYMETRY] = grey
    broken = CornealExam("e-grey", exam.patient_code, exam.eye, CornealMaps(paths))

    report = AnalysisService.from_run(trained_run).analyse(broken)
    assert report.status is AnalysisStatus.REJECTED
    assert report.assessment is None  # aucun score donné
    assert "Pachymétrie" in report.summary() and "sans couleurs" in report.summary()


def test_unusual_image_is_flagged_as_unreliable(trained_run: Path) -> None:
    service = AnalysisService.from_run(trained_run)
    # seuil à -inf : tout œil est « inhabituel » (on teste le branchement, pas le détecteur)
    reference = np.full((5, 2048), 1 / np.sqrt(2048), np.float32)  # empreintes normalisées
    service.novelty = NoveltyDetector(reference=reference, threshold=float("-inf"))
    report = service.analyse(first_exam(trained_run))
    assert report.status is AnalysisStatus.UNUSUAL
    assert report.assessment is not None  # le score existe…
    assert report.summary().startswith("⚠ SCORE NON FIABLE")  # … mais il est signalé


def test_triage_puts_problems_then_high_risk_first(trained_run: Path) -> None:
    service = AnalysisService.from_run(trained_run)
    report = service.analyse(first_exam(trained_run))
    refused = type(report)(report.exam, AnalysisStatus.REJECTED, ("x",))
    assert triage_order(refused) < triage_order(report)
    assert report.assessment is not None
    assert triage_order(report)[0] in {1, 2, 3}
    assert report.assessment.level in set(RiskLevel)


def test_cli_fit_novelty_then_analyse_writes_triage(trained_run: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    fitted = runner.invoke(app, ["fit-novelty", str(trained_run)])
    assert fitted.exit_code == 0, fitted.output
    assert (trained_run / "novelty.npz").is_file()

    out = tmp_path / "analyses"
    result = runner.invoke(
        app, ["analyse", str(trained_run), str(trained_run / "dataset.yaml"),
              "--limit", "3", "--out", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    lines = (out / "triage.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 + 3  # en-tête + 3 yeux
    assert len(list(out.glob("*_analyse.png"))) == 3
