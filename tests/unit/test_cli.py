from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from qarniya import __version__
from qarniya.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_data_inspect_reports_dataset(cornorb_like: Callable[..., Path]) -> None:
    rows = [
        {"patient_code": "P1", "eye": "OD", "label": "0", "split_80_20": "train", "fold": "2"},
        {"patient_code": "P2", "eye": "OD", "label": "1", "split_80_20": "test", "fold": ""},
    ]
    result = CliRunner().invoke(app, ["data", "inspect", str(cornorb_like(rows)), "--check-images"])
    assert result.exit_code == 0, result.output
    assert "Examens chargés : 2" in result.output
    assert "aucun patient partagé" in result.output
    assert "0 examen(s) refusé(s)" in result.output


def test_data_inspect_bad_config_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["data", "inspect", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 1
