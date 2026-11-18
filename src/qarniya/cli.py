"""Point d'entrée en ligne de commande : `qarniya --help`."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from qarniya import __version__
from qarniya.data import (
    ConfigError,
    DatasetConfig,
    ImageQualityChecker,
    build_adapter,
    split_dataset,
)
from qarniya.domain.exceptions import QarniyaError

app = typer.Typer(help="Qarniya AI — dépistage précoce du kératocône.", no_args_is_help=True)
data_app = typer.Typer(help="Inspecter et vérifier un dataset.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Qarniya AI — dépistage précoce du kératocône."""


@app.command()
def version() -> None:
    """Affiche la version installée."""
    typer.echo(f"qarniya {__version__}")


@data_app.command("inspect")
def data_inspect(
    config_path: Path = typer.Argument(..., help="Fichier YAML du dataset"),  # noqa: B008
    check_images: bool = typer.Option(False, "--check-images", help="Contrôle qualité des images"),
    show_skipped: int = typer.Option(5, help="Nombre d'exemples d'exclusions à afficher"),
) -> None:
    """Charge un dataset via sa configuration et affiche un rapport."""
    try:
        config = DatasetConfig.from_yaml(config_path)
        result = build_adapter(config).load()
    except (ConfigError, QarniyaError) as exc:
        typer.secho(f"Erreur : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Dataset '{config.name}' (adaptateur : {config.adapter})", bold=True)
    typer.echo(f"  Examens chargés : {len(result.exams)}")
    typer.echo(f"  Exclus          : {len(result.skipped)}")
    for reason, count in Counter(s.reason.split(" '")[0] for s in result.skipped).most_common():
        typer.echo(f"    - {reason} : {count}")
    for record in result.skipped[:show_skipped]:
        typer.echo(f"      ex. {record.identifier} -> {record.reason}")

    if all(e.is_labelled for e in result.exams) and result.exams:
        splits = split_dataset(result, config.split)
        typer.secho("Découpage :", bold=True)
        for name, counts in splits.summary().items():
            typer.echo(f"  {name:5s} {counts}")
        leaked = splits.leaked_patients()
        status = "OK, aucun patient partagé" if not leaked else f"FUITE : {sorted(leaked)[:5]}"
        typer.echo(f"  Contrôle de fuite : {status}")

    if check_images:
        checker = ImageQualityChecker()
        bad = [r for r in map(checker.check_exam, result.exams) if not r.is_acceptable]
        typer.secho(f"Contrôle qualité : {len(bad)} examen(s) refusé(s)", bold=True)
        for report in bad[:show_skipped]:
            typer.echo(f"  {report.exam_id} : {[i.message for i in report.issues]}")


if __name__ == "__main__":
    app()
