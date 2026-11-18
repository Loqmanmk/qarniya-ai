"""Point d'entrée en ligne de commande : `qarniya --help`."""

from __future__ import annotations

import logging
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
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Plus de détails")) -> None:
    """Qarniya AI — dépistage précoce du kératocône."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )


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


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Fichier YAML de l'expérience"),  # noqa: B008
    epochs: int | None = typer.Option(None, help="Remplace le nombre d'époques (essai rapide)"),
    limit: int | None = typer.Option(None, help="Nombre max d'yeux d'entraînement (essai rapide)"),
) -> None:
    """Entraîne un modèle à partir d'une configuration d'expérience."""
    from qarniya.training import ExperimentConfig, run_experiment  # import tardif : torch est lourd

    try:
        result = run_experiment(ExperimentConfig.from_yaml(config_path), epochs=epochs, limit=limit)
    except (ConfigError, QarniyaError) as exc:
        typer.secho(f"Erreur : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    best = result.best.val
    typer.secho(f"Meilleure époque : {result.best.epoch}", bold=True)
    typer.echo(
        f"  Validation  AUC={best.auc:.3f}  sensibilité={best.sensitivity:.3f}  "
        f"spécificité={best.specificity:.3f}"
    )
    typer.echo(f"  Modèle      {result.checkpoint}")


if __name__ == "__main__":
    app()
