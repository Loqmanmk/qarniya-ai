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


@app.command()
def evaluate(
    run_dir: Path = typer.Argument(..., help="Dossier de l'entraînement (outputs/runs/...)"),  # noqa: B008
    target_sensitivity: float = typer.Option(0.95, help="Sensibilité visée sur la validation"),
    bootstrap: int = typer.Option(2000, help="Nombre de tirages pour les fourchettes"),
    force: bool = typer.Option(False, "--force", help="Ré-évaluer un test déjà évalué"),
) -> None:
    """Évalue un modèle entraîné sur le jeu de TEST (une seule fois)."""
    from qarniya.evaluation.runner import evaluate_run  # import tardif : torch est lourd

    try:
        result = evaluate_run(run_dir, target_sensitivity, bootstrap, force)
    except (ConfigError, QarniyaError) as exc:
        typer.secho(f"Erreur : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.secho(f"Jeu de test : {result.n_test} yeux", bold=True)
    for evaluation in result.evaluations:
        t, ci = evaluation.test, evaluation.confidence
        sens = ci["sensitivity"]
        typer.echo(f"  {evaluation.name}")
        typer.echo(
            f"    AUC {t.auc:.3f} [{ci['auc'].low:.3f}-{ci['auc'].high:.3f}]  "
            f"sensibilité {t.sensitivity:.1%} [{sens.low:.1%}-{sens.high:.1%}]  "
            f"spécificité {t.specificity:.1%}  KC manqués {t.fn}  fausses alertes {t.fp}"
        )
    typer.echo(f"  Rapport : {result.output_dir / 'report.md'}")


@app.command("fit-novelty")
def fit_novelty(
    run_dir: Path = typer.Argument(..., help="Dossier de l'entraînement (outputs/runs/...)"),  # noqa: B008
    quantile: float = typer.Option(0.99, help="Part des yeux de validation jugés « habituels »"),
) -> None:
    """Construit le détecteur d'images inhabituelles (entraînement + validation, jamais le test)."""
    from qarniya.models import Predictor  # import tardif : torch est lourd
    from qarniya.services import NOVELTY_FILE, fit_novelty_detector

    try:
        data_config = DatasetConfig.from_yaml(run_dir / "dataset.yaml")
        splits = split_dataset(build_adapter(data_config).load(), data_config.split)
        predictor = Predictor.from_checkpoint(run_dir / "model.pt")
        detector, distances = fit_novelty_detector(
            predictor, list(splits.train), list(splits.val), quantile
        )
    except (ConfigError, QarniyaError) as exc:
        typer.secho(f"Erreur : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    detector.save(run_dir / NOVELTY_FILE)
    flagged = int((distances > detector.threshold).sum())
    typer.secho("Détecteur d'images inhabituelles enregistré", bold=True)
    typer.echo(f"  Référence : {len(detector.reference)} yeux d'entraînement")
    typer.echo(f"  Seuil     : distance {detector.threshold:.4f} ({quantile:.0%} de la validation)")
    typer.echo(f"  Contrôle  : {flagged}/{len(distances)} yeux de validation au-delà du seuil")
    typer.echo(f"  Fichier   : {run_dir / NOVELTY_FILE}")


@app.command()
def analyse(
    run_dir: Path = typer.Argument(..., help="Dossier du modèle (outputs/runs/...)"),  # noqa: B008
    data_config_path: Path = typer.Argument(..., help="YAML des examens à analyser"),  # noqa: B008
    only: list[str] = typer.Option([], "--only", help="Codes patients à analyser (répétable)"),  # noqa: B008
    limit: int | None = typer.Option(None, help="Nombre maximal d'yeux"),
    out_dir: Path = typer.Option(Path("outputs/analyses"), "--out", help="Dossier des résultats"),  # noqa: B008
) -> None:
    """Analyse des yeux : score, niveau de risque, carte de chaleur, liste triée par urgence."""
    import csv

    from qarniya.services import AnalysisService, AnalysisStatus, triage_order

    try:
        service = AnalysisService.from_run(run_dir)
        exams = list(build_adapter(DatasetConfig.from_yaml(data_config_path)).load().exams)
    except (ConfigError, QarniyaError) as exc:
        typer.secho(f"Erreur : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    if only:
        exams = [e for e in exams if e.patient_code in set(only)]
    exams = exams[:limit] if limit else exams
    if not exams:
        typer.secho("Aucun examen à analyser.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    if service.novelty is None:
        typer.secho("(pas de détecteur d'images inhabituelles : lancez `qarniya fit-novelty`)",
                    fg=typer.colors.YELLOW)  # fmt: skip

    reports = sorted((service.analyse(e) for e in exams), key=triage_order)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for report in reports:
        exam = report.exam
        figure = ""
        if report.status is not AnalysisStatus.REJECTED:
            figure_path = out_dir / f"{exam.patient_code}_{exam.eye.value}_analyse.png"
            service.save_figure(report, figure_path)
            figure = str(figure_path)
        rows.append(
            {
                "patient": exam.patient_code,
                "oeil": exam.eye.value,
                "statut": report.status.value,
                "probabilite": f"{report.assessment.probability:.3f}" if report.assessment else "",
                "niveau": report.assessment.level.label if report.assessment else "",
                "resume": report.summary(),
                "figure": figure,
            }
        )
        typer.echo(f"  {exam.patient_code} {exam.eye.value}  {report.summary()}")
    with (out_dir / "triage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    typer.secho(f"Liste triée par urgence : {out_dir / 'triage.csv'}", bold=True)


if __name__ == "__main__":
    app()
