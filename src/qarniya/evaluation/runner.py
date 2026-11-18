"""Évaluation finale d'un entraînement sur le jeu de TEST.

Protocole (fixé AVANT de regarder le test) :
1. le seuil de décision est choisi sur la VALIDATION (sensibilité visée, ex. 95 %) ;
2. il est appliqué tel quel au TEST, jamais réajusté ;
3. la baseline « chiffres seuls » suit exactement le même protocole ;
4. fourchettes de confiance par bootstrap au niveau patient ;
5. le test n'est évalué qu'UNE fois (sinon on finit par « régler sur le test » sans le vouloir).

Résultats dans <run>/evaluation/ : metrics.json, report.md, predictions_test.csv, errors.csv,
roc_test.png, confusion_test.png.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qarniya.data import DatasetConfig, build_adapter, split_dataset
from qarniya.domain import CornealExam, Diagnosis
from qarniya.domain.exceptions import QarniyaError
from qarniya.evaluation.baseline import TabularBaseline
from qarniya.evaluation.bootstrap import ConfidenceInterval, bootstrap_confidence_intervals
from qarniya.evaluation.figures import RocSeries, plot_confusion, plot_roc
from qarniya.evaluation.metrics import BinaryMetrics, compute_binary_metrics
from qarniya.evaluation.thresholds import threshold_for_sensitivity
from qarniya.models.predictor import Predictor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEvaluation:
    name: str
    threshold: float
    test: BinaryMetrics
    test_at_default: BinaryMetrics  # au seuil naïf 0,5, pour comparaison
    confidence: dict[str, ConfidenceInterval]

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "test": self.test.to_dict(),
            "test_at_0.5": self.test_at_default.to_dict(),
            "ci95": {k: v.to_dict() for k, v in self.confidence.items()},
        }


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    model: ModelEvaluation
    baseline: ModelEvaluation | None
    n_test: int


def evaluate_run(
    run_dir: Path,
    target_sensitivity: float = 0.95,
    n_bootstrap: int = 2000,
    force: bool = False,
    device: torch.device | None = None,
) -> EvaluationResult:
    output_dir = run_dir / "evaluation"
    if (output_dir / "metrics.json").exists() and not force:
        raise QarniyaError(
            f"Le test de {run_dir.name} a déjà été évalué. Ré-évaluer après avoir modifié le "
            "modèle revient à le régler sur le test. Utilisez --force seulement si c'est voulu."
        )

    predictor = Predictor.from_checkpoint(run_dir / "model.pt", device)
    positives = {Diagnosis(v) for v in predictor.loaded.metadata.get("positive_labels", [])}
    positives = positives or {Diagnosis.KERATOCONUS, Diagnosis.SUSPECT}

    data_config = DatasetConfig.from_yaml(run_dir / "dataset.yaml")
    splits = split_dataset(build_adapter(data_config).load(), data_config.split)
    if not splits.test:
        raise QarniyaError("le jeu de test est vide")

    def targets(exams: Sequence[CornealExam]) -> list[int]:
        return [int(e.diagnosis in positives) for e in exams]

    y_val, y_test = targets(splits.val), targets(splits.test)
    groups = [e.patient_code for e in splits.test]

    logger.info("Prédiction : %d yeux de validation, %d yeux de test", len(y_val), len(y_test))
    model_eval = _evaluate(
        "Qarniya AI (images)",
        y_val, predictor.predict(splits.val),
        y_test, (p_test := predictor.predict(splits.test)),
        groups, target_sensitivity, n_bootstrap,
    )  # fmt: skip

    baseline_eval, p_base = None, None
    baseline = TabularBaseline()
    if baseline.available_features(splits.train):
        baseline.fit(splits.train, targets(splits.train))
        p_base = baseline.predict(splits.test)
        baseline_eval = _evaluate(
            f"Chiffres seuls ({', '.join(baseline.features)})",
            y_val, baseline.predict(splits.val), y_test, p_base,
            groups, target_sensitivity, n_bootstrap,
        )  # fmt: skip

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(
        output_dir, run_dir.name, splits.test, y_test, p_test, p_base, model_eval, baseline_eval,
        target_sensitivity, n_bootstrap,
    )  # fmt: skip
    return EvaluationResult(output_dir, model_eval, baseline_eval, len(y_test))


def _evaluate(
    name: str,
    y_val: list[int],
    p_val: np.ndarray,
    y_test: list[int],
    p_test: np.ndarray,
    groups: list[str],
    target_sensitivity: float,
    n_bootstrap: int,
) -> ModelEvaluation:
    threshold = threshold_for_sensitivity(y_val, p_val.tolist(), target_sensitivity)
    probs = p_test.tolist()
    return ModelEvaluation(
        name=name,
        threshold=threshold,
        test=compute_binary_metrics(y_test, probs, threshold),
        test_at_default=compute_binary_metrics(y_test, probs, 0.5),
        confidence=bootstrap_confidence_intervals(
            y_test, probs, groups, threshold, n_resamples=n_bootstrap
        ),
    )


def _write_outputs(
    out: Path,
    run_name: str,
    exams: Sequence[CornealExam],
    y_test: list[int],
    p_test: np.ndarray,
    p_base: np.ndarray | None,
    model_eval: ModelEvaluation,
    baseline_eval: ModelEvaluation | None,
    target_sensitivity: float,
    n_bootstrap: int,
) -> None:
    metrics = {
        "run": run_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "threshold_chosen_on": "validation",
            "target_sensitivity": target_sensitivity,
            "bootstrap_resamples": n_bootstrap,
            "bootstrap_unit": "patient",
        },
        "n_test_eyes": len(y_test),
        "n_test_patients": len({e.patient_code for e in exams}),
        "model": model_eval.to_dict(),
        "baseline": baseline_eval.to_dict() if baseline_eval else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), "utf-8")

    rows = _prediction_rows(exams, y_test, p_test, p_base, model_eval.threshold)
    _write_csv(out / "predictions_test.csv", rows)
    errors = [r for r in rows if r["erreur"]]
    _write_csv(out / "errors.csv", errors)

    series = [RocSeries(model_eval.name, y_test, p_test.tolist(), model_eval.test)]
    if baseline_eval is not None and p_base is not None:
        series.append(RocSeries("Chiffres seuls", y_test, p_base.tolist(), baseline_eval.test))
    plot_roc(series, out / "roc_test.png", f"Courbe ROC — test ({len(y_test)} yeux)")
    plot_confusion(model_eval.test, out / "confusion_test.png", "Qarniya AI — jeu de test")
    (out / "report.md").write_text(
        _report(run_name, metrics, model_eval, baseline_eval, errors), encoding="utf-8"
    )


def _prediction_rows(
    exams: Sequence[CornealExam],
    y_test: list[int],
    p_test: np.ndarray,
    p_base: np.ndarray | None,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for i, exam in enumerate(exams):
        predicted = int(p_test[i] >= threshold)
        kind = {(1, 0): "KC manqué (FN)", (0, 1): "fausse alerte (FP)"}.get((y_test[i], predicted))
        rows.append(
            {
                "exam_id": exam.exam_id,
                "patient": exam.patient_code,
                "oeil": exam.eye.value,
                "diagnostic": exam.diagnosis.value if exam.diagnosis else "",
                "prob_qarniya": round(float(p_test[i]), 4),
                "prob_chiffres": round(float(p_base[i]), 4) if p_base is not None else "",
                "prediction": predicted,
                "erreur": kind or "",
                "kmax": exam.clinical.get("kmax", ""),
                "epaisseur_min": exam.clinical.get("pachymetry_thinnest", ""),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["exam_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{value:.1%}".replace(".", ",")


def _line(e: ModelEvaluation) -> str:
    t, ci = e.test, e.confidence
    return (
        f"| {e.name} | {t.auc:.3f} [{ci['auc'].low:.3f}–{ci['auc'].high:.3f}] "
        f"| {_pct(t.sensitivity)} [{_pct(ci['sensitivity'].low)}–{_pct(ci['sensitivity'].high)}] "
        f"| {_pct(t.specificity)} [{_pct(ci['specificity'].low)}–{_pct(ci['specificity'].high)}] "
        f"| {t.fn} | {t.fp} | {e.threshold:.3f} |"
    )


def _report(
    run_name: str,
    metrics: dict[str, Any],
    model_eval: ModelEvaluation,
    baseline_eval: ModelEvaluation | None,
    errors: list[dict[str, Any]],
) -> str:
    target = metrics["protocol"]["target_sensitivity"]
    lines = [
        f"# Évaluation sur le jeu de test — {run_name}",
        "",
        f"- {metrics['n_test_eyes']} yeux de test ({metrics['n_test_patients']} patients), "
        "jamais vus à l'entraînement.",
        f"- Seuil choisi sur la **validation** pour viser {_pct(target)} de sensibilité, "
        "puis appliqué tel quel au test.",
        f"- Fourchettes à 95 % : bootstrap ({metrics['protocol']['bootstrap_resamples']} tirages, "
        "par patient).",
        "",
        "| Modèle | AUC | Sensibilité | Spécificité | KC manqués | Fausses alertes | Seuil |",
        "|---|---|---|---|---|---|---|",
        _line(model_eval),
    ]
    if baseline_eval:
        lines.append(_line(baseline_eval))
    lines += [
        "",
        "![ROC](roc_test.png) ![Matrice de confusion](confusion_test.png)",
        "",
        f"## Erreurs du modèle ({len(errors)})",
        "",
        "| Œil | Diagnostic | Prob. Qarniya | Prob. chiffres | Erreur | Kmax | Épaisseur min |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['patient']} {r['oeil']} | {r['diagnostic']} | {r['prob_qarniya']} "
        f"| {r['prob_chiffres']} | {r['erreur']} | {r['kmax']} | {r['epaisseur_min']} |"
        for r in errors
    ]
    return "\n".join(lines) + "\n"
