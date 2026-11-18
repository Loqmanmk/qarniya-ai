"""Évaluation finale d'un entraînement sur le jeu de TEST.

Protocole (fixé AVANT de regarder le test) :
1. le seuil de décision de chaque modèle est choisi sur la VALIDATION (sensibilité visée) ;
2. il est appliqué tel quel au TEST, jamais réajusté ;
3. tous les modèles comparés suivent exactement le même protocole ;
4. fourchettes de confiance par bootstrap au niveau patient ;
5. le test n'est évalué qu'UNE fois (sinon on finit par « régler sur le test » sans le vouloir).

Modèles comparés : images (réseau), chiffres seuls (baseline), fusion des deux (vote moyen).
Résultats dans <run>/evaluation/ : metrics.json, report.md, predictions_test.csv, errors.csv,
roc_test.png, confusion_test.png (+ confusion_fusion_test.png).
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
from qarniya.evaluation.fusion import soft_vote
from qarniya.evaluation.metrics import BinaryMetrics, compute_binary_metrics
from qarniya.evaluation.thresholds import threshold_for_sensitivity
from qarniya.models.predictor import Predictor

logger = logging.getLogger(__name__)

ERROR_KIND = {(1, 0): "KC manqué (FN)", (0, 1): "fausse alerte (FP)"}


@dataclass(frozen=True)
class ModelEvaluation:
    key: str  # identifiant court : images, chiffres, fusion
    name: str  # nom affiché
    threshold: float
    test_probabilities: np.ndarray
    test: BinaryMetrics
    test_at_default: BinaryMetrics  # au seuil naïf 0,5, pour comparaison
    confidence: dict[str, ConfidenceInterval]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "test": self.test.to_dict(),
            "test_at_0.5": self.test_at_default.to_dict(),
            "ci95": {k: v.to_dict() for k, v in self.confidence.items()},
        }

    def predicted(self, index: int) -> int:
        return int(self.test_probabilities[index] >= self.threshold)


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    evaluations: tuple[ModelEvaluation, ...]  # le modèle images en premier
    n_test: int

    @property
    def model(self) -> ModelEvaluation:
        return self.evaluations[0]

    def get(self, key: str) -> ModelEvaluation | None:
        return next((e for e in self.evaluations if e.key == key), None)


@dataclass(frozen=True)
class _Protocol:
    y_val: list[int]
    y_test: list[int]
    groups: list[str]
    target_sensitivity: float
    n_bootstrap: int

    def evaluate(
        self, key: str, name: str, p_val: np.ndarray, p_test: np.ndarray
    ) -> ModelEvaluation:
        threshold = threshold_for_sensitivity(self.y_val, p_val.tolist(), self.target_sensitivity)
        probs = p_test.tolist()
        return ModelEvaluation(
            key=key,
            name=name,
            threshold=threshold,
            test_probabilities=p_test,
            test=compute_binary_metrics(self.y_test, probs, threshold),
            test_at_default=compute_binary_metrics(self.y_test, probs, 0.5),
            confidence=bootstrap_confidence_intervals(
                self.y_test, probs, self.groups, threshold, n_resamples=self.n_bootstrap
            ),
        )


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

    protocol = _Protocol(
        y_val=targets(splits.val),
        y_test=targets(splits.test),
        groups=[e.patient_code for e in splits.test],
        target_sensitivity=target_sensitivity,
        n_bootstrap=n_bootstrap,
    )
    logger.info("Prédiction : %d yeux de validation, %d de test", len(splits.val), len(splits.test))
    img_val, img_test = predictor.predict(splits.val), predictor.predict(splits.test)
    evaluations = [protocol.evaluate("images", "Qarniya AI (images)", img_val, img_test)]

    baseline = TabularBaseline()
    if baseline.available_features(splits.train):
        baseline.fit(splits.train, targets(splits.train))
        num_val, num_test = baseline.predict(splits.val), baseline.predict(splits.test)
        evaluations += [
            protocol.evaluate("chiffres", "Chiffres seuls", num_val, num_test),
            # vote moyen : aucun paramètre appris, seul le seuil vient de la validation
            protocol.evaluate(
                "fusion",
                "Fusion images + chiffres",
                soft_vote([img_val, num_val]),
                soft_vote([img_test, num_test]),
            ),
        ]
        logger.info("Variables de la baseline : %s", ", ".join(baseline.features))

    output_dir.mkdir(parents=True, exist_ok=True)
    result = EvaluationResult(output_dir, tuple(evaluations), len(splits.test))
    _write_outputs(result, run_dir.name, splits.test, protocol)
    return result


# ---------------------------------------------------------------- sorties
def _write_outputs(
    result: EvaluationResult, run_name: str, exams: Sequence[CornealExam], protocol: _Protocol
) -> None:
    out, y_test = result.output_dir, protocol.y_test
    metrics = {
        "run": run_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "threshold_chosen_on": "validation",
            "target_sensitivity": protocol.target_sensitivity,
            "bootstrap_resamples": protocol.n_bootstrap,
            "bootstrap_unit": "patient",
            "fusion": "moyenne des probabilités (aucun paramètre appris)",
        },
        "n_test_eyes": len(y_test),
        "n_test_patients": len(set(protocol.groups)),
        "models": {e.key: e.to_dict() for e in result.evaluations},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), "utf-8")

    rows = _prediction_rows(exams, y_test, result.evaluations)
    _write_csv(out / "predictions_test.csv", rows)
    errors = [r for r in rows if any(r[f"erreur_{e.key}"] for e in result.evaluations)]
    _write_csv(out / "errors.csv", errors)

    plot_roc(
        [
            RocSeries(e.name, y_test, e.test_probabilities.tolist(), e.test)
            for e in result.evaluations
        ],
        out / "roc_test.png",
        f"Courbe ROC — test ({len(y_test)} yeux)",
    )
    plot_confusion(result.model.test, out / "confusion_test.png", "Qarniya AI (images) — test")
    fusion = result.get("fusion")
    if fusion:
        plot_confusion(
            fusion.test, out / "confusion_fusion_test.png", "Fusion images + chiffres — test"
        )
    (out / "report.md").write_text(_report(run_name, metrics, result, errors), encoding="utf-8")


def _prediction_rows(
    exams: Sequence[CornealExam], y_test: list[int], evaluations: Sequence[ModelEvaluation]
) -> list[dict[str, Any]]:
    rows = []
    for i, exam in enumerate(exams):
        row: dict[str, Any] = {
            "exam_id": exam.exam_id,
            "patient": exam.patient_code,
            "oeil": exam.eye.value,
            "diagnostic": exam.diagnosis.value if exam.diagnosis else "",
            "kmax": exam.clinical.get("kmax", ""),
            "epaisseur_min": exam.clinical.get("pachymetry_thinnest", ""),
        }
        for e in evaluations:
            row[f"prob_{e.key}"] = round(float(e.test_probabilities[i]), 4)
            row[f"erreur_{e.key}"] = ERROR_KIND.get((y_test[i], e.predicted(i)), "")
        rows.append(row)
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
    sens, spec = ci["sensitivity"], ci["specificity"]
    return (
        f"| {e.name} | {t.auc:.3f} [{ci['auc'].low:.3f}–{ci['auc'].high:.3f}] "
        f"| {_pct(t.sensitivity)} [{_pct(sens.low)}–{_pct(sens.high)}] "
        f"| {_pct(t.specificity)} [{_pct(spec.low)}–{_pct(spec.high)}] "
        f"| {t.fn} | {t.fp} | {e.threshold:.3f} |"
    )


def _report(
    run_name: str, metrics: dict[str, Any], result: EvaluationResult, errors: list[dict[str, Any]]
) -> str:
    protocol = metrics["protocol"]
    keys = [e.key for e in result.evaluations]
    lines = [
        f"# Évaluation sur le jeu de test — {run_name}",
        "",
        f"- {metrics['n_test_eyes']} yeux de test ({metrics['n_test_patients']} patients), "
        "jamais vus à l'entraînement.",
        f"- Seuils choisis sur la **validation** pour viser {_pct(protocol['target_sensitivity'])} "
        "de sensibilité, puis appliqués tels quels au test.",
        f"- Fourchettes à 95 % : bootstrap ({protocol['bootstrap_resamples']} tirages, "
        "par patient).",
        f"- Fusion : {protocol['fusion']}.",
        "",
        "| Modèle | AUC | Sensibilité | Spécificité | KC manqués | Fausses alertes | Seuil |",
        "|---|---|---|---|---|---|---|",
        *(_line(e) for e in result.evaluations),
        "",
        "![ROC](roc_test.png) ![Matrice de confusion](confusion_test.png)",
        "",
        f"## Yeux mal classés par au moins un modèle ({len(errors)})",
        "",
        "| Œil | Diagnostic | Kmax | Épaisseur min | "
        + " | ".join(f"Prob. {k}" for k in keys)
        + " | "
        + " | ".join(f"Erreur {k}" for k in keys)
        + " |",
        "|---|---|---|---|" + "---|" * (2 * len(keys)),
    ]
    for r in errors:
        probs = " | ".join(str(r[f"prob_{k}"]) for k in keys)
        errs = " | ".join(r[f"erreur_{k}"] or "—" for k in keys)
        lines.append(
            f"| {r['patient']} {r['oeil']} | {r['diagnostic']} | {r['kmax']} "
            f"| {r['epaisseur_min']} | {probs} | {errs} |"
        )
    return "\n".join(lines) + "\n"
