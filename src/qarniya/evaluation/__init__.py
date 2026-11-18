"""Métriques médicales, choix du seuil, intervalles de confiance et évaluation finale.

`evaluate_run` (qui charge PyTorch) s'importe explicitement :
    from qarniya.evaluation.runner import evaluate_run
"""

from qarniya.evaluation.bootstrap import ConfidenceInterval, bootstrap_confidence_intervals
from qarniya.evaluation.fusion import soft_vote
from qarniya.evaluation.metrics import BinaryMetrics, compute_binary_metrics
from qarniya.evaluation.thresholds import threshold_for_sensitivity

__all__ = [
    "BinaryMetrics",
    "ConfidenceInterval",
    "bootstrap_confidence_intervals",
    "compute_binary_metrics",
    "soft_vote",
    "threshold_for_sensitivity",
]
