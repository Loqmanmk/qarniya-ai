"""Les métriques médicales doivent être exactes : ce sont elles qu'on présente au jury."""

import math

import pytest

from qarniya.evaluation import compute_binary_metrics


def test_metrics_on_a_hand_computed_example() -> None:
    # 4 malades (1), 4 sains (0) ; au seuil 0,5 : 3 malades détectés, 1 manqué, 1 fausse alerte
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.05]
    m = compute_binary_metrics(y_true, y_prob)
    assert (m.tp, m.fn, m.fp, m.tn) == (3, 1, 1, 3)
    assert m.sensitivity == pytest.approx(0.75)
    assert m.specificity == pytest.approx(0.75)
    assert m.accuracy == pytest.approx(0.75)


def test_lower_threshold_catches_more_patients() -> None:
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.3, 0.2, 0.1]
    assert compute_binary_metrics(y_true, y_prob, threshold=0.5).sensitivity == 0.5
    assert compute_binary_metrics(y_true, y_prob, threshold=0.25).sensitivity == 1.0


def test_perfect_ranking_gives_auc_one() -> None:
    assert compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).auc == pytest.approx(1.0)


def test_auc_is_undefined_with_a_single_class() -> None:
    assert math.isnan(compute_binary_metrics([1, 1], [0.6, 0.9]).auc)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_binary_metrics([1, 0], [0.5])
