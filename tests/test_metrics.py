"""M5 support: scoring recovered models against ground truth."""

import numpy as np

from esindy.metrics import (
    active_set,
    coefficient_error,
    support_scores,
    trajectory_error,
)


def test_active_set_respects_tolerance():
    Xi = np.array([[0.0, 1e-9], [2.0, -3.0]])
    np.testing.assert_array_equal(active_set(Xi), [[False, False], [True, True]])


def test_perfect_recovery_scores_one():
    Xi = np.array([[1.0, 0.0], [0.0, 2.0]])
    s = support_scores(Xi, Xi)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == 1.0


def test_missing_term_lowers_recall():
    true = np.array([[1.0], [2.0]])
    pred = np.array([[1.0], [0.0]])  # dropped a true term
    s = support_scores(true, pred)
    assert s.recall < 1.0
    assert s.precision == 1.0
    assert s.false_negatives == 1


def test_extra_term_lowers_precision():
    true = np.array([[1.0], [0.0]])
    pred = np.array([[1.0], [5.0]])  # invented a term
    s = support_scores(true, pred)
    assert s.precision < 1.0
    assert s.recall == 1.0
    assert s.false_positives == 1


def test_all_zero_prediction_is_degenerate_precision_one():
    true = np.array([[1.0], [2.0]])
    pred = np.zeros((2, 1))
    s = support_scores(true, pred)
    assert s.precision == 1.0  # no false positives
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_coefficient_error_zero_when_identical():
    Xi = np.array([[1.0, -2.0], [0.0, 3.0]])
    assert coefficient_error(Xi, Xi) == 0.0


def test_coefficient_error_relative_scale():
    true = np.array([[2.0]])
    pred = np.array([[3.0]])
    assert coefficient_error(true, pred) == 0.5


def test_trajectory_error_zero_when_identical():
    X = np.random.default_rng(0).random((10, 3))
    assert trajectory_error(X, X) == 0.0
