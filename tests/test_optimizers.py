"""M4: sparse regression via STLSQ.

On exact, well-conditioned synthetic data STLSQ must recover the planted sparse
coefficients to machine precision (with ridge alpha = 0). The threshold's drop/keep
behaviour is pinned with coefficients planted just below and just above lambda, and the
column-normalization path is shown to rescue a true-but-small coefficient on a
large-magnitude column that a single global threshold would wrongly discard.

Convention note (for the M5 oracle vs PySINDy): PySINDy's STLSQ defaults to ridge
alpha = 0.05 and applies the threshold in normalized space when normalize_columns is on.
We default alpha = 0.0 (so exact recovery is exact) and match PySINDy's normalization
convention; oracle tests must set alpha equally on both sides.
"""

import numpy as np
import pytest

from esindy.optimizers import STLSQ, LeastSquares


def test_exact_recovery_well_conditioned():
    rng = np.random.default_rng(0)
    Theta = rng.standard_normal((200, 8))
    xi = np.zeros(8)
    xi[[2, 5, 7]] = [3.5, -2.0, 1.2]
    y = Theta @ xi

    coef = STLSQ(threshold=0.5).fit(Theta, y).coef_[:, 0]
    np.testing.assert_allclose(coef, xi, atol=1e-9)


def test_threshold_drops_below_keeps_above():
    Theta = np.eye(4)
    y = np.array([0.09, 0.11, 5.0, 0.0])  # identity design => coef == y before threshold
    coef = STLSQ(threshold=0.1).fit(Theta, y).coef_[:, 0]
    assert coef[0] == 0.0  # 0.09 < lambda  -> dropped
    assert coef[1] == pytest.approx(0.11)  # 0.11 > lambda  -> survives
    assert coef[2] == pytest.approx(5.0)
    assert coef[3] == 0.0


def test_all_below_threshold_gives_zero():
    Theta = np.eye(3)
    y = np.array([0.01, 0.02, 0.03])
    coef = STLSQ(threshold=0.1).fit(Theta, y).coef_[:, 0]
    np.testing.assert_array_equal(coef, np.zeros(3))


def test_multi_target_shape_and_recovery():
    rng = np.random.default_rng(1)
    Theta = rng.standard_normal((150, 6))
    Xi = np.zeros((6, 2))
    Xi[1, 0] = 4.0
    Xi[3, 1] = -1.5
    Xi[4, 1] = 2.2
    Xdot = Theta @ Xi
    coef = STLSQ(threshold=0.5).fit(Theta, Xdot).coef_
    assert coef.shape == (6, 2)
    np.testing.assert_allclose(coef, Xi, atol=1e-9)


def test_convergence_is_bounded():
    rng = np.random.default_rng(2)
    Theta = rng.standard_normal((100, 10))
    y = rng.standard_normal(100)
    # Should simply terminate and return a finite coefficient vector.
    coef = STLSQ(threshold=0.3, max_iter=5).fit(Theta, y).coef_[:, 0]
    assert np.all(np.isfinite(coef))


def test_normalize_columns_rescues_small_coeff_on_large_column():
    """A true but small coefficient on a large-magnitude column is wrongly dropped by a
    single global threshold; normalize_columns recovers it."""
    rng = np.random.default_rng(3)
    c0 = rng.standard_normal(300)
    c1 = 1000.0 * rng.standard_normal(300)  # huge scale
    c2 = rng.standard_normal(300)
    Theta = np.column_stack([c0, c1, c2])
    xi = np.array([2.0, 0.005, 0.0])  # small true coeff on the big column
    y = Theta @ xi

    naive = STLSQ(threshold=0.1, normalize_columns=False).fit(Theta, y).coef_[:, 0]
    assert naive[1] == 0.0  # dropped for the wrong reason

    scaled = STLSQ(threshold=0.1, normalize_columns=True).fit(Theta, y).coef_[:, 0]
    assert scaled[1] != 0.0
    np.testing.assert_allclose(scaled, xi, atol=1e-6)


def test_ridge_alpha_shrinks_coefficients():
    rng = np.random.default_rng(4)
    Theta = rng.standard_normal((200, 4))
    xi = np.array([5.0, 0.0, -3.0, 0.0])
    y = Theta @ xi
    plain = STLSQ(threshold=0.5, alpha=0.0).fit(Theta, y).coef_[:, 0]
    ridged = STLSQ(threshold=0.5, alpha=10.0).fit(Theta, y).coef_[:, 0]
    # same support, but ridge pulls magnitudes toward zero
    assert np.abs(ridged[0]) < np.abs(plain[0])


def test_least_squares_is_dense_solution():
    rng = np.random.default_rng(5)
    Theta = rng.standard_normal((50, 3))
    xi = np.array([1.0, -2.0, 0.5])
    y = Theta @ xi
    coef = LeastSquares().fit(Theta, y).coef_[:, 0]
    np.testing.assert_allclose(coef, xi, atol=1e-9)


def test_fit_accepts_1d_and_2d_targets():
    Theta = np.eye(3)
    assert STLSQ().fit(Theta, np.ones(3)).coef_.shape == (3, 1)
    assert STLSQ().fit(Theta, np.ones((3, 2))).coef_.shape == (3, 2)
