"""M3: derivative estimation — the #1 failure mode of SINDy.

Two kinds of test:
  1. On *clean* analytic data, every estimator recovers the true derivative to a
     tight tolerance (d/dt sin = cos), and stays length-preserving.
  2. On *noisy* data, smoothing estimators (Savitzky-Golay, smoothing spline) must
     beat naive finite differences. This is an *inequality* test, not an absolute
     one — it encodes why M3 exists at all.
"""

import numpy as np
import pytest

from esindy.differentiation import (
    FiniteDifference,
    SavitzkyGolay,
    SplineDifferentiation,
    get_differentiator,
)

ALL_METHODS = [
    FiniteDifference(),
    SavitzkyGolay(window_length=9, polyorder=3),
    SplineDifferentiation(),
]


def _grid(n=2001, span=2 * np.pi):
    t = np.linspace(0.0, span, n)
    return t, t[1] - t[0]


@pytest.fixture(params=ALL_METHODS, ids=lambda d: type(d).__name__)
def method(request):
    return request.param


# --- clean-data recovery ----------------------------------------------------------


def test_recovers_cosine_from_sine_clean(method):
    t, _ = _grid()
    x = np.sin(t)
    x_dot = method(x, t)
    assert np.max(np.abs(x_dot - np.cos(t))) < 1e-3


def test_length_preserving_1d_and_2d(method):
    t, _ = _grid(n=500)
    x1 = np.sin(t)
    assert method(x1, t).shape == x1.shape
    X2 = np.column_stack([np.sin(t), np.cos(t)])
    assert method(X2, t).shape == X2.shape


def test_accepts_scalar_dt(method):
    t, dt = _grid(n=500)
    x = np.sin(t)
    from_dt = method(x, dt)
    from_t = method(x, t)
    np.testing.assert_allclose(from_dt, from_t, atol=1e-9)


def test_columns_are_independent(method):
    t, _ = _grid(n=800)
    X = np.column_stack([np.sin(t), 2.0 * np.cos(t)])
    out = method(X, t)
    np.testing.assert_allclose(out[:, 0], method(np.sin(t), t), atol=1e-9)
    np.testing.assert_allclose(out[:, 1], method(2.0 * np.cos(t), t), atol=1e-9)


# --- finite-difference polynomial exactness ---------------------------------------


def test_finite_difference_exact_on_linear_including_endpoints():
    t = np.linspace(0.0, 5.0, 50)
    d = FiniteDifference()(3.0 * t + 1.0, t)
    np.testing.assert_allclose(d, np.full_like(t, 3.0), atol=1e-9)


def test_finite_difference_exact_on_quadratic():
    """2nd-order central + 2nd-order one-sided ends are exact for quadratics."""
    t = np.linspace(0.0, 5.0, 50)
    d = FiniteDifference()(t**2, t)
    np.testing.assert_allclose(d, 2.0 * t, atol=1e-8)


# --- the motivating inequality: smoothing beats finite-diff on noise --------------


def _rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_savgol_beats_finite_difference_on_noisy_data():
    t, _ = _grid(n=2001)
    truth = np.cos(t)
    rng = np.random.default_rng(0)
    x_noisy = np.sin(t) + 0.01 * rng.standard_normal(t.size)

    fd_err = _rms(FiniteDifference()(x_noisy, t), truth)
    sg_err = _rms(SavitzkyGolay(window_length=51, polyorder=3)(x_noisy, t), truth)
    assert sg_err < fd_err


def test_spline_beats_finite_difference_on_noisy_data():
    t, _ = _grid(n=2001)
    truth = np.cos(t)
    rng = np.random.default_rng(1)
    x_noisy = np.sin(t) + 0.01 * rng.standard_normal(t.size)

    fd_err = _rms(FiniteDifference()(x_noisy, t), truth)
    sp_err = _rms(SplineDifferentiation(smoothing=0.5)(x_noisy, t), truth)
    assert sp_err < fd_err


# --- factory ----------------------------------------------------------------------


def test_get_differentiator_by_name():
    assert isinstance(get_differentiator("finite_difference"), FiniteDifference)
    assert isinstance(get_differentiator("savgol", window_length=7), SavitzkyGolay)
    assert isinstance(get_differentiator("spline"), SplineDifferentiation)
    with pytest.raises(KeyError):
        get_differentiator("nope")


def test_too_few_samples_raises():
    with pytest.raises(ValueError):
        FiniteDifference()(np.array([1.0, 2.0]), 1.0)
