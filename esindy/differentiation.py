"""Derivative estimators: X(t) -> Ẋ.

Estimating Ẋ from sampled (and, in practice, noisy) X is the dominant failure mode of
SINDy — naive differentiation amplifies noise badly. So this module is pluggable: every
estimator is a callable ``(X, t) -> Ẋ`` that preserves array length, and downstream code
(M5+) takes the estimator as a strategy object.

``t`` may be a scalar sample spacing ``dt`` or a 1-D array of sample times.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


def _resolve_times(t: float | np.ndarray, m: int) -> np.ndarray:
    """Turn a scalar dt or a times array into a length-``m`` times vector."""
    if np.isscalar(t):
        return np.arange(m) * float(t)
    t = np.asarray(t, dtype=float)
    if t.shape != (m,):
        raise ValueError(f"t must have shape ({m},); got {t.shape}")
    return t


class Differentiator(ABC):
    """Base class. Subclasses implement :meth:`_diff_column` on a single 1-D signal."""

    def __call__(self, X: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim not in (1, 2):
            raise ValueError(f"X must be 1-D or 2-D; got shape {X.shape}")
        m = X.shape[0]
        if m < 3:
            raise ValueError(f"need at least 3 samples to differentiate, got {m}")
        times = _resolve_times(t, m)
        if X.ndim == 1:
            return self._diff_column(X, times)
        return np.column_stack([self._diff_column(X[:, j], times) for j in range(X.shape[1])])

    @abstractmethod
    def _diff_column(self, x: np.ndarray, times: np.ndarray) -> np.ndarray: ...


class FiniteDifference(Differentiator):
    """Second-order central differences, second-order one-sided at the boundaries.

    Implemented from scratch on a uniform grid (the case our datasets produce); for a
    non-uniform grid it falls back to ``numpy.gradient``. Exact for quadratics, including
    the endpoints — which the tests assert.
    """

    def _diff_column(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        diffs = np.diff(times)
        dt = diffs[0]
        if not np.allclose(diffs, dt):
            return np.gradient(x, times)  # non-uniform grid: defer to numpy
        d = np.empty_like(x)
        d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        d[0] = (-3.0 * x[0] + 4.0 * x[1] - x[2]) / (2.0 * dt)
        d[-1] = (3.0 * x[-1] - 4.0 * x[-2] + x[-3]) / (2.0 * dt)
        return d


class SavitzkyGolay(Differentiator):
    """Savitzky-Golay: fit a low-order polynomial in a sliding window, read off its
    derivative. The smoothing makes it far more noise-robust than finite differences.
    """

    def __init__(self, window_length: int = 9, polyorder: int = 3) -> None:
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be < window_length")
        self.window_length = window_length
        self.polyorder = polyorder

    def _diff_column(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        diffs = np.diff(times)
        dt = diffs[0]
        if not np.allclose(diffs, dt):
            raise ValueError("SavitzkyGolay requires a uniform time grid")
        window = min(self.window_length, x.size if x.size % 2 else x.size - 1)
        return savgol_filter(x, window, self.polyorder, deriv=1, delta=dt, mode="interp")


class SplineDifferentiation(Differentiator):
    """Fit a smoothing cubic spline, then differentiate it analytically.

    ``smoothing=0`` interpolates (best for clean data); larger values trade fidelity for
    noise rejection. The spline naturally handles non-uniform sampling.
    """

    def __init__(self, smoothing: float = 0.0, order: int = 3) -> None:
        self.smoothing = smoothing
        self.order = order

    def _diff_column(self, x: np.ndarray, times: np.ndarray) -> np.ndarray:
        # UnivariateSpline's `s` is an absolute sum-of-squares budget; scale it by the
        # series length so `smoothing` behaves like a per-sample knob.
        s = self.smoothing * x.size
        spline = UnivariateSpline(times, x, k=self.order, s=s)
        return spline.derivative()(times)


_REGISTRY: dict[str, type[Differentiator]] = {
    "finite_difference": FiniteDifference,
    "savgol": SavitzkyGolay,
    "spline": SplineDifferentiation,
}


def get_differentiator(name: str, **kwargs) -> Differentiator:
    """Construct a differentiator by name (``finite_difference``, ``savgol``, ``spline``)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown differentiator {name!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
