"""Sparse-regression backends: solve Θ · Ξ ≈ Ẋ for a sparse Ξ.

The workhorse is **STLSQ** (Sequentially Thresholded Least Squares): least-squares,
zero everything below a threshold, refit on the survivors, repeat until the support
stops changing.

Two conventions are decided here and matter for matching PySINDy later (M5 oracle):
  - ``alpha`` is ridge (Tikhonov) regularization. We default to ``0.0`` so exact
    synthetic recovery is exact; PySINDy's STLSQ defaults to ``0.05``.
  - ``normalize_columns`` scales each library column to unit L2 norm, applies the
    threshold in that normalized space, then de-normalizes the coefficients — the same
    convention PySINDy uses, so a single global ``threshold`` is meaningful even when
    columns differ by orders of magnitude.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _as_2d_targets(Xdot: np.ndarray) -> np.ndarray:
    Xdot = np.asarray(Xdot, dtype=float)
    return Xdot[:, None] if Xdot.ndim == 1 else Xdot


def _ridge_solve(A: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Least squares with optional Tikhonov regularization, via an augmented system."""
    if alpha > 0.0:
        n = A.shape[1]
        A = np.vstack([A, np.sqrt(alpha) * np.eye(n)])
        y = np.concatenate([y, np.zeros(n)])
    return np.linalg.lstsq(A, y, rcond=None)[0]


class Optimizer(ABC):
    """Base class. ``fit`` populates ``coef_`` with Ξ of shape ``(p, n_targets)``."""

    coef_: np.ndarray

    def fit(self, Theta: np.ndarray, Xdot: np.ndarray) -> Optimizer:
        Theta = np.asarray(Theta, dtype=float)
        targets = _as_2d_targets(Xdot)
        self.coef_ = np.column_stack(
            [self._fit_target(Theta, targets[:, k]) for k in range(targets.shape[1])]
        )
        return self

    @abstractmethod
    def _fit_target(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class LeastSquares(Optimizer):
    """Plain (optionally ridge) least squares — a dense baseline for comparison."""

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def _fit_target(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _ridge_solve(Theta, y, self.alpha)


class STLSQ(Optimizer):
    """Sequentially Thresholded Least Squares."""

    def __init__(
        self,
        threshold: float = 0.1,
        *,
        alpha: float = 0.0,
        max_iter: int = 20,
        normalize_columns: bool = False,
    ) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize_columns = normalize_columns

    def _fit_target(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.normalize_columns:
            norms = np.linalg.norm(Theta, axis=0)
            norms = np.where(norms == 0.0, 1.0, norms)
            coef = self._stlsq(Theta / norms, y)
            return coef / norms  # de-normalize back to original column scale
        return self._stlsq(Theta, y)

    def _stlsq(self, Theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = Theta.shape[1]
        coef = _ridge_solve(Theta, y, self.alpha)
        support = np.abs(coef) >= self.threshold

        for _ in range(self.max_iter):
            coef = np.zeros(p)
            if not support.any():
                break
            coef[support] = _ridge_solve(Theta[:, support], y, self.alpha)
            new_support = np.abs(coef) >= self.threshold
            if np.array_equal(new_support, support):
                break
            support = new_support

        coef[~support] = 0.0
        return coef
