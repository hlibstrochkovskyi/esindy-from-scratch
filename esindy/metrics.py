"""How "good" is a recovered model — so success is measured, not eyeballed.

Support recovery (which terms are nonzero) matters *more* than coefficient error: the
whole point of SINDy is recovering the right *form* of the equation. These functions
compare a recovered Ξ against the ground-truth Ξ from ``datasets``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def active_set(Xi: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of nonzero (active) coefficients."""
    return np.abs(np.asarray(Xi, dtype=float)) > tol


@dataclass(frozen=True)
class SupportScores:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def support_scores(Xi_true: np.ndarray, Xi_pred: np.ndarray, tol: float = 1e-6) -> SupportScores:
    """Precision / recall / F1 over which coefficients are active.

    Precision is 1.0 when nothing was predicted active (no false positives); recall is
    1.0 when there is nothing to find — the usual degenerate-case conventions.
    """
    true = active_set(Xi_true, tol)
    pred = active_set(Xi_pred, tol)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")

    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return SupportScores(precision, recall, f1, tp, fp, fn)


def coefficient_error(Xi_true: np.ndarray, Xi_pred: np.ndarray) -> float:
    """Relative Frobenius-norm error ‖Ξ_pred − Ξ_true‖ / ‖Ξ_true‖."""
    Xi_true = np.asarray(Xi_true, dtype=float)
    Xi_pred = np.asarray(Xi_pred, dtype=float)
    denom = np.linalg.norm(Xi_true)
    if denom == 0.0:
        return float(np.linalg.norm(Xi_pred))
    return float(np.linalg.norm(Xi_pred - Xi_true) / denom)


def trajectory_error(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """Normalized RMS error between two trajectories (short-horizon use for chaos)."""
    X_true = np.asarray(X_true, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    rms = np.sqrt(np.mean((X_pred - X_true) ** 2))
    scale = np.sqrt(np.mean(X_true**2))
    return float(rms / scale) if scale else float(rms)
