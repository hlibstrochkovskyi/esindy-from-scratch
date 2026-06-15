"""Noise-sweep harness — the shared scaffolding for M6 and M7.

It runs an estimator (vanilla SINDy now, E-SINDy later) across increasing noise levels
and reports support-recovery F1 and coefficient error, averaged over seeded trials. M6
uses it to document where vanilla SINDy breaks; M7 uses the *same* harness to show
E-SINDy degrades more gracefully.

Any estimator with the SINDy interface works here: ``fit(X, t=...)`` followed by the
``coefficients_`` and ``feature_names_`` attributes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from esindy import datasets
from esindy._seed import child_seeds
from esindy.datasets import System
from esindy.metrics import coefficient_error, support_scores

# A factory so each trial gets a fresh, unfitted estimator.
EstimatorFactory = Callable[[], object]


@dataclass(frozen=True)
class SweepPoint:
    noise_level: float
    f1: float
    f1_std: float
    coefficient_error: float
    n_trials: int


def evaluate_once(
    make_estimator: EstimatorFactory,
    system: System,
    X_clean: np.ndarray,
    t: np.ndarray,
    *,
    noise_level: float,
    seed,
) -> tuple[float, float]:
    """Fit one estimator on one noisy realization; return (F1, coefficient_error)."""
    X = datasets.add_noise(X_clean, noise_level, seed=seed)
    est = make_estimator()
    est.fit(X, t=t)
    Xi_true = system.coefficient_matrix(est.feature_names_)
    f1 = support_scores(Xi_true, est.coefficients_).f1
    return f1, coefficient_error(Xi_true, est.coefficients_)


def noise_sweep(
    make_estimator: EstimatorFactory,
    system: System,
    levels: Sequence[float],
    *,
    n_trials: int = 5,
    base_seed: int = 0,
) -> list[SweepPoint]:
    """Average F1 / coefficient error over ``n_trials`` seeded realizations per level.

    The clean trajectory is integrated once and reused; trial seeds are derived by index
    so the whole sweep is reproducible (and noise realizations are shared across levels,
    only rescaled, which keeps the degradation curve smooth).
    """
    traj = datasets.simulate(system)
    seeds = child_seeds(base_seed, n_trials)

    points: list[SweepPoint] = []
    for level in levels:
        f1s, errs = [], []
        for seed in seeds:
            f1, err = evaluate_once(
                make_estimator, system, traj.X, traj.t, noise_level=level, seed=seed
            )
            f1s.append(f1)
            errs.append(err)
        points.append(
            SweepPoint(
                noise_level=level,
                f1=float(np.mean(f1s)),
                f1_std=float(np.std(f1s)),
                coefficient_error=float(np.mean(errs)),
                n_trials=n_trials,
            )
        )
    return points


def breakdown_level(points: Sequence[SweepPoint], f1_floor: float = 1.0) -> float | None:
    """The first noise level whose mean F1 drops below ``f1_floor`` (None if never)."""
    for p in points:
        if p.f1 < f1_floor:
            return p.noise_level
    return None
