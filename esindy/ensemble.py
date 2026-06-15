"""E-SINDy: bootstrap-ensemble SINDy.

Wrap STLSQ in bootstrapping to get noise robustness and uncertainty estimates:

  - **Data bagging** — resample time points with replacement each run.
  - **Library bagging** — randomly drop candidate columns each run.
  - For each of ``n_models`` bootstraps, run the optimizer, collect one Ξ^(b).
  - **Aggregate**: a term's *inclusion probability* is how often it is nonzero (over
    the bootstraps where it was even a candidate); its *value* is the median over the
    bootstraps where it is included. A second threshold on inclusion probability is
    what gives E-SINDy its robustness.

Two design choices that matter downstream:
  - Each bootstrap's RNG is derived from its *index* via ``child_seeds`` — so M8 can run
    the loop in parallel and get bit-identical results to this serial version.
  - With library bagging, every Ξ^(b) lives in a different sub-space; we scatter each
    sub-result back into the full library layout, marking absent columns with NaN so
    inclusion probabilities are computed over availability, not over a fixed ``n_models``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from joblib import Parallel, delayed, parallel_config

from esindy._seed import as_generator, child_seeds
from esindy.differentiation import Differentiator, FiniteDifference
from esindy.equations import format_equations, format_equations_latex
from esindy.library import BaseLibrary, PolynomialLibrary
from esindy.optimizers import STLSQ, Optimizer


def aggregate_bootstraps(
    coef_samples: np.ndarray, inclusion_threshold: float = 0.6, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce bootstrap coefficient samples to (coefficients, inclusion_probabilities).

    ``coef_samples`` has shape ``(n_models, p, n_targets)``; entries that are ``NaN``
    mark terms that were *not candidates* in that bootstrap (library bagging) and are
    excluded from both numerator and denominator of the inclusion probability.
    """
    present = ~np.isnan(coef_samples)
    active = present & (np.abs(coef_samples) > tol)
    n_present = present.sum(axis=0)
    n_active = active.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        inclusion = np.where(n_present > 0, n_active / n_present, 0.0)

    # Median of each coefficient over the bootstraps where it is active. Terms that are
    # never active produce an all-NaN slice (warns, returns NaN) -> coalesced to 0.
    masked = np.where(active, coef_samples, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(masked, axis=0)
    median = np.nan_to_num(median, nan=0.0)

    support = inclusion >= inclusion_threshold
    coefficients = np.where(support, median, 0.0)
    return coefficients, inclusion


def _clone_optimizer(optimizer: Optimizer) -> Optimizer:
    # Each bootstrap needs its own optimizer instance (fresh coef_); rebuild from the
    # template's public attributes rather than sharing mutable state across runs.
    return type(optimizer)(**{k: v for k, v in vars(optimizer).items() if not k.endswith("_")})


def _run_one_bootstrap(
    child_seed,
    Theta: np.ndarray,
    x_dot: np.ndarray,
    optimizer: Optimizer,
    *,
    n_subset: int,
    replace: bool,
    library_ensemble: bool,
    n_candidates_to_drop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One bootstrap fit. Module-level and pure so it is picklable for the loky backend.

    The RNG is rebuilt from ``child_seed`` (an index-derived ``SeedSequence``), which is
    what makes the result independent of execution order — so parallel == serial.
    """
    m, p = Theta.shape
    rng = as_generator(child_seed)
    rows = rng.integers(0, m, size=n_subset) if replace else rng.permutation(m)[:n_subset]
    if library_ensemble and n_candidates_to_drop > 0:
        cols = np.sort(rng.choice(p, size=p - n_candidates_to_drop, replace=False))
    else:
        cols = np.arange(p)
    optimizer = _clone_optimizer(optimizer).fit(Theta[np.ix_(rows, cols)], x_dot[rows])
    return cols, optimizer.coef_


class ESINDy:
    """Bootstrap-ensemble SINDy with a SINDy-compatible fit/attribute interface."""

    def __init__(
        self,
        *,
        library: BaseLibrary | None = None,
        optimizer: Optimizer | None = None,
        differentiation: Differentiator | None = None,
        input_names: Sequence[str] | None = None,
        n_models: int = 100,
        n_subset: int | None = None,
        replace: bool = True,
        library_ensemble: bool = False,
        n_candidates_to_drop: int = 0,
        inclusion_threshold: float = 0.6,
        seed=0,
        n_jobs: int = 1,
    ) -> None:
        self.library = library if library is not None else PolynomialLibrary(degree=2)
        self.optimizer = optimizer if optimizer is not None else STLSQ()
        self.differentiation = (
            differentiation if differentiation is not None else FiniteDifference()
        )
        self.input_names = None if input_names is None else tuple(input_names)
        self.n_models = n_models
        self.n_subset = n_subset
        self.replace = replace
        self.library_ensemble = library_ensemble
        self.n_candidates_to_drop = n_candidates_to_drop
        self.inclusion_threshold = inclusion_threshold
        self.seed = seed
        self.n_jobs = n_jobs

    def fit(
        self,
        X: np.ndarray,
        t: float | np.ndarray | None = None,
        x_dot: np.ndarray | None = None,
    ) -> ESINDy:
        X = np.asarray(X, dtype=float)
        if x_dot is None:
            if t is None:
                raise ValueError("provide t (dt or times) so derivatives can be estimated")
            x_dot = self.differentiation(X, t)
        x_dot = np.asarray(x_dot, dtype=float)
        if x_dot.ndim == 1:
            x_dot = x_dot[:, None]

        Theta = self.library.fit_transform(X, input_names=self.input_names)
        self.input_names_ = self.library.input_names_
        self.feature_names_ = self.library.get_feature_names()

        m, p = Theta.shape
        n_targets = x_dot.shape[1]
        n_subset = self.n_subset if self.n_subset is not None else m
        if self.library_ensemble and self.n_candidates_to_drop >= p:
            raise ValueError("n_candidates_to_drop must be < number of library terms")

        results = self._run_bootstraps(Theta, x_dot, n_subset)

        samples = np.full((self.n_models, p, n_targets), np.nan)
        for b, (cols, coef_sub) in enumerate(results):
            samples[b, cols, :] = coef_sub

        self.coef_samples_ = samples
        self.coefficients_, self.inclusion_probabilities_ = aggregate_bootstraps(
            samples, self.inclusion_threshold
        )
        return self

    def _run_bootstraps(
        self, Theta: np.ndarray, x_dot: np.ndarray, n_subset: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        seeds = child_seeds(self.seed, self.n_models)
        kwargs = dict(
            n_subset=n_subset,
            replace=self.replace,
            library_ensemble=self.library_ensemble,
            n_candidates_to_drop=self.n_candidates_to_drop,
        )
        if self.n_jobs == 1:
            return [_run_one_bootstrap(s, Theta, x_dot, self.optimizer, **kwargs) for s in seeds]
        # Pin BLAS to one thread per worker so the loky processes don't oversubscribe the
        # CPU (each STLSQ fit already calls into multithreaded LAPACK). joblib preserves
        # submission order, so scattering results by index stays deterministic.
        with parallel_config(backend="loky", inner_max_num_threads=1):
            return Parallel(n_jobs=self.n_jobs)(
                delayed(_run_one_bootstrap)(s, Theta, x_dot, self.optimizer, **kwargs)
                for s in seeds
            )

    def equations(self, precision: int = 3) -> list[str]:
        """Render the aggregated dynamics, one string per state (mirrors SINDy)."""
        return format_equations(
            self.feature_names_, self.coefficients_, self.input_names_, precision
        )

    def equations_latex(self, precision: int = 3) -> list[str]:
        """Render the aggregated dynamics as LaTeX, one string per state."""
        return format_equations_latex(
            self.feature_names_, self.coefficients_, self.input_names_, precision
        )
