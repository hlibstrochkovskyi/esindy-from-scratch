"""Weak (integral) formulation of SINDy — the noise-robust extension.

The dominant failure mode of SINDy is differentiating noisy data. The weak form sidesteps
it entirely. Multiply ``ẋ = Θ(x)ξ`` by a smooth, compactly-supported test function
``φ_k(t)`` and integrate over the domain. Integration by parts moves the derivative onto
``φ_k`` (which we know analytically), and because ``φ_k`` vanishes at the endpoints the
boundary term drops:

    ∫ φ_k ẋ dt = -∫ φ_k' x dt = ∫ φ_k Θ(x) ξ dt

So with one row per test function we assemble a linear system ``G ξ = b`` where

    G[k, j] = ∫ φ_k(t) θ_j(x(t)) dt          (no derivative of x)
    b[k, i] = -∫ φ_k'(t) x_i(t) dt           (derivative is on φ, not on noisy x)

and solve it with the same STLSQ optimizer. The integrals average over the data, which is
what suppresses noise. We use bump test functions φ(t) = (1 - u²)^p with u = (t-c)/r.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from esindy.equations import format_equations, format_equations_latex
from esindy.library import BaseLibrary, PolynomialLibrary
from esindy.optimizers import STLSQ, Optimizer


def _trapezoid_weights(t: np.ndarray) -> np.ndarray:
    """Quadrature weights so that ``w @ f`` is the trapezoidal integral of ``f`` over t."""
    w = np.empty_like(t)
    w[0] = (t[1] - t[0]) / 2
    w[-1] = (t[-1] - t[-2]) / 2
    w[1:-1] = (t[2:] - t[:-2]) / 2
    return w


def build_test_functions(t: np.ndarray, n_test_functions: int, support_fraction: float, p: int):
    """Return (Phi, dPhi) of shape (m, K): bump functions and their derivatives.

    Centers are spaced so each bump's support ``[c-r, c+r]`` stays inside the domain, so
    every φ_k vanishes at the global endpoints and the integration-by-parts boundary term
    is exactly zero.
    """
    t0, t1 = t[0], t[-1]
    r = support_fraction * (t1 - t0)
    centers = np.linspace(t0 + r, t1 - r, n_test_functions)

    Phi = np.zeros((t.size, n_test_functions))
    dPhi = np.zeros((t.size, n_test_functions))
    for k, c in enumerate(centers):
        u = (t - c) / r
        inside = np.abs(u) < 1
        ui = u[inside]
        Phi[inside, k] = (1 - ui**2) ** p
        dPhi[inside, k] = p * (1 - ui**2) ** (p - 1) * (-2 * ui / r)
    return Phi, dPhi


class WeakSINDy:
    """SINDy via the weak formulation. Same fit/attribute interface as :class:`SINDy`."""

    def __init__(
        self,
        *,
        library: BaseLibrary | None = None,
        optimizer: Optimizer | None = None,
        input_names: Sequence[str] | None = None,
        n_test_functions: int = 200,
        support_fraction: float = 0.05,
        test_poly_order: int = 4,
    ) -> None:
        self.library = library if library is not None else PolynomialLibrary(degree=2)
        self.optimizer = optimizer if optimizer is not None else STLSQ()
        self.input_names = None if input_names is None else tuple(input_names)
        self.n_test_functions = n_test_functions
        self.support_fraction = support_fraction
        self.test_poly_order = test_poly_order

    def fit(self, X: np.ndarray, t: np.ndarray) -> WeakSINDy:
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        if t.ndim != 1 or t.shape[0] != X.shape[0]:
            raise ValueError("weak form needs a 1-D times array matching X's length")

        Theta = self.library.fit_transform(X, input_names=self.input_names)
        self.input_names_ = self.library.input_names_
        self.feature_names_ = self.library.get_feature_names()

        Phi, dPhi = build_test_functions(
            t, self.n_test_functions, self.support_fraction, self.test_poly_order
        )
        w = _trapezoid_weights(t)[:, None]
        G = (Phi * w).T @ Theta  # (K, p)  integral of phi_k * theta_j
        b = -(dPhi * w).T @ X  # (K, n)  weak derivative, no diff of noisy X

        self.optimizer.fit(G, b)
        self.coefficients_ = self.optimizer.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.library.transform(np.asarray(X, dtype=float)) @ self.coefficients_

    def equations(self, precision: int = 3) -> list[str]:
        return format_equations(
            self.feature_names_, self.coefficients_, self.input_names_, precision
        )

    def equations_latex(self, precision: int = 3) -> list[str]:
        return format_equations_latex(
            self.feature_names_, self.coefficients_, self.input_names_, precision
        )
