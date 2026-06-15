"""SINDy: the end-to-end model tying library + differentiation + optimizer together.

    Ẋ = Θ(X) · Ξ

``fit`` builds the candidate library Θ(X), estimates Ẋ (unless supplied), and solves for
a sparse Ξ. ``predict`` evaluates the learned RHS, ``simulate`` integrates it, and
``equations`` renders the discovered dynamics as human-readable strings.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from esindy.differentiation import Differentiator, FiniteDifference
from esindy.equations import format_equations, format_equations_latex
from esindy.library import BaseLibrary, PolynomialLibrary
from esindy.optimizers import STLSQ, Optimizer


class SINDy:
    """Sparse Identification of Nonlinear Dynamics."""

    def __init__(
        self,
        *,
        library: BaseLibrary | None = None,
        optimizer: Optimizer | None = None,
        differentiation: Differentiator | None = None,
        input_names: Sequence[str] | None = None,
    ) -> None:
        self.library = library if library is not None else PolynomialLibrary(degree=2)
        self.optimizer = optimizer if optimizer is not None else STLSQ()
        self.differentiation = (
            differentiation if differentiation is not None else FiniteDifference()
        )
        self.input_names = None if input_names is None else tuple(input_names)

    def fit(
        self,
        X: np.ndarray,
        t: float | np.ndarray | None = None,
        x_dot: np.ndarray | None = None,
    ) -> SINDy:
        X = np.asarray(X, dtype=float)
        if x_dot is None:
            if t is None:
                raise ValueError("provide t (dt or times) so derivatives can be estimated")
            x_dot = self.differentiation(X, t)
        x_dot = np.asarray(x_dot, dtype=float)

        Theta = self.library.fit_transform(X, input_names=self.input_names)
        self.input_names_ = self.library.input_names_
        self.feature_names_ = self.library.get_feature_names()
        self.optimizer.fit(Theta, x_dot)
        self.coefficients_ = self.optimizer.coef_
        return self

    # --- using the fitted model ---------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the learned RHS Θ(X)·Ξ at each row of ``X``."""
        return self.library.transform(np.asarray(X, dtype=float)) @ self.coefficients_

    def simulate(self, x0: np.ndarray, t: np.ndarray, method: str = "RK45") -> np.ndarray:
        """Integrate the discovered model from ``x0`` over the times ``t``."""
        t = np.asarray(t, dtype=float)

        def rhs(_t: float, u: np.ndarray) -> np.ndarray:
            return self.predict(u[None, :])[0]

        sol = solve_ivp(
            rhs,
            (t[0], t[-1]),
            np.asarray(x0, dtype=float),
            t_eval=t,
            method=method,
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"simulation failed: {sol.message}")
        return sol.y.T

    def equations(self, precision: int = 3) -> list[str]:
        """Render the discovered dynamics, one string per state variable."""
        return format_equations(
            self.feature_names_, self.coefficients_, self.input_names_, precision
        )

    def equations_latex(self, precision: int = 3) -> list[str]:
        """Render the discovered dynamics as LaTeX, one string per state variable."""
        return format_equations_latex(
            self.feature_names_, self.coefficients_, self.input_names_, precision
        )

    def print(self, precision: int = 3) -> None:
        for line in self.equations(precision):
            print(line)
