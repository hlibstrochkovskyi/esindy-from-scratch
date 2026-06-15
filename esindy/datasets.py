"""Known dynamical systems with exact ground truth — the test oracle.

Each :class:`System` ships the *exact* coefficient matrix Ξ of its governing equations,
expressed in named candidate terms (``"x0"``, ``"x0 x1"``, ``"x y"``, ``"x^2"`` …).
Tests diff recovered models against this, never against eyeballed output.

Feature-name convention (shared with the library in M2):
  - the constant term is ``"1"``;
  - a power is ``"<var>^<k>"`` (e.g. ``"x0^2"``);
  - a product is space-joined in state order (e.g. ``"x0 x1"``, ``"x z"``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from esindy._seed import as_generator

# state -> {feature_name: coefficient}
Equations = dict[str, dict[str, float]]


@dataclass(frozen=True)
class System:
    """A dynamical system together with its exact symbolic dynamics."""

    name: str
    state_names: tuple[str, ...]
    rhs: Callable[[float, np.ndarray], np.ndarray]
    equations: Equations
    default_x0: np.ndarray
    default_t_span: tuple[float, float]
    default_dt: float
    params: dict[str, float] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return len(self.state_names)

    def feature_names(self) -> list[str]:
        """Unique candidate terms appearing in the true dynamics, in a stable order.

        This is the *minimal* basis that exactly expresses the system — not the full
        polynomial library (that is M2's job). Ordered by (number of factors, name) so
        the ordering is deterministic across runs.
        """
        names = {name for terms in self.equations.values() for name in terms}

        def sort_key(name: str) -> tuple[int, str]:
            n_factors = 0 if name == "1" else len(name.split())
            return (n_factors, name)

        return sorted(names, key=sort_key)

    def coefficient_matrix(self, feature_names: list[str]) -> np.ndarray:
        """Materialize Ξ (shape ``(p, n_states)``) against a given feature ordering.

        Terms absent from a state's equation are zero — this is what lets the same
        system describe its truth against any superset library (used from M5 on).
        """
        Xi = np.zeros((len(feature_names), self.n_states))
        for j, name in enumerate(feature_names):
            for i, state in enumerate(self.state_names):
                Xi[j, i] = self.equations[state].get(name, 0.0)
        return Xi


@dataclass(frozen=True)
class Trajectory:
    """A sampled state trajectory plus a back-reference to its source system."""

    t: np.ndarray
    X: np.ndarray
    system: System

    @property
    def x_dot_exact(self) -> np.ndarray:
        """Exact derivatives from the RHS at the sampled states (clean-data tests)."""
        return np.array([self.system.rhs(0.0, row) for row in self.X])


# --- system definitions ----------------------------------------------------------


def _linear2d() -> System:
    """Damped linear oscillator: ẋ = -0.1x + 2y, ẏ = -2x - 0.1y."""
    a, w = 0.1, 2.0

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        x, y = u
        return np.array([-a * x + w * y, -w * x - a * y])

    equations: Equations = {
        "x0": {"x0": -a, "x1": w},
        "x1": {"x0": -w, "x1": -a},
    }
    return System(
        name="linear2d",
        state_names=("x0", "x1"),
        rhs=rhs,
        equations=equations,
        default_x0=np.array([2.0, 0.0]),
        default_t_span=(0.0, 10.0),
        default_dt=0.01,
        params={"a": a, "w": w},
    )


def _lotka_volterra() -> System:
    """Predator-prey: ẋ = a·x − b·x·y, ẏ = −c·y + d·x·y."""
    a, b, c, d = 1.5, 1.0, 3.0, 1.0

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        x, y = u
        return np.array([a * x - b * x * y, -c * y + d * x * y])

    equations: Equations = {
        "x0": {"x0": a, "x0 x1": -b},
        "x1": {"x1": -c, "x0 x1": d},
    }
    return System(
        name="lotka_volterra",
        state_names=("x0", "x1"),
        rhs=rhs,
        equations=equations,
        default_x0=np.array([10.0, 5.0]),
        default_t_span=(0.0, 15.0),
        default_dt=0.01,
        params={"a": a, "b": b, "c": c, "d": d},
    )


def _lorenz() -> System:
    """Lorenz attractor: ẋ = σ(y−x), ẏ = x(ρ−z) − y, ż = xy − βz."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        x, y, z = u
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    equations: Equations = {
        "x": {"x": -sigma, "y": sigma},
        "y": {"x": rho, "y": -1.0, "x z": -1.0},
        "z": {"z": -beta, "x y": 1.0},
    }
    return System(
        name="lorenz",
        state_names=("x", "y", "z"),
        rhs=rhs,
        equations=equations,
        default_x0=np.array([-8.0, 7.0, 27.0]),
        default_t_span=(0.0, 8.0),
        default_dt=0.002,
        params={"sigma": sigma, "rho": rho, "beta": beta},
    )


_REGISTRY: dict[str, Callable[[], System]] = {
    "linear2d": _linear2d,
    "lotka_volterra": _lotka_volterra,
    "lorenz": _lorenz,
}


def available_systems() -> list[str]:
    """Names of all registered systems."""
    return list(_REGISTRY)


def get_system(name: str) -> System:
    """Construct a registered system by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown system {name!r}; available: {available_systems()}")
    return _REGISTRY[name]()


# --- simulation ------------------------------------------------------------------


def simulate(
    system: System,
    *,
    x0: np.ndarray | None = None,
    t_span: tuple[float, float] | None = None,
    dt: float | None = None,
    method: str = "RK45",
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Trajectory:
    """Integrate ``system`` and return a clean trajectory sampled on a uniform grid.

    Tight default tolerances keep Lorenz on its attractor and the Lotka–Volterra
    invariant flat enough for the conservation test to bite.
    """
    x0 = system.default_x0 if x0 is None else np.asarray(x0, dtype=float)
    t0, t1 = system.default_t_span if t_span is None else t_span
    dt = system.default_dt if dt is None else dt

    t_eval = np.arange(t0, t1, dt)
    sol = solve_ivp(system.rhs, (t0, t1), x0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration of {system.name!r} failed: {sol.message}")
    return Trajectory(t=sol.t, X=sol.y.T, system=system)


def add_noise(X: np.ndarray, level: float, *, seed=None) -> np.ndarray:
    """Add column-scaled Gaussian noise: σ_j = level · std(X[:, j]).

    Scaling per column makes ``level`` a dimensionless noise fraction comparable
    across states — the knob the M6 noise sweep turns.
    """
    X = np.asarray(X, dtype=float)
    if level == 0.0:
        return X.copy()
    rng = as_generator(seed)
    scale = level * X.std(axis=0, keepdims=True)
    return X + rng.normal(size=X.shape) * scale
