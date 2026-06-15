"""M1: datasets are the ground-truth oracle.

Every system ships its exact coefficient matrix Ξ expressed in named candidate terms.
The headline test (`test_xi_matches_symbolic_form`) independently re-derives the RHS
from (Ξ, feature names) and asserts it equals the integrator's RHS — so a typo in a
coefficient dict can never silently become the "truth" the rest of the suite trusts.
"""

import numpy as np
import pytest

from esindy import datasets

ALL_SYSTEMS = ["linear2d", "lotka_volterra", "lorenz"]


@pytest.fixture(params=ALL_SYSTEMS)
def system(request):
    return datasets.get_system(request.param)


def _eval_feature(name: str, state: np.ndarray, state_names: tuple[str, ...]) -> float:
    """Minimal, independent evaluator for a feature-name string.

    Deliberately hand-rolled rather than importing the (not-yet-built) library, so this
    test is an external check on the dataset's declared symbolic form.
    """
    if name == "1":
        return 1.0
    value = 1.0
    for factor in name.split():
        if "^" in factor:
            base, exponent = factor.split("^")
            value *= state[state_names.index(base)] ** int(exponent)
        else:
            value *= state[state_names.index(factor)]
    return float(value)


# --- registry --------------------------------------------------------------------


def test_get_system_unknown_raises():
    with pytest.raises(KeyError):
        datasets.get_system("not_a_system")


def test_available_systems_listed():
    assert set(datasets.available_systems()) == set(ALL_SYSTEMS)


# --- shapes & integration --------------------------------------------------------


def test_trajectory_shapes(system):
    traj = datasets.simulate(system)
    m = traj.t.shape[0]
    assert traj.t.ndim == 1
    assert traj.X.shape == (m, system.n_states)
    assert m > 10
    assert traj.t[0] == pytest.approx(system.default_t_span[0])
    assert traj.t[-1] <= system.default_t_span[1] + 1e-12


def test_simulate_is_deterministic(system):
    a = datasets.simulate(system).X
    b = datasets.simulate(system).X
    np.testing.assert_array_equal(a, b)


def test_x_dot_exact_matches_rhs(system):
    traj = datasets.simulate(system)
    manual = np.array([system.rhs(0.0, row) for row in traj.X])
    np.testing.assert_allclose(traj.x_dot_exact, manual)


# --- the ground-truth contract ---------------------------------------------------


def test_coefficient_matrix_shape(system):
    names = system.feature_names()
    Xi = system.coefficient_matrix(names)
    assert Xi.shape == (len(names), system.n_states)


def test_feature_names_cover_all_nonzero_terms(system):
    names = set(system.feature_names())
    for terms in system.equations.values():
        assert set(terms) <= names


def test_xi_matches_symbolic_form(system):
    """Θ_true · Ξ must reproduce the integrator's RHS at every sampled point."""
    traj = datasets.simulate(system)
    names = system.feature_names()
    Xi = system.coefficient_matrix(names)
    Theta = np.array(
        [[_eval_feature(nm, row, system.state_names) for nm in names] for row in traj.X]
    )
    np.testing.assert_allclose(Theta @ Xi, traj.x_dot_exact, rtol=1e-9, atol=1e-9)


# --- per-system sanity / conservation checks -------------------------------------


def test_linear2d_is_damped():
    traj = datasets.simulate(datasets.get_system("linear2d"))
    assert np.linalg.norm(traj.X[-1]) < np.linalg.norm(traj.X[0])


def test_lotka_volterra_conserves_invariant():
    """LV has an exact conserved quantity V = d·x − c·ln x + b·y − a·ln y."""
    sys = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(sys)
    x, y = traj.X[:, 0], traj.X[:, 1]
    assert np.all(x > 0) and np.all(y > 0)
    a, b, c, d = (sys.params[k] for k in ("a", "b", "c", "d"))
    V = d * x - c * np.log(x) + b * y - a * np.log(y)
    assert np.std(V) / abs(np.mean(V)) < 1e-4


def test_lorenz_stays_on_attractor():
    """Bounded chaotic attractor: trajectory neither blows up nor collapses."""
    traj = datasets.simulate(datasets.get_system("lorenz"))
    assert np.all(np.isfinite(traj.X))
    assert traj.X[:, 2].max() < 100  # z stays in the attractor's range
    assert np.ptp(traj.X[:, 0]) > 10  # and it actually moves around


# --- noise (groundwork for M6) ---------------------------------------------------


def test_add_noise_is_reproducible_and_changes_data():
    X = datasets.simulate(datasets.get_system("linear2d")).X
    n1 = datasets.add_noise(X, level=0.05, seed=0)
    n2 = datasets.add_noise(X, level=0.05, seed=0)
    np.testing.assert_array_equal(n1, n2)
    assert not np.array_equal(n1, X)
    assert n1.shape == X.shape


def test_add_noise_zero_level_is_identity():
    X = datasets.simulate(datasets.get_system("linear2d")).X
    np.testing.assert_array_equal(datasets.add_noise(X, level=0.0, seed=0), X)
