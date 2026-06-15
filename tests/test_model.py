"""M5: vanilla SINDy, end-to-end. This is the headline "it works" milestone.

With clean data and exact derivatives, SINDy must recover the *exact support* and
coefficients of each known system. We then show it still works with *estimated*
derivatives on clean, finely-sampled non-chaotic systems, and that simulate/predict
behave. Chaotic Lorenz is checked by support + short-horizon trajectory, never by
long-horizon pointwise error (the trap in the plan).
"""

import numpy as np
import pytest

from esindy import datasets
from esindy.differentiation import SavitzkyGolay
from esindy.metrics import coefficient_error, support_scores, trajectory_error
from esindy.model import SINDy
from esindy.optimizers import STLSQ

# threshold chosen below each system's smallest true coefficient
THRESHOLDS = {"linear2d": 0.05, "lotka_volterra": 0.5, "lorenz": 0.5}


def _true_xi(system, model):
    """Ground-truth Ξ in the model's own feature ordering."""
    return system.coefficient_matrix(model.feature_names_)


@pytest.mark.parametrize("name", datasets.available_systems())
def test_exact_recovery_with_clean_derivatives(name):
    system = datasets.get_system(name)
    traj = datasets.simulate(system)
    model = SINDy(
        optimizer=STLSQ(threshold=THRESHOLDS[name]),
        input_names=system.state_names,
    )
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    Xi_true = _true_xi(system, model)
    scores = support_scores(Xi_true, model.coefficients_)
    assert scores.f1 == 1.0  # exact support recovery
    assert coefficient_error(Xi_true, model.coefficients_) < 1e-6


@pytest.mark.parametrize("name", ["linear2d", "lotka_volterra"])
def test_recovery_with_estimated_derivatives_clean(name):
    system = datasets.get_system(name)
    traj = datasets.simulate(system)
    model = SINDy(
        optimizer=STLSQ(threshold=THRESHOLDS[name]),
        differentiation=SavitzkyGolay(window_length=11, polyorder=4),
        input_names=system.state_names,
    )
    model.fit(traj.X, t=traj.t)

    Xi_true = _true_xi(system, model)
    scores = support_scores(Xi_true, model.coefficients_)
    assert scores.f1 == 1.0
    assert coefficient_error(Xi_true, model.coefficients_) < 0.05


def test_predict_matches_derivatives():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    np.testing.assert_allclose(model.predict(traj.X), traj.x_dot_exact, atol=1e-6)


def test_simulate_short_horizon_matches_truth():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    horizon = slice(0, 300)  # short horizon
    X_sim = model.simulate(traj.X[0], traj.t[horizon])
    assert trajectory_error(traj.X[horizon], X_sim) < 1e-3


def test_lorenz_short_horizon_only():
    """Chaotic: trust support + short horizon, never long-horizon pointwise error."""
    system = datasets.get_system("lorenz")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    X_sim = model.simulate(traj.X[0], traj.t[:200])
    assert trajectory_error(traj.X[:200], X_sim) < 1e-2


def test_equations_are_human_readable():
    system = datasets.get_system("linear2d")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.05), input_names=system.state_names)
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    eqs = model.equations(precision=3)
    assert len(eqs) == system.n_states
    # ẋ0 = -0.1 x0 + 2 x1 : both terms present, constant absent
    assert "x0" in eqs[0] and "x1" in eqs[0]
    assert "2.000 x1" in eqs[0]


def test_fit_requires_t_when_derivatives_absent():
    model = SINDy()
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, 2)))


def test_default_input_names_when_unspecified():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.5))
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    assert model.input_names_ == ("x0", "x1")
