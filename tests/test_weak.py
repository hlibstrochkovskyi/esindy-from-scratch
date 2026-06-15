"""M10: weak/integral formulation.

The point of the weak form is noise robustness without pointwise derivatives. So the
load-bearing test recovers Lorenz at a noise level where pointwise SINDy has already
failed. The quadrature and test-function machinery is also unit-tested directly.
"""

import numpy as np
import pytest

from esindy import datasets
from esindy.differentiation import FiniteDifference
from esindy.experiments import noise_sweep
from esindy.metrics import coefficient_error, support_scores
from esindy.model import SINDy
from esindy.optimizers import STLSQ
from esindy.weak import WeakSINDy, _trapezoid_weights, build_test_functions

# --- quadrature & test functions (fast, pure) -------------------------------------


def test_trapezoid_weights_integrate_polynomials():
    t = np.linspace(0.0, 2.0, 201)
    w = _trapezoid_weights(t)
    assert w @ np.ones_like(t) == pytest.approx(2.0)  # ∫1 dt
    assert w @ t == pytest.approx(2.0)  # ∫t dt over [0,2]
    assert w @ t**2 == pytest.approx(8 / 3, abs=1e-3)  # ∫t² dt


def test_test_functions_vanish_at_endpoints():
    t = np.linspace(0.0, 10.0, 600)
    Phi, dPhi = build_test_functions(t, n_test_functions=20, support_fraction=0.05, p=4)
    assert np.allclose(Phi[0], 0.0)
    assert np.allclose(Phi[-1], 0.0)
    assert Phi.shape == dPhi.shape == (t.size, 20)


def test_test_function_derivative_matches_numerical():
    t = np.linspace(0.0, 10.0, 4000)
    Phi, dPhi = build_test_functions(t, n_test_functions=5, support_fraction=0.1, p=4)
    k = 2
    numerical = np.gradient(Phi[:, k], t)
    assert np.max(np.abs(numerical - dPhi[:, k])) < 1e-2


# --- clean recovery ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["lotka_volterra", "lorenz"])
def test_weak_recovers_clean_systems(name):
    system = datasets.get_system(name)
    traj = datasets.simulate(system)
    model = WeakSINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
    model.fit(traj.X, traj.t)

    Xi_true = system.coefficient_matrix(model.feature_names_)
    assert support_scores(Xi_true, model.coefficients_).f1 == 1.0
    assert coefficient_error(Xi_true, model.coefficients_) < 0.05


def test_fit_rejects_mismatched_times():
    model = WeakSINDy()
    with pytest.raises(ValueError):
        model.fit(np.zeros((100, 2)), np.linspace(0, 1, 50))


# --- the payoff: weak beats pointwise under noise (slow) --------------------------


@pytest.mark.slow
def test_weak_beats_pointwise_sindy_under_noise():
    system = datasets.get_system("lorenz")
    levels = [0.1, 0.2]

    def mk_pointwise():
        return SINDy(
            optimizer=STLSQ(threshold=0.5),
            differentiation=FiniteDifference(),
            input_names=system.state_names,
        )

    def mk_weak():
        return WeakSINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)

    pointwise = noise_sweep(mk_pointwise, system, levels, n_trials=4)
    weak = noise_sweep(mk_weak, system, levels, n_trials=4)

    for p_pt, w_pt in zip(pointwise, weak, strict=True):
        assert w_pt.f1 >= p_pt.f1
    # clearly better at 10% noise, where pointwise has already degraded
    assert weak[0].f1 > pointwise[0].f1
