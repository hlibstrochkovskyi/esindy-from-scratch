"""M7: E-SINDy (serial).

The aggregation bookkeeping is pinned with a pure-function test (including the
library-bagging case where inclusion probability is over *availability*, not n_models).
Then: true terms get high inclusion probability and false terms low; results are
reproducible under a fixed seed; and the load-bearing premise holds — on a rich library
with noisy derivatives, E-SINDy's support F1 is >= vanilla SINDy's across a noise sweep.
"""

import numpy as np
import pytest

from esindy import datasets
from esindy.differentiation import FiniteDifference
from esindy.ensemble import ESINDy, aggregate_bootstraps
from esindy.experiments import noise_sweep
from esindy.library import PolynomialLibrary
from esindy.metrics import support_scores
from esindy.model import SINDy
from esindy.optimizers import STLSQ

# --- aggregation bookkeeping (fast, pure) -----------------------------------------


def test_aggregate_data_bagging_inclusion_and_median():
    # p=2, n=1, B=4: term 0 active in all four, term 1 active only once.
    samples = np.array([[[1.0], [0.0]], [[1.1], [0.0]], [[0.9], [5.0]], [[1.0], [0.0]]])
    coef, incl = aggregate_bootstraps(samples, inclusion_threshold=0.6)
    assert incl[0, 0] == 1.0
    assert incl[1, 0] == 0.25
    assert coef[0, 0] == pytest.approx(1.0)  # median([0.9,1.0,1.0,1.1])
    assert coef[1, 0] == 0.0  # inclusion below threshold -> dropped


def test_aggregate_library_bagging_uses_availability_not_n_models():
    nan = np.nan
    # term present in only 2 of 4 bootstraps (NaN = not a candidate), active in both.
    samples = np.array([[[1.0]], [[nan]], [[1.0]], [[nan]]])
    coef, incl = aggregate_bootstraps(samples, inclusion_threshold=0.6)
    assert incl[0, 0] == 1.0  # 2 active / 2 present, NOT 2/4
    assert coef[0, 0] == pytest.approx(1.0)


def test_aggregate_never_active_term_is_zero():
    samples = np.zeros((5, 3, 1))
    coef, incl = aggregate_bootstraps(samples)
    np.testing.assert_array_equal(coef, np.zeros((3, 1)))
    np.testing.assert_array_equal(incl, np.zeros((3, 1)))


# --- inclusion probabilities ------------------------------------------------------


def test_true_terms_high_inclusion_false_terms_low():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = ESINDy(
        optimizer=STLSQ(threshold=0.5), input_names=system.state_names, n_models=50, seed=0
    ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    true = system.coefficient_matrix(model.feature_names_) != 0
    assert model.inclusion_probabilities_[true].min() > 0.9
    assert model.inclusion_probabilities_[~true].max() < 0.5
    assert np.all((model.inclusion_probabilities_ >= 0) & (model.inclusion_probabilities_ <= 1))


# --- reproducibility (the property M8 will rely on) -------------------------------


def test_reproducible_under_same_seed():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)

    def fit():
        return ESINDy(
            optimizer=STLSQ(threshold=0.5),
            input_names=system.state_names,
            n_models=30,
            seed=7,
        ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    a, b = fit(), fit()
    np.testing.assert_array_equal(a.coefficients_, b.coefficients_)
    np.testing.assert_array_equal(a.inclusion_probabilities_, b.inclusion_probabilities_)
    np.testing.assert_array_equal(a.coef_samples_, b.coef_samples_)


# --- library bagging --------------------------------------------------------------


def test_library_bagging_still_recovers_on_clean_data():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = ESINDy(
        library=PolynomialLibrary(degree=3),
        optimizer=STLSQ(threshold=0.5),
        input_names=system.state_names,
        n_models=100,
        seed=0,
        library_ensemble=True,
        n_candidates_to_drop=3,
        inclusion_threshold=0.5,
    ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    Xi_true = system.coefficient_matrix(model.feature_names_)
    assert support_scores(Xi_true, model.coefficients_).f1 == 1.0


def test_dropping_all_candidates_raises():
    system = datasets.get_system("linear2d")
    traj = datasets.simulate(system)
    model = ESINDy(
        library=PolynomialLibrary(degree=1),  # p = 3
        input_names=system.state_names,
        library_ensemble=True,
        n_candidates_to_drop=3,
    )
    with pytest.raises(ValueError):
        model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)


def test_equations_render():
    system = datasets.get_system("linear2d")
    traj = datasets.simulate(system)
    model = ESINDy(
        optimizer=STLSQ(threshold=0.05), input_names=system.state_names, n_models=40, seed=0
    ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    eqs = model.equations()
    assert len(eqs) == 2 and "x1" in eqs[0]


# --- the justifying premise: E-SINDy F1 >= SINDy F1 under noise (slow) ------------


@pytest.mark.slow
def test_esindy_beats_sindy_on_noisy_rich_library():
    """Rich library + noisy finite-difference derivatives: single-shot SINDy overfits
    with false positives, while E-SINDy's inclusion threshold filters them out."""
    system = datasets.get_system("lotka_volterra")
    levels = [0.02, 0.05, 0.1]

    def mk_sindy():
        return SINDy(
            library=PolynomialLibrary(degree=4),
            optimizer=STLSQ(threshold=0.5),
            differentiation=FiniteDifference(),
            input_names=system.state_names,
        )

    def mk_esindy():
        return ESINDy(
            library=PolynomialLibrary(degree=4),
            optimizer=STLSQ(threshold=0.5),
            differentiation=FiniteDifference(),
            input_names=system.state_names,
            n_models=60,
            seed=0,
            inclusion_threshold=0.6,
        )

    sindy = noise_sweep(mk_sindy, system, levels, n_trials=5)
    esindy = noise_sweep(mk_esindy, system, levels, n_trials=5)

    for s_pt, e_pt in zip(sindy, esindy, strict=True):
        assert e_pt.f1 >= s_pt.f1  # ensemble never worse on support
    # and strictly better on average — the whole reason the ensemble layer exists
    assert np.mean([p.f1 for p in esindy]) > np.mean([p.f1 for p in sindy])
