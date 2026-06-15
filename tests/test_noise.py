"""M6: document where vanilla SINDy breaks under noise.

This is the motivation artifact for the ensemble. The slow tests fit real systems
across a noise sweep and assert the breakdown is real (Lorenz loses exact support by
~5% noise) and that coefficient error degrades monotonically. The fast tests cover the
harness logic without integrating any ODEs.
"""

import pytest

from esindy import datasets
from esindy.differentiation import SavitzkyGolay
from esindy.experiments import SweepPoint, breakdown_level, noise_sweep
from esindy.model import SINDy
from esindy.optimizers import STLSQ


def _sindy_factory(system, window=15):
    def make():
        return SINDy(
            optimizer=STLSQ(threshold=0.5),
            differentiation=SavitzkyGolay(window_length=window, polyorder=3),
            input_names=system.state_names,
        )

    return make


# --- harness logic (fast) ---------------------------------------------------------


def test_breakdown_level_finds_first_drop():
    pts = [
        SweepPoint(0.0, 1.0, 0.0, 0.0, 5),
        SweepPoint(0.05, 1.0, 0.0, 0.01, 5),
        SweepPoint(0.1, 0.8, 0.05, 0.2, 5),
    ]
    assert breakdown_level(pts) == 0.1


def test_breakdown_level_none_when_never_breaks():
    pts = [SweepPoint(0.0, 1.0, 0.0, 0.0, 5), SweepPoint(0.1, 1.0, 0.0, 0.1, 5)]
    assert breakdown_level(pts) is None


# --- breakdown documentation (slow) -----------------------------------------------


@pytest.mark.slow
def test_vanilla_sindy_breaks_under_noise_on_lorenz():
    system = datasets.get_system("lorenz")
    points = noise_sweep(_sindy_factory(system), system, [0.0, 0.01, 0.05], n_trials=5)
    by_level = {p.noise_level: p.f1 for p in points}

    assert by_level[0.0] == 1.0  # clean data: exact support recovery
    assert by_level[0.01] == 1.0  # still robust at 1% noise
    assert by_level[0.05] < 1.0  # documented breakdown around 5% noise
    assert breakdown_level(points) is not None


@pytest.mark.slow
def test_coefficient_error_grows_with_noise_on_lotka_volterra():
    """LV support is robust, but coefficient accuracy decays monotonically with noise —
    the quieter failure mode the ensemble should also improve."""
    system = datasets.get_system("lotka_volterra")
    points = noise_sweep(_sindy_factory(system), system, [0.0, 0.1, 0.4], n_trials=5)
    errs = [p.coefficient_error for p in points]

    assert errs[0] < errs[1] < errs[2]
    assert all(p.f1 == 1.0 for p in points)
