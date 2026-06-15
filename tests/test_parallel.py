"""M8: parallel E-SINDy must be bit-identical to serial.

This is the payoff of seeding every bootstrap by its index (M7): execution order can
change freely, results cannot. Without it, the benchmark in scripts/ would be comparing
two different computations.
"""

import numpy as np
import pytest

from esindy import datasets
from esindy.ensemble import ESINDy
from esindy.optimizers import STLSQ


@pytest.mark.slow
def test_parallel_matches_serial_bitwise():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)

    def fit(n_jobs):
        return ESINDy(
            optimizer=STLSQ(threshold=0.5),
            input_names=system.state_names,
            n_models=40,
            seed=3,
            n_jobs=n_jobs,
        ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)

    serial = fit(1)
    parallel = fit(2)

    np.testing.assert_array_equal(serial.coef_samples_, parallel.coef_samples_)
    np.testing.assert_array_equal(serial.coefficients_, parallel.coefficients_)
    np.testing.assert_array_equal(
        serial.inclusion_probabilities_, parallel.inclusion_probabilities_
    )
