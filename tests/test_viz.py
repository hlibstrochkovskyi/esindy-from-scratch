"""M9 (optional): smoke tests for plotting. Skipped unless the viz extra is installed."""

import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend for tests

from esindy import datasets, viz  # noqa: E402
from esindy.ensemble import ESINDy  # noqa: E402
from esindy.optimizers import STLSQ  # noqa: E402


def test_plot_trajectory_draws_one_line_per_state():
    system = datasets.get_system("linear2d")
    traj = datasets.simulate(system)
    fig = viz.plot_trajectory(traj.t, traj.X, system.state_names)
    assert len(fig.axes[0].get_lines()) == system.n_states


def test_plot_phase_portrait_3d_runs():
    traj = datasets.simulate(datasets.get_system("lorenz"))
    fig = viz.plot_phase_portrait(traj.X)
    assert fig is not None


def test_plot_f1_vs_noise_draws_a_line_per_method():
    curves = {
        "SINDy": ([0.0, 0.1, 0.2], [1.0, 0.8, 0.6]),
        "E-SINDy": ([0.0, 0.1, 0.2], [1.0, 0.95, 0.9]),
    }
    fig = viz.plot_f1_vs_noise(curves, title="demo")
    assert len(fig.axes[0].get_lines()) == 2


def test_plot_inclusion_probabilities_has_axis_per_state():
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = ESINDy(
        optimizer=STLSQ(threshold=0.5), input_names=system.state_names, n_models=20, seed=0
    ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    fig = viz.plot_inclusion_probabilities(model, threshold=0.6)
    assert len(fig.axes) == system.n_states
