"""Generate the figures embedded in the README.

    uv run --extra viz python scripts/make_figures.py

Everything here is reproducible from the library's own ground-truth systems — the plots
are the test results made visual, not hand-drawn illustrations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from esindy import datasets, viz  # noqa: E402
from esindy.differentiation import FiniteDifference  # noqa: E402
from esindy.ensemble import ESINDy  # noqa: E402
from esindy.experiments import noise_sweep  # noqa: E402
from esindy.library import PolynomialLibrary  # noqa: E402
from esindy.model import SINDy  # noqa: E402
from esindy.optimizers import STLSQ  # noqa: E402
from esindy.weak import WeakSINDy  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2]


def _f1s(points):
    return ([p.noise_level for p in points], [p.f1 for p in points])


def figure_noise_robustness():
    """Two panels: weak vs pointwise on Lorenz, and E-SINDy vs SINDy on a rich library."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    lorenz = datasets.get_system("lorenz")
    pointwise = noise_sweep(
        lambda: SINDy(
            optimizer=STLSQ(threshold=0.5),
            differentiation=FiniteDifference(),
            input_names=lorenz.state_names,
        ),
        lorenz,
        LEVELS,
        n_trials=5,
    )
    weak = noise_sweep(
        lambda: WeakSINDy(optimizer=STLSQ(threshold=0.5), input_names=lorenz.state_names),
        lorenz,
        LEVELS,
        n_trials=5,
    )
    viz.plot_f1_vs_noise(
        {"pointwise SINDy": _f1s(pointwise), "weak SINDy": _f1s(weak)},
        ax=ax1,
        title="Lorenz: weak form resists noise",
    )

    lv = datasets.get_system("lotka_volterra")
    sindy = noise_sweep(
        lambda: SINDy(
            library=PolynomialLibrary(degree=4),
            optimizer=STLSQ(threshold=0.5),
            differentiation=FiniteDifference(),
            input_names=lv.state_names,
        ),
        lv,
        LEVELS,
        n_trials=5,
    )
    esindy = noise_sweep(
        lambda: ESINDy(
            library=PolynomialLibrary(degree=4),
            optimizer=STLSQ(threshold=0.5),
            differentiation=FiniteDifference(),
            input_names=lv.state_names,
            n_models=60,
            seed=0,
        ),
        lv,
        LEVELS,
        n_trials=5,
    )
    viz.plot_f1_vs_noise(
        {"vanilla SINDy": _f1s(sindy), "E-SINDy": _f1s(esindy)},
        ax=ax2,
        title="Lotka–Volterra (degree-4 library): E-SINDy vs vanilla",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_robustness.png", dpi=120)
    plt.close(fig)


def figure_lorenz_attractor():
    """True attractor vs a model discovered from data, simulated on a short horizon."""
    system = datasets.get_system("lorenz")
    traj = datasets.simulate(system)
    model = SINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    horizon = slice(0, 1500)
    sim = model.simulate(traj.X[0], traj.t[horizon])

    fig = plt.figure(figsize=(9, 4))
    for i, (X, title) in enumerate([(traj.X[horizon], "true"), (sim, "discovered model")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], lw=0.5)
        ax.set_title(title)
    fig.suptitle("Lorenz attractor — discovered from data (short horizon)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lorenz_attractor.png", dpi=120)
    plt.close(fig)


def figure_inclusion_probabilities():
    """E-SINDy inclusion probabilities: true terms ~1, everything else ~0."""
    system = datasets.get_system("lotka_volterra")
    traj = datasets.simulate(system)
    model = ESINDy(
        optimizer=STLSQ(threshold=0.5), input_names=system.state_names, n_models=200, seed=0
    ).fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    fig = viz.plot_inclusion_probabilities(model, threshold=model.inclusion_threshold)
    fig.suptitle("E-SINDy inclusion probabilities (Lotka–Volterra, clean)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "inclusion_probabilities.png", dpi=120)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_noise_robustness()
    figure_lorenz_attractor()
    figure_inclusion_probabilities()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
