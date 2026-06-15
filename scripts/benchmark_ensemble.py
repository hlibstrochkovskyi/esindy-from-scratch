"""Honest serial-vs-parallel benchmark for E-SINDy.

Run with:  uv run python scripts/benchmark_ensemble.py

The point is *not* to show parallel always wins — it doesn't. A single STLSQ fit is
sub-millisecond, so for small ensembles the loky process pool's startup and pickling
overhead dominates and parallel is slower. This script measures where the crossover
actually is, rather than assuming the "peg all the threads" story.
"""

from __future__ import annotations

import time

from esindy import datasets
from esindy.ensemble import ESINDy
from esindy.library import PolynomialLibrary
from esindy.optimizers import STLSQ


def time_fit(system, traj, n_models: int, n_jobs: int, degree: int) -> float:
    model = ESINDy(
        library=PolynomialLibrary(degree=degree),
        optimizer=STLSQ(threshold=0.5),
        input_names=system.state_names,
        n_models=n_models,
        n_jobs=n_jobs,
        seed=0,
    )
    start = time.perf_counter()
    model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
    return time.perf_counter() - start


def run(system_name: str, n_models_list, n_jobs_list, degree: int = 4, repeats: int = 3):
    system = datasets.get_system(system_name)
    traj = datasets.simulate(system)
    print(f"\nsystem={system_name}  m={traj.X.shape[0]}  degree={degree}  (best of {repeats})")
    header = "n_models | " + " | ".join(f"n_jobs={j:>2}" for j in n_jobs_list)
    print(header)
    print("-" * len(header))
    for n_models in n_models_list:
        cells = []
        baseline = None
        for n_jobs in n_jobs_list:
            best = min(time_fit(system, traj, n_models, n_jobs, degree) for _ in range(repeats))
            if baseline is None:
                baseline = best
            speedup = baseline / best
            cells.append(f"{best * 1e3:6.1f}ms ({speedup:4.2f}x)")
        print(f"{n_models:>8} | " + " | ".join(cells))


if __name__ == "__main__":
    run("lotka_volterra", n_models_list=[20, 100, 500, 2000], n_jobs_list=[1, 2, 4])
    run("lorenz", n_models_list=[100, 500, 2000], n_jobs_list=[1, 4])
