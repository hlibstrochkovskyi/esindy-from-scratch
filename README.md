# esindy

A from-scratch, test-first implementation of **SINDy** (Sparse Identification of
Nonlinear Dynamics) and its bootstrap-ensemble extension **E-SINDy**.

The whole project is organized around one fact: we generate data from systems whose
exact governing equations we already know, so every test can assert *"did we recover
the true equation?"* — crisp and deterministic. `esindy/datasets.py` is the source of
truth; tests diff against the known coefficient matrix `Ξ`, never against eyeballed
output.

See [`docs/plan/esindy_plan.md`](docs/plan/esindy_plan.md) for the full design and the
milestone breakdown.

## Setup

```bash
uv sync                 # core + dev dependencies
uv sync --extra oracle  # also install PySINDy for the oracle tests
```

## Tests

```bash
uv run pytest                    # core suite (no PySINDy needed)
uv run pytest -m "not slow"      # skip the statistical sweeps
uv run pytest -m oracle          # compare against PySINDy (needs --extra oracle)
```

## Quality

```bash
uv run ruff check .
uv run ruff format .
```

## Quick example

```python
from esindy import datasets, SINDy, STLSQ

system = datasets.get_system("lorenz")
traj = datasets.simulate(system)

model = SINDy(optimizer=STLSQ(threshold=0.5), input_names=system.state_names)
model.fit(traj.X, t=traj.t, x_dot=traj.x_dot_exact)
model.print()
# x' = -10.000 x + 10.000 y
# y' = 28.000 x - 1.000 y - 1.000 x z
# z' = -2.667 z + 1.000 x y
```

For noisy data, reach for `ESINDy` (bootstrap ensemble) or `WeakSINDy` (integral
formulation) — both share the same interface.

## Status — all milestones complete

| # | Milestone | Module |
|---|-----------|--------|
| M0 | Scaffolding & CI | tooling, `_seed.py` |
| M1 | Datasets + ground truth | `datasets.py` |
| M2 | Candidate library | `library.py` |
| M3 | Differentiation | `differentiation.py` |
| M4 | STLSQ optimizer | `optimizers.py` |
| M5 | SINDy end-to-end | `model.py`, `metrics.py` |
| M6 | Noise + breakdown baseline | `experiments.py` |
| M7 | E-SINDy (serial) | `ensemble.py` |
| M8 | Parallelization + benchmark | `ensemble.py`, [`docs/benchmarks.md`](docs/benchmarks.md) |
| M9 | Equation output + viz | `equations.py`, `viz.py` |
| M10 | Weak / integral formulation | `weak.py` |
| — | PySINDy oracle validation | `tests/test_oracle.py` |

Headline results, all asserted by the test suite:
- Vanilla SINDy recovers `linear2d`, `lotka_volterra`, and `lorenz` exactly on clean data.
- Vanilla SINDy loses exact support on Lorenz by ~5% noise (the breakdown baseline).
- E-SINDy beats vanilla on a rich library under noise (F1 1.00 vs 0.73 at 2% noise).
- The weak form beats pointwise SINDy at every noise level on Lorenz.
- Our STLSQ and SINDy match PySINDy to 1e-8.
