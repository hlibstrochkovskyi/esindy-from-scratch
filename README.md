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

## Status

Under active development.
- **M0 — scaffolding & CI** ✅
- **M1 — datasets + ground truth** ✅ (`linear2d`, `lotka_volterra`, `lorenz`)
- **M2 — candidate library** ✅ (polynomial + custom/trig + concat)
- **M3 — differentiation** ← next (the hard milestone)
