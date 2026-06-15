# E-SINDy parallelization benchmark

Reproduce with:

```bash
uv run python scripts/benchmark_ensemble.py
```

The bootstrap loop is embarrassingly parallel (each fit is independent), but a single
STLSQ fit is sub-millisecond, so process-pool startup and array pickling can easily cost
more than the work itself. The point of this benchmark is to find *where parallel
actually pays off* rather than assume it always does.

## Results

Measured on a 6-core / 12-thread Zen 2 laptop, BLAS pinned to one thread per worker
(`inner_max_num_threads=1`), best of 3 runs. Speedups are relative to `n_jobs=1`.

**Lotka–Volterra** (m = 1500 points, degree-4 library):

| n_models | n_jobs=1 | n_jobs=2 | n_jobs=4 |
|---------:|---------:|---------:|---------:|
| 20  |   32 ms | 1.12× | **0.77×** (slower) |
| 100 |  137 ms | 1.46× | 1.21× |
| 500 |  641 ms | 1.66× | **3.16×** |
| 2000| 1700 ms | 1.84× | 2.49× |

**Lorenz** (m = 4000 points, degree-4 library):

| n_models | n_jobs=1 | n_jobs=4 |
|---------:|---------:|---------:|
| 100  |  1338 ms | 2.79× |
| 500  |  6229 ms | 4.01× |
| 2000 | 24443 ms | 3.93× |

## Takeaways

- **Small ensembles lose.** At `n_models=20` the loky pool overhead dominates and
  `n_jobs=4` is *slower* than serial. Don't parallelize tiny jobs.
- **Crossover is around `n_models≈100`** for the small (LV) problem; beyond that,
  parallel wins and keeps improving.
- **Bigger per-fit work shifts the crossover left.** Lorenz fits are heavier
  (m = 4000), so parallel already pays off at `n_models=100` and reaches ~4×.
- **SMT is not a free 2×.** On this 6-core/12-thread chip, real speedups plateau near
  3–4× — consistent with physical-core scaling plus a modest SMT bump, not 12×. This is
  why `n_jobs` defaults to `1`: parallelism is opt-in, for the regimes where it helps.

Crucially, parallel results are **bit-identical** to serial (see
`tests/test_parallel.py`) because every bootstrap is seeded by its index, not by
execution order.
