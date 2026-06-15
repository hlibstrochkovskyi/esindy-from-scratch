# Ensemble-SINDy from Scratch — Test-Driven Implementation Plan

**Goal:** Build a from-scratch implementation of Sparse Identification of Nonlinear
Dynamics (SINDy) and its bootstrap-ensemble extension (E-SINDy), developed
test-first, validated against known ground-truth systems and against PySINDy as a
reference oracle.

**Why this project, honestly.** SINDy is the right choice *for a TDD project*
because it has ground truth: you generate data from a system whose exact ODE you
already know, so every test asserts "did we recover the true equation?" — crisp and
deterministic. The catch the original pitch hid: PySINDy already implements all of
this, and the genuinely hard part is **estimating derivatives from noisy data**, not
the regression. This plan treats both facts as load-bearing rather than sweeping them
under the rug.

---

## 0. The math, stated precisely (so tests have something exact to check)

Given a state trajectory sampled at `m` time points with `n` state variables,
`X ∈ ℝ^{m×n}`, SINDy assumes the dynamics are sparse in some library of candidate
functions:

```
Ẋ = Θ(X) · Ξ
```

- `Θ(X) ∈ ℝ^{m×p}` — library matrix; each column is a candidate function (1, x, y,
  x², xy, y², sin x, …) evaluated at every sample.
- `Ξ ∈ ℝ^{p×n}` — coefficient matrix; column `k` gives the dynamics of state `k`.
  The whole premise is that `Ξ` is **sparse** (most candidate terms are zero).

Solved one state variable (one column) at a time via **STLSQ — Sequentially
Thresholded Least Squares** (note: this is the correct name; the source doc's
"Sparsified Sequential Thresholded Least Squares" is garbled):

```
1. ξ ← lstsq(Θ, Ẋ_k)               # ordinary least squares
2. small ← |ξ| < λ                  # threshold
3. ξ[small] ← 0                     # drop small terms
4. re-run lstsq on the surviving columns only
5. repeat 2–4 until the support (set of nonzero terms) stops changing
```

**E-SINDy** wraps STLSQ in bootstrapping:

- **Data bagging:** resample rows (time points) with replacement.
- **Library bagging:** randomly drop a subset of candidate columns each run.
- For each of `B` bootstraps, run STLSQ → get one `Ξ^{(b)}`.
- **Aggregate:**
  - *Inclusion probability* of each coefficient = fraction of bootstraps in which it
    is nonzero.
  - *Value* = median (robust) of the coefficient across bootstraps where included.
- **Threshold by inclusion probability** (e.g. keep terms included in > 60–90% of
  bootstraps). This second threshold is what gives E-SINDy its noise robustness.

The two facts that decide success or failure, and that the original pitch omitted:

1. **`Ẋ` is not given — it must be estimated from `X`.** Numerically
   differentiating noisy data amplifies noise badly. This is the #1 failure mode.
2. **Column scaling matters.** `x` and `x³` differ by orders of magnitude, so `Θ` is
   ill-conditioned and a single global `λ` is meaningless unless columns are
   normalized. PySINDy exposes `normalize_columns` for exactly this reason.

---

## 1. Architecture (modules, each independently testable)

```
esindy/
  datasets.py        # known systems → trajectories + true Ξ (the test oracle)
  library.py         # X → Θ, plus human-readable feature names
  differentiation.py # pluggable Ẋ estimators (finite-diff, Savitzky-Golay, spline)
  optimizers.py      # STLSQ (and plain LSQ / Lasso for comparison)
  model.py           # SINDy: fit / predict / simulate / equations (str + LaTeX)
  ensemble.py        # E-SINDy: bootstrap, aggregate, inclusion probs, parallelism
  metrics.py         # support recovery (P/R/F1), coeff error, simulation error
  viz.py             # plots / optional terminal or web dashboard
tests/
  ...                # mirrors the module layout
```

Design rule: **`datasets.py` is the source of truth.** Every system there ships with
its exact `Ξ` matrix, so tests never hand-eyeball results — they diff against the
known answer.

---

## 2. Test strategy (the core of what was asked)

Four layers, fast → slow, deterministic → statistical.

### 2.1 Unit tests — deterministic, millisecond-fast
- **Library:** for a tiny hand-built input (e.g. `X = [[2, 3]]`), assert each `Θ`
  column equals the hand-computed value (`1, 2, 3, 4, 6, 9, …`). Assert the polynomial
  feature **count** matches the combinatorial formula `C(n+d, d)` for `n` variables up
  to degree `d`. Assert feature *names* are correct (`"x0^2"`, `"x0 x1"`).
- **Differentiation:** feed an analytic function (`sin`) sampled finely; assert the
  estimated derivative matches the analytic one (`cos`) within tolerance on **clean**
  data. Then add noise and assert Savitzky-Golay beats naive finite-difference
  (an inequality test, not an absolute-error test).
- **STLSQ:** construct a *synthetic* well-conditioned `Θ` and a known sparse `ξ`, set
  `Ẋ = Θξ` **exactly** (no integration, no noise), assert STLSQ recovers `ξ` to
  machine precision. Plant a coefficient just **below** `λ` and assert it gets
  dropped; plant one just **above** and assert it survives.

### 2.2 Integration tests — components combined, clean data
- Full SINDy on **clean** Lotka-Volterra / Lorenz → assert **exact support recovery**
  (right terms nonzero, all others zero) and coefficients within a tight relative
  tolerance. This is the headline "it works" test.

### 2.3 Statistical / property tests — E-SINDy, seeded for reproducibility
- Inclusion probability of **true** terms → ~1.0; of **false** terms → low. Assert
  with margins, under a fixed RNG seed.
- **Noise sweep:** across increasing noise, assert E-SINDy's support-recovery F1 is
  **≥** vanilla SINDy's. This is the test that justifies the whole ensemble layer
  existing — if it ever fails, the project's premise is wrong and you want to know.
- Property-based (use `hypothesis`): for random sparse `ξ` and random well-conditioned
  `Θ`, clean recovery *always* succeeds.

### 2.4 Oracle tests — against PySINDy
- On identical data and hyperparameters, assert your STLSQ coefficients match
  PySINDy's `STLSQ` within tolerance. This converts "the library already exists" into
  a rigorous external check on your implementation. Keep these in a separate, optional
  test marker (`@pytest.mark.oracle`) so the core suite doesn't depend on PySINDy.

### 2.5 Regression tests
- Golden coefficients for a fixed seed + fixed data; assert byte-stable across
  refactors. Guards against silent behavior drift.

> **Subtle testing trap — chaotic systems.** For Lorenz you cannot test by comparing
> simulated trajectories point-by-point against truth: sensitive dependence on initial
> conditions makes them diverge even with a *perfect* model. Test Lorenz by **support
> + coefficient recovery**, or by comparing **short-horizon** trajectories, or by
> attractor statistics — never by long-horizon pointwise error. Bake this into the
> test design or you'll chase phantom "bugs."

---

## 3. Milestones (each: write the failing test first, then implement)

| # | Milestone | Done-when (test gate) |
|---|-----------|------------------------|
| **M0** | Scaffolding & CI | repo, venv, `pytest`, `ruff`, `coverage`, `pre-commit`, global seed control. CI runs green on a trivial test. |
| **M1** | Datasets + ground truth | `datasets.py` integrates known systems via `scipy.integrate.solve_ivp` and ships each system's exact `Ξ`. Tests: trajectory shapes, a conservation/sanity check, `Ξ` matches the symbolic form. |
| **M2** | Candidate library | Tests first (hand-computed `Θ`, `C(n+d,d)` count, names). Implement polynomial library, then trig/custom terms. |
| **M3** | Differentiation | **The hard milestone.** Tests first (analytic-derivative recovery on clean data; SavGol < finite-diff on noisy data). Implement finite-difference, Savitzky-Golay, spline. |
| **M4** | STLSQ optimizer | Tests first (exact synthetic recovery; threshold drop/keep; convergence; column-normalization effect). Implement. |
| **M5** | SINDy end-to-end | Tests: exact recovery on clean Lorenz / Lotka-Volterra; `predict`; `simulate` and compare on a short horizon. **Vanilla SINDy now works.** Optionally validate vs PySINDy here. |
| **M6** | Noise + breakdown baseline | Add noise to datasets; run a noise sweep; tests **document the noise level where vanilla SINDy breaks**. This is the motivation artifact for the ensemble. |
| **M7** | E-SINDy (serial first) | Bootstrap (data + library), aggregation, inclusion probabilities, probability threshold. Tests: true-term inclusion high / false low; **E-SINDy F1 ≥ SINDy F1 on the noise sweep.** Correctness before speed — serial only. |
| **M8** | Parallelization + benchmark | Add `joblib` (`loky` backend). **Set `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` in workers to avoid BLAS oversubscription.** Deliverable is an honest benchmark table: serial vs parallel vs vectorized across problem sizes, showing *where parallel helps and where it doesn't*. Tests: parallel results identical to serial under same seed; a perf-regression guard. |
| **M9** | Equation output + viz | Pretty-print + LaTeX of discovered equations. Tests: LaTeX matches expected for a known system; round-trip (parse printed equation back, compare to truth). Optional live dashboard. |
| **M10** | Stretch: weak/integral formulation | The genuinely advanced, noise-robust extension that avoids pointwise derivatives entirely (multiply by smooth test functions, integrate by parts). Test: recovers Lorenz at noise levels where pointwise SINDy fails. |

Critical path is **M2 → M3 → M4 → M5 → M7**. M8/M9/M10 are independent and can be
reordered or dropped without blocking a working tool.

---

## 4. Metrics — how "good" is defined (so success isn't a vibe)

- **Support recovery** — precision / recall / F1 over which terms are nonzero. This
  matters *more* than coefficient error: getting the right *form* of the equation is
  the whole point.
- **Coefficient relative error** on recovered terms.
- **Simulation / forecast error** — integrate the discovered model and compare
  (short-horizon for chaotic systems; see the trap above).
- **Ensemble calibration** — are inclusion probabilities meaningful?
- **Performance** — wall-time as a function of `B` (bootstraps) and thread count.

---

## 5. Risk register (the honest part)

1. **Derivative noise is the dominant failure mode.** Budget real time for M3; it is
   harder than the regression. If you skip it, everything downstream looks broken for
   the wrong reason.
2. **`λ` does not auto-tune.** The threshold is data-dependent with no universal
   value. Plan a `λ`-sweep and a sparsity-vs-error Pareto plot. Don't pretend it's
   automatic.
3. **Library choice is decisive.** If the true dynamics aren't expressible in your
   candidate basis, *no* method recovers them. Include a deliberate "basis miss" test
   that is **expected to fail recovery**, so you understand and document the limit
   rather than discover it as a surprise.
4. **Parallel speedup may be modest or negative for small problems.** A single STLSQ
   fit is sub-millisecond; ensemble overhead can dominate. M8 *measures* this instead
   of assuming the "peg all 12 threads" story from the source doc — which, on a
   6-core/12-thread Zen 2 chip, was overstated anyway (SMT adds ~10–30%, not 2×).
5. **Chaotic-system testing trap** (Section 2.5) — guard against it in test design.
6. **Reproducibility.** Thread *one* seed through `numpy`'s `Generator`, the
   integrator, and every bootstrap, or your statistical tests will flake and your
   golden tests will drift.

---

## 6. Stack

- Core: `numpy`, `scipy` (`solve_ivp`, `savgol_filter`, `lstsq`).
- Parallel: `joblib` (loky), with BLAS thread env vars pinned in workers.
- Tests: `pytest`, `pytest-cov`, `hypothesis`; `pysindy` behind an `oracle` marker.
- Quality: `ruff`, `pre-commit`.
- Viz (optional): `matplotlib`, or `rich`/`textual` for a terminal dashboard.

Everything here is CPU-only, low-memory, and pure linear algebra — so the hardware is
genuinely a non-issue, but for the unglamorous reason that the problem is small, not
because these methods are uniquely suited to a GPU-less laptop.
