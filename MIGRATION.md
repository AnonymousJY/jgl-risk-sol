# Environment migration: PyMC 5 → PyMC 6

Written 29 August 2026, against PyMC 6.3.1 (released 16 August 2026).

**Read §1 before upgrading.** There is a change that will alter your stored
parameter estimates without raising an error.

---

## 1. The silent breakage

`Library/RiskEngineKimYi2025.py` extracts credible intervals **by column
position**:

```python
params_sys_df = az.summary(idata_systematic, stat_focus="mean")
columns = params_sys_df.columns
column_mean      = columns[0]
column_ci_lower  = columns[2]
column_ci_upper  = columns[3]
```

PyMC 6.0 requires **ArviZ 1.0**, which changes two things at once:

1. `arviz.InferenceData` no longer exists — it is replaced by
   `xarray.DataTree`, and statistics move to `arviz-stats`, reached through an
   `.azstats` accessor. `az.summary(...)` as written will not survive.
2. **The default interval changed from a 94% highest-density interval to an 89%
   equal-tailed interval.**

Point 2 is the dangerous one. If the summary call is patched so that it *runs*
but the interval default is not pinned, every `_CI_LOWER` and `_CI_UPPER` in
`Study/Estimated Parameters PMLE/` silently changes meaning — from 94% HDI to
89% ETI. Nothing errors. The numbers just mean something different.

That matters here specifically because `poc/prior_diagnostics.py` converts
interval width to a standard deviation using an HDI-94 factor, and because the
whole identification finding rests on comparing posterior width to prior width.
An 89% ETI is narrower than a 94% HDI, so a naive upgrade would make every
parameter look *better* identified than it is.

**And the incremental design makes it worse.** `estimate_systematic.py` skips
dates that already have a CSV. Upgrade mid-run and you get a single parameter
series in which early dates are 94% HDI and later dates are 89% ETI, with
nothing marking the boundary.

### Resolved: the convention is now fixed in this repository

`Library/PosteriorSummary.py` defines the project-wide convention and computes
summaries from the posterior draws directly, so nothing depends on any ArviZ
default:

    95% EQUAL-TAILED interval - the 2.5% and 97.5% posterior quantiles
    sd = width / 3.919928        (2 * Phi^-1(0.975); NOT the 3.7616 HDI factor)
    stamp: CI_CONVENTION = "eti_0.95"

Equal-tailed rather than highest-density, because "2.5% and 97.5%" is quantile
language and the two differ for the skewed posteriors in this model.

`Library/RiskEngineKimYi2025.py` should be switched from positional `az.summary`
columns to `PosteriorSummary.summarize()`. That call also fixes a second bug:
the current code exponentiates the *summary* for alpha and pprob
(`np.exp(mean)`), which reports a geometric rather than an arithmetic mean and
shifts the interval. `summarize(..., transform=np.exp)` applies the transform to
the draws instead, which is correct.

### Required actions

- Route every summary through `Library.PosteriorSummary.summarize()`. Do not
  read `az.summary` columns positionally in either version.
- Replace position-based column access with named access.
- **Do not mix estimates across versions.** Either delete
  `Study/Estimated Parameters PMLE/` and re-run everything under the new stack,
  or record the pymc/arviz version and interval convention in each CSV.
- Re-run `poc/prior_diagnostics.py` after migrating and confirm the ratios are
  unchanged. If dETA1 stops looking prior-driven, suspect the interval
  convention before believing the result.

---

## 2. Other breaking changes in PyMC 6.0

- **PyTensor 3.0, default backend is now numba.** The custom likelihood
  (`_dist_loglike_systematic`, passed to `pm.CustomDist(..., logp=...)`) is the
  most likely thing to fail or slow down. If it does, revert the backend with
  `pytensor.config.linker = "cvm"` and treat numba compatibility as separate
  work.
- **Sampler tuning defaults differ** — PyMC NUTS 1000, nutpie 400. The code
  passes `tune=1000` explicitly, which is good; keep it explicit.
- Several names removed from the PyMC root namespace.
- `sample_vars` / `freeze_vars` added to control resampling; variables
  previously resampled automatically now warn instead.

## 3. Python version

PyMC 6.3.1 requires **Python ≥ 3.12** and supports 3.12, 3.13 and 3.14.

**This interacts with the multiprocessing setup.** Both
`Scripts/run_pmle_kimyi2025.py` and `poc/estimate_systematic.py` call:

```python
multiprocessing.set_start_method("fork", force=True)
```

`fork` in a process that has already started threads is unsafe, Python 3.12+
warns about it, and Python 3.14 changes the default start method on Linux to
`forkserver` for exactly this reason. PyMC with a numba backend does start
threads. Forking a threaded process is a classic source of hangs that look like
slow sampling.

Change it to `forkserver` (or `spawn`). The worker functions are module-level
and their arguments are plain arrays, so they should pickle without trouble —
but test with two dates before launching 78.

## 4. Suggested environment

```yaml
name: jgl-risk-sol
channels: [conda-forge]
dependencies:
  - python=3.12          # 3.13 also fine; 3.12 is the conservative choice
  - pymc>=6.3
  - nutpie
  - arviz>=1.2
  - numpy>=2
  - scipy
  - pandas
  - pyarrow
  - matplotlib
  - numba
  - pip
  - pip:
    - finance-datareader>=0.9.50
```

conda-forge is worth insisting on: PyMC recommends it because PyTensor's C and
BLAS toolchain builds cleanly there, and the pip-only path compiles from source.

## 5. A sequencing suggestion

You are mid-diagnosis on parameter identification. Upgrading the stack at the
same time means changing two things at once, and if the numbers move you will
not know which change moved them.

Consider establishing the rolling-252-day baseline on the **current** pinned
stack first, then migrating and confirming the new stack reproduces it. That
turns the upgrade into a verifiable step rather than a confound — and given
§1, reproduction is worth checking explicitly rather than assuming.

If the current environment is not practical to keep alive, migrate first, but
delete the existing parameter CSVs so nothing is mixed.


---

## 6. GPU

The machine has a CUDA GPU. It will not help the estimation run, and may hurt.

**Why not.** Each P-MLE fit has six parameters and 252 observations. That is far
too small for a GPU to pay back kernel-launch overhead - GPUs win on large
vectorised work, and there is none here. The parallelism this workload actually
has is *across* fits: 82 to 245 independent one-year windows. That is a CPU-core
scheduling problem, not a GPU problem.

`pm.sample` does accept JAX-backed samplers (`nuts_sampler="numpyro"` or
`"blackjax"`) which can run on GPU, and `RiskEngineKimYi2025.py` already exposes
that argument. Two cautions before trying it: the custom likelihood is written
in PyTensor and every op must transpile to JAX, which is likely but not
guaranteed; and even when it works, a six-parameter model is not where GPU wins.
Test on one date before assuming.

**What actually speeds this up.** `ProcessPoolExecutor()` with no `max_workers`
defaults to `os.cpu_count()`, and each worker then calls
`pm.sample(chains=4, cores=4)`. On a 16-core box that is 16 x 4 = 64 sampling
processes competing for 16 cores. `poc/estimate_systematic.py` now defaults to
`cpu_count // 4` workers and prints the arithmetic; `--workers` overrides it.
`Scripts/run_pmle_kimyi2025.py` still has the unbounded pool and would benefit
from the same change.

**Where the GPU is genuinely worth using: the reconstruction, later.**
`poc/backfill_poc.py` draws roughly 40 names x 7 windows x 200 bootstrap draws x
100 paths x ~750 days - on the order of four billion normal variates plus
arithmetic. That is exactly the vectorised workload CuPy exists for, and
`environment.yml` has a commented `cupy` line ready.

**This is now wired.** `Library/ArrayBackend.py` selects CuPy when it is
importable and a device is visible, and `reconstruct()` in
`poc/backfill_poc.py` runs on whichever backend is active. It also announces
the device at the start of a run.

`reconstruct()` was rewritten to generate the entire (draws, paths, days) block
in one call rather than looping over draws - about 15M variates per call. That
is what makes a GPU worth using, and it is faster on CPU too. Verified against
the previous per-draw loop: max absolute difference 2.8e-17.

Two things to know:

- **Seeds are not portable across backends.** NumPy and CuPy do not share an RNG
  stream, so the same seed gives different draws on CPU and GPU. Results are
  statistically equivalent, not bit-identical. Within one backend a seed
  reproduces. Any table or figure in a validation pack must record which device
  produced it - `ArrayBackend.BACKEND` exists for that. Set `JGL_FORCE_CPU=1` to
  check a GPU result against CPU.
- `Library/ImportLibs.py` is untouched and still rebinds `np = cp` inside its own
  namespace only, which is why `ArrayBackend` exists separately. While there:
  `ImportLibs.py` sets `cpx = sc`, assigning scipy to the cupyx name, which looks
  like a typo.
