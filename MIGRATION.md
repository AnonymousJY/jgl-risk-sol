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

### Required actions

- Pin the interval explicitly wherever the summary is computed. Do not rely on
  any library default, in either version.
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
