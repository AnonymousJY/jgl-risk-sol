"""
ArrayBackend.py — explicit CPU/GPU array module selection.

Use this instead of the global rebinding in ImportLibs.py. That module does
`np = cp` inside its own namespace, which silently affects only code that
imports numpy *through* it and leaves every other module on the CPU. Here the
choice is explicit at the call site.

    from Library.ArrayBackend import xp, asnumpy, BACKEND, rng_for

    z = rng_for(seed).standard_normal((d, p, t))    # on whichever device
    result = asnumpy(z.mean())                      # back to the host

Set JGL_FORCE_CPU=1 to disable the GPU without uninstalling CuPy - useful when
checking that a GPU result matches the CPU one.

REPRODUCIBILITY WARNING
-----------------------
NumPy and CuPy do not share a random number stream. The same seed gives
DIFFERENT draws on the two backends. Results are statistically equivalent, not
bit-identical. Within one backend a seed is reproducible.

Any figure or table in a validation pack must therefore record which backend
produced it. BACKEND is exported for exactly that purpose.
"""

import os

import numpy as _np

BACKEND = "numpy"
xp = _np

if not os.environ.get("JGL_FORCE_CPU"):
    try:
        # Optional dependency: absent on CPU-only machines and on macOS, where
        # conda-forge has no build. Type checkers flag this as unresolved -
        # that is expected, hence the ignore. The except branch is the CPU path.
        import cupy as _cp  # type: ignore[import-not-found]

        if _cp.cuda.is_available():
            xp = _cp
            BACKEND = "cupy"
    except Exception:                                            # noqa: BLE001
        pass


def is_gpu():
    return BACKEND == "cupy"


def asnumpy(a):
    """Bring an array back to the host, whatever it currently is."""
    if BACKEND == "cupy":
        import cupy as _c  # type: ignore[import-not-found]
        if isinstance(a, _c.ndarray):
            return _c.asnumpy(a)
    return _np.asarray(a)


def rng_for(seed):
    """A generator on the active backend.

    CuPy's Generator API mirrors NumPy's for the methods used here
    (standard_normal, normal), but the streams differ - see the warning above.
    """
    if BACKEND == "cupy":
        import cupy as _c  # type: ignore[import-not-found]
        return _c.random.default_rng(seed)
    return _np.random.default_rng(seed)


def describe():
    if BACKEND != "cupy":
        return "CPU (numpy)"
    try:
        import cupy as _c  # type: ignore[import-not-found]
        d = _c.cuda.runtime.getDeviceProperties(_c.cuda.runtime.getDevice())
        name = d["name"].decode() if isinstance(d["name"], bytes) else str(d["name"])
        free, total = _c.cuda.runtime.memGetInfo()
        return "GPU (cupy) - %s, %.1f/%.1f GB free" % (
            name, free / 1e9, total / 1e9)
    except Exception:                                            # noqa: BLE001
        return "GPU (cupy)"
