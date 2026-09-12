"""Pick a multiprocessing start method that exists on the running platform.

Both estimation drivers hard-coded

    multiprocessing.set_start_method(os.environ.get("JGL_MP_START",
                                                    "forkserver"), force=True)

which raises ValueError on Windows before anything else can happen:
get_all_start_methods() there is ["spawn"] and nothing else. macOS has all
three but defaults to spawn for the same thread-safety reason Linux is moving
to.

    Linux    ['fork', 'spawn', 'forkserver']
    macOS    ['spawn', 'fork', 'forkserver']
    Windows  ['spawn']

WHY forkserver IS STILL THE PREFERENCE WHERE IT EXISTS. fork() in a process
that has already started threads is unsafe - Python 3.12 warns and 3.14
changes the Linux default - and PyMC with a numba backend does start threads.
The failure mode is a hang indistinguishable from slow sampling, which this
project has already paid for once. forkserver forks from a clean helper that
never ran the samplers, so it has neither fork's thread problem nor spawn's
cost.

WHAT SPAWN COSTS, AND WHAT IT DOES NOT BREAK. A spawn child re-imports the
main module from scratch, so every worker pays the pymc/pytensor import - tens
of seconds - and pays it again each time POOL_CHUNK recycles the pool. Raise
JGL_POOL_CHUNK on Windows so that happens less often.

Correctness is a separate question and the answer is that the drivers are
already spawn-safe, though by accident rather than design:

  - main() is behind `if __name__ == "__main__"`, and a spawn child imports
    the script as "__mp_main__", so the body runs and main() does not.
  - Nothing the worker needs comes from a module global. PRIORS_IN_FORCE,
    STORE_ID, LOOKBACK and IDIO_PRIORS_IN_FORCE are all threaded through the
    task tuple. A comment in estimate_systematic.py used to say those globals
    existed "so the forkserver workers inherit" them; that was never how the
    values reached a worker, and under spawn a child inherits nothing at all,
    so acting on it would have broken the run.
  - The helper functions are module-level and importable by name, and the
    arguments are numpy arrays, dicts and strings - all picklable.

So the only thing that needed fixing was the method name itself.
"""

import multiprocessing
import os

from Library.Logging import setup_logging

_LOG = setup_logging(__name__)

_PREFERENCE = ("forkserver", "spawn", "fork")


def choose_start_method(preferred=None):
    """The method to use, without setting it. Never raises."""
    available = multiprocessing.get_all_start_methods()
    requested = os.environ.get("JGL_MP_START") or preferred

    if requested:
        if requested in available:
            return requested
        _LOG.warning(
            "JGL_MP_START=%r is not available on this platform (have %s); "
            "falling back.", requested, ", ".join(available))

    for method in _PREFERENCE:
        if method in available:
            return method
    return available[0]


def set_start_method(preferred=None):
    """Set it and return what was set. Safe to call at import time."""
    method = choose_start_method(preferred)
    try:
        multiprocessing.set_start_method(method, force=True)
    except (ValueError, RuntimeError) as exc:                    # noqa: BLE001
        # Nothing left to fall back to; run with whatever the platform default
        # is rather than refusing to start.
        _LOG.warning("could not set start method %r (%s); using the default %r",
                     method, exc, multiprocessing.get_start_method())
        return multiprocessing.get_start_method()
    if method == "fork":
        _LOG.warning(
            "using fork: this platform has neither forkserver nor spawn. "
            "fork() after threads have started is unsafe and PyMC starts "
            "threads; a hang here will look like slow sampling.")
    return method


def reap_forkserver():
    """Kill the forkserver helper, where there is one.

    os._exit skips atexit, which is the point - but multiprocessing's atexit
    handler is also what shuts the forkserver helper down, and an orphaned
    helper holds the terminal's pty open, so the shell never gets its prompt
    back even though the process is gone.

    A no-op under spawn and on Windows, where there is no helper and where
    signal.SIGKILL does not exist. The callers' bare `except Exception` used
    to swallow that AttributeError, which worked but only by accident.
    """
    if multiprocessing.get_start_method(allow_none=True) != "forkserver":
        return
    try:
        import signal
        from multiprocessing import forkserver
        pid = getattr(forkserver._forkserver, "_forkserver_pid", None)
        if pid:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    except Exception:                                            # noqa: BLE001
        pass                       # never let cleanup stop the exit
