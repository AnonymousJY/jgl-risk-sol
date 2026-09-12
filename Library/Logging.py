"""
Library.Logging - shared logging setup for the reproduction scripts.

Every replication script imports :func:`setup_logging` and calls it at
module load time. The result is a consistent, timestamped log going to
stdout, so a replicator can see progress in real time and (optionally)
tee it to a file.

Usage::

    from Library.Logging import setup_logging
    logger = setup_logging(__name__)
    logger.info("Starting Table 2 pipeline")

Environment overrides::

    LIQUIDITY_LOG_LEVEL   INFO | DEBUG | WARNING | ERROR
    LIQUIDITY_LOG_FILE    optional path; if set, log lines are duplicated
                          to this file in append mode.
"""

import logging
import os
import sys
from multiprocessing import parent_process


_DEFAULT_LEVEL = "INFO"
_FMT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# The REPORT channel carries the aligned tables, and it must add nothing to
# the line. _FMT above prepends about 55 characters - timestamp, level and
# module name - which is right for an event ("216 fits to run") and fatal for
# a table: every column in this project is hand-aligned to a width of 66 to
# 100, and a prefix whose own width varies with the module name does not just
# shift the block, it makes the alignment meaningless.
_REPORT_FMT = "%(message)s"


def _level():
    name = os.environ.get("LIQUIDITY_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    return getattr(logging, name, logging.INFO)


def _log_file():
    """The log file, for the PARENT process only.

    estimate_systematic.py and estimate_idiosyncratic.py fan out over a
    forkserver pool. A file handler configured in the module body is created
    again in every worker, so six processes would hold six independent file
    objects open on the same path in append mode. Small O_APPEND writes do not
    usually tear on Linux, but "usually" is not a guarantee and it is a hazard
    print never had, because no file handler existed before this.

    Workers keep the stdout handler, which is the same inherited fd 1 they
    already wrote to with print - no change in exposure there. They emit
    exactly one line type between them (_warn_low_ess), so nothing of value is
    missing from the file.
    """
    path = os.environ.get("LIQUIDITY_LOG_FILE")
    return path if (path and parent_process() is None) else None


def _drop_inherited_file_handlers(lg):
    """Under fork, a child inherits the PARENT's configured logger object.

    The first-configure guard then does nothing - the flag is inherited set -
    so a file handler the parent opened is still attached, and the child
    writes through it. Verified: with JGL_MP_START=fork and a child logging
    under the parent's own logger name, six workers put six lines into the
    file through six inherited handlers.

    forkserver, the default here, does not have the problem: the child is a
    fresh interpreter, re-imports, and _log_file() returns None in it. This
    exists for the fork path and for anyone who sets JGL_MP_START.
    """
    if parent_process() is None:
        return
    for h in list(lg.handlers):
        if isinstance(h, logging.FileHandler):
            lg.removeHandler(h)
            try:
                h.close()
            except Exception:                                 # noqa: BLE001
                pass


def setup_logging(name=None):
    """Return a configured logger for EVENTS - timestamped and prefixed.

    Use this for things that happen: how many fits are queued, a pool
    recycling, a file being written, a diagnostic warning. Use report() below
    for anything whose column positions matter.
    """
    level = _level()

    root = logging.getLogger()
    # Configure root only once; subsequent calls just return the named logger.
    if not getattr(root, "_liquidity_configured", False):
        root.setLevel(level)
        # Remove any pre-existing handlers to avoid duplicate lines
        # (e.g., when re-imported in a Jupyter kernel).
        for h in list(root.handlers):
            root.removeHandler(h)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(stream_handler)

        log_file = _log_file()
        if log_file:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
            root.addHandler(file_handler)

        # Silence overly chatty libraries by default.
        for noisy in ("matplotlib", "PIL", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        root._liquidity_configured = True
    else:
        _drop_inherited_file_handlers(root)

    return logging.getLogger(name if name else "liquidity")


def report(name=None):
    """Return a logger for aligned REPORT output - no prefix of any kind.

    Same machinery as setup_logging - level from LIQUIDITY_LOG_LEVEL, tee to
    LIQUIDITY_LOG_FILE in the parent - but the formatter is bare, so a table
    row reaches the terminal exactly as print would have written it. That is
    the point: converting a report to logging should change what you can DO
    with the output (level it, filter it, tee it) and not one character of
    what it looks like.

    propagate is off so a record does not also travel to the root handler and
    appear a second time with a timestamp in front of it.
    """
    lg = logging.getLogger("report.%s" % (name or "liquidity"))
    if not getattr(lg, "_liquidity_report", False):
        lg.setLevel(_level())
        lg.propagate = False
        for h in list(lg.handlers):
            lg.removeHandler(h)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(_REPORT_FMT))
        lg.addHandler(stream_handler)

        log_file = _log_file()
        if log_file:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(logging.Formatter(_REPORT_FMT))
            lg.addHandler(file_handler)

        lg._liquidity_report = True
    else:
        _drop_inherited_file_handlers(lg)
    return lg
