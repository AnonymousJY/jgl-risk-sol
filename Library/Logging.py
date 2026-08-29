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


_DEFAULT_LEVEL = "INFO"
_FMT = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name=None):
    """Return a configured logger. Safe to call multiple times."""
    level_name = os.environ.get("LIQUIDITY_LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

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

        log_file = os.environ.get("LIQUIDITY_LOG_FILE")
        if log_file:
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
            root.addHandler(file_handler)

        # Silence overly chatty libraries by default.
        for noisy in ("matplotlib", "PIL", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        root._liquidity_configured = True

    return logging.getLogger(name if name else "liquidity")
