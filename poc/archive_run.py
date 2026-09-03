"""Move a completed estimation run out of the untagged drawer.

Until now every rolling run wrote to Study/Estimated Parameters PMLE/^SPX/
regardless of which priors produced it, and run() skips any date already on
disk. So changing a prior and rerunning skipped every date and reprinted the
OLD posteriors under the NEW prior's header. estimate_systematic.py now stores
each prior set separately (^SPX__gaps, ^SPX__recentred, ...), which fixes it
going forward but leaves whatever is currently in ^SPX/ unlabelled.

This moves that run into its proper drawer so it is kept, correctly named, and
out of the way of the next one. The per-date CSVs are RENAMED as well as moved:
DataAccess builds each filename from the underlying id, so a directory rename
alone would leave the archive on disk but unreadable by --report-only.

    python poc/archive_run.py --tag gaps            # dry run, shows the plan
    python poc/archive_run.py --tag gaps --apply

Nothing is deleted and nothing is overwritten: if the destination already
exists the script refuses and tells you.
"""
import argparse
import os
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import PMLE_DIR                       # noqa: E402

# Mirrored from poc/estimate_systematic.py rather than imported. Importing it
# drags in pymc, and a script whose entire job is to rename four paths should
# run in any environment - including one where the estimation stack is broken,
# which is exactly when you most want to rescue a finished run. The assertion
# below catches drift whenever the real module happens to be importable.
SYSTEMATIC_ID = "^SPX"
FULL_SAMPLE_ID = "^SPX_FULLSAMPLE"
STORE_SUFFIX = {"paper": "", "recentred": "__recentred", "capped": "__capped",
                "capped-beta": "__cappedbeta", "gaps": "__gaps"}

try:                                                          # pragma: no cover
    from poc import estimate_systematic as _es
    assert _es.SYSTEMATIC_ID == SYSTEMATIC_ID
    assert _es.FULL_SAMPLE_ID == FULL_SAMPLE_ID
    assert _es.STORE_SUFFIX == STORE_SUFFIX, (
        "poc/archive_run.py is out of step with estimate_systematic.py")
except ImportError:
    pass


def plan(tag):
    """Every (src, dst) this archive would perform, skipping absent sources."""
    suffix = STORE_SUFFIX[tag]
    if not suffix:
        raise SystemExit("--tag paper is the untagged drawer itself; nothing "
                         "to move. Archive the run that is IN it under the "
                         "prior that actually produced it.")

    poc = os.path.join(_REPO_ROOT, "poc")
    pairs = [
        # the per-date CSVs: the expensive part, one file per valuation date
        (os.path.join(PMLE_DIR, SYSTEMATIC_ID),
         os.path.join(PMLE_DIR, SYSTEMATIC_ID + suffix)),
        # the full-sample fit, keyed on its own underlying id
        (os.path.join(PMLE_DIR, FULL_SAMPLE_ID),
         os.path.join(PMLE_DIR, FULL_SAMPLE_ID + suffix)),
        # the assembled series and the raw full-sample dump
        (os.path.join(poc, "systematic_params.csv"),
         os.path.join(poc, "systematic_params%s.csv" % suffix)),
        (os.path.join(poc, "full_sample_params.json"),
         os.path.join(poc, "full_sample_params%s.json" % suffix)),
    ]
    return [(s, d) for s, d in pairs if os.path.exists(s)]


def _retag_contents(src_dir, dst_dir):
    """Rename the CSVs inside a moved drawer so they can still be read.

    available_pmle_dates() and pmle_params_path() both build the filename from
    the underlying id: estimated_params_pmle_<id>_<YYYYMMDD>.csv. Move
    ^SPX/ to ^SPX__gaps/ without touching the files and every date inside
    becomes invisible - the archive survives as bytes but not as data, which is
    the worst of both outcomes. Rename them to match their new drawer.
    """
    if not os.path.isdir(dst_dir):
        return 0
    old_id = os.path.basename(src_dir)
    new_id = os.path.basename(dst_dir)
    old_prefix = "estimated_params_pmle_%s_" % old_id
    new_prefix = "estimated_params_pmle_%s_" % new_id
    n = 0
    for fn in os.listdir(dst_dir):
        if fn.startswith(old_prefix) and fn.endswith(".csv"):
            os.rename(os.path.join(dst_dir, fn),
                      os.path.join(dst_dir, new_prefix + fn[len(old_prefix):]))
            n += 1
    return n


def describe(path):
    if os.path.isdir(path):
        n = sum(len(f) for _, _, f in os.walk(path))
        return "directory, %d file(s)" % n
    return "%.1f KB" % (os.path.getsize(path) / 1024.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, choices=sorted(STORE_SUFFIX),
                    help="the priors that produced the run currently sitting "
                         "in the untagged drawer")
    ap.add_argument("--apply", action="store_true",
                    help="actually move. Without this the plan is printed and "
                         "nothing is touched.")
    a = ap.parse_args()

    moves = plan(a.tag)
    if not moves:
        print("Nothing to archive - the untagged drawer is already empty.")
        print("A rerun will estimate from scratch, which is what you want.")
        return

    print("=" * 72)
    print("Archive the current run as '%s'%s"
          % (a.tag, "" if a.apply else "   (DRY RUN)"))
    print("=" * 72)

    collisions = [(s, d) for s, d in moves if os.path.exists(d)]
    for src, dst in moves:
        mark = "BLOCKED" if os.path.exists(dst) else "move   "
        print("  %s  %s" % (mark, os.path.relpath(src, _REPO_ROOT)))
        print("           -> %s" % os.path.relpath(dst, _REPO_ROOT))
        print("           (%s)" % describe(src))

    if collisions:
        print("\n  %d destination(s) already exist. Refusing to overwrite an"
              % len(collisions))
        print("  earlier archive. Rename or remove them first, then rerun.")
        raise SystemExit(1)

    if not a.apply:
        print("\n  Dry run. Add --apply to move.")
        return

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        n = _retag_contents(src, dst)
        print("  moved %s%s"
              % (os.path.relpath(dst, _REPO_ROOT),
                 "  (%d file(s) retagged)" % n if n else ""))

    print("\nDone. The untagged drawer is now empty, so")
    print("  python poc/estimate_systematic.py --priors <name> --step 21")
    print("will estimate every date afresh instead of skipping them, and will")
    print("write to its own tagged drawer from here on.")


if __name__ == "__main__":
    main()
