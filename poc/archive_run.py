"""Give a finished estimation run a readable name.

estimate_systematic.py now stores each run in a drawer named after the priors
that produced it, fingerprinted by their VALUES:

    Study/Estimated Parameters PMLE/^SPX__gaps_3f9a1c2b/

so editing a prior opens a new drawer and a stale run can no longer be silently
resumed. That makes archiving unnecessary for safety. It is still useful for
memory: a digest tells you two runs differ, not which was which.

This renames a drawer to something you will still recognise later, and moves
the loose poc/ artefacts that belong with it:

    python poc/archive_run.py --list
    python poc/archive_run.py --drawer '^SPX__gaps_3f9a1c2b' --label alpha0.037
    python poc/archive_run.py --drawer '^SPX__gaps_3f9a1c2b' --label alpha0.037 --apply

The per-date CSVs are RENAMED as well as moved: DataAccess builds each filename
from the underlying id, so a directory rename alone would leave the archive on
disk but unreadable by --report-only. Nothing is deleted, and an existing
destination is refused rather than overwritten.
"""
import argparse
import json
import os
import re
import shutil
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from Library.DataAccess import PMLE_DIR                       # noqa: E402

# Mirrored from poc/estimate_systematic.py rather than imported. Importing it
# drags in pymc, and a script whose entire job is to rename paths should run in
# any environment - including one where the estimation stack is broken, which
# is exactly when you most want to rescue a finished run.
SYSTEMATIC_ID = "^SPX"


def drawers():
    """Every estimate drawer on disk, with what is known about each."""
    if not os.path.isdir(PMLE_DIR):
        return []
    out = []
    for name in sorted(os.listdir(PMLE_DIR)):
        folder = os.path.join(PMLE_DIR, name)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        csvs = [f for f in files if f.endswith(".csv")]
        # Systematic runs drop _priors.json, idiosyncratic runs
        # _conditioning.json. Read whichever is there - a drawer you cannot
        # identify is a drawer you cannot resume, which is the whole reason
        # to look at this listing.
        manifest = None
        for fn in ("_priors.json", "_conditioning.json"):
            mpath = os.path.join(folder, fn)
            if not os.path.exists(mpath):
                continue
            try:
                with open(mpath) as fh:
                    manifest = json.load(fh)
            except (ValueError, OSError):
                manifest = {"tag": "(unreadable %s)" % fn}
            break
        out.append({"id": name, "n": len(csvs), "manifest": manifest})
    return out


def resume_command(m):
    """The command that continues an idiosyncratic drawer, from its manifest."""
    if not m or "anchor" not in m:
        return None
    return ("python poc/estimate_idiosyncratic.py --names %s --anchor %s "
            "--priors %s" % (m.get("name", "?"), m["anchor"],
                             m.get("systematic_priors", "?")))


def describe_priors(manifest):
    """One line naming the priors, so --list is readable without opening files."""
    if not manifest:
        return "no manifest - predates content tagging"
    if "anchor" in manifest:                      # an idiosyncratic drawer
        return ("idiosyncratic  name=%s  anchor=%s  systematic=%s (%s)"
                % (manifest.get("name", "?"), manifest["anchor"],
                   manifest.get("systematic_priors", "?"),
                   manifest.get("systematic_digest", "?")))
    spec = manifest.get("priors")
    if not isinstance(spec, dict):
        return "%s (%s)" % (manifest.get("tag", "?"), spec)
    bits = []
    for k in ("alpha_rv", "sigma", "pprob_rv", "lamb", "eta1", "eta2"):
        if k in spec:
            kind, kw = spec[k]
            args = ",".join("%g" % v for _, v in sorted(kw.items()))
            bits.append("%s=%s(%s)" % (k, kind, args))
    out = "%s  %s" % (manifest.get("tag", "?"), " ".join(bits))
    sm = manifest.get("sampler")
    if isinstance(sm, dict):
        out += "\n      sampler: draws %s  chains %s  cores %s  tune %s" % (
            sm.get("draws"), sm.get("chains"), sm.get("cores"), sm.get("tune"))
    return out


def do_list():
    found = drawers()
    if not found:
        print("No estimate drawers under %s" % PMLE_DIR)
        return
    print("=" * 72)
    print("Estimate drawers")
    print("=" * 72)
    for d in found:
        print("  %-34s %4d date(s)" % (d["id"], d["n"]))
        print("      %s" % describe_priors(d["manifest"]))
        cmd = resume_command(d["manifest"])
        if cmd:
            print("      resume: %s" % cmd)
    print("\n  Rename one with --drawer <id> --label <name>.")
    print("  A 'resume' line re-runs that drawer's remaining dates; the run")
    print("  skips what is already there, so it is safe to repeat.")


def _retag_contents(old_id, dst_dir):
    """Rename the CSVs inside a moved drawer so they can still be read.

    available_pmle_dates() and pmle_params_path() both build the filename from
    the underlying id: estimated_params_pmle_<id>_<YYYYMMDD>.csv. Rename the
    directory without touching the files and every date inside becomes
    invisible - the archive survives as bytes but not as data, which is the
    worst of both outcomes.
    """
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
        return "directory, %d file(s)" % len(os.listdir(path))
    return "%.1f KB" % (os.path.getsize(path) / 1024.0)


def plan(drawer, label):
    """Every (src, dst) this rename would perform, skipping absent sources."""
    src_dir = os.path.join(PMLE_DIR, drawer)
    if not os.path.isdir(src_dir):
        raise SystemExit("No such drawer: %s\n  Run --list to see what exists."
                         % src_dir)
    new_id = "%s__%s" % (drawer, label)

    # The loose poc/ artefacts carry the drawer's suffix. Strip the leading
    # underlying id to recover it, then move any file that matches.
    suffix = drawer[len(SYSTEMATIC_ID):] if drawer.startswith(SYSTEMATIC_ID) else ""
    poc = os.path.join(_REPO_ROOT, "poc")
    pairs = [(src_dir, os.path.join(PMLE_DIR, new_id))]
    for stem, ext in (("systematic_params", ".csv"),
                      ("full_sample_params", ".json")):
        src = os.path.join(poc, stem + suffix + ext)
        if os.path.exists(src):
            pairs.append((src, os.path.join(poc, stem + suffix + "__" + label + ext)))
    return [(s, d) for s, d in pairs if os.path.exists(s)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="show every drawer on disk and the priors behind it")
    ap.add_argument("--drawer",
                    help="the drawer to rename, as --list prints it")
    ap.add_argument("--label",
                    help="what to call it, e.g. alpha0.037 or pre-bugfix. "
                         "Letters, digits, dot, dash and underscore.")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename. Without this the plan is printed "
                         "and nothing is touched.")
    a = ap.parse_args()

    if a.list or not (a.drawer or a.label):
        do_list()
        return
    if not (a.drawer and a.label):
        raise SystemExit("--drawer and --label go together.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.label):
        raise SystemExit("--label must be letters, digits, dot, dash, "
                         "underscore. It becomes a path.")

    moves = plan(a.drawer, a.label)
    print("=" * 72)
    print("Rename '%s' -> '%s__%s'%s"
          % (a.drawer, a.drawer, a.label, "" if a.apply else "   (DRY RUN)"))
    print("=" * 72)

    collisions = [(s, d) for s, d in moves if os.path.exists(d)]
    for src, dst in moves:
        print("  %s  %s" % ("BLOCKED" if os.path.exists(dst) else "move   ",
                            os.path.relpath(src, _REPO_ROOT)))
        print("           -> %s" % os.path.relpath(dst, _REPO_ROOT))
        print("           (%s)" % describe(src))

    if collisions:
        print("\n  %d destination(s) already exist. Refusing to overwrite."
              % len(collisions))
        raise SystemExit(1)

    if not a.apply:
        print("\n  Dry run. Add --apply to rename.")
        return

    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        n = _retag_contents(a.drawer, dst) if os.path.isdir(dst) else 0
        print("  moved %s%s" % (os.path.relpath(dst, _REPO_ROOT),
                                "  (%d file(s) retagged)" % n if n else ""))
    print("\nDone. The original drawer name is free again, so a rerun under "
          "those\npriors will estimate from scratch rather than resume.")


if __name__ == "__main__":
    main()
