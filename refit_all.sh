#!/usr/bin/env bash
# Refit everything at the current sampler settings, in dependency order.
#
#   ./refit_all.sh                 # dry run - prints the plan, touches nothing
#   ./refit_all.sh --apply
#
#   ARMS="skew skew-tight"  NAMES="C,BAC,JPM"  ANCHOR=hybrid  ./refit_all.sh --apply
#
# Order is not cosmetic. Idiosyncratic fits CONDITION on the systematic
# estimate for the same date, so the systematic arms must be refitted first or
# every name is fitted against the old numbers.
#
# Existing drawers are ARCHIVED, not overwritten. --force would be one flag
# shorter and would destroy the 10,000-draw set that every result quoted so far
# rests on; archiving keeps both readable and correctly labelled, which is the
# only way to check that cutting draws did not move anything that matters.
set -euo pipefail

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

ARMS="${ARMS:-skew skew-tight}"
NAMES="${NAMES:-C,BAC,JPM}"
ANCHOR="${ANCHOR:-hybrid}"
LABEL="${LABEL:-draws10000}"
WORKERS="${WORKERS:-$(python -c 'import os;print(os.cpu_count() or 4)')}"
export JGL_CORES="${JGL_CORES:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="refit_all_${STAMP}.log"

say() { echo "$@" | tee -a "$LOG"; }
run() {
    say "+ $*"
    [ "$APPLY" = "1" ] && { "$@" >>"$LOG" 2>&1 || { say "  FAILED: $*"; exit 1; }; }
    return 0
}

say "========================================================================"
say "refit all   arms='$ARMS'  names='$NAMES'  anchor=$ANCHOR"
say "            JGL_CORES=$JGL_CORES  workers=$WORKERS  draws=default (1000)"
say "            $([ "$APPLY" = 1 ] && echo 'APPLYING' || echo 'DRY RUN - nothing will be touched')"
say "========================================================================"

say ""
say "-- 1. what is on disk now --"
python poc/archive_run.py --list 2>&1 | tee -a "$LOG"

say ""
say "-- 2. archive the existing drawers as '$LABEL' --"
DRAWERS="$(python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from Library.DataAccess import PMLE_DIR
if os.path.isdir(PMLE_DIR):
    for d in sorted(os.listdir(PMLE_DIR)):
        if os.path.isdir(os.path.join(PMLE_DIR, d)) and "__" + os.environ.get("LABEL_SKIP","zzz") not in d:
            print(d)
PY
)"
if [ -z "$DRAWERS" ]; then
    say "   (no drawers on disk - nothing to archive)"
else
    for d in $DRAWERS; do
        run python poc/archive_run.py --drawer "$d" --label "$LABEL" --apply
    done
fi

say ""
say "-- 3. systematic, one run per arm (must precede the names) --"
for arm in $ARMS; do
    run python -u poc/estimate_systematic.py --priors "$arm" --step 21 \
        --workers "$WORKERS" --no-color
done

say ""
say "-- 4. full-sample calibration, per arm (the anchor for --anchor full/hybrid) --"
for arm in $ARMS; do
    run python -u poc/estimate_systematic.py --priors "$arm" --full-sample --no-color
done

say ""
say "-- 5. idiosyncratic --"
for arm in $ARMS; do
    run python -u poc/estimate_idiosyncratic.py --names "$NAMES" \
        --priors "$arm" --anchor "$ANCHOR" --workers "$WORKERS"
done

say ""
say "-- done --"
python poc/archive_run.py --list 2>&1 | tee -a "$LOG"
say ""
say "log: $LOG"
if [ "$APPLY" = "0" ]; then
    say ""
    say "That was a DRY RUN. Re-run with --apply to execute."
    say "Detach it, because the whole thing is 30-60 minutes:"
    say "    ./run_bg.sh /dev/null  # (or simply:)  setsid nohup ./refit_all.sh --apply &"
fi
