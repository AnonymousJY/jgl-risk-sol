#!/usr/bin/env bash
# Daily (--step 1) backfill of the P-MLE drawers, one YEAR per estimator call.
#
#   ./run_daily_backfill.sh                          # systematic, 2007-2026
#   ARM=skew-tight BEG_Y=2007 END_Y=2012 ./run_daily_backfill.sh
#   WITH_NAMES=C,BAC,JPM ./run_daily_backfill.sh     # names too, same year
#
# Detach it - this is hours, not minutes:
#
#   setsid nohup ./run_daily_backfill.sh > backfill.out 2>&1 < /dev/null &
#   tail -f backfill.out
#
# WHY A YEAR AT A TIME. estimate_systematic.py skips any date already on disk,
# so the whole job is resumable at any granularity - but one 5,131-date call
# gives no progress signal for hours and no place to stop. A year is ~260 fits,
# which is a real checkpoint, an ETA after the first one, and a clean point to
# kill it. The cost of splitting is one process start per year.
#
# ORDER. Idiosyncratic fits with --anchor rolling condition on the systematic
# estimate for the SAME date and silently skip dates that have none, so the
# systematic year must finish before the names for that year start. That is why
# WITH_NAMES runs inside the year loop rather than as a second pass.
#
# The drawer is MID-FILL until this finishes. Do not take vintage means from a
# partially filled drawer: the filled years carry ~260 dates each and the rest
# carry ~12, so any average over the whole window is weighted towards whatever
# happened to be done first. Wait for the run to end, or read the monthly
# __draws10000 drawers, which this does not touch.
set -euo pipefail

ARM="${ARM:-skew-tight}"
BEG_Y="${BEG_Y:-2007}"
END_Y="${END_Y:-2026}"
LAST_DAY="${LAST_DAY:-20260831}"          # study window end, from StudyWindow.py
WITH_NAMES="${WITH_NAMES:-}"
ANCHOR="${ANCHOR:-rolling}"
WORKERS="${WORKERS:-$(python -c 'import os; print(os.cpu_count() or 4)')}"
export JGL_CORES="${JGL_CORES:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="backfill_${ARM}_${STAMP}.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "daily backfill  arm=$ARM  years $BEG_Y-$END_Y  workers=$WORKERS"
say "names: ${WITH_NAMES:-none}  anchor=$ANCHOR"
say "log: $LOG"

total=0
done_years=0
for Y in $(seq "$BEG_Y" "$END_Y"); do
    b="${Y}0101"
    e="${Y}1231"
    [ "$b" \> "$LAST_DAY" ] && { say "year $Y is past $LAST_DAY - stopping"; break; }
    [ "$e" \> "$LAST_DAY" ] && e="$LAST_DAY"

    t0=$(date +%s)
    say "year $Y  $b -> $e  systematic"
    python -u poc/estimate_systematic.py --priors "$ARM" --step 1 \
        --beg "$b" --end "$e" --workers "$WORKERS" --no-color \
        >>"$LOG" 2>&1 < /dev/null

    if [ -n "$WITH_NAMES" ]; then
        say "year $Y  $b -> $e  names $WITH_NAMES"
        python -u poc/estimate_idiosyncratic.py --names "$WITH_NAMES" \
            --priors "$ARM" --anchor "$ANCHOR" --step 1 \
            --beg "$b" --end "$e" --workers "$WORKERS" \
            >>"$LOG" 2>&1 < /dev/null
    fi

    t1=$(date +%s)
    took=$((t1 - t0))
    total=$((total + took))
    done_years=$((done_years + 1))
    left=$((END_Y - Y))
    say "year $Y done in ${took}s   avg $((total / done_years))s/yr   ETA $(( left * total / done_years / 60 )) min for $left more"
done

say "backfill complete"
