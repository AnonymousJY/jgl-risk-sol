#!/usr/bin/env bash
# Full daily systematic estimation, 2007-01-01 -> 2026-08-31.
#
#   tmux new -s pmle
#   ./run_daily.sh
#   Ctrl-B then D to detach; `tmux attach -t pmle` to come back
#
# ~5,131 fits, roughly 11 hours at 6 workers. Fully resumable: dates already
# written to Study/Estimated Parameters PMLE/ are skipped, so an interruption
# costs only the fit in flight. Re-running after a crash is safe.
set -euo pipefail
WORKERS="${1:-6}"
LOG="pmle_daily_$(date +%Y%m%d_%H%M).log"
echo "workers=$WORKERS  log=$LOG"
python -u poc/estimate_systematic.py \
    --beg 20070101 --end 20260831 --step 1 \
    --workers "$WORKERS" --no-color 2>&1 | tee "$LOG"
