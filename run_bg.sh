#!/usr/bin/env bash
# Run any long estimation DETACHED, so it survives the terminal.
#
#   ./run_bg.sh poc/estimate_idiosyncratic.py --names COIN --priors skew
#   ./run_bg.sh poc/estimate_systematic.py --priors skew-tight --step 21
#
# Why this exists: a run started in a VS Code integrated terminal dies when
# that terminal is recreated - a window reload, an extension restart, the
# "History restored" message. It is not a hang and not an OOM; the process is
# simply gone, mid-fit, with no traceback. tmux solves it too (see
# run_daily.sh), but this needs nothing installed.
#
# setsid detaches from the controlling terminal; nohup ignores SIGHUP; the
# redirect means output survives even though nothing is watching.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: ./run_bg.sh <script.py> [args...]" >&2; exit 2; }

SCRIPT="$1"; shift
STEM="$(basename "${SCRIPT%.py}")"
LOG="${STEM}_$(date +%Y%m%d_%H%M%S).log"

setsid nohup python -u "$SCRIPT" "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
sleep 1
echo "started  pid $PID"
echo "log      $LOG"
echo
echo "  watch     tail -f $LOG"
echo "  alive?    ps -p $PID -o pid,etime,rss,cmd"
echo "  stop      kill $PID"
echo
echo "It is now detached: closing this terminal, reloading the window, or"
echo "logging out will NOT stop it. Every estimation script here is"
echo "resumable, so even a kill costs only the fits in flight."
