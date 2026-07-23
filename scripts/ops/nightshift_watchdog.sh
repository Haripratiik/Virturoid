#!/bin/sh
# Night-shift watchdog (plan v2 §5.5, belt-and-braces). Idempotent relauncher: flock prevents double-start; if
# the night runner is already up it no-ops. Add to cron on the WSL2 box:  */5 * * * * /home/YOUR_USER/virturoid/scripts/ops/nightshift_watchdog.sh
# Use ONLY if the systemd unit proves flaky; safe to run alongside it (lock + the JSONL journal are idempotent).
set -eu
REPO=/home/YOUR_USER/virturoid
PY=/home/YOUR_USER/rl/bin/python
LOCK=/tmp/nightshift.lock
LOG="$REPO/runs/night_watchdog.log"

# already running? (bracket trick avoids matching the grep itself; do not leak full args)
if pgrep -f "[n]ight_runner" >/dev/null 2>&1; then
  exit 0
fi

# disk guard: refuse to start a new run below ~10 GB free (avoid a half-written checkpoint filling the VHD)
FREE_KB=$(df -Pk "$REPO" | awk 'NR==2{print $4}')
if [ "${FREE_KB:-0}" -lt 10485760 ]; then
  echo "$(date -u +%FT%TZ) watchdog: low disk (${FREE_KB}KB), not starting" >> "$LOG"
  exit 0
fi

cd "$REPO"
export PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=.85 XLA_FLAGS=--xla_gpu_autotune_level=0
# flock -n: acquire-or-skip; --resume continues from the journal (completed candidates are skipped, crash-safe)
flock -n "$LOCK" "$PY" -m virturoid.night_runner --resume --journal runs/night.jsonl >> "$LOG" 2>&1 || true
