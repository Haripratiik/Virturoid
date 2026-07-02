# Night-shift unattended-ops stack (breakthrough plan v2 §5.5 / gaps G8, N11)

Deployment templates for running the Dreamer-mode night shift unattended on the one WSL2 Ubuntu GPU box.
These are **reference artifacts** (they run on the box, not in CI); the mechanisms are from research stream 4.
Ranked; use the simplest that holds.

## 1. systemd SYSTEM unit (primary supervisor)
systemd works in WSL ≥ 0.67.6 and is default-on in current Ubuntu (`wsl --version`; `wsl --update` if absent;
enable via `/etc/wsl.conf` → `[boot]\nsystemd=true`). Use a **system** unit with `User=` (avoids user-unit
linger flakiness in WSL). Install `nightshift.service` to `/etc/systemd/system/`, then:
```
sudo systemctl daemon-reload && sudo systemctl enable --now nightshift
journalctl -fu nightshift          # follow logs
```
`Restart=on-failure` + `RestartSec=30` + `StartLimitBurst` restart a crashed run; `MemoryMax` caps the trainer.

## 2. The WSL keep-alive trap (MANDATORY — Microsoft docs)
**systemd services do NOT keep a WSL instance alive** — the instance dies shortly after the last client session
ends, so the unit alone won't survive an idle host. Pin the instance with a **Windows Task Scheduler** job at
startup (run whether logged on or not, highest privileges):
```
wsl.exe -d Ubuntu --exec /bin/sh -c "sleep infinity"
```
This permanent session keeps the VM up so systemd keeps sshd/nightshift running. Also set **Windows Update
active hours** / pause — a host reboot is the #1 unattended killer; with checkpoint-resume it becomes a resume,
not a dead night.

## 3. cron + flock watchdog (belt-and-braces, optional)
If systemd proves flaky, add `nightshift_watchdog.sh` to cron (`*/5 * * * *`): an idempotent relauncher that
can't double-start (flock) and no-ops if the run is already up. Safe to run alongside systemd (lock + journal
idempotency). `tmux` is for a debugging window only — NOT supervision (no restart semantics).

## 4. JAX OOM + disk guards
- One trainer at a time (the JSONL journal serializes candidates). `export XLA_PYTHON_CLIENT_MEM_FRACTION=.85`.
- NEVER `XLA_PYTHON_CLIENT_PREALLOCATE=false` for long runs (fragmentation). `ALLOCATOR=platform` only for OOM
  debugging.
- Run each candidate's training as a **child subprocess** (crash-only): an XLA OOM aborts the child, the loop
  journals `failed:oom` and moves on.
- In `~/.wslconfig`: pin `memory=` (leave Windows headroom; WSL grabs 50% RAM by default), `sparseVhd=true`
  (stop VHD bloat). Pre-candidate disk check: refuse below ~10 GB (`shutil.disk_usage`); keep-last-k checkpoints.

## 5. Resume / reconcile
The night loop's JSONL journal already skips completed candidate ids on restart, so a crash mid-candidate simply
re-runs that candidate (crash-only design). The TDR watchdog killing long GPU kernels (observed ~iter 20) is why
the trainer checkpoints every 10 iters (`--keep-checkpoints`) — a killed run still leaves a fetchable,
deploy-sim-selectable policy.
