"""GPU training access for the app: train a composed body on the remote GPU box (Tailscale + MJX PPO),
so learn-on-request can use real GPU training (far stronger gaits than the on-device ES) when the box is
reachable — with a clean CPU fallback when it isn't.

The box is configured via env (``VIRTUROID_GPU_SSH``, e.g. ``tailscale ssh user@host``), python at ``~/rl/bin/python`` (jax+mjx CUDA), repo at
``~/virturoid``. We ship the current src+scripts (isolated tar) + the gene JSON, launch
``mjx_morph_attention.py --gene-json`` detached, poll its log for progress, and fetch the saved policy
npz (same format the CPU MorphPolicy loads, so it banks into the SAME flywheel). Relates to
[[morph-policy-theme1]], [[mjx-rl-debugging-method]] (isolated tar sync; periodic checkpoint).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

try:                                                   # load .env so VIRTUROID_GPU_SSH resolves before _HOST is read
    from virturoid.services.llm_client import _load_local_env as _load_env
    _load_env()
except Exception:  # noqa: BLE001 - best-effort; falls back to env / placeholder
    pass
# The GPU box is configured via env, NOT a hardcoded personal host (privacy + reusability). Set VIRTUROID_GPU_SSH
# in your .env (e.g. user@host). The placeholder default fails the reachability check gracefully -> CPU fallback.
_HOST = os.environ.get("VIRTUROID_GPU_SSH") or "user@gpu-box"
_PY = os.environ.get("VIRTUROID_GPU_PYTHON", "~/rl/bin/python")
_REPO_ROOT = Path(__file__).resolve().parents[3]      # .../src/virturoid/services -> repo root


def _ssh(cmd: str, *, timeout: float = 60, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["tailscale", "ssh", _HOST, cmd], capture_output=True, timeout=timeout, input=stdin)


def gpu_available(*, timeout: float = 20) -> bool:
    """True if the GPU box answers over Tailscale (quick liveness check)."""
    try:
        return b"BOX_OK" in _ssh("echo BOX_OK", timeout=timeout).stdout
    except Exception:  # noqa: BLE001
        return False


def _sync_repo(say) -> None:
    say("syncing the project to the GPU box…")
    with tempfile.TemporaryDirectory() as td:
        tgz = Path(td) / "sync.tgz"
        with tarfile.open(tgz, "w:gz") as t:
            t.add(_REPO_ROOT / "src", arcname="src")
            t.add(_REPO_ROOT / "scripts", arcname="scripts")
        _ssh("cat > ~/app_sync.tgz", timeout=180, stdin=tgz.read_bytes())
    _ssh("cd ~/virturoid && tar xzf ~/app_sync.tgz", timeout=90)


def _poll_and_fetch(remote_npz: str, out_path: str, *, progress, timeout: float, say,
                    proc_name: str = "mjx_morph_attention") -> str | None:
    deadline = time.time() + timeout
    started = time.time()
    misses = 0
    last_iter = None
    # The first iteration includes the one-off MJX/XLA kernel compile (~1-2 min) and emits no log line, so
    # tell the user that up front  -  otherwise the UI looks frozen during the compile.
    say("compiling on the GPU — the first iteration takes ~1-2 min, then it trains live…")
    while time.time() < deadline:
        time.sleep(8)
        try:
            out = _ssh(f"pgrep -f '[{proc_name[0]}]{proc_name[1:]}' >/dev/null && echo RUN || echo DONE; "
                       "grep -E 'iter|fwd_vel|reward|success|lift|saved|Error|Traceback' ~/app_train.log "
                       "2>/dev/null | tail -1", timeout=40).stdout.decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001 - transient Tailscale hiccup; keep polling
            misses += 1
            if misses > 12:
                break
            continue
        misses = 0
        m = re.search(r"iter\s+(\d+).*?(fwd_vel=[-+]?\d*\.?\d+|reward=\s*[-+]?\d*\.?\d+|success[=:\s]+[-+]?\d*\.?\d+)", out)
        if m:
            last_iter = m.group(1)
            say(f"training on the GPU — iter {m.group(1)}, {m.group(2)}  ({int(time.time() - started)}s elapsed)")
        elif last_iter is None:
            # still compiling (no iter line yet): heartbeat so the user sees it's alive, not hung
            say(f"compiling on the GPU — {int(time.time() - started)}s elapsed (first iteration coming up)…")
        if out.startswith("DONE"):
            break
    say("fetching the trained policy…")
    npz = _ssh(f"cat ~/virturoid/{remote_npz}", timeout=120).stdout
    if len(npz) < 200:                                   # no/short file -> training didn't produce one
        return None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(npz)
    import numpy as np
    np.load(out_path)                                    # validate it loads
    return out_path


def _launch_ok(launch_cmd: str, proc_name: str) -> bool:
    """Start a detached (setsid) trainer and confirm it is running. tailscale ssh frequently does NOT return
    promptly after backgrounding a job (the channel lingers until the call times out), so a slow / timed-out
    launch call is NOT a failure -- the setsid'd process detaches and trains regardless. Treat 'LAUNCHED echoed'
    OR 'pgrep finds the trainer' as a successful launch (this fixes silent None returns on a healthy box)."""
    try:
        if b"LAUNCHED" in _ssh(launch_cmd, timeout=20).stdout:
            return True
    except Exception:  # noqa: BLE001 - channel hung after backgrounding; the job still detached, confirm below
        pass
    time.sleep(5)
    try:
        return b"RUN" in _ssh(f"pgrep -f '[{proc_name[0]}]{proc_name[1:]}' >/dev/null && echo RUN || echo NO", timeout=25).stdout
    except Exception:  # noqa: BLE001
        return False


# reward-shaping flags the MJX trainer accepts (gait_critic's bounded design space -> trainer CLI flags).
_REWARD_FLAGS = ("prog_w", "clear_w", "swing_w", "slip_w", "alt_w", "smooth_w", "air_w", "vz_w", "wxy_w",
                 "torque_w", "back_w", "fwd_gate_w")   # back_w = backward penalty; fwd_gate_w = G4 forward-gate fix


def train_gene_on_gpu(gene, *, out_path: str, iters: int = 80, envs: int = 1024, progress=None,
                      timeout: float = 2400.0, reward_weights: dict | None = None,
                      cpg: bool = False, dr: bool = False, init_npz: str | None = None,
                      adaptive: bool = False, ep_len: int | None = None,
                      film: bool = False, topo_bias: bool = False,
                      calf_phase: float | None = None, cpg_freq: float | None = None,
                      decimation: int | None = None, action_lpf: float | None = None,
                      sphere_feet: bool = False, contact_dr: bool = False,
                      keep_checkpoints: bool = False) -> str | None:
    """Sync repo+gene to the box, run MJX PPO, fetch the trained policy to ``out_path``. Returns the local
    npz path, or ``None`` on any failure so the caller can fall back to CPU. ``reward_weights`` (the AI gait
    critic's redesigned weights) are passed as trainer flags; ``cpg``/``dr`` enable the trot-CPG prior and
    sim2real domain randomization; ``init_npz`` warm-starts from a banked policy on the box."""
    def say(m: str):
        if progress:
            progress(m)
    try:
        _sync_repo(say)
        _ssh("cat > ~/app_gene.json", timeout=60, stdin=json.dumps(gene.to_dict()).encode("utf-8"))
        extra = " --cpg" if cpg else ""
        extra += " --dr" if dr else ""
        extra += " --adaptive" if adaptive else ""           # inertia-scaled per-joint PD gains — needed to train a humanoid
        extra += " --film" if film else ""                    # Phase-5 FiLM joint-attribute conditioning
        extra += " --topo-bias" if topo_bias else ""          # Phase-5 topology-aware attention bias
        if calf_phase is not None:                            # per-body CPG gait direction (hexapod fwd at 0.0)
            extra += f" --calf-phase {float(calf_phase)}"
        if cpg_freq is not None:
            extra += f" --cpg-freq {float(cpg_freq)}"
        if decimation and int(decimation) > 1:               # plan v2 T1.1: 50Hz control (deploy-gap root-cause fix)
            extra += f" --decimation {int(decimation)}"
        if action_lpf and float(action_lpf) > 0.0:           # plan v2 T1.2: action low-pass (train==deploy)
            extra += f" --action-lpf {float(action_lpf)}"
        if sphere_feet:                                      # plan v2 T1.4: manifold-invariant sphere feet (train==deploy)
            extra += " --sphere-feet"
        if contact_dr:                                       # plan v2 T1.5: per-env contact-model DR (the MJX<->CPU gap)
            extra += " --contact-dr"
        if keep_checkpoints:                                 # plan v2 T0.1: numbered checkpoints for deploy-sim selection
            extra += " --keep-checkpoints"
        if ep_len:
            extra += f" --ep-len {int(ep_len)}"
        if init_npz:
            extra += f" --init-npz {init_npz}"
        for k, v in (reward_weights or {}).items():          # critic weights -> --clear-w / --prog-w / ... flags
            if k in _REWARD_FLAGS:
                extra += f" --{k.replace('_', '-')} {float(v)}"
        say(f"launching MJX PPO on the GPU ({iters} iters{', critic-tuned reward' if reward_weights else ''})…")
        launch = (f"cd ~/virturoid && rm -f runs/app_gene.npz; PYTHONPATH=src XLA_FLAGS='--xla_gpu_autotune_level=0' setsid {_PY} "
                  f"scripts/mjx_morph_attention.py --gene-json ~/app_gene.json --iters {iters} --envs {envs} "
                  f"--save runs/app_gene.npz{extra} </dev/null >~/app_train.log 2>&1 & echo LAUNCHED")
        if not _launch_ok(launch, "mjx_morph_attention"):
            return None
        return _poll_and_fetch("runs/app_gene.npz", out_path, progress=progress, timeout=timeout, say=say)
    except Exception:  # noqa: BLE001 - any failure -> caller uses the CPU trainer
        return None


# Manipulation MJX trainers (the plan's #1 priority: arms first). Each is a residual policy on a scripted
# FK-IK base with a process-aware, hack-resistant reward read from the MJX contact arrays — the team's proven
# residual recipe ([[mjx-rl-debugging-method]]), now MJX-SAFE via compile_gene_with_scene(physics_only=True).
_MANIP_SCRIPT = {
    "grasp": "mjx_residual_grasp.py", "grasp_lift": "mjx_residual_grasp.py",
    "pick_place": "mjx_residual_grasp.py", "pick_place_sort": "mjx_residual_grasp.py",
    "push": "mjx_residual_push.py", "reach": "mjx_reach_train.py",
}


def train_manipulation_on_gpu(*, task: str = "grasp", out_path: str, iters: int = 200, envs: int = 192,
                              ep_len: int = 200, progress=None, timeout: float = 2400.0) -> str | None:
    """Train a MANIPULATION policy on the GPU via MJX PPO (the plan's #1 priority — arms first). Routes ``task``
    to the residual-grasp / residual-push / reach MJX script, runs it on the box, polls + fetches the saved npz.
    Returns the local npz path, or ``None`` on any failure (so the caller can fall back to the scripted
    baseline). The scripts build their own gripper-arm body; the MJX-safe compile is wired in each."""
    script = _MANIP_SCRIPT.get((task or "").lower(), "mjx_residual_grasp.py")
    proc = script[:-3]

    def say(m: str):
        if progress:
            progress(m)
    try:
        _sync_repo(say)
        say(f"launching MJX manipulation PPO on the GPU ({task} via {script}, {iters} iters)…")
        # XLA autotuning intermittently fails on this box with CUDA_ERROR_NOT_READY (esp. right after another
        # job was killed); disabling it (level 0) makes the grasp/push kernels compile reliably (verified: a
        # tiny jax op + the locomotion run are unaffected). The kernels are slightly less tuned, not wrong.
        launch = (f"cd ~/virturoid && PYTHONPATH=src XLA_FLAGS='--xla_gpu_autotune_level=0' setsid {_PY} "
                  f"scripts/{script} --iters {iters} --envs {envs} --ep-len {ep_len} --save runs/app_manip.npz "
                  f"</dev/null >~/app_train.log 2>&1 & echo LAUNCHED")
        if not _launch_ok(launch, proc):
            return None
        return _poll_and_fetch("runs/app_manip.npz", out_path, progress=progress, timeout=timeout, say=say,
                               proc_name=proc)
    except Exception:  # noqa: BLE001 - any failure -> caller uses the scripted baseline
        return None


def train_mjcf_on_gpu(mjcf: str, *, out_path: str, iters: int = 80, envs: int = 1024, progress=None,
                      timeout: float = 2400.0, reward_weights: dict | None = None,
                      cpg: bool = False, dr: bool = False) -> str | None:
    """Like ``train_gene_on_gpu`` but for an IMPORTED model: ship the raw MJCF and train it on the GPU via
    ``--mjcf-file``. ``cpg`` enables the (quad/biped) gait prior, ``dr`` sim2real domain randomization, and
    ``reward_weights`` the AI critic's redesigned reward — same wiring as the generated-body trainer."""
    def say(m: str):
        if progress:
            progress(m)
    try:
        _sync_repo(say)
        _ssh("cat > ~/app_model.xml", timeout=60, stdin=mjcf.encode("utf-8"))
        extra = " --cpg" if cpg else ""
        extra += " --dr" if dr else ""
        for k, v in (reward_weights or {}).items():
            if k in _REWARD_FLAGS:
                extra += f" --{k.replace('_', '-')} {float(v)}"
        say(f"launching MJX PPO on the GPU ({iters} iters{', CPG' if cpg else ''})…")
        launch = (f"cd ~/virturoid && PYTHONPATH=src XLA_FLAGS='--xla_gpu_autotune_level=0' setsid {_PY} "
                  f"scripts/mjx_morph_attention.py --mjcf-file ~/app_model.xml --iters {iters} --envs {envs} "
                  f"--save runs/app_model.npz{extra} </dev/null >~/app_train.log 2>&1 & echo LAUNCHED")
        if not _launch_ok(launch, "mjx_morph_attention"):
            return None
        return _poll_and_fetch("runs/app_model.npz", out_path, progress=progress, timeout=timeout, say=say)
    except Exception:  # noqa: BLE001 - any failure -> caller uses the CPU trainer
        return None
