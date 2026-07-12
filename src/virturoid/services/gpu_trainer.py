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


def _read_complete_npz(read_bytes, *, attempts: int = 4, wait_s: float = 2.0) -> bytes | None:
    """Read a remote NPZ only after it is both non-trivial and parseable.

    A detached GPU process can disappear from ``pgrep`` a moment before the remote filesystem exposes its final
    bytes. Treating that short window as a hard training failure silently sends the user to CPU fallback. This
    bounded retry accepts only a complete NumPy archive, so it fixes the race without accepting corruption.
    """
    import io
    import numpy as np

    n_attempts = max(1, int(attempts))
    for attempt in range(n_attempts):
        try:
            payload = read_bytes()
            if len(payload) >= 200:
                np.load(io.BytesIO(payload))
                return payload
        except Exception:  # noqa: BLE001 - short/incomplete archive; a bounded retry may see the final flush
            pass
        if attempt + 1 < n_attempts:
            time.sleep(max(0.0, float(wait_s)))
    return None


def _poll_and_fetch(remote_npz: str, out_path: str, *, progress, timeout: float, say,
                    proc_name: str = "mjx_morph_attention", fetch_checkpoints: bool = False) -> str | None:
    deadline = time.time() + timeout
    started = time.time()
    misses = 0
    last_iter = None
    completed = False
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
            completed = True
            break
    if not completed:
        # v7-F4: a deadline-exceeded trainer must not keep running detached — it pins GPU memory (the measured
        # ~9 GB leak) and shadows the NEXT launch's pgrep/log. Kill it explicitly; best-effort (box may be down).
        say("training did not finish inside the deadline — stopping the remote trainer (frees GPU memory)…")
        try:
            _ssh(f"pkill -f '[{proc_name[0]}]{proc_name[1:]}'", timeout=30)
        except Exception:  # noqa: BLE001
            pass
    say("fetching the trained policy…")
    npz = _read_complete_npz(lambda: _ssh(f"cat ~/virturoid/{remote_npz}", timeout=120).stdout)
    if npz is None:                                      # bounded retries still found no complete file
        return None
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(npz)
    if fetch_checkpoints:
        # also fetch the numbered checkpoints (plan v2 T0.1) to local siblings <out_stem>_it{N}.npz so the caller
        # can DEPLOY-SELECT the best (never trust the final by train-reward -- it can deploy WORSE than an earlier
        # one, the measured MJX->CPU divergence). Best-effort: a missing checkpoint just isn't a candidate.
        base = remote_npz.split("/")[-1][:-4]            # e.g. "app_gene"
        out_stem = out_path[:-4]
        try:
            listing = _ssh(f"ls -1 ~/virturoid/{remote_npz[:-4]}_it*.npz 2>/dev/null",
                           timeout=40).stdout.decode("utf-8", "ignore")
            for remote in listing.split():
                remote = remote.strip()
                nm = remote.split("/")[-1]               # "app_gene_it30.npz"
                if not nm.startswith(base):
                    continue
                data = _ssh(f"cat {remote}", timeout=120).stdout
                if len(data) >= 200:
                    Path(out_stem + nm[len(base):]).write_bytes(data)   # <out_stem>_it30.npz
        except Exception:  # noqa: BLE001 - checkpoint fetch is best-effort; the final still verifies
            pass
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


# v7-F3 (master_plan_v7 §1): the WS-F.1 parity precondition as an ENFORCED launch gate. The historical deploy
# disaster was a policy trained on a stack whose engines disagreed (+0.535 m CPU vs −0.138 m MJX — the sign
# flip); the harness existed but nothing CALLED it before spending compute. This runs the REAL comparison ON THE
# BOX (both engines, identical control bytes) and refuses to train on red. Two legs: (1) fixed control, strict
# state tolerance — catches numeric/model mismatch; (2) a REAL walking controller's recorded torques
# (``crawl_gait_rollout(record_ctrl=True)``) checked for the forward SIGN — random sinusoids cannot displace a
# stable body past the 0.05 m sign floor (measured |fwd|≈0.02 even at 2.5× amplitude), so only a gait trace makes
# the sign check non-vacuous. Long-horizon state tol is meaningless under contact chaos → leg 2 is sign-only.
_PARITY_GATE_CACHE: dict = {}
_PARITY_REMOTE_PY = """
import base64, json, sys
import numpy as np
p = json.load(sys.stdin)
xml = base64.b64decode(p['xml_b64']).decode()
from virturoid.services.parity_harness import parity_check, rollout_cpu_mujoco, rollout_mjx, compare
rep = parity_check(xml, backends=('cpu_mujoco', 'mjx_single'), steps=int(p.get('steps', 300)))
legs = [{'leg': 'fixed_ctrl', 'parity_ok': bool(rep.parity_ok), 'comparisons': rep.comparisons,
         'backends': rep.backends_run}]
ok = bool(rep.parity_ok)
if p.get('trace_b64'):
    tr = np.frombuffer(base64.b64decode(p['trace_b64'])).reshape(p['trace_shape'])
    c = compare(rollout_cpu_mujoco(xml, tr), rollout_mjx(xml, tr), state_tol=float('inf'))
    vac = abs(c['forward_a']) < 0.05 and abs(c['forward_b']) < 0.05
    legs.append({'leg': 'gait_trace', 'parity_ok': bool(c['parity_ok']), 'vacuous': bool(vac), 'comparison': c})
    ok = ok and (vac or c['parity_ok'])
print('PARITY_GATE_JSON ' + json.dumps({'parity_ok': ok, 'legs': legs}))
"""


def parity_gate_for_gene(gene, *, say=lambda m: None, steps: int = 300, timeout: float = 420.0) -> dict:
    """Run the CPU-vs-MJX parity check for ``gene`` ON THE GPU BOX; return ``{"parity_ok": bool, "legs": [...]}``.
    A green result is cached per compiled-XML hash for the process (a training night retraining the same body
    pays the ~60-90 s gate once); a red/unreachable result is NEVER cached (transient failures may clear).
    Requires the repo already synced to the box (``_sync_repo``) so the box runs the same harness bytes."""
    import base64
    import hashlib

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
    key = hashlib.sha256(xml.encode()).hexdigest()[:16]
    if key in _PARITY_GATE_CACHE:
        return _PARITY_GATE_CACHE[key]
    payload: dict = {"xml_b64": base64.b64encode(xml.encode()).decode(), "steps": int(steps)}
    try:                                                     # the non-vacuous leg: a real walking controller's bytes
        from virturoid.services.morph_policy import crawl_gait_rollout
        import numpy as np
        tr = crawl_gait_rollout(gene, steps=600, record_ctrl=True).get("ctrl_trace")
        if tr is not None and len(tr) >= 50:
            trace = np.asarray(tr, dtype=np.float64)
            payload["trace_b64"] = base64.b64encode(trace.tobytes()).decode()
            payload["trace_shape"] = list(trace.shape)
    except Exception:  # noqa: BLE001 - a body with no crawl gait still gets the strict fixed-ctrl leg
        pass
    try:
        r = _ssh(f'cd ~/virturoid && PYTHONPATH=src {_PY} -c "{_PARITY_REMOTE_PY}"', timeout=timeout,
                 stdin=json.dumps(payload).encode("utf-8"))
        for line in r.stdout.decode("utf-8", "ignore").splitlines():
            if line.startswith("PARITY_GATE_JSON"):
                gate = json.loads(line.split("PARITY_GATE_JSON", 1)[1])
                if gate.get("parity_ok"):
                    _PARITY_GATE_CACHE[key] = gate
                return gate
        return {"parity_ok": False, "note": "gate produced no verdict (remote error) — refusing to train blind",
                "stderr_tail": r.stderr.decode("utf-8", "ignore")[-400:]}
    except Exception as exc:  # noqa: BLE001 - unreachable box: fail CLOSED (never train unverified)
        return {"parity_ok": False, "note": f"parity gate unreachable ({type(exc).__name__}) — refusing to train blind"}


# reward-shaping flags the MJX trainer accepts (gait_critic's bounded design space -> trainer CLI flags).
_REWARD_FLAGS = ("prog_w", "clear_w", "swing_w", "slip_w", "alt_w", "smooth_w", "air_w", "vz_w", "wxy_w",
                 "torque_w", "back_w", "fwd_gate_w", "periodic_w", "wtw_w", "upright_hi", "upright_w")   # back_w = backward penalty;
#   fwd_gate_w = G4; upright_hi = anti-CROUCH stance-height (raise toward 0.9 for bodies that farm air-time low)
#   forward-gate fix; periodic_w = P1 periodic gait-contact reward (plan v4)


def default_training_recipe(gene) -> dict:
    """The AUTOMATIC GPU-training recipe for a body — the product's SINGLE source of truth, so cpg / adaptive
    gains / the deploy-gap deltas are chosen by the SYSTEM per morphology, NOT hand-toggled differently at each
    call-site. Enforces the deploy-safe INVARIANT ``phase_obs == cpg``: a gait-phase clock is only ever fed at
    deploy WITH a CPG source (recipe_rollout_morph gates the clock on cpg_on), so phase_obs WITHOUT cpg silently
    drops the clock at deploy and the gait drifts out of phase and falls — the deploy gap diagnosed 2026-07-09.
    Returns the kwargs to splat into ``train_gene_on_gpu(gene, **default_training_recipe(gene))``."""
    legged = False
    adaptive = False
    n_legs = 0
    try:
        import mujoco

        from virturoid.services.appendage_map import build_appendage_map
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morph_policy import adaptive_recommended
        from virturoid.services.task_matched_eval import robot_kind
        legged = robot_kind(gene) == "legged"                # bodies driven by a leg gait (quad/hexapod/humanoid)
        if legged:
            model = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
            adaptive = adaptive_recommended(model)           # inertia-scaled gains auto-gate on for heavy/humanoid
            n_legs = build_appendage_map(model).n_legs
    except Exception:  # noqa: BLE001 - recipe selection is best-effort; fall back to the safe legged defaults
        legged, n_legs = True, 4
    # the trot-CPG prior fits >=3-leg bodies (measured: a quad/hexapod trains BETTER with it); it does NOT fit a
    # BIPED (2 legs) — a humanoid trained WITHOUT cpg deployed 0.34 m stable vs 0.16 m with the trot-CPG. So cpg
    # (and its paired phase clock) is on only for >=3 legs.
    cpg = bool(legged and n_legs >= 3)
    return {
        "cpg": cpg,                                          # trot-CPG gait prior (>=3-leg bodies only)
        "phase_obs": cpg,                                    # INVARIANT phase_obs==cpg (deploy-safe; see docstring)
        "adaptive": bool(adaptive),                          # per-joint inertia-scaled gains (auto; humanoid needs it)
        "dr": True, "contact_dr": True, "sphere_feet": True, "real_actuator": True,   # sim2real + deploy-gap deltas
        "decimation": 4,                                     # 50 Hz control (train==deploy)
    }


def train_gene_on_gpu(gene, *, out_path: str, iters: int = 80, envs: int = 1024, progress=None,
                      timeout: float = 2400.0, reward_weights: dict | None = None,
                      cpg: bool = False, dr: bool = False, init_npz: str | None = None,
                      adaptive: bool = False, ep_len: int | None = None,
                      film: bool = False, topo_bias: bool = False,
                      calf_phase: float | None = None, cpg_freq: float | None = None,
                      decimation: int | None = None, action_lpf: float | None = None,
                      sphere_feet: bool = False, contact_dr: bool = False,
                      phase_obs: bool = False, dr_scale: float = 1.0,
                      real_actuator: bool = False,
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
        # v7-F3: the WS-F parity precondition is ENFORCED, not advisory — never spend GPU compute on a stack
        # where the engines disagree (the historical sign flip trained a policy that deployed BACKWARD).
        # VIRTUROID_SKIP_PARITY_GATE=1 is the explicit test/emergency escape hatch, never a default.
        if os.environ.get("VIRTUROID_SKIP_PARITY_GATE") != "1":
            say("parity gate: verifying CPU MuJoCo == MJX on the box (identical control bytes)…")
            gate = parity_gate_for_gene(gene, say=say)
            if not gate.get("parity_ok"):
                say(f"PARITY GATE RED — refusing to train: {gate.get('note') or gate.get('legs')}")
                return None                                  # caller falls back to CPU (deploy==train there)
            say("parity gate GREEN — engines agree; training may spend.")
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
        if phase_obs:                                        # P1 (plan v4): CPG master-phase gait clock as percept token
            extra += " --phase-obs"
        if dr_scale != 1.0:                                  # P3 (plan v4): soften/widen the contact-DR range width
            extra += f" --dr-scale {float(dr_scale)}"
        if real_actuator:                                    # B1 (plan v5): real 4-quadrant torque-speed clamp
            extra += " --real-actuator"
        if keep_checkpoints:                                 # plan v2 T0.1: numbered checkpoints for deploy-sim selection
            extra += " --keep-checkpoints"
        if ep_len:
            extra += f" --ep-len {int(ep_len)}"
        if init_npz:
            # WARM-START seed: if it is a LOCAL file (e.g. build/models/quaddec_fwd.npz from the flywheel), UPLOAD
            # it to the box first -- the trainer's --init-npz runs box-side and can't see a local path (the night's
            # transfer warm-start silently fell here: the box had no such file -> trainer errored -> candidate fell).
            # A box-relative path (runs/…) is passed through as-is.
            from pathlib import Path
            _seed = Path(init_npz)
            if _seed.is_file():
                _ssh("cat > ~/app_init.npz", timeout=120, stdin=_seed.read_bytes())
                extra += " --init-npz ~/app_init.npz"
            else:
                extra += f" --init-npz {init_npz}"
        for k, v in (reward_weights or {}).items():          # critic weights -> --clear-w / --prog-w / ... flags
            if k in _REWARD_FLAGS:
                extra += f" --{k.replace('_', '-')} {float(v)}"
        say(f"launching MJX PPO on the GPU ({iters} iters{', critic-tuned reward' if reward_weights else ''})…")
        # v7-F4: MEM_FRACTION caps XLA's arena so a killed/leaked trainer can't pin the whole 12 GB (the measured
        # ~9 GB leak-on-kill); previously this guard lived only in the nightshift ops track, not interactive runs.
        launch = (f"cd ~/virturoid && rm -f runs/app_gene.npz; PYTHONPATH=src XLA_FLAGS='--xla_gpu_autotune_level=0' "
                  f"XLA_PYTHON_CLIENT_MEM_FRACTION=.85 setsid {_PY} "
                  f"scripts/mjx_morph_attention.py --gene-json ~/app_gene.json --iters {iters} --envs {envs} "
                  f"--save runs/app_gene.npz{extra} </dev/null >~/app_train.log 2>&1 & echo LAUNCHED")
        if not _launch_ok(launch, "mjx_morph_attention"):
            return None
        return _poll_and_fetch("runs/app_gene.npz", out_path, progress=progress, timeout=timeout, say=say,
                               fetch_checkpoints=keep_checkpoints)
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
