"""Sim-backend parity harness — the hard precondition for any WS-F GPU training (master_plan_v6 WS-F.1).

WS-F history is stalls and a diagnosed **CPU/GPU sign flip**: an identical zero-residual policy read +0.535 m on
CPU MuJoCo and −0.138 m on GPU MJX. The plan's discipline: *before any training run, make the sign flip a FAILING
TEST, not a mystery.* This module is that test. It replays a **fixed open-loop control sequence** (identical bytes
across backends, so any divergence is the ENGINE, never the policy) on the same body through each available
backend and compares the state trajectories + the forward-displacement SIGN.

Backends are pluggable: ``cpu_mujoco`` (always available), ``mjx_single`` / ``mjx_batch`` (guarded — they run on
the GPU box; here they report unavailable rather than crash). So the harness is CPU-testable now (cpu-vs-cpu is
bit-parity; a sign-flipping backend trips the detector) and becomes the go/no-go gate on the GPU box: parity must
be green before a single training step is spent. No GPU guessing — one diagnosis, one bundled retrain.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ParityBackendUnavailable(RuntimeError):
    """Raised when a requested backend (e.g. MJX) isn't importable in this environment."""


def fixed_ctrl_sequence(nu: int, steps: int, *, seed: int = 0):
    """A deterministic open-loop control trajectory (steps × nu) — the SAME bytes fed to every backend. A seeded
    per-actuator sinusoid: reproducible, non-trivial (it moves the body), and independent of any policy/engine."""
    import numpy as np
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=nu)
    freq = rng.uniform(0.5, 1.5, size=nu)
    amp = rng.uniform(0.3, 0.8, size=nu)
    t = np.arange(steps)[:, None]
    return (amp * np.sin(2 * np.pi * freq * (t / 100.0) + phase)).astype(float)


def _xml_for(gene_or_xml) -> str:
    if isinstance(gene_or_xml, str):
        return gene_or_xml
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    return compile_gene_to_mjcf(gene_or_xml, include_floor=True, spawn_z=standing_spawn_z(gene_or_xml))


def rollout_cpu_mujoco(gene_or_xml, ctrl_seq, *, record_every: int = 5) -> dict:
    """Replay ``ctrl_seq`` on CPU MuJoCo, returning the sub-sampled qpos trajectory + the base forward travel."""
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_string(_xml_for(gene_or_xml))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
    nu = m.nu
    seq = np.asarray(ctrl_seq, float)
    base = 0
    x0 = float(d.xpos[base][0]) if m.nbody > 1 else 0.0
    traj = []
    for i in range(len(seq)):
        d.ctrl[:nu] = seq[i, :nu]
        mujoco.mj_step(m, d)
        if i % record_every == 0:
            traj.append(np.array(d.qpos, float))
    xf = float(d.xpos[base][0]) if m.nbody > 1 else 0.0
    # base body forward: use the free-base qpos[0] (world x of the root) when present, else body 1
    fwd = float(d.qpos[0]) if m.nq >= 1 else (xf - x0)
    return {"backend": "cpu_mujoco", "qpos_traj": np.array(traj), "forward": fwd, "nu": nu,
            "n_steps": len(seq), "finite": bool(np.all(np.isfinite(d.qpos)))}


def rollout_mjx(gene_or_xml, ctrl_seq, *, batch: int = 1, record_every: int = 5) -> dict:
    """Replay ``ctrl_seq`` on MJX (single-env or batched). Guarded: raises ParityBackendUnavailable when MJX isn't
    importable (this CPU box) — it runs on the GPU box, where this leg exercises the exact CPU/GPU comparison."""
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
    except Exception as exc:  # noqa: BLE001
        raise ParityBackendUnavailable(f"MJX not available here ({type(exc).__name__}); run on the GPU box") from exc
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_string(_xml_for(gene_or_xml))
    mx = mjx.put_model(m)
    dx = mjx.make_data(mx)
    seq = jnp.asarray(np.asarray(ctrl_seq, float))
    nu = m.nu

    def step(dx, u):
        dx = dx.replace(ctrl=u[:nu])
        return mjx.step(mx, dx), dx.qpos
    traj = []
    for i in range(len(ctrl_seq)):
        dx, qpos = step(dx, seq[i])
        if i % record_every == 0:
            traj.append(np.asarray(qpos, float))
    fwd = float(np.asarray(dx.qpos, float)[0]) if m.nq >= 1 else 0.0
    return {"backend": f"mjx_{'batch' if batch > 1 else 'single'}", "qpos_traj": np.array(traj),
            "forward": fwd, "nu": nu, "n_steps": len(ctrl_seq), "finite": bool(np.all(np.isfinite(dx.qpos)))}


BACKENDS = {"cpu_mujoco": rollout_cpu_mujoco,
            "mjx_single": lambda g, c, **k: rollout_mjx(g, c, batch=1, **k),
            "mjx_batch": lambda g, c, **k: rollout_mjx(g, c, batch=64, **k)}


def compare(a: dict, b: dict, *, state_tol: float = 1e-3, forward_floor: float = 0.05) -> dict:
    """Compare two backend rollouts: max state divergence + the forward-displacement SIGN-FLIP check (the WS-F
    bug). ``parity_ok`` iff states agree within ``state_tol`` AND the forward signs agree."""
    import numpy as np
    ta, tb = a.get("qpos_traj"), b.get("qpos_traj")
    n = min(len(ta), len(tb)) if (ta is not None and tb is not None) else 0
    if n == 0:
        max_diff = float("inf")
    else:
        w = min(ta.shape[1], tb.shape[1])
        max_diff = float(np.max(np.abs(ta[:n, :w] - tb[:n, :w])))
    fa, fb = float(a.get("forward", 0.0)), float(b.get("forward", 0.0))
    sign_flip = bool((fa * fb < 0) and abs(fa) >= forward_floor and abs(fb) >= forward_floor)
    parity_ok = bool(max_diff <= state_tol and not sign_flip)
    return {"pair": f"{a.get('backend')} vs {b.get('backend')}", "max_state_abs_diff": round(max_diff, 6),
            "forward_a": round(fa, 4), "forward_b": round(fb, 4), "sign_flip": sign_flip,
            "parity_ok": parity_ok,
            "note": ("SIGN FLIP — the WS-F bug: the same control travels opposite directions across backends; do "
                     "NOT train until this is green (frame/velocity-index mismatch)" if sign_flip
                     else "engines agree within tolerance" if parity_ok
                     else "state trajectories diverge beyond tolerance (numerics/model mismatch)")}


@dataclass
class ParityReport:
    body: str = ""
    backends_run: list = field(default_factory=list)
    backends_unavailable: list = field(default_factory=list)
    comparisons: list = field(default_factory=list)
    parity_ok: bool = False

    def to_dict(self) -> dict:
        return {"body": self.body, "backends_run": self.backends_run,
                "backends_unavailable": self.backends_unavailable, "comparisons": self.comparisons,
                "parity_ok": self.parity_ok,
                "gate": ("PARITY GREEN — safe to spend training compute" if self.parity_ok else
                         "PARITY NOT ESTABLISHED — fix before any GPU training (or only CPU backend available)")}


def parity_check(gene_or_xml, *, backends=("cpu_mujoco", "mjx_single"), steps: int = 300, seed: int = 0,
                 body: str = "") -> ParityReport:
    """Run the fixed control through each backend and compare every backend against the first available one.
    Unavailable backends (e.g. MJX here) are recorded honestly, never crash. ``parity_ok`` requires >=2 backends
    that all agree (so a CPU-only run reports 'not established' — it cannot prove cross-engine parity by itself)."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(_xml_for(gene_or_xml))
    ctrl = fixed_ctrl_sequence(m.nu, steps, seed=seed)
    rep = ParityReport(body=body or getattr(gene_or_xml, "id", "body"))
    rollouts = {}
    for name in backends:
        fn = BACKENDS.get(name)
        if fn is None:
            rep.backends_unavailable.append(name); continue
        try:
            rollouts[name] = fn(gene_or_xml, ctrl)
            rep.backends_run.append(name)
        except ParityBackendUnavailable:
            rep.backends_unavailable.append(name)
    if len(rep.backends_run) >= 2:
        ref = rep.backends_run[0]
        rep.comparisons = [compare(rollouts[ref], rollouts[other]) for other in rep.backends_run[1:]]
        rep.parity_ok = all(c["parity_ok"] for c in rep.comparisons)
    else:
        rep.parity_ok = False        # cannot prove cross-engine parity with a single backend
    return rep
