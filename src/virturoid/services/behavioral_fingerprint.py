"""Behavioral fingerprint (``z_dyn``) — embed a body by its RESPONSE, not just its shape.

Measured: static geometry ranks CLASS-level gait transfer at ~0.90 but WITHIN-class transfer near chance —
because whether gait (freq, amps) ports to a body is set by DYNAMICS geometry cannot express (resonance, mass
distribution, stance scale — the dynamic-similarity quantities). v1 tried a free-ring probe (release → count
oscillation peaks); measured on real bodies it was DEGENERATE — the deploy PD (kp=32, kd=1.5) is near-critically
damped, so most bodies never ring (f_res=0 everywhere). v2 measures resonance the honest way: FORCE the body at
three standard frequencies through the SAME crawl-gait law gaits deploy on, and read which regime it responds in.

  * SETTLE probe — zero-amplitude gait (= PD hold at stance): settled stance height (the Froude length scale;
    measured to track leg scale 0.14→0.25→0.49) + spawn-drop depth (compliance).
  * SWEEP probe — short bursts at f ∈ {1.0, 1.8, 2.6} Hz: forward response per frequency (the body's transfer
    curve through the actual gait mechanism), which frequency wins (argmax = the resonance analog), support
    fraction and height retention per frequency, and a UNIT-CORRECT step lock: cadence is foot-liftoffs/sec
    summed over all feet (morph_policy:897-905), so lock = cadence / (n_feet · f) ≈ 1 when each foot steps once
    per commanded cycle.

12 features, ~4 short CPU rollouts, deterministic, cached per gene content. Consumed by ``body_metric`` spaces
``dyn`` / ``rich_dyn`` — and gated like everything else: adopted only if it PROVABLY beats baseline held-out.
"""
from __future__ import annotations

import hashlib
import json

_SWEEP_F = (1.0, 1.8, 2.6)
DYN_FEATURE_NAMES = (
    "stance_h", "settle_drop",
    "fwd_f1", "fwd_f2", "fwd_f3",          # forward response at each drive frequency (the transfer curve)
    "best_f",                              # argmax frequency (normalized position 0/0.5/1) — the resonance analog
    "fwd_ratio_hi_lo",                     # response asymmetry: >1 likes fast drive, <1 likes slow
    "lock_f1", "lock_f2", "lock_f3",       # per-foot per-cycle step lock at each frequency (~1 = locked)
    "support_f2", "height_f2",             # stepping support + height retention at the middle frequency
)
_cache: dict[str, list[float]] = {}


def _gene_key(gene) -> str:
    try:
        return hashlib.md5(json.dumps(gene.to_dict(), sort_keys=True).encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return str(id(gene))


def _n_feet(gene) -> int:
    """Ground-contact appendages ≈ limbs off the root (the same limb_count the rich embedding uses)."""
    root = gene.root()
    if root is None:
        return 1
    return max(1, sum(1 for s in gene.segments if s.parent == root.name))


def dyn_fingerprint(gene, *, settle_steps: int = 300, burst_steps: int = 420) -> dict:
    """The named behavioral features for one body (dict). Deterministic; ~4 short CPU rollouts."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    out = dict.fromkeys(DYN_FEATURE_NAMES, 0.0)
    try:
        feet = _n_feet(gene)
        # SETTLE: zero-amplitude gait = PD hold -> stance height (Froude scale) + drop depth (compliance)
        r0 = crawl_gait_rollout(gene, steps=settle_steps, hip_amp=0.0, knee_amp=0.0, record_qpos=True)
        zs = [float(q[2]) for q in (r0.get("qpos_frames") or []) if len(q) > 2]
        if zs:
            tail = zs[-max(5, len(zs) // 10):]
            out["stance_h"] = round(sum(tail) / len(tail), 4)
            out["settle_drop"] = round(max(0.0, zs[0] - min(zs)), 4)
        # SWEEP: forced response at three standard drive frequencies through the deploy gait law
        fwd = []
        for i, f in enumerate(_SWEEP_F, start=1):
            rb = crawl_gait_rollout(gene, steps=burst_steps, freq=f)
            fw = float(rb.get("forward", 0.0))
            fwd.append(fw)
            out[f"fwd_f{i}"] = round(fw, 4)
            cad = float(rb.get("cadence", 0.0))
            out[f"lock_f{i}"] = round(cad / (feet * f), 4) if f > 0 else 0.0
            if i == 2:
                out["support_f2"] = round(float(rb.get("support_frac", 0.0)), 4)
                out["height_f2"] = round(float(rb.get("height_ratio", 0.0)), 4)
        best = max(range(len(fwd)), key=lambda k: fwd[k])
        out["best_f"] = round(best / (len(fwd) - 1), 4)           # 0=slow regime, 1=fast regime
        lo, hi = fwd[0], fwd[-1]
        out["fwd_ratio_hi_lo"] = round((hi + 0.05) / (lo + 0.05), 4) if lo > -0.04 else 0.0
    except Exception:  # noqa: BLE001 - a body the probes can't run (fixed-base arm, compile issue) -> zeros
        pass
    return out


def z_dyn(gene) -> list[float]:
    """The fixed-order behavioral fingerprint vector (cached per gene content)."""
    key = _gene_key(gene)
    if key in _cache:
        return list(_cache[key])
    fp = dyn_fingerprint(gene)
    vec = [float(fp[k]) for k in DYN_FEATURE_NAMES]
    _cache[key] = vec
    return list(vec)
