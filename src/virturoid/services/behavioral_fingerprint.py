"""Behavioral fingerprint (``z_dyn``) — embed a body by its RESPONSE, not just its shape.

Measured: static geometry ranks CLASS-level gait transfer at ~0.90 but WITHIN-class transfer at ~0.57 (chance) —
because whether gait (freq, amps) ports to a body is set by DYNAMIC quantities geometry cannot express: the
body's natural (SLIP/pendular) frequency under its deploy PD, its damping, its stance height (the Froude length),
and whether its step cadence actually locks to a commanded drive frequency. This module measures exactly those
with two cheap, deterministic probes that reuse the SAME rollout machinery gaits deploy on (so the fingerprint is
measured in the frame that matters):

  * Probe 0 — SETTLE/RELEASE: run the crawl-gait rollout with ZERO amplitudes (= pure PD hold at stance) from the
    spawn drop; the root-height trace rings at the body's effective vertical resonance -> f_res, damping ratio,
    settled stance height, mass-normalized stiffness.
  * Probe 3 — REFERENCE BURSTS: two short standard-gait bursts (low/high drive frequency); does the body's
    measured cadence LOCK to the commanded frequency (the dynamic-similarity signal), how much support/height it
    keeps, how far it travels per cycle.

~11 features, CPU MuJoCo, deterministic, a few seconds per body, cached per gene. Consumed by ``body_metric``
feature spaces ``dyn`` / ``rich_dyn`` (and gated exactly like everything else: adopted only if it PROVABLY beats
the baseline held-out on physics-verified transfer labels).
"""
from __future__ import annotations

import hashlib
import json
import math

DYN_FEATURE_NAMES = (
    "f_res_hz", "damping_ratio", "stance_h", "k_eff_norm", "settle_drop",
    "lock_lo", "support_lo", "height_lo", "fwd_lo",
    "lock_hi", "support_hi", "fwd_hi",
)
_LO_F, _HI_F = 1.2, 2.6          # the two reference drive frequencies (span the banked-gait range)
_cache: dict[str, list[float]] = {}


def _gene_key(gene) -> str:
    try:
        return hashlib.md5(json.dumps(gene.to_dict(), sort_keys=True).encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return str(id(gene))


def _ring_features(zs: list[float], dt: float) -> tuple[float, float, float, float]:
    """(f_res_hz, damping_ratio, settled_h, drop) from a root-height trace: find the post-drop oscillation peaks;
    frequency from peak spacing, damping from the log-decrement of successive peak amplitudes."""
    if len(zs) < 10:
        return 0.0, 0.0, (zs[-1] if zs else 0.0), 0.0
    settled = sum(zs[-max(5, len(zs) // 10):]) / max(5, len(zs) // 10)
    drop = max(0.0, zs[0] - min(zs))
    # peaks of |z - settled| -> the ringing period; ignore the first samples (spawn transient)
    dev = [z - settled for z in zs]
    peaks = []
    for i in range(2, len(dev) - 2):
        if abs(dev[i]) >= abs(dev[i - 1]) and abs(dev[i]) > abs(dev[i + 1]) and abs(dev[i]) > 1e-4:
            if not peaks or i - peaks[-1][0] > 3:
                peaks.append((i, abs(dev[i])))
    if len(peaks) < 2:
        return 0.0, 1.0, settled, drop                            # overdamped: no ring
    # successive |peaks| are half-periods of the damped oscillation
    gaps = [(peaks[i + 1][0] - peaks[i][0]) for i in range(len(peaks) - 1)]
    half_period = (sum(gaps) / len(gaps)) * dt
    f_res = 1.0 / (2.0 * half_period) if half_period > 1e-9 else 0.0
    # log-decrement over successive half-cycles -> damping ratio
    decs = [math.log(max(1e-9, peaks[i][1]) / max(1e-9, peaks[i + 1][1])) for i in range(len(peaks) - 1)]
    d = max(0.0, sum(decs) / len(decs))                           # per half-cycle
    zeta = d / math.sqrt(math.pi ** 2 + d * d)
    return f_res, min(1.0, zeta), settled, drop


def dyn_fingerprint(gene, *, settle_steps: int = 360, burst_steps: int = 420) -> dict:
    """The named behavioral features for one body (dict). Deterministic; ~3 short CPU rollouts."""
    from virturoid.services.morph_policy import crawl_gait_rollout
    out = dict.fromkeys(DYN_FEATURE_NAMES, 0.0)
    try:
        # Probe 0: zero-amplitude gait = PD hold at stance -> the spawn drop rings at the body's resonance
        r0 = crawl_gait_rollout(gene, steps=settle_steps, hip_amp=0.0, knee_amp=0.0, record_qpos=True)
        zs = [float(q[2]) for q in (r0.get("qpos_frames") or []) if len(q) > 2]
        dt = 0.01 * max(1, settle_steps // max(1, len(zs)))       # frame_every spacing in sim time
        f_res, zeta, h, drop = _ring_features(zs, dt)
        m = sum(s.mass_kg for s in gene.segments) or 1.0
        out.update(f_res_hz=round(f_res, 4), damping_ratio=round(zeta, 4), stance_h=round(h, 4),
                   k_eff_norm=round(math.log1p(m * (2 * math.pi * f_res) ** 2), 4), settle_drop=round(drop, 4))
        # Probe 3: two reference bursts; cadence LOCK = measured step frequency / commanded frequency
        for tag, f in (("lo", _LO_F), ("hi", _HI_F)):
            rb = crawl_gait_rollout(gene, steps=burst_steps, freq=f)
            cad = float(rb.get("cadence", 0.0))
            out[f"lock_{tag}"] = round(cad / f, 4) if f > 0 else 0.0
            out[f"support_{tag}"] = round(float(rb.get("support_frac", 0.0)), 4)
            out[f"fwd_{tag}"] = round(float(rb.get("forward", 0.0)), 4)
            if tag == "lo":
                out["height_lo"] = round(float(rb.get("height_ratio", 0.0)), 4)
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
