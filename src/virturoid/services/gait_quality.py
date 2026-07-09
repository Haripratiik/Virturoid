"""Honest gait quality: turn an anti-Goodhart rollout's metrics into an explicit WALK/CROUCH/SLIDE/FELL verdict.

This lives in the PACKAGE (not ``scripts/``) because the product's legged motion verdict (``ai_native_tools``)
depends on it — importing it from ``scripts/`` broke the flagship path in any real deployment (installed package
or an MCP server launched from another cwd, where the repo root isn't on ``sys.path``, so ``import scripts`` fails
with ModuleNotFoundError and every legged robot silently read as "could not simulate"). ``scripts/verify_gait.py``
now re-exports these for back-compat.

A credible WALK requires ALL of: survived (didn't fall) + upright at the body's true stance height (not a crouch)
+ real foot-lift cadence + genuine stepping support + forward travel. A body that stands still scores cadence 0 ->
SLIDE; a low unstable body scores low upright_frac -> CROUCH; a fall is named by its dominant orientation mode.
"""
from __future__ import annotations


def orientation_summary(qpos_frames) -> dict:
    """Max + final roll/pitch/yaw (deg) from the base quaternion trace. The anti-Goodhart signal for a
    LEGGED body's true failure mode: a trot that loses lateral balance ROLLS over, an over-driven gait
    PITCH-dives over its front legs, an asymmetric gait YAW-drifts off a straight line. Height alone can't
    tell these apart (all three end with a low base) — the classifier needs the orientation to name the fall."""
    import numpy as np
    if not qpos_frames:
        return {}
    R = P = Y = 0.0
    rr = pp = yy = 0.0
    for f in qpos_frames:
        w, x, y, z = (float(f[3]), float(f[4]), float(f[5]), float(f[6]))
        rr = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pp = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        yy = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        R = max(R, abs(rr)); P = max(P, abs(pp)); Y = max(Y, abs(yy))
    return {"roll_max": round(R, 1), "pitch_max": round(P, 1), "yaw_max": round(Y, 1),
            "roll_final": round(rr, 1), "pitch_final": round(pp, 1), "yaw_final": round(yy, 1)}


_CLEAN_ROLL_MAX = 35.0     # a trot/tripod rocks side-to-side some; beyond this it's tipping, not walking
_CLEAN_PITCH_MAX = 20.0    # a clean walker holds its body level; a big pitch = REARING/DIVING to fake forward
                           # (measured: a clean quad trot pitches ~9deg; a hexapod that games `forward` via a
                           # huge-stride lurch pitches ~25deg — un-gameable gate rejects the lurch)


def classify(r: dict) -> str:
    """Explicit, honest verdict from the anti-Goodhart metrics + orientation. A CREDIBLE WALK requires the scalar
    gates AND a LEVEL body: a gait that clears forward/upright/cadence by REARING or PITCH-DIVING (which games the
    forward metric with a violent lurch, while ``upright_frac`` — a z-height/up-vector check — misses the pitch) is
    NOT a credible walk. Orientation is only judged when the qpos trace is available; a metric-only call keeps the
    scalar verdict."""
    survived = bool(r.get("survived"))
    up = float(r.get("upright_frac", 0.0)); hr = float(r.get("height_ratio", 0.0))
    cad = float(r.get("cadence", 0.0)); sup = float(r.get("support_frac", 0.0))
    fwd = float(r.get("forward", 0.0))
    o = orientation_summary(r.get("qpos_frames") or [])
    level = (not o) or (o["roll_max"] < _CLEAN_ROLL_MAX and o["pitch_max"] < _CLEAN_PITCH_MAX)
    scalars_ok = survived and up >= 0.6 and cad >= 1.0 and sup >= 0.25 and fwd >= 0.3
    if scalars_ok and level:
        return "CREDIBLE WALK"
    # not a credible walk: name the DOMINANT fall mode from the orientation trace (if replayable) so the
    # verdict is diagnostic, not just "bad" — a roll-over reads as CROUCH by height alone, which misleads.
    if o:
        modes = [("ROLL-OVER", o["roll_max"]), ("PITCH-DIVE", o["pitch_max"]), ("YAW-DRIFT", o["yaw_max"])]
        name, val = max(modes, key=lambda t: t[1])
        if not survived and val >= 45.0:
            return f"FELL by {name} (roll {o['roll_max']:.0f} / pitch {o['pitch_max']:.0f} / yaw {o['yaw_max']:.0f} deg max)"
    if scalars_ok and not level:   # travels + stays up by height, but REARS/PITCHES — a lurch, not a clean walk
        return f"LURCHES (pitch {o['pitch_max']:.0f} / roll {o['roll_max']:.0f} deg — rears/rocks, not a clean walk)"
    if up < 0.6 or hr < 0.6:
        return "CROUCH (low/unstable stance)"
    if cad < 1.0 or sup < 0.25:
        return "SLIDE (feet barely lift / no real stepping)"
    return "FORWARD BUT SHORT / not a credible walk"
