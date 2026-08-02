"""Honest gait quality: turn an anti-Goodhart rollout's metrics into an explicit WALK/CROUCH/SLIDE/FELL verdict.

This lives in the PACKAGE (not ``scripts/``) because the product's legged motion verdict (``ai_native_tools``)
depends on it — importing it from ``scripts/`` broke the flagship path in any real deployment (installed package
or an MCP server launched from another cwd, where the repo root isn't on ``sys.path``, so ``import scripts`` fails
with ModuleNotFoundError and every legged robot silently read as "could not simulate"). ``scripts/verify_gait.py``
now re-exports these for back-compat.

A credible WALK requires ALL of: survived (didn't fall) + upright at the body's true stance height (not a crouch)
+ real foot-lift cadence + genuine stepping support + forward travel + A RATE THAT HOLDS. A body that stands still
scores cadence 0 -> SLIDE; a low unstable body scores low upright_frac -> CROUCH; a body that covers ground and
then stops scores a decaying travel rate -> DRIFTS THEN STALLS; a fall is named by its dominant orientation mode.
"""
from __future__ import annotations


def orientation_summary(qpos_frames) -> dict:
    """Max + P95 + final roll/pitch/yaw (deg) from the base quaternion trace. The anti-Goodhart signal for a
    LEGGED body's true failure mode: a trot that loses lateral balance ROLLS over, an over-driven gait
    PITCH-dives over its front legs, an asymmetric gait YAW-drifts off a straight line. Height alone can't
    tell these apart (all three end with a low base) — the classifier needs the orientation to name the fall.

    P95 IS HERE BECAUSE A MAX IS NOT COMPARABLE ACROSS HORIZONS, and ``classify``'s horizon stopped being a
    constant the moment the settling gate below was added. ``roll_max``/``pitch_max`` are monotone
    non-decreasing in the number of frames, so a max-based level gate gets strictly stricter the longer you
    look: look long enough at any body and it eventually clips any threshold. MEASURED 2026-08-01 (task #267)
    on the grounded authored hexapod at its own fitted op-point — pitch_max 17.5 (1500 steps) -> 21.5 (6000)
    -> 22.7 (12000) against a 20 deg gate, while its P95 sits at 17.0 and only 22 of 2400 recorded frames ever
    exceed 20 deg, the FIRST of them at step 4280. That body walks 8.14 m dead straight at a steady
    0.68 m / 1000 steps; a horizon change alone would have relabelled it a LURCH. "Rears/rocks" means a
    SUSTAINED attitude, so the gate reads a sustained statistic and keeps the single worst frame only as an
    absolute ceiling (``_HARD_*`` below). Cross-checked on 10 grounded bodies at 800 and 1500 steps: ZERO
    verdict changes at the horizons already in use, so this is inert everywhere except where the max was wrong.
    """
    import numpy as np
    if not qpos_frames:
        return {}
    R: list[float] = []; P: list[float] = []; Y: list[float] = []
    for f in qpos_frames:
        w, x, y, z = (float(f[3]), float(f[4]), float(f[5]), float(f[6]))
        R.append(abs(float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))))
        P.append(abs(float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))))
        Y.append(abs(float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))))
    f = qpos_frames[-1]
    w, x, y, z = (float(f[3]), float(f[4]), float(f[5]), float(f[6]))
    rr = float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))
    pp = float(np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))
    yy = float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))
    return {"roll_max": round(max(R), 1), "pitch_max": round(max(P), 1), "yaw_max": round(max(Y), 1),
            "roll_p95": round(float(np.percentile(R, 95)), 1),
            "pitch_p95": round(float(np.percentile(P, 95)), 1),
            "yaw_p95": round(float(np.percentile(Y, 95)), 1),
            "roll_final": round(rr, 1), "pitch_final": round(pp, 1), "yaw_final": round(yy, 1)}


_CLEAN_ROLL_MAX = 35.0     # a trot/tripod rocks side-to-side some; beyond this it's tipping, not walking
_CLEAN_PITCH_MAX = 20.0    # a clean walker holds its body level; a big pitch = REARING/DIVING to fake forward
                           # (measured: a clean quad trot pitches ~9deg; a hexapod that games `forward` via a
                           # huge-stride lurch pitches ~25deg — un-gameable gate rejects the lurch)
# ...applied to the SUSTAINED (P95) attitude. The single worst frame keeps an absolute ceiling at 2x, so
# reading a percentile cannot let a genuine tip through on the technicality that it was brief.
_HARD_ROLL_MAX = 70.0
_HARD_PITCH_MAX = 40.0

# ---------------------------------------------------------------------------------------------------------
# SETTLING: is it still walking at the end, or did it move and then stop?
# ---------------------------------------------------------------------------------------------------------
#: below this many steps a trace CANNOT show settling, so the gate stays off and the verdict is exactly what it
#: has always been. This is what keeps every short diagnostic rollout in the codebase byte-identical.
_SETTLE_MIN_STEPS = 3000
#: travel-per-1000-steps over the WHOLE run must be at least this fraction of the rate over its first quarter.
#: 0.5 is not a tuned constant — it separates every measured case by a wide margin (task #267): template
#: default 0.92, template searched 1.19, fitted hexapod 1.23, the rate-holding cat found by settling-aware
#: search 0.84 — against the authored dog's adopted point at 0.24 and the authored cat's at 0.20.
_RATE_HOLD_FRAC = 0.5


def travel_rates(r: dict, horizons=None) -> dict:
    """Travel-per-1000-steps at each PREFIX horizon of ONE rollout, read off the recorded base-x trace.

    A single rollout answers "how fast was it still going at step h?" for every h up to its own horizon, because
    ``crawl_gait_rollout`` records the base pose every ``frame_every`` steps and its direction/frequency probe is
    horizon-INDEPENDENT (it always probes ``min(350, steps)``, so the same wave direction and frequency are
    chosen whatever horizon is requested). Verified exact 2026-08-01: the 1500-step prefix of a 6000-step
    grounded-cat rollout gives rate +0.6379, and an independently-run 1500-step rollout gives +0.6379. That is
    why judging at a longer horizon costs ONE longer rollout and not several.

    Returns ``{}`` for any rollout that did not record a trace, or one that predates the ``steps``/``frame_every``
    disclosure — never a guess.
    """
    frames = r.get("qpos_frames") or []
    steps = int(r.get("steps") or 0)
    fe = int(r.get("frame_every") or 0)
    if not frames or steps <= 0 or fe <= 0:
        return {}
    if horizons is None:
        horizons = (max(1, steps // 4), max(1, steps // 2), steps)
    x0 = float(frames[0][0])
    out: dict[int, float] = {}
    for h in horizons:
        h = int(h)
        if h <= 0 or h > steps:
            continue
        i = min(len(frames) - 1, max(0, h // fe - 1))
        out[h] = (float(frames[i][0]) - x0) / (h / 1000.0)
    return out


def settling(r: dict) -> dict | None:
    """Did this body KEEP walking, or move and then stop? ``None`` when the trace is too short to say.

    THE DEFECT THIS EXISTS FOR (measured, task #267). The grounded authored cat's adopted operating point is
    reported as a CREDIBLE WALK at 1500 steps and FALLS at step 2014: the credible walk was the fall happening
    514 steps after we stopped looking. The optimised quantity — NET DISPLACEMENT AT A FIXED HORIZON — measures
    locomotion on the canonical template and DRIFT-BEFORE-FALLING on the animals, and at one horizon the two are
    indistinguishable. Across horizons they are trivially distinguishable, by their RATE PROFILE
    (m per 1000 steps, at 1500 / 2500 / 4000 / 6000):

        template DEFAULT   1.385 -> 1.440 -> 1.395 -> 1.278   CONSTANT   -> +7.67 m, alive at 6000
        template SEARCHED  2.58  -> 2.80  -> 2.99  -> 3.08    CONSTANT   -> +18.50 m, alive
        cat ADOPTED        0.639 -> 0.300 -> 0.188 -> 0.125   1/t DECAY  -> fell at 2014
        dog ADOPTED        0.953 -> 0.836 -> 0.503 -> 0.230   DECAY      -> survives, stops advancing

    A 1/t profile IS the signature of "moved, then stopped": the numerator froze and only the denominator kept
    growing. So the honest question is not "did it travel far" but "was it STILL TRAVELLING at the end", and the
    dog shows why surviving the longer horizon is not on its own enough — it is alive at 6000 and at 12000, and
    its net travel there is +0.05 m because it walked out and wandered back.

    The gate is OFF below ``_SETTLE_MIN_STEPS``: a short rollout cannot demonstrate settling either way, and
    silently failing bodies for a question their trace cannot answer would be a second false verdict.
    """
    steps = int(r.get("steps") or 0)
    if steps < _SETTLE_MIN_STEPS:
        return None
    q = max(1, steps // 4)
    rates = travel_rates(r, horizons=(q, max(1, steps // 2), steps))
    if q not in rates or steps not in rates:
        return None
    early, full = rates[q], rates[steps]
    # A COMPARISON, not a ratio, so the signs behave: a body that started backwards (early < 0) and ended up
    # advancing must PASS, and `full / early` would report that as a large negative number.
    return {"rates": rates, "early_steps": q, "steps": steps, "rate_early": early, "rate_full": full,
            "hold": (full / early if early > 1e-9 else None),
            "holds_rate": bool(full >= _RATE_HOLD_FRAC * early)}


def classify(r: dict) -> str:
    """Explicit, honest verdict from the anti-Goodhart metrics + orientation + SETTLING. A CREDIBLE WALK requires
    the scalar gates AND a LEVEL body AND a RATE THAT HOLDS:

    * LEVEL — a gait that clears forward/upright/cadence by REARING or PITCH-DIVING (which games the forward
      metric with a violent lurch, while ``upright_frac`` — a z-height/up-vector check — misses the pitch) is not
      a credible walk. Judged on the SUSTAINED attitude; see ``orientation_summary``.
    * RATE — a body whose travel-per-1000-steps decays has not walked, it has drifted; see ``settling``.

    Both extra gates are judged only from what the rollout actually recorded: orientation needs the qpos trace,
    settling needs a trace of at least ``_SETTLE_MIN_STEPS``. A metric-only call keeps the scalar verdict, and a
    short rollout keeps exactly the verdict it has always had.
    """
    survived = bool(r.get("survived"))
    up = float(r.get("upright_frac", 0.0)); hr = float(r.get("height_ratio", 0.0))
    cad = float(r.get("cadence", 0.0)); sup = float(r.get("support_frac", 0.0))
    fwd = float(r.get("forward", 0.0))
    o = orientation_summary(r.get("qpos_frames") or [])
    level = (not o) or (o["roll_p95"] < _CLEAN_ROLL_MAX and o["pitch_p95"] < _CLEAN_PITCH_MAX
                        and o["roll_max"] < _HARD_ROLL_MAX and o["pitch_max"] < _HARD_PITCH_MAX)
    s = settling(r)
    holds = s is None or bool(s["holds_rate"])
    scalars_ok = survived and up >= 0.6 and cad >= 1.0 and sup >= 0.25 and fwd >= 0.3
    if scalars_ok and level and holds:
        return "CREDIBLE WALK"
    # not a credible walk: name the DOMINANT fall mode from the orientation trace (if replayable) so the
    # verdict is diagnostic, not just "bad" — a roll-over reads as CROUCH by height alone, which misleads.
    if o:
        modes = [("ROLL-OVER", o["roll_max"]), ("PITCH-DIVE", o["pitch_max"]), ("YAW-DRIFT", o["yaw_max"])]
        name, val = max(modes, key=lambda t: t[1])
        if not survived and val >= 45.0:
            return f"FELL by {name} (roll {o['roll_max']:.0f} / pitch {o['pitch_max']:.0f} / yaw {o['yaw_max']:.0f} deg max)"
    if scalars_ok and level and not holds:  # upright the whole way and never fell — but it stopped advancing
        return (f"DRIFTS THEN STALLS (travel-per-1000 steps fell {s['rate_early']:+.2f} -> {s['rate_full']:+.2f} m "
                f"between step {s['early_steps']} and {s['steps']} — it moved, then stopped)")
    if scalars_ok and not level:   # travels + stays up by height, but REARS/PITCHES — a lurch, not a clean walk
        return (f"LURCHES (pitch {o['pitch_p95']:.0f} / roll {o['roll_p95']:.0f} deg sustained — "
                f"rears/rocks, not a clean walk)")
    if up < 0.6 or hr < 0.6:
        return "CROUCH (low/unstable stance)"
    if cad < 1.0 or sup < 0.25:
        return "SLIDE (feet barely lift / no real stepping)"
    return "FORWARD BUT SHORT / not a credible walk"
