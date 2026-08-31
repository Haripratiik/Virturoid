"""Honest gait quality: turn an anti-Goodhart rollout's metrics into an explicit WALK/CROUCH/SLIDE/FELL verdict.

This lives in the PACKAGE (not ``scripts/``) because the product's legged motion verdict (``ai_native_tools``)
depends on it — importing it from ``scripts/`` broke the flagship path in any real deployment (installed package
or an MCP server launched from another cwd, where the repo root isn't on ``sys.path``, so ``import scripts`` fails
with ModuleNotFoundError and every legged robot silently read as "could not simulate"). ``scripts/verify_gait.py``
now re-exports these for back-compat.

A credible WALK requires ALL of: survived (didn't fall) + upright at the body's true stance height (not a crouch)
+ real foot-lift cadence + genuine stepping support + forward travel + A RATE THAT HOLDS + A COURSE IT HELD. A body
that stands still scores cadence 0 -> SLIDE; a low unstable body scores low upright_frac -> CROUCH; a body that
covers ground and then stops scores a decaying travel rate -> DRIFTS THEN STALLS; a body that goes round in a
circle scores a swung heading / a path much longer than its net displacement -> WALKS IN A CIRCLE; a fall is named
by its dominant orientation mode.
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


# ---------------------------------------------------------------------------------------------------------
# COURSE: did it go somewhere, or did it go round?
# ---------------------------------------------------------------------------------------------------------
#: SUSTAINED heading deviation from the heading it set out on, in degrees. 90 is not a taste parameter: at a
#: right angle the body is travelling PERPENDICULAR to where it started, so progress along its original heading
#: has stopped, and past it the body is coming back. Below 90 every further step still increases how far it has
#: got from the start along that heading; above, more time buys no more distance. That is exactly the property
#: "walked N metres" is claiming, and the property a circle does not have.
_CLEAN_HEADING_DEV = 90.0
#: ...and the single worst frame keeps an absolute ceiling at 2x, the same rule (and the same reason) as
#: ``_HARD_ROLL_MAX``/``_HARD_PITCH_MAX``: reading a percentile must not let a body that turned right around
#: through 180 deg pass on the technicality that it did not stay there.
_HARD_HEADING_DEV = 180.0
#: net displacement / ground covered. 0.5 = "you walked twice as far as you got". A perfect circle scores 0 (it
#: ends where it began), a half-circle 0.64, a straight walk ~1. MEASURED on this checkout: the straight
#: hexapod scores 0.969 and pays only 3% to intra-stride body sway, and the WORST credible row in the whole
#: re-simulated bank (83 of the 102 locomotion rows, at 800/1500/6000 steps each) scores 0.725 — so this gate
#: sits 1.45x below the corpus it must not break. It is also what catches a body that mills about without its
#: heading ever swinging far (measured: the template quad under a hard steer keeps heading dev at 35 deg and
#: still scores 0.370), which the heading gate alone never sees.
_MIN_STRAIGHTNESS = 0.5
#: ...and below this much ground covered, ``straightness`` is 0/0 and the question is UNANSWERABLE, so it is
#: reported as ``None`` and the milling gate does not fire. Same rule ``settling`` follows: never fail a body for
#: a question its own trace cannot answer. 0.05 m is far below the 0.3 m ``forward`` gate, so no rollout whose
#: scalars pass can land here — ``path_m >= |net_m| >= |forward|`` by construction. What DOES land here is a
#: metrics dict carrying an orientation-only trace (several diagnostics build one), and before this floor existed
#: such a dict read as MILLS AROUND on a path of exactly 0.00 m.
_MIN_PATH_M = 0.05


def course_summary(r: dict) -> dict | None:
    """WHERE DID IT GO — net displacement, ground actually covered, and how far the heading swung from its start.

    THE DEFECT THIS EXISTS FOR (measured 2026-08-08, task D5). ``morph_policy``'s ``forward`` is world-frame
    delta-x, and until this function existed nothing else about the PATH reached the verdict: a body that walked a
    2.2 m circle and came back past its own start line reported ``forward 1.582 m`` and printed CREDIBLE WALK,
    with ``yaw_max 179.4 deg`` computed and used only to name a fall mode on a body that had already failed.
    Delta-x is not wrong — it is the honest projection on the start heading, and it is BOUNDED for a circle — but
    the 0.3 m gate it is tested against is trivially cleared by any loop wider than that.

    That ``yaw_max 179.4`` is itself worth a sentence, because it is the smaller half of the same bug.
    ``orientation_summary``'s yaw comes from ``arctan2``, so it is the heading MODULO a half-turn and SATURATES at
    180: it cannot count a full revolution. Integrated properly, the same episode swings **440 deg** — it goes
    round MORE THAN ONCE — so even the diagnostic number that did exist understated the defect by 2.5x, and no
    threshold placed on it could ever have separated "turned 179" from "turned 719".

    Two numbers, because there are two ways to not-go-anywhere and neither implies the other:

      * ``heading_dev_p95_deg`` — the SUSTAINED unwrapped heading deviation from the start heading. Catches
        turning. P95 and not max for the reason ``orientation_summary`` gives at length: a max is monotone in the
        number of frames, and a body's heading oscillates within every stride (measured: up to ~30 deg peak on
        banked rows whose net course change is under 20 deg), so a max-based gate would read stride oscillation
        as a course change and get stricter the longer you looked.
      * ``straightness`` — net displacement / path length. Catches milling: a body can wander a long way for
        little net gain while its heading barely swings, and a heading gate alone never sees it.

    Prefers the rollout's OWN per-physics-step accumulation (``path_m``/``heading_dev_p95_deg``, emitted by
    ``crawl_gait_rollout`` since this task) because that is exact and independent of ``frame_every``; falls back
    to integrating the recorded ``qpos_frames`` so a legacy result, a banked row's replay, a serpentine run or a
    learned-policy rollout is auditable too. MEASURED, the two agree to 0.3% (straight hexapod 0.969 per-step vs
    0.971 off a 300-frame trace), and straightness is flat to within 1% from 60 samples up to every step.

    ``None`` when the rollout recorded neither — never a guess.
    """
    import numpy as np
    if r.get("path_m") is not None and r.get("heading_dev_p95_deg") is not None:
        path = float(r["path_m"]); net = float(r.get("net_m") or 0.0)
        return {"net_m": round(net, 3), "path_m": round(path, 3),
                "straightness": round(net / path, 3) if path >= _MIN_PATH_M else None,
                "heading_dev_p95_deg": float(r["heading_dev_p95_deg"]),
                "heading_dev_max_deg": float(r.get("heading_dev_max_deg") or r["heading_dev_p95_deg"]),
                "source": "rollout"}
    frames = r.get("qpos_frames") or []
    if len(frames) < 3:
        return None
    P = np.asarray([[float(f[0]), float(f[1])] for f in frames])
    Q = np.asarray([[float(f[3]), float(f[4]), float(f[5]), float(f[6])] for f in frames])
    yaw = np.arctan2(2 * (Q[:, 0] * Q[:, 3] + Q[:, 1] * Q[:, 2]), 1 - 2 * (Q[:, 2] ** 2 + Q[:, 3] ** 2))
    d = (np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi          # per-frame wrap, then integrate: an UNWRAPPED
    dev = np.abs(np.concatenate([[0.0], np.cumsum(d)]))       # heading, so a full turn reads 360 and not 0
    net = float(np.hypot(*(P[-1] - P[0])))
    path = float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    return {"net_m": round(net, 3), "path_m": round(path, 3),
            "straightness": round(net / path, 3) if path >= _MIN_PATH_M else None,
            "heading_dev_p95_deg": round(float(np.degrees(np.percentile(dev, 95))), 1),
            "heading_dev_max_deg": round(float(np.degrees(np.max(dev))), 1),
            "source": "trace"}


def classify(r: dict) -> str:
    """Explicit, honest verdict from the anti-Goodhart metrics + orientation + SETTLING + COURSE. A CREDIBLE WALK
    requires the scalar gates AND a LEVEL body AND a RATE THAT HOLDS AND A COURSE IT HELD:

    * LEVEL — a gait that clears forward/upright/cadence by REARING or PITCH-DIVING (which games the forward
      metric with a violent lurch, while ``upright_frac`` — a z-height/up-vector check — misses the pitch) is not
      a credible walk. Judged on the SUSTAINED attitude; see ``orientation_summary``.
    * RATE — a body whose travel-per-1000-steps decays has not walked, it has drifted; see ``settling``.
    * COURSE — a body that goes ROUND rather than ANYWHERE has not walked either. ``forward`` is world-frame
      delta-x, so a circular path whose far side happens to sit at +x books that delta-x as travel: measured, the
      offline lynx's fragile operating point walks a 2.2 m loop back past its own start line and reported
      CREDIBLE WALK, 1.582 m. See ``course_summary``.

    All three extra gates are judged only from what the rollout actually recorded: orientation and course need the
    qpos trace (or, for course, the rollout's own per-step accumulation), settling needs a trace of at least
    ``_SETTLE_MIN_STEPS``. A metric-only call keeps the scalar verdict.
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
    c = course_summary(r)
    turned = c is not None and (c["heading_dev_p95_deg"] >= _CLEAN_HEADING_DEV
                                or c["heading_dev_max_deg"] >= _HARD_HEADING_DEV)
    milled = c is not None and c["straightness"] is not None and c["straightness"] < _MIN_STRAIGHTNESS
    on_course = c is None or not (turned or milled)
    scalars_ok = survived and up >= 0.6 and cad >= 1.0 and sup >= 0.25 and fwd >= 0.3
    if scalars_ok and level and holds and on_course:
        return "CREDIBLE WALK"
    # not a credible walk: name the DOMINANT fall mode from the orientation trace (if replayable) so the
    # verdict is diagnostic, not just "bad" — a roll-over reads as CROUCH by height alone, which misleads.
    if o:
        modes = [("ROLL-OVER", o["roll_max"]), ("PITCH-DIVE", o["pitch_max"]), ("YAW-DRIFT", o["yaw_max"])]
        name, val = max(modes, key=lambda t: t[1])
        if not survived and val >= 45.0:
            return f"FELL by {name} (roll {o['roll_max']:.0f} / pitch {o['pitch_max']:.0f} / yaw {o['yaw_max']:.0f} deg max)"
    if scalars_ok and level and not on_course:
        # It is up, it is stepping, it covered ground — and it did not GO anywhere. Reported before the
        # rate/lurch branches because it is the more specific and more actionable diagnosis: a circler usually
        # also fails the rate hold (its net displacement stops growing once it comes back round), and "it walked
        # in a circle" tells an engineer what to fix in a way "it moved, then stopped" does not.
        where = (f"{c['net_m']:.2f} m net for {c['path_m']:.2f} m walked, heading swung "
                 f"{c['heading_dev_p95_deg']:.0f} deg")
        if turned and milled:
            return (f"WALKS IN A CIRCLE ({where} — it came back on itself; `forward` is world-frame "
                    f"displacement and a loop can point part of itself down +x)")
        if turned:
            return (f"TURNS OFF COURSE ({where} — its heading left the direction it set out in, so more "
                    f"steps buy no more distance)")
        return (f"MILLS AROUND ({where} — most of the ground it covered was not progress)")
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
