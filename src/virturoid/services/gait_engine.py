"""GEN-2 (docs/generality_plan.md) — ONE wave-gait engine for ANY morphology, over the GEN-1 AppendageMap.

Replaces the three hard-coded gait code paths (quad diagonal trot, quad-only crawl, hexapod tripod special
case) with a single symmetry-group formula (Golubitsky/Stewart/Collins; Wilson wave gait; McGhee & Frank
static stability). For N legs at fore-aft stations, side L/R, duty factor β:

    phi_leg = (segment_rank * (1 - β)  +  0.5 * is_right)   mod 1

VERIFIED (docs/generality_plan.md §3): β=0.5 → quad diagonal trot {0,π,π,0} AND hexapod alternating tripod;
β=(N-1)/N → the crawl/wave that lifts one leg at a time (3+ always planted → statically stable). ONE
parameter spans every gait, and β is CHOSEN from the static-stability margin of the actual foot positions —
so a wide/many-legged body gets a fast trot/tripod and a tippy few-legged body gets a stable crawl, computed
per body, never a per-class recipe. A limbless body (snake) gets a serpenoid travelling wave down its spine.

Output matches ``_trot_cpg_tokens``: per-token ``(amp, phase)`` arrays + a gate, so ``recipe_rollout_morph``
drives it unchanged and it slots in as that function's GENERAL fallback (the quad/biped name-paths stay
byte-identical for banked policies; only bodies the name-regex MISSED — hexapod/spider/snake — reach here).
"""
from __future__ import annotations


def select_duty(amap, model) -> float:
    """Pick the duty factor β from static stability, generalizing the fan+crawl finding to any body.
    Many legs (>=5) → alternating tripod/wave is statically stable at β=0.5. A quad/tripod is chosen by its
    STANCE RATIO (support-polygon half-width / CoM height — the 0.22-tippy vs 0.44-stable measurement): a wide
    body trots (β=0.5), a narrow one crawls (β=(N-1)/N, one leg up at a time). Biped stays β=0.5 (needs
    learned balance regardless)."""
    import numpy as np

    n = amap.n_legs
    if n >= 5:
        return 0.5                                          # tripod/wave: >=3 feet down at β=0.5, statically stable
    if n <= 2:
        return 0.5
    # quad / tripod: measure the lateral stance ratio from the actual foot positions + CoM height
    try:
        data = _forward(model)
        com = np.average(np.asarray(data.xipos[1:]), axis=0, weights=np.asarray(model.body_mass[1:]))
        fy = [p[1] for p in amap.foot_xy()]
        half_width = (max(fy) - min(fy)) / 2.0
        ratio = half_width / max(1e-3, float(com[2]))
    except Exception:  # noqa: BLE001
        ratio = 0.0
    return 0.5 if ratio >= 0.30 else (n - 1) / n            # wide -> trot; narrow -> crawl


def _forward(model):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return data


def _segment_ranks(legs):
    """Rear→front station rank per leg (legs at the same fore-aft x share a station). Uses the world tip x."""
    xs = sorted({round(lg.tip_xy[0], 2) for lg in legs})     # distinct fore-aft stations, rear (low x)→front
    rank_of = {x: i for i, x in enumerate(xs)}
    return [rank_of[round(lg.tip_xy[0], 2)] for lg in legs]


def wave_gait(model, graph, params, *, duty: float | None = None):
    """General wave gait for the body's AppendageMap. Returns ``(amp, phase, gate)`` in graph-token order —
    the exact contract of ``_trot_cpg_tokens`` — plus sets nothing else. Legs get the symmetry-formula phases;
    a spine gets a serpenoid wave; anything with no drivable appendage returns ``gate=False`` (scalar recipe)."""
    import numpy as np

    from virturoid.services.appendage_map import build_appendage_map
    n = graph.n_tokens
    amp = np.zeros(n); phase = np.zeros(n)
    amap = build_appendage_map(model)
    th_a, ca_a, ca_ph = params["thigh_amp"], params["calf_amp"], params["calf_phase"]
    two_pi = 2.0 * np.pi

    # LIMBLESS: a serpenoid travelling wave down the spine's lateral (yaw) joints -> undulation propels it
    if amap.spine is not None and not amap.legs:
        toks = amap.spine.tokens
        if not toks:
            return amp, phase, False
        for i, t in enumerate(toks):
            amp[t] = th_a                                    # every spine joint undulates
            phase[t] = two_pi * (i / max(1, len(toks))) * 2.0   # ~2 spatial wavelengths along the body
        return amp, phase, True

    if not amap.legs:
        return amp, phase, False

    beta = float(duty) if duty is not None else select_duty(amap, model)
    ranks = _segment_ranks(amap.legs)
    for lg, seg in zip(amap.legs, ranks):
        is_right = 1.0 if lg.side < 0 else 0.0
        phi = (seg * (1.0 - beta) + 0.5 * is_right) % 1.0
        for t, role in zip(lg.tokens, lg.roles):
            if role == "stride":
                amp[t] = th_a; phase[t] = two_pi * phi
            elif role == "lift":
                amp[t] = ca_a; phase[t] = two_pi * phi + ca_ph
            # abduction / yaw held at the PD default (amp 0) — lateral balance, not propulsion
    return amp, phase, True
