"""Analytic physics features for a body — the cheap signal the 29-D STRUCTURAL embedding is blind to.

Measured (flywheel_breakthrough_plan §3.I2): at cosine 0.989 the nearest banked body FALLS while the query body
STANDS under the identical gait. The structural embedding cannot see stance/CoM/leg-count differences that decide
walk-vs-fall. The research (§3.E5) says the fix is NOT a learned encoder (overfits at our N) but CLOSED-FORM
physics features that target the stability boundary directly — computable from the compiled MuJoCo model at its
standing pose, with zero training signal:

  * static stability margin  — signed distance from the CoM ground-projection to the foot support polygon
  * tip ratio                — support half-width / CoM height (the classic tip-over ratio)
  * CoM-height / leg-length  — tall-and-tippy vs low-and-stable
  * pendulum natural freq    — sqrt(g / CoM height): a physics prior for gait frequency (Froude/SLIP family)
  * control authority        — sum(actuator peak torque) / (m·g·leg): can the motors move this body at all
  * raw stance/mass/girth    — the axes §3.I2 measured diverging at high cosine

This module ONLY measures; it designs nothing. Pure, deterministic, dependency-light (a hand-rolled 2-D convex
hull, no scipy). Whether these features actually PREDICT walk-vs-fall is an empirical question answered by
``scripts/probe_dynamics_separation.py`` — do not wire them into retrieval until that probe says they beat cosine.
"""
from __future__ import annotations

import math

_G = 9.81
FEATURE_NAMES = ("stability_margin_m", "support_area_m2", "tip_ratio", "com_height_m",
                 "com_over_leg", "pendulum_freq_hz", "control_authority", "n_feet",
                 "total_mass_kg", "mean_leg_len_m", "min_leg_girth_m", "n_limb_chains")


# ------------------------------------------------------------------ 2-D convex hull + signed distance (no scipy)
def _convex_hull(pts: list[tuple]) -> list[tuple]:
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _polygon_area(hull: list[tuple]) -> float:
    if len(hull) < 3:
        return 0.0
    a = 0.0
    for i in range(len(hull)):
        x1, y1 = hull[i]; x2, y2 = hull[(i + 1) % len(hull)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _signed_dist_to_hull(pt: tuple, hull: list[tuple]) -> float:
    """Signed distance from ``pt`` to the polygon boundary: + inside, - outside. Robust to degenerate hulls."""
    if len(hull) < 3:
        # degenerate support (a line/point): distance to the nearest vertex, always negative (unstable)
        return -min(math.hypot(pt[0] - hx, pt[1] - hy) for hx, hy in hull) if hull else -1.0
    px, py = pt
    inside = True
    min_edge = float("inf")
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]; x2, y2 = hull[(i + 1) % n]
        # is pt left of edge (hull is CCW from _convex_hull)?
        cr = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cr < 0:
            inside = False
        # distance from pt to this edge segment
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg2))
        cx, cy = x1 + t * dx, y1 + t * dy
        min_edge = min(min_edge, math.hypot(px - cx, py - cy))
    return min_edge if inside else -min_edge


# ------------------------------------------------------------------ the feature vector
def dynamics_features(gene) -> dict:
    """Closed-form analytic features from the compiled model at its standing pose. Best-effort: a body that will
    not compile returns a conservative all-unstable row rather than raising (so a caller can rank it last)."""
    from virturoid.services.heldout_set import _leg_chain_count
    try:
        import numpy as np
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    except Exception:  # noqa: BLE001
        return {k: (-1.0 if k == "stability_margin_m" else 0.0) for k in FEATURE_NAMES}

    # whole-robot CoM (mass-weighted body positions, skip world body 0)
    masses = m.body_mass[1:]
    xpos = d.xpos[1:]
    total_mass = float(masses.sum()) or 1e-6
    com = (masses[:, None] * xpos).sum(0) / total_mass

    # feet = the geoms whose world-z sits in the lowest band (the ground-contact set), excluding the floor plane
    GT = mujoco.mjtGeom
    body_geoms = [g for g in range(m.ngeom) if int(m.geom_type[g]) != int(GT.mjGEOM_PLANE)]
    if not body_geoms:
        return {k: (-1.0 if k == "stability_margin_m" else 0.0) for k in FEATURE_NAMES}
    gz = np.array([float(d.geom_xpos[g][2] - m.geom_rbound[g]) for g in body_geoms])   # geom bottom
    zmin = float(gz.min())
    feet = [g for g, z in zip(body_geoms, gz) if z < zmin + 0.06]
    foot_xy = [(float(d.geom_xpos[g][0]), float(d.geom_xpos[g][1])) for g in feet]
    hull = _convex_hull(foot_xy)
    margin = _signed_dist_to_hull((float(com[0]), float(com[1])), hull)
    support_area = _polygon_area(hull)

    floor_z = zmin
    com_height = max(1e-3, float(com[2]) - floor_z)
    # characteristic leg length: mean length of actuated segments (the limbs), fallback to body extent
    legs = [s for s in gene.segments if s.joint_type in ("revolute", "prismatic")]
    leg_lens = [float(getattr(s, "length_m", 0.0)) for s in legs] or [0.1]
    mean_leg = sum(leg_lens) / len(leg_lens)
    min_girth = min((float(getattr(s, "radius_m", 0.02)) for s in legs), default=0.02)
    support_halfwidth = math.sqrt(support_area / math.pi) if support_area > 0 else 0.0
    tip_ratio = support_halfwidth / com_height
    tau_sum = float(np.abs(m.actuator_forcerange[:, 1]).sum()) if m.nu else 0.0
    control_authority = tau_sum / (total_mass * _G * max(mean_leg, 0.05))

    return {
        "stability_margin_m": round(margin, 4),
        "support_area_m2": round(support_area, 4),
        "tip_ratio": round(tip_ratio, 3),
        "com_height_m": round(com_height, 4),
        "com_over_leg": round(com_height / max(mean_leg, 1e-3), 3),
        "pendulum_freq_hz": round(math.sqrt(_G / com_height) / (2 * math.pi), 3),
        "control_authority": round(control_authority, 3),
        "n_feet": len(feet),
        "total_mass_kg": round(total_mass, 3),
        "mean_leg_len_m": round(mean_leg, 4),
        "min_leg_girth_m": round(min_girth, 4),
        "n_limb_chains": _leg_chain_count(gene),
    }


def dynamics_vector(gene) -> list[float]:
    """The features as an ordered vector (FEATURE_NAMES order) — for appending to a retrieval key once proven."""
    f = dynamics_features(gene)
    return [float(f[k]) for k in FEATURE_NAMES]
