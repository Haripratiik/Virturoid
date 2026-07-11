"""In-place stance repair for a legged body — the measured R1 lever (flywheel_breakthrough_plan §1.3, §3.M).

The DOMINANT reason LLM-authored quadrupeds fail is morphological, not the flywheel: a narrow stance / missing
lateral splay → the body ROLLS OVER (measured: the demo dog at 82°). No banked gait prior fixes a stance.

The fix is NOT `ensure_walkable_quad(force=True)` — that discards the LLM's body and swaps in a *generic* fanned
quad (a silent template substitution the architecture forbids). Instead this WIDENS THE BODY'S OWN STANCE:
it adds lateral roll to each leg's proximal segment so the feet splay outward, keeping EVERY authored part
(geometry, part count, proportions). Measured: this alone converted fall→walk on 2/4 rolling bodies (small dog
0.17→0.68 m, heavy quad 0.25→0.53 m) while being a no-op on an already-walking body.

It is GATED: a few splay angles are physics-evaluated and the widened body is adopted ONLY when it earns a
*better* verdict than as-authored — so a wrong splay axis on an odd body simply isn't kept.

HONEST SCOPE (measured, §5d): this was wired into ``create_robot``/``submit_design`` and then REVERTED — the
product-path walk-rate lift was **0/5**. Two measured reasons: (1) the composer already fans offline bodies via
``ensure_walkable``, so extra splay is redundant-to-over-fanning there; (2) the dominant remaining failure is
fore-aft **LURCH (rearing/pitching)**, which lateral splay cannot fix — that needs active balance (learned
control), not geometry. The gate here (default crawl gait) also disagrees with the deploy verdict (flywheel-hint
gait), so a "win" under one controller degraded under the other. The module is retained as a **factory
verify-build helper** (where the search gate and the deploy controller are the same), NOT a hot-path repair.
"""
from __future__ import annotations

import math
from dataclasses import replace

DEFAULT_GAIT = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
SPLAY_ANGLES_DEG = (0.0, 15.0, 25.0, 35.0)   # 0 = as-authored (the incumbent the repair must beat)


def _leg_sides(gene) -> dict:
    """Map each root-child (limb-proximal) segment to its lateral side (+1 left / -1 right) from the COMPILED
    world-y of its body — robust to naming/convention, so the splay pushes each leg AWAY from the centreline."""
    root = gene.root() if hasattr(gene, "root") else None
    if root is None:
        return {}
    sides: dict = {}
    try:
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene)))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for child in gene.children_of(root.name):
            try:
                y = float(d.xpos[m.body(child.name).id][1])
                sides[child.name] = 1.0 if y >= 0 else -1.0
            except Exception:  # noqa: BLE001
                sides[child.name] = 1.0
    except Exception:  # noqa: BLE001 - fall back to the authored mount-offset sign
        for child in gene.children_of(root.name):
            sides[child.name] = 1.0 if (getattr(child, "mount_offset", (0, 0, 0))[1] >= 0) else -1.0
    return sides


def widen_stance(gene, roll_deg: float, sides: dict | None = None):
    """Return a copy of ``gene`` with each leg's proximal segment rolled ``roll_deg`` outward (feet splay wider).
    Only the leg-proximal segments' mount rotation changes; all parts/geometry/topology are preserved."""
    root = gene.root() if hasattr(gene, "root") else None
    if root is None or roll_deg == 0.0:
        return gene
    sides = sides if sides is not None else _leg_sides(gene)
    out = []
    for s in gene.segments:
        if s.name in sides and s.parent == root.name:
            me = list(s.mount_euler)
            me[0] = me[0] + sides[s.name] * math.radians(roll_deg)   # roll about fore-aft axis -> lateral splay
            out.append(replace(s, mount_euler=tuple(me)))
        else:
            out.append(replace(s))
    return replace(gene, segments=out)


def _walk_score(gene, *, steps: int) -> tuple:
    """(credible_bool, forward_m) under the default crawl — the gate's un-gameable metric."""
    from virturoid.services.gait_search import evaluate_gait
    r = evaluate_gait(gene, dict(DEFAULT_GAIT), steps=steps)
    fwd = float(r.get("forward", 0.0))
    cred = bool(r.get("survived")) and fwd >= 0.25 and float(r.get("height_ratio", 0.0)) >= 0.55
    return (1 if cred else 0, round(fwd, 3))


def repair_stance(gene, *, angles=SPLAY_ANGLES_DEG, steps: int = 600) -> tuple:
    """GATED in-place stance repair. Returns ``(gene_out, report)``. ``gene_out`` is the widened body ONLY if a
    splay earns a strictly better verdict than as-authored (0°); otherwise the original gene is returned unchanged.
    Best-effort + bounded (len(angles) short rollouts); a non-legged body is returned untouched."""
    from virturoid.services.task_matched_eval import robot_kind
    try:
        if robot_kind(gene) != "legged":
            return gene, {"applied": False, "reason": "not a legged body"}
    except Exception:  # noqa: BLE001
        return gene, {"applied": False, "reason": "kind unavailable"}
    try:
        sides = _leg_sides(gene)
        base = _walk_score(gene, steps=steps)
        best, best_gene, best_deg = base, gene, 0.0
        for deg in angles:
            if deg == 0.0:
                continue
            cand = widen_stance(gene, deg, sides)
            sc = _walk_score(cand, steps=steps)
            if sc > best:                                    # strictly better verdict (credible first, then forward)
                best, best_gene, best_deg = sc, cand, deg
    except Exception as exc:  # noqa: BLE001 - a repair failure must never sink the build
        return gene, {"applied": False, "reason": f"{type(exc).__name__}"}
    applied = best_deg != 0.0
    report = {"applied": applied, "roll_deg": best_deg,
              "forward_before_m": base[1], "forward_after_m": best[1],
              "credible_before": bool(base[0]), "credible_after": bool(best[0])}
    report["note"] = (f"widened the stance +{best_deg:.0f}° per side to stop roll-over — a physics-required "
                      f"adjustment; every authored part is kept (forward {base[1]:.2f}→{best[1]:.2f} m)"
                      if applied else "stance is already walkable as authored (no change)")
    return best_gene, report
