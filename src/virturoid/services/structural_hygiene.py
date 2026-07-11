"""Structural hygiene — penetration/gap budgets as a MEASUREMENT (master_plan_v6 WS-I).

ISSUES Cat-B (clipping / self-penetration / disconnected pieces) fixes have landed; WS-I turns them into a
STANDING measurement so a regression blocks rather than silently returning. The check is un-gameable and physics-
grounded: compile the body, let it settle briefly under gravity with zero control, then read MuJoCo's own contact
solver for interpenetration between geoms it did NOT already exclude (adjacent links are excluded by construction,
so a remaining negative contact distance is REAL self-clipping). Connectivity is guaranteed by the gene schema
(one root, every parent present, acyclic — ``RobotGene.validate``), so this focuses on the measurable defect:
does the body clip through itself at rest?

Pure/CPU/best-effort: an un-compilable body reports ``ok=False`` with the reason, never a crash.
"""
from __future__ import annotations

# a real robot self-penetrates ~0 at rest; allow a small tolerance for capsule overlap at joints (metres)
DEFAULT_PENETRATION_BUDGET_M = 0.02


def penetration_report(gene, *, settle_steps: int = 60, budget_m: float = DEFAULT_PENETRATION_BUDGET_M) -> dict:
    """Max self-interpenetration depth (m) after a brief settle. ``within_budget`` iff <= ``budget_m``."""
    try:
        import mujoco
        import numpy as np
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        m = mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        for _ in range(max(1, settle_steps)):
            mujoco.mj_step(m, d)
        floor_geoms = {g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE}
        max_pen = 0.0
        n_pen = 0
        for c in d.contact[:d.ncon]:
            if c.geom1 in floor_geoms or c.geom2 in floor_geoms:
                continue                                        # foot-on-floor contact is expected, not clipping
            if c.dist < 0:                                      # negative distance = interpenetration
                depth = float(-c.dist)
                if depth > 1e-4:
                    n_pen += 1
                    max_pen = max(max_pen, depth)
        finite = bool(np.all(np.isfinite(d.qpos)))
        return {"ok": True, "max_penetration_m": round(max_pen, 5), "n_penetrating_contacts": n_pen,
                "budget_m": budget_m, "within_budget": bool(max_pen <= budget_m and finite),
                "finite": finite, "n_geoms": int(m.ngeom)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:160], "within_budget": False}


def connectivity_report(gene) -> dict:
    """Every segment reachable from the single root (the gene schema guarantees it; measured here so a broken
    importer/compiler that produced a floating piece is caught)."""
    issues = gene.validate()
    root = gene.root() if hasattr(gene, "root") else None
    reachable = set()
    if root is not None:
        stack = [root.name]
        while stack:
            nm = stack.pop()
            if nm in reachable:
                continue
            reachable.add(nm)
            stack.extend(c.name for c in gene.children_of(nm))
    n = len(gene.segments)
    disconnected = [s.name for s in gene.segments if s.name not in reachable]
    return {"ok": not issues and not disconnected, "n_segments": n, "n_reachable": len(reachable),
            "disconnected_pieces": disconnected, "schema_issues": issues[:3]}


def hygiene_report(gene, *, budget_m: float = DEFAULT_PENETRATION_BUDGET_M) -> dict:
    """Combined structural-hygiene verdict for one body: penetration budget + connectivity."""
    pen = penetration_report(gene, budget_m=budget_m)
    con = connectivity_report(gene)
    return {"clean": bool(pen.get("within_budget") and con.get("ok")),
            "penetration": pen, "connectivity": con}
