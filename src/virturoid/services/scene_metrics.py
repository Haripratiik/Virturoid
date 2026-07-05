"""Scene diversity + validity metrics (scene-gen plan S7). The practical, simulator-cheap metrics the layout
literature uses, so a scene-quality regression is caught nightly rather than after a wasted training run:

- **Col_obj / Col_scene** (PhyScene): fraction of objects / scenes with an inter-object footprint collision.
- **R_out**: fraction of objects outside the scene bounds.
- **settle_pass_rate**: fraction of scenes whose objects stay physically stable (S4 settle gate).
- **structural_diversity**: fraction of DISTINCT structures in a family (the anti-overfitting metric — a family of
  seed-jitter clones scores ~0; a genuinely varied family approaches 1) + a description-level Self-BLEU proxy.
- **valid_rate**: fraction passing the full S4 physical-validity stack.

Aggregated over a SceneFamily so CI can assert thresholds. Pure-CPU (numpy; MuJoCo only for the optional settle
component, which degrades gracefully when absent)."""

from __future__ import annotations

import numpy as np

from virturoid.services.scene_validity import settle_gate, validate_scene_physical


def _footprint(o):
    hx = (o.size_xyz[0] / 2) if o.size_xyz else 0.03
    hy = (o.size_xyz[1] / 2) if o.size_xyz else 0.03
    return o.pose_xyz_rpy[0], o.pose_xyz_rpy[1], hx, hy


def collision_metrics(scene) -> dict:
    """Col_obj = fraction of (free/prop) objects overlapping another in the xy footprint; collides = any overlap.
    Walls/floor are excluded (a bin abutting a wall is fine)."""
    objs = [o for o in scene.objects if o.object_type in ("cube", "box", "container", "obstacle")]
    if len(objs) < 2:
        return {"col_obj": 0.0, "collides": False, "n": len(objs)}
    hit = set()
    for i in range(len(objs)):
        xi, yi, hxi, hyi = _footprint(objs[i])
        for j in range(i + 1, len(objs)):
            xj, yj, hxj, hyj = _footprint(objs[j])
            if abs(xi - xj) < (hxi + hxj) * 0.98 and abs(yi - yj) < (hyi + hyj) * 0.98:
                hit.add(i); hit.add(j)
    return {"col_obj": round(len(hit) / len(objs), 3), "collides": bool(hit), "n": len(objs)}


def out_of_bounds_rate(scene) -> float:
    """R_out: fraction of objects whose center is outside the scene bounds (xy)."""
    if not getattr(scene, "bounds", None):
        return 0.0
    (xmn, ymn, _), (xmx, ymx, _) = scene.bounds
    objs = scene.objects or []
    out = sum(1 for o in objs if not (xmn <= o.pose_xyz_rpy[0] <= xmx and ymn <= o.pose_xyz_rpy[1] <= ymx))
    return round(out / max(1, len(objs)), 3)


def structural_diversity(family) -> dict:
    """The anti-overfitting metric. distinct_frac = distinct structures / total train scenes (1.0 = every scene is
    structurally unique; ~0 = seed-jitter clones). self_bleu proxy = mean pairwise Jaccard of the structure
    descriptor tokens (lower = more diverse)."""
    keys = list(family.train_keys)
    if not keys:
        return {"distinct_frac": 0.0, "self_similarity": 1.0, "n": 0}
    distinct = len(set(keys))
    toks = [set(str(k).replace("(", " ").replace(")", " ").replace(",", " ").split()) for k in keys]
    sims = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            u = toks[i] | toks[j]
            sims.append(len(toks[i] & toks[j]) / max(1, len(u)))
    return {"distinct_frac": round(distinct / len(keys), 3),
            "self_similarity": round(float(np.mean(sims)) if sims else 0.0, 3), "n": len(keys)}


def settle_pass_rate(scenes) -> float:
    """Fraction of scenes whose objects settle stably (S4 settle gate). Skips gracefully if MuJoCo is absent."""
    if not scenes:
        return 1.0
    res = [settle_gate(s) for s in scenes]
    checked = [r for r in res if "skipped" not in r]
    if not checked:
        return 1.0
    return round(sum(r["ok"] for r in checked) / len(checked), 3)


def run_scene_ci(tasks=("pick_place_sort", "stack", "navigation"), *, n_train: int = 8, n_held_out: int = 3,
                 seed: int = 0, run_settle: bool = False,
                 thresholds=None) -> dict:
    """Nightly scene-quality gate: generate a family per task and assert the quality thresholds a regression would
    break — structural diversity (not seed-clones), a disjoint held-out split, mostly-valid scenes, low collision
    / out-of-bounds. Returns a per-task report + an overall pass, for the CI matrix + readiness ledger."""
    from virturoid.services.scene_family import generate_family
    th = {"min_distinct_frac": 0.8, "min_valid_rate": 0.7, "max_mean_col_obj": 0.15,
          "max_mean_r_out": 0.05, **(thresholds or {})}
    rows, ok = [], True
    for t in tasks:
        fam = generate_family(t, n_train=n_train, n_held_out=n_held_out, seed=seed)
        m = family_metrics(fam, run_settle=run_settle)
        passed = (m["disjoint_split"] and m["structural_diversity"]["distinct_frac"] >= th["min_distinct_frac"]
                  and m["valid_rate"] >= th["min_valid_rate"] and m["mean_col_obj"] <= th["max_mean_col_obj"]
                  and m["mean_r_out"] <= th["max_mean_r_out"])
        ok = ok and passed
        rows.append({**m, "passed": bool(passed)})
    return {"ok": bool(ok), "thresholds": th, "tasks": rows,
            "summary": f"{sum(r['passed'] for r in rows)}/{len(rows)} task scene-families pass quality gates"}


def family_metrics(family, *, robot_radius: float = 0.2, run_settle: bool = True) -> dict:
    """Aggregate scene-quality report for a whole family: diversity, mean collision/out-of-bounds, settle-pass,
    and the full-validity pass rate over the TRAIN pool. This is the dict CI asserts thresholds on."""
    train = family.train
    div = structural_diversity(family)
    col = float(np.mean([collision_metrics(s)["col_obj"] for s in train])) if train else 0.0
    rout = float(np.mean([out_of_bounds_rate(s) for s in train])) if train else 0.0
    valid = [validate_scene_physical(s, robot_radius=robot_radius, run_settle=run_settle)["ok"] for s in train]
    return {"task_type": family.task_type, "n_train": len(train), "n_held_out": len(family.held_out),
            "disjoint_split": family.disjoint,
            "structural_diversity": div, "mean_col_obj": round(col, 3), "mean_r_out": round(rout, 3),
            "settle_pass_rate": settle_pass_rate(train) if run_settle else None,
            "valid_rate": round(sum(valid) / max(1, len(valid)), 3)}
