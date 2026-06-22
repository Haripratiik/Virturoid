"""Parallel co-design search for a working humanoid (uses the idle PC's CPU cores).

Our physics is CPU-bound MuJoCo, so the fastest way to crack the from-scratch humanoid is a
broad PARALLEL search over its body+controller parameters — exactly what an idle multi-core
PC with 32 GB RAM is good at (no GPU/WSL2/SSH needed). Each worker compiles a humanoid gene
variant and runs REAL pick-and-place; the best configuration is printed so it can be baked in.

    python scripts/search_humanoid.py --n 300 --workers 8 --scenes 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Search bounds: body (limb scale, torque scale, arm mount pitch) + controller (kp, kd, steps).
_BOUNDS = {"length_scale": (0.8, 1.25), "torque_scale": (0.8, 2.2), "mount_pitch": (0.0, 1.6),
           "kp": (15.0, 80.0), "kd": (2.0, 10.0), "phase_steps": (260, 420)}


def _make_variant(vec):
    from virturoid.fixtures.gene_library import humanoid_upper_body_gene

    base = humanoid_upper_body_gene()
    pitched = False
    segs = []
    for s in base.segments:
        ns = dataclasses.replace(s)
        if s.joint_type in ("revolute", "prismatic"):
            ns.length_m = round(s.length_m * vec["length_scale"], 5)
            if s.actuator_torque_nm:
                ns.actuator_torque_nm = round(s.actuator_torque_nm * vec["torque_scale"], 2)
            if not pitched and abs(s.joint_axis[1]) > 0.5:
                ns.mount_euler = (s.mount_euler[0], round(vec["mount_pitch"], 3), s.mount_euler[2])
                pitched = True
        elif s.parent is not None:
            ns.length_m = round(s.length_m * vec["length_scale"], 5)
        segs.append(ns)
    return dataclasses.replace(base, id="humanoid_search", segments=segs)


def _evaluate(args):  # top-level for Windows 'spawn' picklability
    vec, n_scenes, seed = args
    from virturoid.services.gene_build import evaluate_gene_pick_place, generate_reachable_scenes

    g = _make_variant(vec)
    if g.validate():
        return {"vec": vec, "success": 0.0, "placed": 0, "invalid": True}
    scenes = generate_reachable_scenes(g, count=n_scenes, seed=seed)
    params = {"kp": vec["kp"], "kd": vec["kd"], "phase_steps": int(vec["phase_steps"])}
    try:
        r = evaluate_gene_pick_place(g, scenes, params=params)
    except Exception as exc:  # noqa: BLE001
        return {"vec": vec, "success": 0.0, "placed": 0, "error": str(exc)}
    return {"vec": vec, "success": r["success_rate"], "placed": r["blocks_placed"], "total": r["blocks_total"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Parallel humanoid co-design search.")
    ap.add_argument("--n", type=int, default=200, help="candidates to evaluate")
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = cpu_count-1)")
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="build/data/humanoid_search.json")
    args = ap.parse_args(argv)

    import multiprocessing as mp

    rng = random.Random(args.seed)
    cands = []
    for _ in range(args.n):
        vec = {k: rng.uniform(lo, hi) for k, (lo, hi) in _BOUNDS.items()}
        cands.append((vec, args.scenes, args.seed))

    workers = args.workers or max(1, (mp.cpu_count() or 2) - 1)
    print(f"searching {args.n} humanoid configs on {workers} workers ({args.scenes} scenes each)...")
    results = []
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_evaluate, cands), 1):
            results.append(res)
            if res["success"] > 0:
                print(f"[{i}/{args.n}] success={res['success']:.0%} placed={res.get('placed')} :: {_short(res['vec'])}")
    results.sort(key=lambda r: (r["success"], r.get("placed", 0)), reverse=True)
    best = results[0]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results[:20], indent=2), encoding="utf-8")
    succ = sum(1 for r in results if r["success"] > 0)
    print(f"\n{succ}/{args.n} configs placed >=1 block in a full episode.")
    print(f"BEST success={best['success']:.0%} placed={best.get('placed')}/{best.get('total')}")
    print(f"BEST config: {json.dumps(best['vec'], indent=2)}")
    print(f"(top 20 saved to {args.out} — paste the best vec back and it gets baked into the gene)")
    return 0


def _short(vec):
    return ", ".join(f"{k}={v:.2f}" for k, v in vec.items())


if __name__ == "__main__":
    raise SystemExit(main())
