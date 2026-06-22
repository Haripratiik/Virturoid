"""Generate a real (gene -> measured MuJoCo success) dataset for the GNN surrogate (Phase C).

Samples gene variants (varying limb lengths, torques, arm mount) from a seed gene, evaluates
each in REAL MuJoCo pick-and-place on robot-matched scenes, and writes JSONL rows
{gene:{...}, success_rate:float}. This is the genuine flywheel data the GNN learns from.

Pure MuJoCo (CPU) — run it on the idle PC to build a dataset, e.g.:
    python scripts/generate_surrogate_data.py --base humanoid --n 60 --scenes 4 \
        --out build/data/humanoid_surrogate.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _variant(base, rng):
    ls = rng.uniform(0.85, 1.2)            # global limb-length scale
    ts = rng.uniform(0.8, 2.2)             # torque scale
    pitch = rng.uniform(0.0, 1.6)          # arm mount pitch
    pitched = False
    segs = []
    for s in base.segments:
        ns = dataclasses.replace(s)
        if s.joint_type in ("revolute", "prismatic"):
            ns.length_m = round(s.length_m * ls, 5)
            if s.actuator_torque_nm:
                ns.actuator_torque_nm = round(s.actuator_torque_nm * ts, 2)
            if not pitched and abs(s.joint_axis[1]) > 0.5:
                ns.mount_euler = (s.mount_euler[0], round(pitch, 3), s.mount_euler[2])
                pitched = True
        elif s.parent is not None:
            ns.length_m = round(s.length_m * ls, 5)
        segs.append(ns)
    return dataclasses.replace(base, id=f"{base.id}_s{rng.randint(0, 1_000_000)}", segments=segs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate a MuJoCo-labeled surrogate dataset.")
    ap.add_argument("--base", choices=["arm", "humanoid"], default="humanoid")
    ap.add_argument("--n", type=int, default=40, help="number of gene variants to evaluate")
    ap.add_argument("--scenes", type=int, default=4)
    ap.add_argument("--out", default="build/data/surrogate.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
    from virturoid.services.gene_build import evaluate_gene_pick_place, generate_reachable_scenes

    base = humanoid_upper_body_gene() if args.base == "humanoid" else tabletop_arm_gene()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            g = _variant(base, rng)
            if g.validate():
                continue
            scenes = generate_reachable_scenes(g, count=args.scenes, seed=args.seed)
            try:
                r = evaluate_gene_pick_place(g, scenes)
            except Exception as exc:  # noqa: BLE001 - skip variants that fail to compile/sim
                print(f"[{i}] skipped: {exc}")
                continue
            f.write(json.dumps({"gene": g.to_dict(), "success_rate": r["success_rate"],
                                "blocks_placed": r["blocks_placed"]}) + "\n")
            n_written += 1
            print(f"[{i + 1}/{args.n}] {g.id} success={r['success_rate']:.0%}")
    print(f"\nwrote {n_written} labeled rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
