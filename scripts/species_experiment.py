"""Experiment: compose a range of NEW robot species from blocks, build + evaluate each in real physics,
auto-place them into a fresh species tree, and report a thorough analysis (per-species results + the
self-organized morphology tree + a distance matrix). Drives the "try new species, analyze" loop.

    PYTHONPATH=src python scripts/species_experiment.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import numpy as np
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.morphology_builder import MorphologyBuilder
    from virturoid.services.morphology_composer import compose_robot, compose_working_robot
    from virturoid.services.morphology_embedding import distance, embed_gene
    from virturoid.services.species_discovery import auto_place_species

    rows = []                      # (label, gene, metric_name, metric_value, note)

    def manipulator(label, prompt):
        try:
            res = compose_working_robot(prompt, iterations=2, population=4, scene_count=3)
            g = res["gene"]
            return (label, g, "task_success", res["success_rate"],
                    f"dof={len(g.actuated_joints())} ee={g.end_effector_type} (baseline {res['baseline_success']})")
        except Exception as exc:  # noqa: BLE001
            return (label, None, "task_success", None, f"FAILED: {exc}")

    def mobile(label, prompt):
        g = compose_robot(prompt)
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        d = mujoco.MjData(mj); mujoco.mj_forward(mj, d)
        for _ in range(400):
            mujoco.mj_step(mj, d)
        x0, y0 = float(d.qpos[0]), float(d.qpos[1])
        for _ in range(800):
            d.ctrl[:] = mj.actuator_forcerange[:, 1]
            mujoco.mj_step(mj, d)
        disp = ((float(d.qpos[0]) - x0) ** 2 + (float(d.qpos[1]) - y0) ** 2) ** 0.5
        return (label, g, "drive_m", round(disp, 3), f"dof={len(g.actuated_joints())} stable={bool(np.all(np.isfinite(d.qpos)))}")

    def quadruped(label):
        b = MorphologyBuilder("quadruped", base_mount="free", gene_id="exp_quad").base(
            "torso", shape="box", length=0.08, radius=0.16, mass=4.0)
        for i, (sx, sy) in enumerate([(0.12, 0.13), (0.12, -0.13), (-0.12, 0.13), (-0.12, -0.13)]):
            b.limb(f"leg{i}", [dict(length=0.12, axis=(0, 1, 0), torque=8.0),
                               dict(length=0.12, axis=(0, 1, 0), torque=6.0)],
                   parent="torso", mount_offset=(sx, sy, -0.08))
        g = b.build()
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        d = mujoco.MjData(mj); mujoco.mj_forward(mj, d)
        for _ in range(400):
            mujoco.mj_step(mj, d)
        return (label, g, "stand_z", round(float(d.qpos[2]), 3),
                f"dof={len(g.actuated_joints())} stable={bool(np.all(np.isfinite(d.qpos)))}")

    print("Composing + evaluating new species (real physics)...\n")
    rows.append(manipulator("3dof-gripper-arm", "tabletop arm to sort red and blue blocks into bins"))
    rows.append(manipulator("6dof-industrial-arm", "a precise 6-dof industrial arm for assembly"))
    rows.append(manipulator("7dof-franka-arm", "a Franka-style 7-dof arm for precise insertion"))
    rows.append(manipulator("dexterous-hand-arm", "pick up delicate irregular fruit without crushing"))
    rows.append(manipulator("spray-arm", "spray paint a panel with 0.8 m reach"))
    rows.append(mobile("mobile-rover", "a mobile base to deliver parts indoors"))
    rows.append(quadruped("quadruped"))

    print(f"{'species':22} {'class':11} {'metric':13} {'value':>7}  note")
    print("-" * 92)
    for label, g, mname, mval, note in rows:
        cls = g.robot_class if g else "?"
        print(f"{label:22} {cls:11} {mname:13} {str(mval):>7}  {note}")

    # Auto-organize into a fresh species tree + analyze the structure.
    print("\nSelf-organized species tree (by morphology):")
    with MemoryDB(Path(tempfile.mkdtemp()) / "exp.db") as db:
        placed = []
        for label, g, *_ in rows:
            if g is None:
                continue
            pl = auto_place_species(g, db)
            placed.append((label, g, pl))
            print(f"  {label:22} -> {pl['action']:10} {pl['species_pattern']:28} "
                  f"parent={pl['parent']} dist={pl['distance']}")

    # Morphology distance matrix (how the species relate in the vector space).
    genes = [(l, g) for l, g, *_ in rows if g is not None]
    print("\nMorphology distance matrix:")
    labels = [l[:10] for l, _ in genes]
    print(" " * 12 + " ".join(f"{l:>10}" for l in labels))
    for la, ga in genes:
        ds = [distance(embed_gene(ga), embed_gene(gb)) for _, gb in genes]
        print(f"{la[:10]:>12}" + " ".join(f"{d:>10.2f}" for d in ds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
