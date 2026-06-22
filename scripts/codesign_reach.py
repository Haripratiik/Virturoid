"""WS-D (co-design the body): amend the arm so its end-effector can actually CONTACT a tabletop box.

CPU-isolated finding: the seed tabletop arm can only bring its ee ~0.08-0.12 m from a tabletop box
(outside contact range), so no controller/residual can push it — the BODY is the limit (skill_milestone_plan.md
WS-D, decision tree). This runs the platform's reasoned-redesign (`design_critic`) directed at
IK-contact-feasibility: try the `lengthen_reach` amendment over a range and keep the body whose FK-IK
reaches the most of the workspace within the contact margin. Offline (CPU FK-IK), fast.

    PYTHONPATH=src python scripts/codesign_reach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MARGIN = 0.06          # ee must reach within this of a tabletop point to contact/push the box
WORKSPACE = ([0.24, -0.14], [0.42, 0.14])
GRID = 4               # GRID x GRID tabletop points to test reachability over


def ik_contact_feasibility(gene) -> float:
    """Fraction of workspace tabletop points the arm's FK-IK can reach within MARGIN (CPU)."""
    import numpy as np
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    box = SceneObject("box", "cube", (0.33, 0.0, 0.05, 0, 0, 0), mass_kg=0.05, material="gray_block",
                      friction=1.0, scale=1.0)
    try:
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [box]))
    except Exception:  # noqa: BLE001 - invalid/uncompilable gene -> infeasible
        return 0.0
    box_z = 0.05
    (xlo, ylo), (xhi, yhi) = WORKSPACE
    xs = np.linspace(xlo, xhi, GRID); ys = np.linspace(ylo, yhi, GRID)
    reached = 0; total = 0
    for x in xs:
        for y in ys:
            total += 1
            _, best = plan_joint_targets(mj, (float(x), float(y), box_z), iterations=6, candidates=40)
            if best < MARGIN:
                reached += 1
    return reached / total


def reachability_map(gene):
    """Print best ee->point distance over a fine grid for the seed arm -> shows the reachable region."""
    import numpy as np
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    box = SceneObject("box", "cube", (0.33, 0.0, 0.05, 0, 0, 0), mass_kg=0.05, material="gray_block", friction=1.0, scale=1.0)
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [box]))
    xs = np.linspace(0.24, 0.42, 7); ys = np.linspace(-0.14, 0.14, 7)
    print("reachability map (best ee->point dist, m; * = contactable <0.06):")
    print("       " + "  ".join(f"x={x:.2f}" for x in xs))
    feas_xy = []
    for y in ys:
        row = []
        for x in xs:
            best = plan_joint_targets(mj, (float(x), float(y), 0.05), iterations=6, candidates=40)[1]
            row.append(best)
            if best < 0.06:
                feas_xy.append((float(x), float(y)))
        print(f"y={y:+.2f} " + "  ".join((f"*{b:.2f}" if b < 0.06 else f" {b:.2f}") for b in row))
    if feas_xy:
        fx = [p[0] for p in feas_xy]; fy = [p[1] for p in feas_xy]
        print(f"reachable region: x in [{min(fx):.2f},{max(fx):.2f}]  y in [{min(fy):.2f},{max(fy):.2f}]  "
              f"({len(feas_xy)} pts)")


def main() -> int:
    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import kinematic_reach, lengthen_reach

    seed = tabletop_arm_gene()
    reachability_map(seed)
    base_feas = ik_contact_feasibility(seed)
    print(f"\nseed: reach={kinematic_reach(seed):.3f}m  contact-feasibility={base_feas:.0%}")

    best_gene, best_feas, best_factor = seed, base_feas, 1.0
    for factor in (1.15, 1.3, 1.5, 1.7, 2.0, 2.4):
        cand = lengthen_reach(seed, factor=factor, new_id=f"arm_reach_x{factor}",
                              species=f"manipulator.reacharm.x{factor}")
        feas = ik_contact_feasibility(cand)
        print(f"  lengthen_reach x{factor}: reach={kinematic_reach(cand):.3f}m  contact-feasibility={feas:.0%}")
        if feas > best_feas:
            best_gene, best_feas, best_factor = cand, feas, factor

    print(f"\nBEST: lengthen_reach x{best_factor}  reach={kinematic_reach(best_gene):.3f}m  "
          f"contact-feasibility={best_feas:.0%}  (seed was {base_feas:.0%})")
    out = Path("build") / "codesign" / "reacharm_gene.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps(best_gene.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
