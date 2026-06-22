"""Local CPU isolation of the operational-space PD base (debug tool for mjx_residual_push.py).

The GPU residual base wasn't bringing the ee to the box (reach pinned ~1.45 regardless of kp). Blind
GPU smokes are too slow to debug a control law. This runs the SAME OSC-PD base on one CPU MuJoCo env
with the known-correct mj_jacSite ((3, nv)) and prints ee->box distance over time. If this drives the
ee onto the box, the law is right and the MJX port (mjx.jac semantics / dof mapping) is the bug.

    PYTHONPATH=src python scripts/osc_base_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import numpy as np
    import mujoco

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    gene = tabletop_arm_gene()
    spec = select_task_spec("place the box on the target")
    objs = [o for o in generate_task_scenes(gene, spec, count=1)[0].objects if o.object_type == "cube"]
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, objs))
    mj.opt.iterations = 10
    mj.opt.ls_iterations = 8
    d = mujoco.MjData(mj)

    ee = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
    bq = int(mj.jnt_qposadr[box_jid])
    act_dof = [int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(mj.nu)]
    fr = mj.actuator_forcerange[:, 1]
    print(f"nv={mj.nv} nu={mj.nu} ee_site={ee} box_qadr={bq} act_dof={act_dof} forcerange={fr.tolist()}")

    # WS-A test: SCRIPTED-PUSHER base. Aim the FK-IK target PAST the box toward the goal, so the base
    # itself pushes the box toward the goal (a zero residual already makes progress). Filter arm-link
    # collisions (the MJX fix) so CPU mirrors the trainer. Ramp the reference for stability.
    from virturoid.services.pick_place_controller import plan_joint_targets

    ee_body = int(mj.site_bodyid[ee]); box_body = int(mj.jnt_bodyid[box_jid])
    for g in range(mj.ngeom):
        if int(mj.geom_bodyid[g]) not in (0, ee_body, box_body):
            mj.geom_contype[g] = 0; mj.geom_conaffinity[g] = 0

    box_xy = np.array([0.30, -0.05]); goal_xy = np.array([0.40, 0.08])      # push from box -> goal
    mujoco.mj_resetData(mj, d)
    d.qpos[bq], d.qpos[bq + 1] = box_xy
    mujoco.mj_forward(mj, d)
    box_z = float(d.qpos[bq + 2])
    push_dir = (goal_xy - box_xy) / (np.linalg.norm(goal_xy - box_xy) + 1e-6)
    push_pt = box_xy + 0.06 * push_dir                                       # aim a bit PAST the box toward goal
    q_star = np.asarray(plan_joint_targets(mj, (push_pt[0], push_pt[1], box_z), iterations=6, candidates=40)[0])
    d0_push = float(np.linalg.norm(box_xy - goal_xy))
    print(f"box={box_xy.tolist()} goal={goal_xy.tolist()} push_pt={np.round(push_pt,3).tolist()} "
          f"q*={np.round(q_star,3).tolist()} initial box->goal={d0_push:.3f}")

    KPJ, KDJ, DQ = 50.0, 5.0, 0.06
    q_ref = d.qpos[np.asarray(act_dof)].copy()
    for t in range(250):
        q_ref = q_ref + np.clip(q_star - q_ref, -DQ, DQ)
        q = d.qpos[np.asarray(act_dof)]; qd = d.qvel[np.asarray(act_dof)]
        tau = d.qfrc_bias[np.asarray(act_dof)] + KPJ * (q_ref - q) - KDJ * qd
        d.ctrl[:] = np.clip(tau, -fr, fr)
        mujoco.mj_step(mj, d)
        if t % 25 == 0 or t == 249:
            box = d.qpos[bq:bq + 3]
            print(f"  t={t:>3} reach={float(np.linalg.norm(d.site_xpos[ee]-box)):.3f} "
                  f"box->goal={float(np.linalg.norm(box[:2]-goal_xy)):.3f} box={np.round(box[:3],3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
