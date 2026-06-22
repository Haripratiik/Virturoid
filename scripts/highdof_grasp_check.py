"""CPU validation: can the TASK-SPACE base grasp + lift a box on a HIGH-DOF gripper arm? (M3 recipe,
now with task-space arm control instead of joint-PD, for a composed 6-7 DOF arm.) Reach the grasp TCP to
the box with Jacobian control, close the fingers, lift — and check the box rises with real finger-box
contact. If it grips+lifts, the task-space base is grasp-competent and a GPU residual is justified;
if not, iterate here (cheap) before any GPU run.

    PYTHONPATH=src python scripts/highdof_grasp_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import numpy as np
    import mujoco

    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import (
        task_space_reach_ctrl, _actuated_joint_map, _actuator_force_clamps, _ee_site_id)
    from virturoid.schemas.scenes import SceneObject

    for label, prompt in [("6dof", "a precise 6-dof arm to grasp and lift a box"),
                          ("7dof", "a Franka-style 7-dof arm to grasp and lift a box")]:
        gx, gy = 0.42, 0.0
        g = compose_robot(prompt)
        cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, friction=1.0, scale=1.0)
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(g, [cube]))
        mj.opt.iterations = 20; mj.opt.ls_iterations = 10
        tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        if tcp < 0:
            tcp = _ee_site_id(mj)
        box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
        bq = int(mj.jnt_qposadr[box_jid])
        box_g = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box")
        fgeoms = {mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("finger_l_geom", "finger_r_geom")}
        act = _actuated_joint_map(mj); clamps = _actuator_force_clamps(mj)
        fin_slots = [k for k, (_, _, jnt) in enumerate(act) if int(mj.jnt_type[jnt]) == 2]
        fr = mj.actuator_forcerange[:, 1]
        d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)
        box_z0 = float(d.qpos[bq + 2])

        def fb_contacts():
            return sum(1 for i in range(d.ncon)
                       if {int(d.contact[i].geom1), int(d.contact[i].geom2)} & fgeoms
                       and box_g in (int(d.contact[i].geom1), int(d.contact[i].geom2)))

        def step(target, finger_cmd, steps):
            for _ in range(steps):
                mujoco.mj_forward(mj, d)
                # position-only task-space reach (the shipped controller). NOTE: this reaches the box but
                # cannot orient the gripper, so the fingers don't straddle it — documenting why real
                # high-DOF grasp needs a 6-DOF op-space solver or a learned policy (findings §9).
                task_space_reach_ctrl(mj, d, tcp, target, act, clamps, site_id=tcp)
                for k in fin_slots:
                    d.ctrl[k] = float(finger_cmd * fr[k])     # +close / -open (override arm-Jacobian's ~0 on fingers)
                mujoco.mj_step(mj, d)

        box = d.qpos[bq:bq + 3]
        step((box[0], box[1], 0.06), -0.4, 300)               # reach TCP to the box, fingers open
        step((box[0], box[1], 0.055), 1.0, 200)               # close fingers on the box
        max_c = max(fb_contacts() for _ in [0])                # contacts at end of close
        step((box[0], box[1], 0.20), 1.0, 300)                # lift, fingers closed
        lifted = float(d.qpos[bq + 2]) - box_z0
        print(f"{label}: lifted {lifted:+.3f} m  finger-box_contacts(after close)={max_c}  "
              + ("GRASP+LIFT WORKS" if lifted > 0.04 else "no lift"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
