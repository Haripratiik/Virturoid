"""Why does the box lift with zero detected finger-box contacts? Find the TRUE lift mechanism
before trusting the grasp base (a scoop/cradle would make the GPU contact-reward fire never / be a
phantom grasp). Steps box_x=0.45 through the sequence and prints, during the hold/lift, EVERY contact
touching the box (by geom name), the box pose, and the finger joint openings.

    PYTHONPATH=src python scripts/grasp_mechanism_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import numpy as np
    import mujoco

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    gx, gy = 0.45, 0.0
    gene = add_parallel_gripper(tabletop_arm_gene())
    cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=1.0, scale=1.0)
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))
    mj.opt.iterations = 20; mj.opt.ls_iterations = 10
    gname = {g: (mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}") for g in range(mj.ngeom)}
    tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
    bq = int(mj.jnt_qposadr[box_jid])
    box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "cube")

    arm_act, fin_act = [], []
    for u in range(mj.nu):
        jid = int(mj.actuator_trnid[u, 0]); jt = int(mj.jnt_type[jid])
        (fin_act if jt == 2 else arm_act).append(u)
    arm_dof = [int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in arm_act]
    fin_qadr = [int(mj.jnt_qposadr[int(mj.actuator_trnid[u, 0])]) for u in fin_act]
    fr = mj.actuator_forcerange[:, 1]
    d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)

    def plan(z):
        return np.asarray(plan_joint_targets(mj, (gx, gy, z), iterations=10, candidates=80,
                                             site_name="grasp_site")[0])[:len(arm_act)]
    above, at, lift = plan(0.15), plan(0.051), plan(0.20)

    def drive(arm_target, finger_cmd, steps):
        for _ in range(steps):
            for k, u in enumerate(arm_act):
                qa = d.qpos[arm_dof[k]]; va = d.qvel[arm_dof[k]]
                d.ctrl[u] = float(np.clip(d.qfrc_bias[arm_dof[k]] + 60 * (arm_target[k] - qa) - 6 * va, -fr[u], fr[u]))
            for u in fin_act:
                d.ctrl[u] = float(finger_cmd * fr[u])
            mujoco.mj_step(mj, d)

    bdof = int(mj.jnt_dofadr[box_jid])

    def dump(label):
        mujoco.mj_forward(mj, d)
        box = d.qpos[bq:bq + 3]; tcpp = d.site_xpos[tcp]
        fopen = [round(float(d.qpos[a]), 4) for a in fin_qadr]
        bvel = float(np.linalg.norm(d.qvel[bdof:bdof + 6]))
        allc = [f"{gname[int(d.contact[i].geom1)]}~{gname[int(d.contact[i].geom2)]}" for i in range(d.ncon)]
        box_c = [c for c in allc if "cube" in c]
        print(f"[{label}] box={np.round(box,3)} tcp={np.round(tcpp,3)} finger_open={fopen} box_vel={bvel:.3f}")
        print(f"         ncon={d.ncon} box_contacts={box_c or '(none)'}  all={allc}")

    drive(above, -0.4, 250); dump("approach")
    drive(at, -0.4, 250); dump("descend (jaws open, straddling?)")
    drive(at, 1.0, 200); dump("closed")
    drive(lift, 1.0, 150); dump("mid-lift")
    drive(lift, 1.0, 150); dump("lifted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
