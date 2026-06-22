"""Geometry probe for the parallel-jaw grasp (M3 G2 — why the scripted base fails to lift).

Prints the actual world geometry so the gripper redesign is grounded in numbers, not guesses:
  - palm-tip (ee_site) world pos + palm body's local +z direction (which way fingers point),
  - each finger body + finger-tip world pos, and the finger inner-face y-gap OPEN vs CLOSED,
  - box center + half-extents,
  - the two go/no-go geometric gates: (a) can the open gap STRADDLE the box?  (b) does the
    approach put the finger TIPS (not the palm) at the object?

    PYTHONPATH=src python scripts/grasp_geom_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import numpy as np
    import mujoco

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper, reachable_workspace
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    gene = add_parallel_gripper(tabletop_arm_gene())
    ws = reachable_workspace(gene, (0.24, -0.12), (0.42, 0.12), z=0.05, grid=5, margin=0.07)
    gx = float(np.clip(0.40, ws["bbox"]["x"][0], ws["bbox"]["x"][1]))
    gy = float(np.clip(0.0, ws["bbox"]["y"][0], ws["bbox"]["y"][1]))
    cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=1.0, scale=1.0)
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))
    d = mujoco.MjData(mj)

    ee = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    box_half = mj.geom_size[box_geom].copy()
    fl = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "finger_l")
    fr = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "finger_r")
    flg = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_geom")
    frg = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_geom")
    fhalf = mj.geom_size[flg].copy()    # box geom half-extents (x,y,z) in finger local frame

    # actuators: arm (revolute) vs finger (slide)
    arm_act, fin_act = [], []
    for u in range(mj.nu):
        jid = int(mj.actuator_trnid[u, 0]); jt = int(mj.jnt_type[jid])
        (fin_act if jt == 2 else arm_act).append(u)
    arm_dof = [int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in arm_act]
    fin_dof = [int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in fin_act]
    forcer = mj.actuator_forcerange[:, 1]

    def snap(label):
        mujoco.mj_forward(mj, d)
        palm = d.site_xpos[ee].copy()
        pmat = d.site_xmat[ee].reshape(3, 3)
        zaxis = pmat[:, 2]                                   # palm local +z in world (finger direction)
        flp = d.xpos[fl].copy(); frp = d.xpos[fr].copy()
        # finger tips: body origin + half-length*local_z*2 along finger +z (geom is along +z, centered at h)
        ftip_l = d.geom_xpos[flg] + d.geom_xmat[flg].reshape(3, 3)[:, 2] * fhalf[2]
        ftip_r = d.geom_xpos[frg] + d.geom_xmat[frg].reshape(3, 3)[:, 2] * fhalf[2]
        inner_gap = abs(flp[1] - frp[1]) - 2 * fhalf[1]      # open space between finger inner faces (y)
        print(f"  [{label}] palm_tip={np.round(palm,3)} palm_+z={np.round(zaxis,2)}")
        print(f"           finger_l_body={np.round(flp,3)} finger_r_body={np.round(frp,3)} inner_gap_y={inner_gap:+.3f}")
        print(f"           finger_tip_l={np.round(ftip_l,3)} finger_tip_r={np.round(ftip_r,3)}")
        return palm, inner_gap

    mujoco.mj_resetData(mj, d)
    print(f"box: center=({gx},{gy},0.051) half-extents={np.round(box_half,3)}  (width_y={2*box_half[1]:.3f})")
    print(f"finger geom half-extents={np.round(fhalf,3)}  span(rest |y|)= see below")
    print("REST pose:")
    palm0, gap_open = snap("rest")

    # drive arm to pre-grasp (grasp TCP to box center), fingers open, and report
    q_arm = np.asarray(plan_joint_targets(mj, (gx, gy, 0.05), iterations=8, candidates=60, site_name="grasp_site")[0])[:len(arm_act)]

    def drive(arm_target, finger_cmd, steps):
        for _ in range(steps):
            for k, u in enumerate(arm_act):
                qa = d.qpos[arm_dof[k]]; va = d.qvel[arm_dof[k]]
                d.ctrl[u] = float(np.clip(d.qfrc_bias[arm_dof[k]] + 40*(arm_target[k]-qa) - 4*va, -forcer[u], forcer[u]))
            for u in fin_act:
                d.ctrl[u] = float(finger_cmd * forcer[u])
            mujoco.mj_step(mj, d)

    drive(q_arm, -0.3, 250)
    print("\nAfter REACH (ee->box center, fingers open):")
    palm_r, gap_r = snap("reach")

    drive(q_arm, 1.0, 150)
    print("\nAfter CLOSE:")
    snap("close")

    print("\n--- gates ---")
    print(f"(a) straddle: open_inner_gap_y={gap_open:+.3f} vs box_width_y={2*box_half[1]:.3f}  "
          f"-> {'OK can enclose' if gap_open > 2*box_half[1] + 0.005 else 'FAIL too narrow'}")
    print(f"(b) approach: finger_tips below palm by ~{abs(palm0[2]-(palm0[2])):.3f}; see palm_+z sign "
          "(want fingers to descend ALONGSIDE the box, tips near table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
