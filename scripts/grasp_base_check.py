"""CPU validation of the grasp BASE on the gripper-arm gene (M3 G2 precursor — validate before GPU).

Runs the standard pick sequence on one CPU MuJoCo env — approach ABOVE the cube (jaws open) ->
descend so the jaws straddle it -> close -> lift — targeting the grasp TCP (``grasp_site``, the
finger-tip midpoint), and instruments it: where the TCP actually lands vs the box, finger<->box
contacts, and box height per phase. If the box rises, the grasp base works and the GPU residual
trainer (G2) is justified; if not, the readouts say which gate fails (reach / straddle / grip).

    PYTHONPATH=src python scripts/grasp_base_check.py
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
    gy = 0.0

    def trial(gx, verbose=False):
        cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                           friction=1.0, scale=1.0)
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, [cube]))
        mj.opt.iterations = 20; mj.opt.ls_iterations = 10
        tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        box_jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")
        bq = int(mj.jnt_qposadr[box_jid])
        box_geom = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box")  # geom name = SceneObject.name
        fgeoms = {mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("finger_l_geom", "finger_r_geom")}
        arm_act, fin_act = [], []
        for u in range(mj.nu):
            jid = int(mj.actuator_trnid[u, 0]); jt = int(mj.jnt_type[jid])
            (fin_act if jt == 2 else arm_act).append(u)
        arm_dof = [int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in arm_act]
        fr = mj.actuator_forcerange[:, 1]
        d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)
        box_z0 = float(d.qpos[bq + 2])

        def fb_contacts():
            return sum(1 for i in range(d.ncon)
                       if {int(d.contact[i].geom1), int(d.contact[i].geom2)} & fgeoms
                       and box_geom in (int(d.contact[i].geom1), int(d.contact[i].geom2)))

        def plan(z):
            return np.asarray(plan_joint_targets(mj, (gx, gy, z), iterations=10, candidates=80,
                                                 site_name="grasp_site")[0])[:len(arm_act)]
        above, at, lift = plan(0.15), plan(0.051), plan(0.20)

        max_grip = [0]

        def drive(arm_target, finger_cmd, steps, track=False):
            for _ in range(steps):
                for k, u in enumerate(arm_act):
                    qa = d.qpos[arm_dof[k]]; va = d.qvel[arm_dof[k]]
                    d.ctrl[u] = float(np.clip(d.qfrc_bias[arm_dof[k]] + 60 * (arm_target[k] - qa) - 6 * va, -fr[u], fr[u]))
                for u in fin_act:
                    d.ctrl[u] = float(finger_cmd * fr[u])
                mujoco.mj_step(mj, d)
                if track:
                    max_grip[0] = max(max_grip[0], fb_contacts())

        drive(above, -0.4, 250)
        drive(at, -0.4, 250)
        mujoco.mj_forward(mj, d)
        tcp_xy = float(np.hypot(d.site_xpos[tcp][0] - d.qpos[bq], d.site_xpos[tcp][1] - d.qpos[bq + 1]))
        drive(at, 1.0, 200, track=True)
        drive(lift, 1.0, 300, track=True)
        lifted = float(d.qpos[bq + 2]) - box_z0
        if verbose:
            print(f"    tcp_landing_x={float(d.site_xpos[tcp][0]):.3f}")
        return tcp_xy, max_grip[0], lifted

    # Body<->task matching for GRASP: sweep box-x to find where the TCP reaches DOWN onto the box and
    # the jaws capture it (the gripper's reach-down envelope is forward of x~0.40, the M2 lesson).
    print(f"{'box_x':>6} {'tcp-box_xy':>11} {'max_grip_contacts':>17} {'lifted_m':>9}")
    best = None
    for gx in [0.40, 0.43, 0.45, 0.47, 0.49]:
        tcp_xy, n_grip, lifted = trial(gx)
        tag = "  <- GRASP+LIFT" if lifted > 0.04 else ""
        print(f"{gx:>6.2f} {tcp_xy:>11.3f} {n_grip:>17d} {lifted:>+9.3f}{tag}")
        if best is None or lifted > best[1]:
            best = (gx, lifted)
    print(f"best: box_x={best[0]:.2f} lifted={best[1]:+.3f} m")
    print("=== GRASP+LIFT WORKS ===" if best[1] > 0.04 else "=== no lift at any x — iterate gripper/control ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
