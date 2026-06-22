"""Single-env MJX trace of the scripted grasp BASE — why does it lift +0.14m on CPU but 0 in MJX?
Mirrors the trainer's model (same contact-filtering), runs the phase base (no residual) on ONE env,
and prints box_z / tcp-box / finger openings / finger<->box contacts through the phases. The method
that cracked push & the CPU grasp: isolate the divergence on one env before touching the GPU loop.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_grasp_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import jax, jax.numpy as jp
    import mujoco
    import numpy as np
    from mujoco import mjx

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    gx, gy = 0.46, 0.0
    gene = add_parallel_gripper(tabletop_arm_gene())
    cube = SceneObject("box", "cube", (gx, gy, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                       friction=1.0, scale=1.0)
    xml = compile_gene_with_scene(gene, [cube])
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.opt.iterations = 10; mj.opt.ls_iterations = 8
    tcp = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    box_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box"))
    BOX_G = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box"))
    FL = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_geom"))
    FR = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_geom"))
    box_body = int(mj.jnt_bodyid[box_jid]); fl_b = int(mj.geom_bodyid[FL]); fr_b = int(mj.geom_bodyid[FR])
    # bitmasks: box<->{table,fingers}; fingers<->box only (NOT table); arm/palm disabled.
    for g in range(mj.ngeom):
        b = int(mj.geom_bodyid[g])
        if b == box_body:
            mj.geom_contype[g] = 0b11; mj.geom_conaffinity[g] = 0b11
        elif b in (fl_b, fr_b):
            mj.geom_contype[g] = 0b10; mj.geom_conaffinity[g] = 0b10
        elif b == 0:
            mj.geom_contype[g] = 0b01; mj.geom_conaffinity[g] = 0b01
        else:
            mj.geom_contype[g] = 0; mj.geom_conaffinity[g] = 0
    mx = mjx.put_model(mj)
    bq = int(mj.jnt_qposadr[box_jid]); bv = int(mj.jnt_dofadr[box_jid])
    nu = int(mj.nu)
    act_dof = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(nu)], dtype=int)
    arm = [u for u in range(nu) if int(mj.jnt_type[int(mj.actuator_trnid[u, 0])]) != 2]
    fin = [u for u in range(nu) if int(mj.jnt_type[int(mj.actuator_trnid[u, 0])]) == 2]
    FCLOSE = 0.045
    fr = jp.asarray(mj.actuator_forcerange[:, 1])
    KP = jp.asarray([120.0 if u in arm else 300.0 for u in range(nu)])
    KD = jp.asarray([12.0 if u in arm else 10.0 for u in range(nu)])
    L, DQ = 200, 0.10
    # arm STARTS pre-folded at q_above (gripper hovering over the box); phases: settle -> descend ->
    # grip -> lift. P_SETTLE, P_GRIP (fingers close), P_LIFT (arm rises).
    P_SETTLE, P_GRIP, P_LIFT = 0.15, 0.50, 0.65
    NACON, NJMAX = 96, 192
    print("arm forcerange:", [round(float(mj.actuator_forcerange[u, 1]), 1) for u in arm])

    def ik(z):
        q = plan_joint_targets(mj, (gx, gy, z), iterations=6, candidates=40, site_name="grasp_site")[0]
        return np.asarray([q[s] for s in arm])
    q_above, q_at, q_lift = ik(0.15), ik(float(mjx.forward(mx, mjx.make_data(mx, naconmax=NACON, njmax=NJMAX)).qpos[bq+2])), ik(0.20)
    q_above, q_at, q_lift = map(jp.asarray, (q_above, q_at, q_lift))
    arm_j = jp.asarray(arm); fin_j = jp.asarray(fin)

    step = jax.jit(lambda d: mjx.step(mx, d))
    d = mjx.make_data(mx, naconmax=NACON, njmax=NJMAX)
    # init: box at rest, arm pre-folded at q_above
    qpos0 = d.qpos.at[bq].set(gx).at[bq+1].set(gy).at[bq+2].set(0.051)
    for k, u in enumerate(arm):
        qpos0 = qpos0.at[int(mj.jnt_qposadr[int(mj.actuator_trnid[u, 0])])].set(float(q_above[k]))
    d = mjx.forward(mx, d.replace(qpos=qpos0))
    box_z0 = float(d.qpos[bq+2])
    q_arm_ref = jp.asarray(q_above)

    def contacts(d):
        c = d._impl.contact if hasattr(d, "_impl") else d.contact
        g, dist = c.geom, c.dist
        a = dist < 1e-3
        g0, g1 = g[:, 0], g[:, 1]
        hb = (g0 == BOX_G) | (g1 == BOX_G)
        fl = int(jp.sum(a & hb & ((g0 == FL) | (g1 == FL))))
        frr = int(jp.sum(a & hb & ((g0 == FR) | (g1 == FR))))
        return fl, frr

    print(f"box_z0={box_z0:.3f} q_above={np.round(np.asarray(q_above),2)} q_at={np.round(np.asarray(q_at),2)} q_lift={np.round(np.asarray(q_lift),2)}")
    for t in range(L):
        frac = t / L
        q_arm_tgt = q_lift if frac >= P_LIFT else (q_above if frac < P_SETTLE else q_at)
        q_arm_ref = q_arm_ref + jp.clip(q_arm_tgt - q_arm_ref, -DQ, DQ)
        f_tgt = 0.0 if frac < P_GRIP else FCLOSE
        q_star = jp.zeros(nu).at[arm_j].set(q_arm_ref).at[fin_j].set(f_tgt)
        q = d.qpos[act_dof]; qd = d.qvel[act_dof]
        tau = d.qfrc_bias[act_dof] + KP * (q_star - q) - KD * qd
        ctrl = jp.clip(tau, -fr, fr)
        d = step(d.replace(ctrl=ctrl))
        if t % 20 == 0 or t == L - 1:
            box = np.asarray(d.qpos[bq:bq+3]); tcpp = np.asarray(d.site_xpos[tcp])
            fl, frr = contacts(d)
            fopen = [round(float(d.qpos[mj.jnt_qposadr[int(mj.actuator_trnid[u,0])]]), 3) for u in fin]
            qa = np.round(np.asarray(d.qpos[act_dof][arm_j]), 3)
            tgt = np.round(np.asarray(q_arm_tgt), 3)
            phase = "lift" if frac >= P_LIFT else ("settle" if frac < P_SETTLE else ("grip" if frac >= P_GRIP else "descend"))
            print(f"t={t:>3} {phase:>8} tcp={np.round(tcpp,3)} box={np.round(box[:2],3)} tcp_xy-box={np.hypot(tcpp[0]-box[0],tcpp[1]-box[1]):.3f} "
                  f"qarm={qa} tgt={tgt} fopen={fopen} c={fl}/{frr}")
    print(f"FINAL lifted {float(d.qpos[bq+2])-box_z0:+.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
