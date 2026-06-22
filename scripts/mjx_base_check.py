"""Pin the CPU-vs-MJX discrepancy in the joint-PD base (1 env, verbose).

Same joint-PD-toward-FK-IK base reaches d_reach 0.08 on CPU MuJoCo but flails to 1.46 in the MJX
trainer. This runs it in a SINGLE MJX env and prints internals (reach, box pos, qfrc_bias, tau) so we
see exactly what diverges — box launched by contact? qfrc_bias wrong? Run on the GPU box.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_base_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import jax
    import jax.numpy as jp
    import numpy as np
    import mujoco
    from mujoco import mjx

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    gene = tabletop_arm_gene(); spec = select_task_spec("place the box on the target")
    objs = [o for o in generate_task_scenes(gene, spec, count=1)[0].objects if o.object_type == "cube"]
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, objs))
    mj.opt.iterations = 10; mj.opt.ls_iterations = 8
    ee = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    box_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box"))
    # HYPOTHESIS TEST: MJX NaNs on arm-link<->table contact. Disable collision for every arm-link geom
    # except the ee body — keep ee<->box, ee<->table(body0), box<->table. If NaN vanishes, this is it.
    ee_body = int(mj.site_bodyid[ee]); box_body = int(mj.jnt_bodyid[box_jid])
    n_off = 0
    for g in range(mj.ngeom):
        b = int(mj.geom_bodyid[g])
        if b not in (0, ee_body, box_body):
            mj.geom_contype[g] = 0; mj.geom_conaffinity[g] = 0; n_off += 1
    print(f"disabled collision on {n_off} arm-link geoms (kept ee={ee_body}, box={box_body}, world)")
    mx = mjx.put_model(mj)
    bq = int(mj.jnt_qposadr[box_jid])
    ad = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(mj.nu)], dtype=int)
    fr = jp.asarray(mj.actuator_forcerange[:, 1])
    box_z = float(mjx.forward(mx, mjx.make_data(mx, naconmax=96, njmax=192)).qpos[bq + 2])
    q_star = jp.asarray(np.asarray(plan_joint_targets(mj, (0.33, 0.0, box_z), iterations=6, candidates=40)[0]))
    print(f"q*={np.round(np.asarray(q_star),3).tolist()} box_z={box_z:.3f}")

    d = mjx.make_data(mx, naconmax=96, njmax=192)
    qpos = d.qpos.at[bq].set(0.33).at[bq + 1].set(0.0).at[bq + 2].set(box_z)
    d = mjx.forward(mx, d.replace(qpos=qpos))
    KPJ, KDJ, DQ = 50.0, 5.0, 0.03   # ramp the reference at <=DQ rad/step so torque stays unsaturated

    @jax.jit
    def step1(d, q_ref):
        q_ref = q_ref + jp.clip(q_star - q_ref, -DQ, DQ)        # reference generator (bounded-rate ramp)
        q = d.qpos[ad]; qd = d.qvel[ad]
        tau = d.qfrc_bias[ad] + KPJ * (q_ref - q) - KDJ * qd
        return mjx.step(mx, d.replace(ctrl=jp.clip(tau, -fr, fr))), q_ref, tau

    q_ref = d.qpos[ad]
    for t in range(120):
        d, q_ref, tau = step1(d, q_ref)
        if t % 12 == 0 or t == 119:
            ee_pos = np.asarray(d.site_xpos[ee]); box = np.asarray(d.qpos[bq:bq + 3])
            print(f"  t={t:>3} reach={float(np.linalg.norm(ee_pos-box)):.3f} "
                  f"ee={np.round(ee_pos,3).tolist()} box={np.round(box,3).tolist()} "
                  f"tau={np.round(np.asarray(tau),2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
