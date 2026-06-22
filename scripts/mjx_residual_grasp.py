"""Phase 3 M3-G2, residual RL: learn GRASP-AND-LIFT as a correction on a scripted FK-IK-PD base.

Builds directly on the proven push recipe (scripts/mjx_residual_push.py, phase3_rl_mjx.md §M2):
FK-IK joint-PD base + reference ramp + MJX contact-filtering + bounded residual + process-aware,
hack-resistant reward + task matched to the body's reachable region. The differences for grasp:

  - BODY: the parallel-jaw gripper gene (add_parallel_gripper, span 0.10, TCP `grasp_site`); nu=5
    (3 arm revolute + 2 finger slide).
  - BASE: a PHASE schedule (approach above -> descend onto -> close jaws -> lift), each phase a
    per-env FK-IK arm target (precomputed grid, targeting grasp_site) + a scripted finger open/close,
    tracked by gravity-comp joint-PD (arm + fingers). CPU-validated in grasp_base_check.py: lifts a
    tabletop cube +0.14 m with sustained two-finger contact, in the region x in [0.43,0.49].
  - REWARD (process-aware, ADR-4): lifting is rewarded ONLY while BOTH fingers actually contact the
    box (real friction grasp, read from the MJX contact arrays) and the box stays at the grasp point,
    with a speed penalty (no flinging). Held-out score is the SPARSE sustained-grasp success.
  - REGION: spawn in the grasp-feasible band (M2's body<->task lesson, now for grasp); curriculum
    widens the band as success grows.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_residual_grasp.py --iters 200 --envs 192 --ep-len 200
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=192)
    ap.add_argument("--ep-len", type=int, default=200)      # approach + descend + close + lift
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=5)         # TDR-safe (push: chunk 5-6 stable @192)
    ap.add_argument("--kpj", type=float, default=120.0)     # arm joint-PD position gain (stiff enough to
    ap.add_argument("--kdj", type=float, default=12.0)      # track the fold/descend in 200 steps — MJX probe)
    ap.add_argument("--kpf", type=float, default=300.0)     # finger PD gain (saturates grip force)
    ap.add_argument("--kdf", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.3)     # residual authority (fraction of forcerange)
    ap.add_argument("--ik-grid", type=int, default=8)       # IK lookup grid resolution per axis
    ap.add_argument("--save", type=str, default=None)       # save trained policy params (npz) for the skill bank (G4)
    ap.add_argument("--warm-start", type=str, default=None) # init policy from a banked skill's params (G4 flywheel)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    import optax
    from mujoco import mjx

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.design_critic import add_parallel_gripper
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.schemas.scenes import SceneObject

    print("backend:", jax.default_backend(), jax.devices())
    gene = add_parallel_gripper(tabletop_arm_gene())
    # Grasp-feasible band (grasp_base_check.py sweep): the gripper reaches DOWN onto the box only for
    # x in [0.43,0.49] (the TCP's reach-down envelope is forward of x~0.40). Match task to body.
    X_LO, X_HI = 0.43, 0.49
    Y_LO, Y_HI = -0.08, 0.08
    cube0 = SceneObject("box", "cube", (0.46, 0.0, 0.05, 0, 0, 0), mass_kg=0.03, material="gray_block",
                        friction=1.0, scale=1.0)
    xml = compile_gene_with_scene(gene, [cube0], physics_only=True)  # MJX-safe: no cylinder housings/collars
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.opt.iterations = 10; mj.opt.ls_iterations = 8
    tcp_site = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    box_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box"))
    if tcp_site < 0 or box_jid < 0:
        print("ERROR: missing grasp_site or free_box."); return 1
    BOX_G = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "box"))   # geom name = SceneObject.name
    FL_G = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_l_geom"))
    FR_G = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, "finger_r_geom"))
    box_body = int(mj.jnt_bodyid[box_jid])
    fl_body = int(mj.geom_bodyid[FL_G]); fr_body = int(mj.geom_bodyid[FR_G])
    # MJX contact-filtering via contype/conaffinity BITMASKS (M2 lesson + MJX grasp-probe finding):
    #   box   <-> table AND fingers      (box rests on the table; jaws grip it)
    #   fingers <-> box ONLY  (NOT the table) — straddling a 0.05 box on a 0.025 table needs the
    #     0.06 jaws to dip to z~0.02 (below the table top); if they collide with the table it shoves
    #     the arm off the box before the jaws close (verified in mjx_grasp_probe.py). Let them pass.
    #   arm links + palm: disabled (folding-arm contacts NaN in MJX within ~12 steps; CPU tolerates).
    for g in range(mj.ngeom):
        b = int(mj.geom_bodyid[g])
        if b == box_body:
            mj.geom_contype[g] = 0b11; mj.geom_conaffinity[g] = 0b11
        elif b in (fl_body, fr_body):
            mj.geom_contype[g] = 0b10; mj.geom_conaffinity[g] = 0b10
        elif b == 0:                                  # world (table)
            mj.geom_contype[g] = 0b01; mj.geom_conaffinity[g] = 0b01
        else:                                         # arm links + palm
            mj.geom_contype[g] = 0; mj.geom_conaffinity[g] = 0
    mx = mjx.put_model(mj)
    bq = int(mj.jnt_qposadr[box_jid]); bv = int(mj.jnt_dofadr[box_jid])
    nu, nq, nv = int(mj.nu), int(mj.nq), int(mj.nv)
    act_dof = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(nu)], dtype=int)
    # arm (revolute) vs finger (slide) actuator slots
    arm_slots = [u for u in range(nu) if int(mj.jnt_type[int(mj.actuator_trnid[u, 0])]) != 2]
    fin_slots = [u for u in range(nu) if int(mj.jnt_type[int(mj.actuator_trnid[u, 0])]) == 2]
    arm_slots_j = jp.asarray(arm_slots, dtype=int); fin_slots_j = jp.asarray(fin_slots, dtype=int)
    arm_qadr = jp.asarray([int(mj.jnt_qposadr[int(mj.actuator_trnid[u, 0])]) for u in arm_slots], dtype=int)
    n_arm = len(arm_slots)
    FCLOSE = float(min(mj.jnt_range[int(mj.actuator_trnid[fin_slots[0], 0])][1], 0.045))
    fr = jp.asarray(mj.actuator_forcerange[:, 1])
    act_scale = args.alpha * fr
    KP = jp.asarray([args.kpj if u in arm_slots else args.kpf for u in range(nu)])
    KD = jp.asarray([args.kdj if u in arm_slots else args.kdf for u in range(nu)])
    N, L, CHUNK = args.envs, args.ep_len, args.chunk
    NACON, NJMAX = 96, 192
    DQ = 0.10                                  # arm reference ramp rate (rad/step) — faster fold (MJX probe)
    CONTACT_MARGIN = 1e-3                       # "in contact" if penetrating within 1mm
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.97, 0.95, 0.2, 1e-2, 0.5, 3e-4
    # The arm STARTS pre-folded at the hover pose q_above (gripper over the box) — folding from rest
    # can't complete in 200 steps (MJX probe). Phases (fractions of L): settle[0,P_SETTLE) hold q_above
    # -> descend[P_SETTLE,P_LIFT) to q_at, fingers close at P_GRIP -> lift[P_LIFT,1) to q_lift.
    P_SETTLE, P_GRIP, P_LIFT = 0.15, 0.50, 0.65
    box_z = float(mjx.forward(mx, mjx.make_data(mx, naconmax=NACON, njmax=NJMAX)).qpos[bq + 2])
    obs_dim = nq + nv + 3 + 3 + 3 + 1 + 1       # qpos,qvel,tcp,box,tcp-box,lifted,phase

    # ---- precompute coarse box(x,y) -> arm FK-IK targets for the 3 grasp heights (CPU, ONE-TIME) ----
    ik_mj = mujoco.MjModel.from_xml_string(xml)
    G = args.ik_grid
    gx = np.linspace(X_LO, X_HI, G); gy = np.linspace(Y_LO, Y_HI, G)
    grid_xy = np.array([(x, y) for x in gx for y in gy], dtype=float)        # (G*G, 2)

    def ik_arm(x, y, z):
        q = plan_joint_targets(ik_mj, (float(x), float(y), z), iterations=6, candidates=40,
                               site_name="grasp_site")[0]
        return [q[s] for s in arm_slots]

    t_ik = time.time()
    grid_above = np.array([ik_arm(x, y, 0.15) for x, y in grid_xy], dtype=float)
    grid_at = np.array([ik_arm(x, y, box_z) for x, y in grid_xy], dtype=float)
    grid_lift = np.array([ik_arm(x, y, 0.20) for x, y in grid_xy], dtype=float)
    print(f"IK grids: 3x{len(grid_xy)} grasp targets precomputed in {time.time()-t_ik:.1f}s", flush=True)
    grid_xy_j = jp.asarray(grid_xy)
    grid_above_j, grid_at_j, grid_lift_j = map(jp.asarray, (grid_above, grid_at, grid_lift))

    def init_mlp(key, sizes):
        ps = []
        for i in range(len(sizes) - 1):
            key, k = jax.random.split(key)
            ps.append((jax.random.normal(k, (sizes[i], sizes[i + 1])) * (1.0 / sizes[i]) ** 0.5,
                       jp.zeros(sizes[i + 1])))
        return ps

    def mlp(ps, x):
        for w, b in ps[:-1]:
            x = jp.tanh(x @ w + b)
        w, b = ps[-1]
        return x @ w + b

    key = jax.random.PRNGKey(0)
    key, kp_, kv = jax.random.split(key, 3)
    params = {"pi": init_mlp(kp_, [obs_dim, 128, 128, nu]),
              "vf": init_mlp(kv, [obs_dim, 256, 256, 1]),
              "logstd": jp.zeros(nu) - 1.2}

    def flatten_params(p):
        flat = {"logstd": np.asarray(p["logstd"])}
        for net in ("pi", "vf"):
            for i, (w, b) in enumerate(p[net]):
                flat[f"{net}_w{i}"] = np.asarray(w); flat[f"{net}_b{i}"] = np.asarray(b)
        return flat

    def load_params(path):
        data = np.load(path); out = {"pi": [], "vf": [], "logstd": jp.asarray(data["logstd"])}
        for net in ("pi", "vf"):
            i = 0
            while f"{net}_w{i}" in data:
                out[net].append((jp.asarray(data[f"{net}_w{i}"]), jp.asarray(data[f"{net}_b{i}"]))); i += 1
        return out

    if args.warm_start:                                  # G4 flywheel: seed from a banked skill
        loaded = load_params(args.warm_start)
        if np.asarray(loaded["pi"][0][0]).shape == (obs_dim, 128) and loaded["logstd"].shape == (nu,):
            params = loaded; print(f"warm-started policy from {args.warm_start}")
        else:
            print(f"warm-start SKIPPED (shape mismatch with {args.warm_start}); cold start")
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = opt.init(params)

    def obs_of(d, t):
        tcp = d.site_xpos[tcp_site]; box = d.qpos[bq:bq + 3]
        phase = jp.array([t / L]); lifted = jp.array([box[2] - box_z])
        obs = jp.concatenate([d.qpos, d.qvel, tcp, box, tcp - box, lifted, phase])
        return jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def phase_arm_target(t, q_above, q_at, q_lift):
        frac = t / L
        is_settle = frac < P_SETTLE
        is_lift = frac >= P_LIFT
        q = jp.where(is_lift[..., None], q_lift, jp.where(is_settle[..., None], q_above, q_at))
        return q                                          # (N, n_arm)

    def phase_finger_target(t):
        return jp.where((t / L) < P_GRIP, 0.0, FCLOSE)    # open through descend, clamp from the grip phase

    def pd_base(d, q_star_full):
        q = d.qpos[act_dof]; qd = d.qvel[act_dof]
        tau = d.qfrc_bias[act_dof] + KP * (q_star_full - q) - KD * qd
        return jp.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    def both_finger_contact(d):
        c = d._impl.contact if hasattr(d, "_impl") else d.contact   # mjx>=0.10 moved contact under _impl
        g = c.geom                                        # (N, nacon, 2)
        dist = c.dist                                     # (N, nacon)
        a = dist < CONTACT_MARGIN
        g0, g1 = g[..., 0], g[..., 1]
        has_box = (g0 == BOX_G) | (g1 == BOX_G)
        fl_box = jp.any(a & has_box & ((g0 == FL_G) | (g1 == FL_G)), axis=1)
        fr_box = jp.any(a & has_box & ((g0 == FR_G) | (g1 == FR_G)), axis=1)
        return (fl_box & fr_box).astype(jp.float32)

    def reset(key, x_lo, x_hi, y_lo, y_hi):
        k1, = jax.random.split(key, 1)
        box_xy = jax.random.uniform(k1, (N, 2), minval=jp.array([x_lo, y_lo]), maxval=jp.array([x_hi, y_hi]))
        d2 = jp.sum((box_xy[:, None, :] - grid_xy_j[None, :, :]) ** 2, axis=-1)    # (N, G*G)
        idx = jp.argmin(d2, axis=1)
        q_above = grid_above_j[idx]; q_at = grid_at_j[idx]; q_lift = grid_lift_j[idx]   # (N, n_arm)
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        qpos = data.qpos.at[:, bq].set(box_xy[:, 0]).at[:, bq + 1].set(box_xy[:, 1]).at[:, bq + 2].set(box_z)
        qpos = qpos.at[:, arm_qadr].set(q_above)        # arm STARTS pre-folded at the hover pose
        data = data.replace(qpos=qpos)
        data = jax.vmap(lambda d: mjx.forward(mx, d))(data)
        return data, q_above, q_at, q_lift

    def reward_of(d):
        box = d.qpos[:, bq:bq + 3]; tcp = d.site_xpos[:, tcp_site]
        box_speed = jp.clip(jp.linalg.norm(jp.nan_to_num(d.qvel[:, bv:bv + 3], posinf=0.0, neginf=0.0), axis=-1), 0.0, 10.0)
        d_tcp = jp.clip(jp.nan_to_num(jp.linalg.norm(tcp - box, axis=-1), nan=1.5, posinf=1.5), 0.0, 1.5)
        lifted = jp.clip(jp.nan_to_num(box[:, 2] - box_z, nan=0.0), -0.1, 0.5)
        both = both_finger_contact(d)                         # real two-finger grip (process gate)
        controlled_lift = jp.clip(lifted, 0.0, 0.3) * both    # lift only counts while truly gripped
        speed_excess = jp.clip(box_speed - 0.5, 0.0, 3.0)     # dense anti-fling penalty
        # success: gripped by BOTH fingers AND lifted off the table AND still HELD at the grasp point
        # (d_tcp small) — a flung box leaves the TCP, so this can't be hacked, and (unlike a speed
        # gate) it fires during an honest in-progress lift.
        grasped = ((both > 0.5) & (lifted > 0.05) & (d_tcp < 0.06)).astype(jp.float32)
        rew = 10.0 * controlled_lift + 1.0 * both - 0.3 * speed_excess - 0.15 * d_tcp + 6.0 * grasped
        return rew, grasped, lifted, both

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, carry, q_above, q_at, q_lift, keys):
        std = jp.exp(params["logstd"])

        def body(carry, key):
            data, q_arm_ref, t = carry
            q_arm_tgt = phase_arm_target(t, q_above, q_at, q_lift)
            q_arm_ref = q_arm_ref + jp.clip(q_arm_tgt - q_arm_ref, -DQ, DQ)       # ramp arm toward phase target
            f_tgt = phase_finger_target(t)
            q_star = jp.zeros((N, nu)).at[:, arm_slots_j].set(q_arm_ref)
            q_star = q_star.at[:, fin_slots_j].set(f_tgt)
            obs = jax.vmap(obs_of, in_axes=(0, None))(data, t)
            mean = mlp(params["pi"], obs); val = mlp(params["vf"], obs)[:, 0]
            act = mean + std * jax.random.normal(key, mean.shape)
            logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            base = jax.vmap(pd_base)(data, q_star)
            ctrl = jp.clip(base + jp.clip(act, -1, 1) * act_scale, -fr, fr)
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            rew, grasped, lifted, both = reward_of(data2)
            return (data2, q_arm_ref, t + 1), (obs, act, logp, rew, val, grasped, lifted, both)

        return jax.lax.scan(body, carry, keys)

    @jax.jit
    def finish(params, data_last, t_last, traj):
        obs, act, logp, rew, val, grasped, lifted, both = traj
        last_val = mlp(params["vf"], jax.vmap(obs_of, in_axes=(0, None))(data_last, t_last))[:, 0]

        def gae(carry, x):
            nextval, adv = carry; rew_t, val_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = delta + GAMMA * LAM * adv
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val), reverse=True)
        return (obs, act, logp, advs, advs + val, rew.sum(0).mean(),
                grasped[-10:].mean(), lifted[-10:].mean(), both[-10:].mean())

    def rollout(params, key, region):
        rkey, key = jax.random.split(key)
        data, q_above, q_at, q_lift = reset(rkey, *region)
        carry = (data, q_above, jp.float32(0.0)); chunks = []   # reference starts at the pre-folded hover pose
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            carry, out = roll_chunk(params, carry, q_above, q_at, q_lift, jax.random.split(ck, CHUNK))
            jax.block_until_ready(carry[0].qpos)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(8)]
        return finish(params, carry[0], carry[2], traj)

    def loss_fn(params, obs, act, old_logp, adv, ret):
        std = jp.exp(params["logstd"]); mean = mlp(params["pi"], obs)
        logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
        ratio = jp.exp(logp - old_logp); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio, -adv * jp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        vf = ((mlp(params["vf"], obs)[:, 0] - ret) ** 2).mean()
        ent = (params["logstd"] + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
        return pg + VF * vf - ENT * ent

    @jax.jit
    def update(params, opt_state, obs, act, logp, adv, ret, key):
        B = obs.shape[0]; mb = B // args.minibatches

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key); perm = jax.random.permutation(pk, B)

            def mbstep(carry, i):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                g = jax.grad(loss_fn)(params, obs[idx], act[idx], logp[idx], adv[idx], ret[idx])
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), None
            (params, opt_state), _ = jax.lax.scan(mbstep, (params, opt_state), jp.arange(args.minibatches))
            return (params, opt_state, key), None
        (params, opt_state, _), _ = jax.lax.scan(epoch, (params, opt_state, key), jp.arange(args.epochs))
        return params, opt_state

    print(f"residual grasp (phase FK-IK joint-PD base): nu={nu} arm={arm_slots} fingers={fin_slots} "
          f"obs={obs_dim} FCLOSE={FCLOSE:.3f} | {N} envs L={L} CHUNK={CHUNK} alpha={args.alpha}")
    # curriculum: start in the most-reliable sub-band (x~0.45, y~0), widen as sustained-grasp grows.
    cx = 0.02; cy = 0.02; t0 = time.time()
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        region = (0.46 - cx, 0.46 + cx, -cy, cy)
        obs, act, logp, adv, ret, ep_rew, grasped, lifted, both = rollout(params, rk, region)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if float(grasped) > 0.5:                            # widen the spawn band once reliably grasping
            cx = min(0.03, cx + 0.002); cy = min(0.08, cy + 0.004)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  grasped={float(grasped):.0%}  "
                  f"lifted={float(lifted):+.3f}m  both_contact={float(both):.0%}  "
                  f"band=x±{cx:.3f},y±{cy:.3f}  ({(time.time()-t0):.0f}s)", flush=True)
    print(f"done: residual grasp, {args.iters} iters, grasped {float(grasped):.0%}, lifted {float(lifted):+.3f}m, "
          f"both_contact {float(both):.0%}, band x±{cx:.3f},y±{cy:.3f}, {time.time()-t0:.0f}s")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, **flatten_params(params))
        # SKILL CARD (G4): everything the bank needs to reuse/warm-start this skill. Emitted as a JSON
        # line so banking is a one-liner (skill_flywheel.bank_skill_from_card).
        import json as _json
        card = {"task_type": "grasp", "robot_class": gene.robot_class, "species": gene.species,
                "gene_id": gene.id, "success_rate": round(float(grasped), 4),
                "region": {"x": [X_LO, X_HI], "y": [Y_LO, Y_HI]},
                "base_config": {"kpj": args.kpj, "kdj": args.kdj, "kpf": args.kpf, "kdf": args.kdf,
                                "alpha": args.alpha, "dq": DQ, "phases": [P_SETTLE, P_GRIP, P_LIFT],
                                "ep_len": L, "fclose": FCLOSE},
                "reward_spec": {"controlled_lift": 10.0, "both_contact": 1.0, "speed_penalty": 0.3,
                                "d_tcp_shaping": 0.15, "grasped_bonus": 6.0,
                                "success_gate": "both & lifted>0.05 & d_tcp<0.06"},
                "obs_dim": obs_dim, "act_dim": nu, "params_path": str(args.save)}
        print("SKILL_CARD " + _json.dumps(card))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
