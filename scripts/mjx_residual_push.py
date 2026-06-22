"""Phase 3 M2, residual RL (roadmap N1, ADR-3): learn PUSH as a correction on an FK-IK-PD base.

History (all recorded in phase3_rl_mjx.md §M2):
  - from-scratch PPO: arm never reaches the box (6 reward iterations, 0%).
  - residual on a task-space OSC base: the base reaches forward/over the box but won't FOLD DOWN to
    the table (CPU-isolated in osc_base_check.py) — geometrically inadequate.
  - THIS: the base is the scripted FK-IK reacher's joint targets + JOINT-space PD. FK-IK finds the
    folded config that actually puts the ee on a tabletop box (it's why scripted pick-place hits
    67-91%). IK is numpy/CEM (not JAX), so we precompute a coarse box(x,y) -> q* grid ONCE, then at
    reset look up each env's target (vectorized) and run a gravity-compensated joint-space PD base
    toward it (fully vmappable). Action = tau_base + alpha*act_scale*pi_resid; the residual learns
    the push, the process-aware reward stops it kicking.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_residual_push.py --iters 150 --envs 192 --ep-len 150
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
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--envs", type=int, default=192)
    ap.add_argument("--ep-len", type=int, default=150)     # base needs time to fold to the box, then push
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--kpj", type=float, default=50.0)     # joint-space PD position gain (base)
    ap.add_argument("--kdj", type=float, default=5.0)      # joint-space PD damping (base)
    ap.add_argument("--alpha", type=float, default=0.3)    # residual authority (fraction of forcerange)
    ap.add_argument("--ik-grid", type=int, default=12)     # IK lookup grid resolution per axis
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    import optax
    from mujoco import mjx

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import plan_joint_targets
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    print("backend:", jax.default_backend(), jax.devices())
    gene = tabletop_arm_gene()
    spec = select_task_spec("place the box on the target")
    objs = [o for o in generate_task_scenes(gene, spec, count=1)[0].objects if o.object_type == "cube"]
    xml = compile_gene_with_scene(gene, objs, physics_only=True)  # MJX-safe: no cylinder housings/collars
    mj = mujoco.MjModel.from_xml_string(xml)
    mj.opt.iterations = 10; mj.opt.ls_iterations = 8
    ee = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    box_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box"))
    if box_jid < 0:
        print("ERROR: no free_box joint."); return 1
    # MJX contact-filtering (VERIFIED fix): MJX NaNs within ~12 steps on arm-link<->table contact when
    # the arm folds down (CPU MuJoCo tolerates it). Keep ONLY the meaningful contacts — ee<->box and
    # box<->table — by disabling collision on arm-link geoms. This is the roadmap §0 MJX-safe pass.
    _ee_body = int(mj.site_bodyid[ee]); _box_body = int(mj.jnt_bodyid[box_jid])
    for g in range(mj.ngeom):
        if int(mj.geom_bodyid[g]) not in (0, _ee_body, _box_body):
            mj.geom_contype[g] = 0; mj.geom_conaffinity[g] = 0
    mx = mjx.put_model(mj)
    bq = int(mj.jnt_qposadr[box_jid]); bv = int(mj.jnt_dofadr[box_jid])
    nu, nq, nv = int(mj.nu), int(mj.nq), int(mj.nv)
    act_dof = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(mj.nu)], dtype=int)
    fr = jp.asarray(mj.actuator_forcerange[:, 1])
    act_scale = args.alpha * fr
    # Reachable CONTACT region (CPU reachability map, codesign_reach.py): the arm only folds onto the
    # box for x>=0.33 — spawning over the full table left half the boxes uncontactable. Match task to body.
    lo = jp.array([0.34, -0.12]); hi = jp.array([0.42, 0.12])
    lo_b = lo + 0.01; hi_b = hi - 0.01
    SEP_FLOOR, SEP_START, SEP_CAP = 0.07, 0.10, 0.22
    KPJ, KDJ, DQ = args.kpj, args.kdj, 0.06   # DQ: ref ramp rate (rad/step) — faster fold = more contact time
    N, L, CHUNK = args.envs, args.ep_len, args.chunk
    NACON, NJMAX = 96, 192
    box_z = float(mjx.forward(mx, mjx.make_data(mx, naconmax=NACON, njmax=NJMAX)).qpos[bq + 2])
    obs_dim = nq + nv + 3 + 2 + 3 + 3 + 2
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.97, 0.95, 0.2, 1e-2, 0.5, 3e-4

    # ---- precompute a coarse box(x,y) -> folded-arm joint targets grid (CPU FK-IK, ONE-TIME) -------
    ik_mj = mujoco.MjModel.from_xml_string(xml)
    G = args.ik_grid
    gx = np.linspace(float(lo[0]), float(hi[0]), G)
    gy = np.linspace(float(lo[1]), float(hi[1]), G)
    grid_xy = np.array([(x, y) for x in gx for y in gy], dtype=float)             # (G*G, 2)
    t_ik = time.time()
    grid_q = np.array([plan_joint_targets(ik_mj, (float(x), float(y), box_z), iterations=6, candidates=40)[0]
                       for x, y in grid_xy], dtype=float)                          # (G*G, nu)
    print(f"IK grid: {len(grid_xy)} folded targets precomputed in {time.time()-t_ik:.1f}s", flush=True)
    grid_xy_j = jp.asarray(grid_xy); grid_q_j = jp.asarray(grid_q)

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
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = opt.init(params)

    def obs_of(d, target):
        ee_pos = d.site_xpos[ee]; box = d.qpos[bq:bq + 3]
        obs = jp.concatenate([d.qpos, d.qvel, ee_pos, target, box, ee_pos - box, box[:2] - target])
        return jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def pd_base(d, q_star):
        """Gravity-compensated JOINT-space PD toward the folded FK-IK target (per env) -> ctrl (nu,)."""
        q = d.qpos[act_dof]; qd = d.qvel[act_dof]
        tau = d.qfrc_bias[act_dof] + KPJ * (q_star - q) - KDJ * qd
        return jp.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    def reset(key, sep_max):
        k1, k2, k3 = jax.random.split(key, 3)
        box_xy = jax.random.uniform(k1, (N, 2), minval=lo_b, maxval=hi_b)
        ang = jax.random.uniform(k2, (N,), minval=0.0, maxval=2 * jp.pi)
        rad = jax.random.uniform(k3, (N,), minval=SEP_FLOOR, maxval=sep_max)
        targets = jp.clip(box_xy + jp.stack([jp.cos(ang) * rad, jp.sin(ang) * rad], -1), lo, hi)
        # nearest precomputed FK-IK target for each env's box position (vectorized)
        d2 = jp.sum((box_xy[:, None, :] - grid_xy_j[None, :, :]) ** 2, axis=-1)     # (N, G*G)
        q_star = grid_q_j[jp.argmin(d2, axis=1)]                                     # (N, nu)
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        qpos = data.qpos.at[:, bq].set(box_xy[:, 0]).at[:, bq + 1].set(box_xy[:, 1]).at[:, bq + 2].set(box_z)
        data = data.replace(qpos=qpos)
        data = jax.vmap(lambda d: mjx.forward(mx, d))(data)
        d0 = jp.clip(jp.linalg.norm(box_xy - targets, axis=-1), 0.0, 1.5)
        return data, targets, d0, q_star

    def reward_of(d, targets, prev_d):
        box = d.qpos[:, bq:bq + 3]; ee_pos = d.site_xpos[:, ee]
        box_speed = jp.clip(jp.linalg.norm(jp.nan_to_num(d.qvel[:, bv:bv + 3], posinf=0.0, neginf=0.0), axis=-1), 0.0, 10.0)
        d_push = jp.clip(jp.nan_to_num(jp.linalg.norm(box[:, :2] - targets, axis=-1), nan=1.5, posinf=1.5), 0.0, 1.5)
        d_reach = jp.clip(jp.nan_to_num(jp.linalg.norm(ee_pos - box, axis=-1), nan=1.5, posinf=1.5), 0.0, 1.5)
        in_contact = (d_reach < 0.12).astype(jp.float32)   # base steady-states ~0.12-0.15 in-episode; register it
        controlled_progress = (prev_d - d_push) * in_contact
        speed_excess = jp.clip(box_speed - 0.5, 0.0, 3.0)
        settled = ((d_push < 0.06) & (box_speed < 0.10)).astype(jp.float32)
        rew = 8.0 * controlled_progress - 0.5 * speed_excess - 0.1 * d_reach + 5.0 * settled
        return rew, settled, d_push, d_reach

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, carry, targets, q_star, keys):
        std = jp.exp(params["logstd"])

        def body(carry, key):
            data, prev_d, q_ref = carry
            q_ref = q_ref + jp.clip(q_star - q_ref, -DQ, DQ)                   # ramp reference toward FK-IK target
            obs = jax.vmap(obs_of)(data, targets)
            mean = mlp(params["pi"], obs); val = mlp(params["vf"], obs)[:, 0]
            act = mean + std * jax.random.normal(key, mean.shape)
            logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            base = jax.vmap(pd_base)(data, q_ref)                             # (N, nu) joint-PD base toward the ramp
            ctrl = jp.clip(base + jp.clip(act, -1, 1) * act_scale, -fr, fr)    # base + bounded residual
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            rew, succ, d_push, d_reach = reward_of(data2, targets, prev_d)
            return (data2, d_push, q_ref), (obs, act, logp, rew, val, succ, d_reach)

        return jax.lax.scan(body, carry, keys)

    @jax.jit
    def finish(params, data_last, targets, traj):
        obs, act, logp, rew, val, succ, d_reach = traj
        last_val = mlp(params["vf"], jax.vmap(obs_of)(data_last, targets))[:, 0]

        def gae(carry, x):
            nextval, adv = carry; rew_t, val_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = delta + GAMMA * LAM * adv
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val), reverse=True)
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), succ[-5:].mean(), d_reach.mean()

    def rollout(params, key, sep_max):
        rkey, key = jax.random.split(key)
        data, targets, d0, q_star = reset(rkey, sep_max)
        q_ref0 = data.qpos[:, act_dof]                 # start the reference at the arm's rest pose
        carry = (data, d0, q_ref0); chunks = []
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            carry, out = roll_chunk(params, carry, targets, q_star, jax.random.split(ck, CHUNK))
            jax.block_until_ready(carry[0].qpos)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(7)]
        return finish(params, carry[0], targets, traj)

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

    print(f"residual push (FK-IK joint-PD base): nu={nu} obs={obs_dim} act_dof={list(map(int, act_dof))} "
          f"| {N} envs L={L} CHUNK={CHUNK} kpj={KPJ} kdj={KDJ} alpha={args.alpha}")
    cur_sep = SEP_START; t0 = time.time()
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        obs, act, logp, adv, ret, ep_rew, placed, reach = rollout(params, rk, cur_sep)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if float(placed) > 0.4 and cur_sep < SEP_CAP:   # advance once reliably placing at this difficulty
            cur_sep = min(SEP_CAP, cur_sep + 0.02)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  placed={float(placed):.0%}  "
                  f"reach={float(reach):.3f}m  sep={cur_sep:.2f}m  ({(time.time()-t0):.0f}s)", flush=True)
    print(f"done: residual push, {args.iters} iters, placed {float(placed):.0%}, reach {float(reach):.3f}m, "
          f"sep={cur_sep:.2f}m, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
