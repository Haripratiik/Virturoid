"""Phase 3 Milestone 2 (first contact skill): learn to PUSH a box to a target — contacts ON.

This is the honest first contact skill. A real grasp needs an actuated parallel-jaw gripper gene
+ a friction grasp (MJX-JAX has no adhesion and ignores eq_active, so the scripted _pin_block weld
has no port — see docs/phase4_codesign_plan.md). Non-prehensile PUSHING needs no gripper and no
grasp constraint: the arm makes real frictional contact with the box and shoves it to a goal. It
exercises the whole contact pipeline (the part reach disabled) on the 3060.

Built on mjx_ppo_min.py's machinery (pure-JAX MLP + optax PPO + GAE + the TDR-safe CHUNK host loop)
— the only changes are: contacts ENABLED, obs/reward include the box, reset randomizes box+target,
and conservative envs (contacts make each step heavier, so the TDR budget shrinks).

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_push_skill.py --iters 200 --envs 64 --chunk 2

CONTACT TDR LESSON (2026-06-16): a 192-env / CHUNK=5 contact run hung the WSL2 GPU box (the
default 2s TDR watchdog couldn't recover the heavier contact kernel -> WSL went offline). The
single-step 64-env contact probe was fine. So defaults here are CONSERVATIVE — start at 64 envs /
CHUNK=2 and scale up ONLY after raising TdrDelay on the PC (see docs/phase3_rl_mjx.md). Contacts
make each step far heavier than free-space reach, so the env x chunk budget is much smaller.
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
    ap.add_argument("--envs", type=int, default=192)       # safe now: TdrDelay=60s + capped solver + bounded contact buffers
    ap.add_argument("--ep-len", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=6)        # contacts heavy, but 60s TDR budget tolerates a 6-step fused kernel
    # SCENE-COUNT SCALING ABLATION (scene-gen plan S5/S6): restrict training box-start layouts to a discrete pool
    # of K points and evaluate on a DISJOINT held-out pool. K=0 -> continuous uniform (the original, infinite
    # scenes). Small K overfits to those starts; larger K generalizes -> the ProcTHOR/CoinRun scaling law, on us.
    ap.add_argument("--n-train-layouts", type=int, default=0, help="0=continuous; K>0 restricts box starts to K fixed points")
    ap.add_argument("--eval-heldout-layouts", type=int, default=0, help="if >0, eval held-out box-start success after training")
    ap.add_argument("--layout-seed", type=int, default=0)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jp
    import mujoco
    import optax
    from mujoco import mjx

    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    print("backend:", jax.default_backend(), jax.devices())
    gene = tabletop_arm_gene()
    spec = select_task_spec("place the box on the target")
    scenes = generate_task_scenes(gene, spec, count=1)
    objs = [o for o in scenes[0].objects if o.object_type == "cube"]  # just the box (no bins/containers)
    mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(gene, objs, physics_only=True))  # MJX-safe, contacts ON
    # Cap the Newton solver: bounds per-step solve time so one pathological contact frame can't run a
    # multi-second kernel (the likely cause of the earlier box hang) — stability + TDR headroom.
    mj.opt.iterations = 10
    mj.opt.ls_iterations = 8
    mx = mjx.put_model(mj)

    ee = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    box_jid = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box"))
    if box_jid < 0:
        print("ERROR: no 'free_box' joint in the scene; cannot push."); return 1
    bq = int(mj.jnt_qposadr[box_jid])           # box free-joint qpos start (x,y,z,quat)
    bv = int(mj.jnt_dofadr[box_jid])            # box free-joint qvel start (linear xyz, angular xyz)
    nu, nq, nv = int(mj.nu), int(mj.nq), int(mj.nv)
    act_scale = jp.asarray(mj.actuator_forcerange[:, 1])   # full torque (needed to fold the arm down to the table)
    # Table-top workspace the arm can actually reach (matches the reach script's envelope).
    lo = jp.array([0.24, -0.14]); hi = jp.array([0.42, 0.14])
    N, L, CHUNK = args.envs, args.ep_len, args.chunk
    NACON, NJMAX = 96, 192    # bound per-env contact/constraint buffers (no overflow surprises)
    # Reverse/EXPANDING curriculum (Florensa et al.; COHER): the goal starts just OUTSIDE the success
    # ball and grows only as the policy succeeds. This is the research-backed fix for sparse pushing
    # where a distant goal is never discovered from scratch (from-scratch needs ~100x our compute).
    SEP_FLOOR, SEP_START, SEP_CAP = 0.07, 0.10, 0.22   # >0.06 success ball, so no spawn-on-goal phantom success
    lo_b = lo + 0.05; hi_b = hi - 0.05   # spawn box in an INNER region so target stays in-bounds (no clip-onto-box bug)

    # scene-count ablation pools: deterministic box-start layouts. TRAIN pool = K points; HELD-OUT pool = M points
    # sampled from a different sub-seed so they never coincide with train (the honest structural split).
    import numpy as _np
    def _pool(n, seed):
        r = _np.random.default_rng(seed)
        return jp.asarray(_np.array(lo_b) + r.random((n, 2)) * (_np.array(hi_b) - _np.array(lo_b)), dtype=jp.float32)
    TRAIN_POOL = _pool(args.n_train_layouts, 1000 + args.layout_seed) if args.n_train_layouts > 0 else None
    HELD_POOL = _pool(args.eval_heldout_layouts, 9000 + args.layout_seed) if args.eval_heldout_layouts > 0 else None

    # default box rest height (let the scene decide z; we only randomize x,y).
    _d0 = mjx.forward(mx, mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))
    box_z = float(_d0.qpos[bq + 2])
    obs_dim = nq + nv + 3 + 2 + 3 + 3 + 2   # qpos,qvel, ee(3), target(2), box(3), ee-box(3), box_xy-target(2)
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.97, 0.95, 0.2, 1e-2, 0.5, 3e-4

    def init_mlp(key, sizes):
        ps = []
        for i in range(len(sizes) - 1):
            key, k = jax.random.split(key)
            w = jax.random.normal(k, (sizes[i], sizes[i + 1])) * (1.0 / sizes[i]) ** 0.5
            ps.append((w, jp.zeros(sizes[i + 1])))
        return ps

    def mlp(ps, x):
        for w, b in ps[:-1]:
            x = jp.tanh(x @ w + b)
        w, b = ps[-1]
        return x @ w + b

    key = jax.random.PRNGKey(0)
    key, kp, kv = jax.random.split(key, 3)
    params = {"pi": init_mlp(kp, [obs_dim, 128, 128, nu]),
              "vf": init_mlp(kv, [obs_dim, 256, 256, 1]),
              "logstd": jp.zeros(nu) - 0.5}
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = opt.init(params)

    def obs_of(d, target):  # single env (vmapped)
        ee_pos = d.site_xpos[ee]
        box = d.qpos[bq:bq + 3]
        obs = jp.concatenate([d.qpos, d.qvel, ee_pos, target, box, ee_pos - box, box[:2] - target])
        return jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    def reset(key, sep_max, pool=None):
        k1, k2, k3 = jax.random.split(key, 3)
        if pool is not None:                                     # draw each env's box start from the K-point pool
            idx = jax.random.randint(k1, (N,), 0, pool.shape[0])
            box_xy = pool[idx]
        else:                                                   # continuous uniform = infinite scenes (default)
            box_xy = jax.random.uniform(k1, (N, 2), minval=lo_b, maxval=hi_b)
        # Target a curriculum distance AWAY from the box (random direction): the band [SEP_FLOOR, sep_max]
        # starts tight and widens as the policy improves, so success always demands a real (but currently
        # achievable) push — never the spawn-coincidence that pinned the old baseline.
        ang = jax.random.uniform(k2, (N,), minval=0.0, maxval=2 * jp.pi)
        rad = jax.random.uniform(k3, (N,), minval=SEP_FLOOR, maxval=sep_max)
        targets = jp.clip(box_xy + jp.stack([jp.cos(ang) * rad, jp.sin(ang) * rad], -1), lo, hi)
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        qpos = data.qpos.at[:, bq].set(box_xy[:, 0]).at[:, bq + 1].set(box_xy[:, 1]).at[:, bq + 2].set(box_z)
        data = data.replace(qpos=qpos)
        data = jax.vmap(lambda d: mjx.forward(mx, d))(data)
        d0 = jp.clip(jp.linalg.norm(box_xy - targets, axis=-1), 0.0, 1.5)   # initial box->target dist (for progress reward)
        return data, targets, d0

    def reward_of(d, targets, prev_d):  # BATCHED (N envs); prev_d = last step's box->target distance
        # PROCESS-AWARE, KICK-RESISTANT reward. A naive -distance (or even Δ-distance) reward is
        # hackable: KICKING the box toward the goal scores as well as pushing it under control. We
        # separate the two using physical process signals the sim can measure:
        box = d.qpos[:, bq:bq + 3]
        ee_pos = d.site_xpos[:, ee]
        # bound box_speed (nan_to_num default turns +inf into 3.4e38 -> the iter-0 -7e9 blow-up).
        box_speed = jp.clip(jp.linalg.norm(jp.nan_to_num(d.qvel[:, bv:bv + 3], posinf=0.0, neginf=0.0), axis=-1), 0.0, 10.0)
        d_push = jp.clip(jp.nan_to_num(jp.linalg.norm(box[:, :2] - targets, axis=-1), nan=1.5, posinf=1.5), 0.0, 1.5)
        d_reach = jp.clip(jp.nan_to_num(jp.linalg.norm(ee_pos - box, axis=-1), nan=1.5, posinf=1.5), 0.0, 1.5)
        in_contact = (d_reach < 0.08).astype(jp.float32)        # hand actually on the box
        progress = prev_d - d_push                               # box moved toward goal this step
        controlled_progress = progress * in_contact              # only count progress made WHILE pushing (a free-flying kick earns ~0)
        speed_excess = jp.clip(box_speed - 0.5, 0.0, 3.0)        # BOUNDED anti-kick (v2's unbounded penalty made the policy avoid the box entirely)
        settled = ((d_push < 0.06) & (box_speed < 0.10)).astype(jp.float32)   # PLACED at rest, not flying through
        # Balanced: a STRONG pull to the box (-0.5*d_reach) so the arm engages, generous reward for
        # in-contact progress, a bounded gentleness penalty, and a big settle bonus. Engaging the box
        # must be net-positive or the policy just hides from it.
        rew = 8.0 * controlled_progress - 0.5 * speed_excess - 0.5 * d_reach + 5.0 * settled
        return rew, settled, d_push

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, carry, targets, keys):
        std = jp.exp(params["logstd"])

        def body(carry, key):
            data, prev_d = carry
            obs = jax.vmap(obs_of)(data, targets)
            mean = mlp(params["pi"], obs)
            val = mlp(params["vf"], obs)[:, 0]
            act = mean + std * jax.random.normal(key, mean.shape)
            logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            ctrl = jp.clip(act, -1, 1) * act_scale
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            rew, succ, d_push = reward_of(data2, targets, prev_d)
            return (data2, d_push), (obs, act, logp, rew, val, succ)

        return jax.lax.scan(body, carry, keys)

    @jax.jit
    def finish(params, data_last, targets, traj):
        obs, act, logp, rew, val, succ = traj
        last_val = mlp(params["vf"], jax.vmap(obs_of)(data_last, targets))[:, 0]

        def gae(carry, x):
            nextval, adv = carry
            rew_t, val_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = delta + GAMMA * LAM * adv
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val), reverse=True)
        # success = PLACED at the episode end (box settled in zone on the last few steps) — not "ever
        # within range", which a spawn-near-goal or a fly-through would satisfy. This is the true metric.
        placed = succ[-5:].mean(0)            # fraction of last 5 steps settled, per env
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), placed.mean()

    def rollout(params, key, sep_max, pool=None):
        rkey, key = jax.random.split(key)
        data, targets, d0 = reset(rkey, sep_max, pool)
        carry = (data, d0)
        chunks = []
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            carry, out = roll_chunk(params, carry, targets, jax.random.split(ck, CHUNK))
            jax.block_until_ready(carry[0].qpos)   # short kernel boundary each chunk (TDR)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(6)]
        return finish(params, carry[0], targets, traj)

    def loss_fn(params, obs, act, old_logp, adv, ret):
        std = jp.exp(params["logstd"])
        mean = mlp(params["pi"], obs)
        logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
        ratio = jp.exp(logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio, -adv * jp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        v = mlp(params["vf"], obs)[:, 0]
        vf = ((v - ret) ** 2).mean()
        ent = (params["logstd"] + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
        return pg + VF * vf - ENT * ent

    @jax.jit
    def update(params, opt_state, obs, act, logp, adv, ret, key):
        B = obs.shape[0]; mb = B // args.minibatches

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, B)

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

    print(f"push skill: nu={nu} nq={nq} nv={nv} obs={obs_dim} box_qadr={bq} box_z={box_z:.3f} "
          f"| {N} envs, L={L}, CHUNK={CHUNK} | reverse curriculum {SEP_START:.2f}->{SEP_CAP:.2f}m")
    cur_sep = SEP_START
    t0 = time.time()
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        obs, act, logp, adv, ret, ep_rew, ep_succ = rollout(params, rk, cur_sep, TRAIN_POOL)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        # Expand the curriculum only once the policy is succeeding at the current difficulty.
        if float(ep_succ) > 0.5 and cur_sep < SEP_CAP:
            cur_sep = min(SEP_CAP, cur_sep + 0.02)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  placed={float(ep_succ):.0%}  "
                  f"sep={cur_sep:.2f}m  ({(time.time()-t0):.0f}s)", flush=True)
    print(f"done: push skill, {args.iters} iters, placed {float(ep_succ):.0%} at sep={cur_sep:.2f}m, {time.time()-t0:.0f}s")
    # HELD-OUT EVAL (the scaling-ablation headline): success on box starts NEVER seen in training, at the full
    # curriculum distance, averaged over a few eval batches. A K-overfit policy scores low here; a diverse one high.
    if HELD_POOL is not None:
        succs = []
        for e in range(4):
            key, ek = jax.random.split(key)
            *_, held_succ = rollout(params, ek, SEP_CAP, HELD_POOL)
            succs.append(float(held_succ))
        held = sum(succs) / len(succs)
        train_layouts = args.n_train_layouts if args.n_train_layouts > 0 else -1   # -1 = continuous
        print(f"HELDOUT_SUCCESS n_train_layouts={train_layouts} placed={held:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
