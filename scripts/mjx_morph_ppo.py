"""MJX PPO — learned LOCOMOTION for a COMPOSED body (Theme 1, GPU step 1).

Faithful to ``mjx_ppo_min`` (pure-JAX + optax, no brax/flax; host loop over short CHUNK kernels to dodge
the WSL2 GPU watchdog/TDR), changed from reach → locomotion: floor ENABLED with a capped contact budget,
observation built from the morphology graph (``morph_graph.encode_robot`` — so the obs is the same
body-agnostic token layout the CPU MorphPolicy uses), action = bounded residual on gravity compensation
(our proven recipe), reward = forward base velocity + upright − control cost. Trains a composed quadruped
to walk on the GPU — real learned control, the thing scripted gaits can't generalize.

This is GPU step 1 (single body, MLP over the flattened morph obs — highest-confidence reuse of the
proven PPO). Step 2 swaps the MLP for the attention MorphPolicy (already in ``morph_policy.py``) and
trains across a species body distribution for cross-morphology transfer.

    # on the GPU box (after an isolated tar sync; .env excluded):
    PYTHONPATH=src ~/rl/bin/python scripts/mjx_morph_ppo.py --robot quadruped --iters 300 --envs 2048
    PYTHONPATH=src ~/rl/bin/python scripts/mjx_morph_ppo.py --smoke   # single-env validation first
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
    ap.add_argument("--robot", default="quadruped",
                    help="composed robot class to train locomotion for (quadruped|...)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--ep-len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.6, help="bounded-residual action scale")
    ap.add_argument("--smoke", action="store_true", help="single-env, few-step sanity run (CPU→MJX discipline)")
    ap.add_argument("--save", default=None, help="path to save the trained policy (.npz) for banking/reuse")
    args = ap.parse_args(argv)
    if args.smoke:
        args.envs, args.iters, args.ep_len, args.minibatches = 2, 2, 20, 1

    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    import optax
    from mujoco import mjx

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements

    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    gene = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=args.robot, robot_class=args.robot))
    mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
    # Bound the MJX contact kernel: keep robot↔floor contacts, drop robot self-collision (legs rarely
    # need it and uncapped pairs overflow the kernel → CUDA launch failure). Floor = geom 0.
    for g in range(mj.ngeom):
        if g == 0:
            continue
        mj.geom_contype[g] = 1; mj.geom_conaffinity[g] = 1     # collide with floor (contype/affin 1) only
    mj.geom_contype[0] = 1; mj.geom_conaffinity[0] = 1
    NACON, NJMAX = 64, 256
    mx = mjx.put_model(mj)

    graph = encode_robot(mj)
    n_tok = graph.n_tokens
    act_u = jp.asarray(graph.act_u)
    qadr = jp.asarray(graph.qadr); vadr = jp.asarray(graph.vadr)
    static = jp.asarray(graph.static)                          # (n_tok, S) constant structural features
    clamps = jp.asarray(graph.clamps)
    base_jid = graph.base_jid
    base_qadr = graph.base_qadr
    base_vadr = int(mj.jnt_dofadr[base_jid]) if base_jid >= 0 else -1
    nu = int(mj.nu)
    act_dof = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(nu)])
    feat_per_tok = static.shape[1] + 4 + 9                     # static + (qpos,qvel,sin,cos) + base global(9)
    obs_dim = n_tok * feat_per_tok
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.99, 0.95, 0.2, 5e-3, 0.5, 3e-4
    N, L = args.envs, args.ep_len

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
    params = {"pi": init_mlp(kp, [obs_dim, 256, 256, nu]),
              "vf": init_mlp(kv, [obs_dim, 256, 256, 1]),
              "logstd": jp.zeros(nu) - 0.5}
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = opt.init(params)

    def obs_of(d):
        """Flattened morph-token observation for one env (matches the CPU morph_graph layout)."""
        qpos = d.qpos[qadr]; qvel = d.qvel[vadr]
        dyn = jp.stack([qpos, qvel, jp.sin(qpos), jp.cos(qpos)], axis=1)          # (n_tok, 4)
        q = d.qpos[base_qadr:base_qadr + 7]; qd = d.qvel[base_vadr:base_vadr + 6]
        upright = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
        g = jp.array([1.0, q[2], upright, qd[0], qd[1], qd[2], qd[3], qd[4], qd[5]])
        glob = jp.tile(g, (n_tok, 1))                                            # (n_tok, 9)
        obs = jp.concatenate([static, dyn, glob], axis=1).reshape(-1)            # (obs_dim,)
        return jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    def reset(key):
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        return jax.vmap(lambda d: mjx.forward(mx, d))(data)

    def reward_of(d):
        qd = jp.nan_to_num(d.qvel[:, base_vadr:base_vadr + 3], posinf=0.0, neginf=0.0)
        fwd = jp.clip(qd[:, 0], -3.0, 3.0)                                       # forward (+x) base velocity
        quat = d.qpos[:, base_qadr + 3:base_qadr + 7]
        upright = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
        z = jp.nan_to_num(d.qpos[:, base_qadr + 2], posinf=0.0, neginf=0.0)
        alive = (z > 0.12).astype(jp.float32)
        return fwd + 0.3 * upright + 0.2 * alive, fwd

    CHUNK = 5

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, data, keys):
        std = jp.exp(params["logstd"])

        def body(data, key):
            obs = jax.vmap(obs_of)(data)
            mean = mlp(params["pi"], obs)
            val = mlp(params["vf"], obs)[:, 0]
            act = mean + std * jax.random.normal(key, mean.shape)
            logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            # action = bounded residual on gravity comp, written to the actuated DOFs' ctrl
            grav = data.qfrc_bias[:, act_dof]
            ctrl = jp.clip(grav + jp.clip(act, -1, 1) * args.alpha * clamps, -clamps, clamps)
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            rew, fwd = reward_of(data2)
            return data2, (obs, act, logp, rew, val, fwd)

        return jax.lax.scan(body, data, keys)

    @jax.jit
    def finish(params, data_last, traj):
        obs, act, logp, rew, val, fwd = traj
        last_val = mlp(params["vf"], jax.vmap(obs_of)(data_last))[:, 0]

        def gae(carry, x):
            nextval, adv = carry
            rew_t, val_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = delta + GAMMA * LAM * adv
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val), reverse=True)
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), fwd.mean(0).mean()

    def rollout(params, key):
        rkey, key = jax.random.split(key)
        data = reset(rkey)
        chunks = []
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            data, out = roll_chunk(params, data, jax.random.split(ck, CHUNK))
            jax.block_until_ready(data.qpos)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(6)]
        return finish(params, data, traj)

    def loss_fn(params, obs, act, old_logp, adv, ret):
        std = jp.exp(params["logstd"])
        mean = mlp(params["pi"], obs)
        logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
        ratio = jp.exp(logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio, -adv * jp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        v = mlp(params["vf"], obs)[:, 0]
        ent = (params["logstd"] + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
        return pg + VF * ((v - ret) ** 2).mean() - ENT * ent

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

    print(f"robot={args.robot} tokens={n_tok} obs_dim={obs_dim} nu={nu} envs={N} ep_len={L}", flush=True)
    t0 = time.time()
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        obs, act, logp, adv, ret, ep_rew, ep_fwd = rollout(params, rk)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  fwd_vel={float(ep_fwd):+.3f}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)
    print(f"done: {args.robot} locomotion, {args.iters} iters, fwd_vel {float(ep_fwd):+.3f}, "
          f"{time.time()-t0:.0f}s on {jax.default_backend()}", flush=True)
    if args.save:
        # bank the trained policy (params + metadata) for reuse / the skill flywheel / cross-morphology PEFT
        out = {f"pi_w{i}": np.asarray(w) for i, (w, b) in enumerate(params["pi"])}
        out.update({f"pi_b{i}": np.asarray(b) for i, (w, b) in enumerate(params["pi"])})
        out["logstd"] = np.asarray(params["logstd"])
        out["meta"] = np.asarray([n_tok, feat_per_tok, nu, float(ep_fwd)])
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, **out)
        print(f"saved policy -> {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
