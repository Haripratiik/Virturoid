"""MJX PPO — VELOCITY-COMMAND-conditioned attention MorphPolicy (toward learned steerable navigation).

The low half of the hierarchical-navigation recipe (legged_gym / ViNL): a policy conditioned on a
commanded body velocity (forward v_x, yaw rate w_z), rewarded for TRACKING it (the field-standard
exp(-err^2/sigma) linear- and angular-velocity tracking terms) + an upright bonus. Commands are sampled
per episode. The trained policy then takes commands from a high-level planner (A* maze path →
pure-pursuit → (v_x, w_z)) so a composed LEGGED body follows a path through a maze — the thing scripted
gaits failed at. Same attention arch + obs layout as ``mjx_morph_attention.py`` but with a 2-d command
appended to each token's features (feature_dim 24→26), so the CPU follower feeds the same obs.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_morph_vel.py --robot quadruped --iters 120 --envs 1024 \
        --save runs/morph_quad_vel.npz
    PYTHONPATH=src ~/rl/bin/python scripts/mjx_morph_vel.py --smoke
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
    ap.add_argument("--robot", default="quadruped")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--ep-len", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save", default=None)
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
    if args.robot in ("frog", "steerable"):
        from virturoid.services.steerable_body import steerable_quadruped
        gene = steerable_quadruped(species=args.robot)
    else:
        gene = compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=args.robot, robot_class=args.robot))
    mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
    for g in range(mj.ngeom):
        mj.geom_contype[g] = 1; mj.geom_conaffinity[g] = 1
    NACON, NJMAX = 64, 256
    mx = mjx.put_model(mj)

    graph = encode_robot(mj)
    NT = graph.n_tokens
    qadr = jp.asarray(graph.qadr); vadr = jp.asarray(graph.vadr)
    static = jp.asarray(graph.static)
    clamps = jp.asarray(graph.clamps)
    base_qadr = graph.base_qadr
    base_vadr = int(mj.jnt_dofadr[graph.base_jid])
    nu = int(mj.nu)
    act_dof = jp.asarray([int(mj.jnt_dofadr[int(mj.actuator_trnid[u, 0])]) for u in range(nu)])
    F = static.shape[1] + 4 + 9 + 2                       # + 2-d velocity command
    H = args.hidden
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.99, 0.95, 0.2, 5e-3, 0.5, 3e-4
    N, L = args.envs, args.ep_len
    VX_LO, VX_HI, WZ_ABS = -1.0, 1.0, 1.0                 # forward AND backward + turns: a constant gait
    #                                                       can't win the tracking reward -> forces the
    #                                                       policy to actually FOLLOW the command (legged_gym)

    key = jax.random.PRNGKey(0)
    def W(k, shape, s=0.3):
        return jax.random.normal(k, shape) * s
    ks = jax.random.split(key, 9)
    att = {"We": W(ks[0], (F, H)), "be": jp.zeros(H), "Wq": W(ks[1], (H, H)), "Wk": W(ks[2], (H, H)),
           "Wv": W(ks[3], (H, H)), "Wo": W(ks[4], (H, H)), "Wh": W(ks[5], (H, 1)), "bh": jp.zeros(1)}
    vf = [(W(ks[6], (H, 64)) * (1 / H) ** 0.5, jp.zeros(64)), (W(ks[7], (64, 1)) * (1 / 64) ** 0.5, jp.zeros(1))]
    params = {"att": att, "vf": vf, "logstd": jp.zeros(1) - 0.5}
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = opt.init(params)

    def obs_tokens(d, cmd):                               # cmd (2,) for this env
        qpos = d.qpos[qadr]; qvel = d.qvel[vadr]
        dyn = jp.stack([qpos, qvel, jp.sin(qpos), jp.cos(qpos)], axis=1)
        q = d.qpos[base_qadr:base_qadr + 7]; qd = d.qvel[base_vadr:base_vadr + 6]
        upright = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
        g = jp.array([1.0, q[2], upright, qd[0], qd[1], qd[2], qd[3], qd[4], qd[5]])
        glob = jp.concatenate([g, cmd])                   # (11,)
        return jp.nan_to_num(jp.concatenate([static, dyn, jp.tile(glob, (NT, 1))], axis=1))

    def policy_mean_pool(P, obs):
        a = P["att"]
        e = jp.tanh(obs @ a["We"] + a["be"])
        q, k, v = e @ a["Wq"], e @ a["Wk"], e @ a["Wv"]
        s = (q @ k.T) / jp.sqrt(H); s = s - s.max(axis=1, keepdims=True)
        w = jp.exp(s); w = w / w.sum(axis=1, keepdims=True)
        u = jp.tanh(e + (w @ v) @ a["Wo"])
        return jp.tanh(u @ a["Wh"] + a["bh"])[:, 0], u.mean(0)

    def vmlp(ps, x):
        x = jp.tanh(x @ ps[0][0] + ps[0][1])
        return x @ ps[1][0] + ps[1][1]

    def act_and_value(P, obs_b):
        means, pools = jax.vmap(lambda o: policy_mean_pool(P, o))(obs_b)
        return means, vmlp(P["vf"], pools)[:, 0]

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    def reset(key):
        data = jax.vmap(lambda _: mjx.make_data(mx, naconmax=NACON, njmax=NJMAX))(jp.arange(N))
        data = jax.vmap(lambda d: mjx.forward(mx, d))(data)
        k1, k2 = jax.random.split(key)
        vx = jax.random.uniform(k1, (N, 1), minval=VX_LO, maxval=VX_HI)
        wz = jax.random.uniform(k2, (N, 1), minval=-WZ_ABS, maxval=WZ_ABS)
        return data, jp.concatenate([vx, wz], axis=1)     # commands (N,2), fixed per episode

    def reward_of(d, cmd):
        q = d.qpos[:, base_qadr + 3:base_qadr + 7]
        yaw = jp.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        vw = jp.nan_to_num(d.qvel[:, base_vadr:base_vadr + 3], posinf=0.0, neginf=0.0)
        vx_body = vw[:, 0] * jp.cos(yaw) + vw[:, 1] * jp.sin(yaw)      # forward speed in base frame
        wz = jp.nan_to_num(d.qvel[:, base_vadr + 5], posinf=0.0, neginf=0.0)
        upright = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        track_lin = jp.exp(-((vx_body - cmd[:, 0]) ** 2) / 0.25)
        track_ang = jp.exp(-((wz - cmd[:, 1]) ** 2) / 0.25)   # steering matters as much as forward now
        rew = track_lin + track_ang + 0.2 * upright
        return rew, track_lin

    CHUNK = 5

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, data, cmd, keys):
        std = jp.exp(params["logstd"])[0]

        def body(data, key):
            obs = jax.vmap(obs_tokens)(data, cmd)
            means, val = act_and_value(params, obs)
            act = means + std * jax.random.normal(key, means.shape)
            logp = (-0.5 * (((act - means) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            ctrl = jp.clip(data.qfrc_bias[:, act_dof] + jp.clip(act, -1, 1) * args.alpha * clamps, -clamps, clamps)
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            rew, tl = reward_of(data2, cmd)
            return data2, (obs, act, logp, rew, val, tl)

        return jax.lax.scan(body, data, keys)

    @jax.jit
    def finish(params, data_last, cmd, traj):
        obs, act, logp, rew, val, tl = traj
        _, last_val = act_and_value(params, jax.vmap(obs_tokens)(data_last, cmd))

        def gae(carry, x):
            nextval, adv = carry
            rew_t, val_t = x
            delta = rew_t + GAMMA * nextval - val_t
            adv = delta + GAMMA * LAM * adv
            return (val_t, adv), adv
        _, advs = jax.lax.scan(gae, (last_val, jp.zeros(N)), (rew, val), reverse=True)
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), tl.mean(0).mean()

    def rollout(params, key):
        rkey, key = jax.random.split(key)
        data, cmd = reset(rkey)
        chunks = []
        for _ in range(L // CHUNK):
            key, ck = jax.random.split(key)
            data, out = roll_chunk(params, data, cmd, jax.random.split(ck, CHUNK))
            jax.block_until_ready(data.qpos)
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(6)]
        return finish(params, data, cmd, traj)

    def loss_fn(params, obs, act, old_logp, adv, ret):
        std = jp.exp(params["logstd"])[0]
        means, val = act_and_value(params, obs)
        logp = (-0.5 * (((act - means) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
        ratio = jp.exp(logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio, -adv * jp.clip(ratio, 1 - CLIP, 1 + CLIP)).mean()
        ent = (params["logstd"] + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
        return pg + VF * ((val - ret) ** 2).mean() - ENT * ent

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

    def _save(p, score):
        a = p["att"]
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save, We=np.asarray(a["We"]), be=np.asarray(a["be"]), Wq=np.asarray(a["Wq"]),
                 Wk=np.asarray(a["Wk"]), Wv=np.asarray(a["Wv"]), Wo=np.asarray(a["Wo"]),
                 Wh=np.asarray(a["Wh"]), bh=np.asarray(a["bh"]), meta=np.asarray([F, H, NT, float(score)]))

    print(f"VEL-CMD robot={args.robot} tokens={NT} F={F} H={H} nu={nu} envs={N}", flush=True)
    t0 = time.time(); track = 0.0
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        obs, act, logp, adv, ret, ep_rew, track = rollout(params, rk)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  lin_track={float(track):.3f}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)
        if args.save and it > 0 and it % 30 == 0:
            _save(params, track); print(f"  [checkpoint @ iter {it}]", flush=True)
    print(f"done: vel-cmd {args.robot}, lin_track {float(track):.3f}, {time.time()-t0:.0f}s", flush=True)
    if args.save:
        _save(params, track); print(f"saved -> {args.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
