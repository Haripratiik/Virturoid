"""Minimal, dependency-light PPO on MJX (jax + optax only) — Phase 3 Milestone 1 (reach).

No brax/flax (they fight bleeding-edge jax). Pure-JAX MLPs + optax. Trains a REACH policy for a
gene-compiled robot on the GPU: drive ee_site to a randomized target. Episodic PPO (reset N envs
each iteration, roll out L steps via lax.scan, GAE, clipped surrogate). Validates the full
gene→MJCF→MJX→PPO chain on the 3060 and produces a real learned policy.

    PYTHONPATH=src ~/rl/bin/python scripts/mjx_ppo_min.py --gene arm --iters 200 --envs 1024
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
    ap.add_argument("--gene", choices=["arm", "humanoid"], default="arm")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--ep-len", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jp
    import mujoco
    import optax
    from mujoco import mjx

    from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    print("backend:", jax.default_backend(), jax.devices())
    gene = humanoid_upper_body_gene() if args.gene == "humanoid" else tabletop_arm_gene()
    # Reach is a FREE-SPACE task: no floor, contacts disabled. MJX has no broadphase and
    # uncapped contacts overflow the contact kernel (CUDA launch failure) when links collide;
    # disabling contacts removes that and is faster. (Contact-rich skills will set max_contacts.)
    mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))
    mj.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    mx = mjx.put_model(mj)
    ee = int(mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site"))
    nu, nq, nv = int(mj.nu), int(mj.nq), int(mj.nv)
    # action [-1,1] -> ctrl scaled to the actuator force range. Reach runs with NO floor and
    # contacts DISABLED, so the old "0.3x to avoid whipping links into the floor" concern doesn't
    # apply — and 0.3x (3.6 Nm on a 12 Nm joint) is too weak to fold the arm DOWN against gravity
    # to reach low/forward targets (it plateaued at ~0.5 m, 0% success). The scripted pick-place
    # controller reaches this same workspace with FULL torque, so give the policy full authority.
    # (nan_to_num on obs + the dist clamp in reward_of guard the rare diverged env.)
    ACT_FRACTION = 1.0
    act_scale = ACT_FRACTION * jp.asarray(mj.actuator_forcerange[:, 1])
    lo = jp.array([0.20, -0.20, 0.05]); hi = jp.array([0.45, 0.20, 0.35])
    obs_dim = nq + nv + 9
    GAMMA, LAM, CLIP, ENT, VF, LR = 0.97, 0.95, 0.2, 1e-2, 0.5, 3e-4
    N, L = args.envs, args.ep_len

    # ---- pure-JAX MLPs (params = pytrees of (W,b)) ----------------------------------------
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

    def obs_of(d, target):
        ee_pos = d.site_xpos[ee]
        obs = jp.concatenate([d.qpos, d.qvel, ee_pos, target, ee_pos - target])
        return jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)  # a diverged env can't poison the net

    step_v = jax.vmap(mjx.step, in_axes=(None, 0))

    def reset(key):
        data = jax.vmap(lambda _: mjx.make_data(mx))(jp.arange(N))
        data = jax.vmap(lambda d: mjx.forward(mx, d))(data)
        targets = jax.random.uniform(key, (N, 3), minval=lo, maxval=hi)
        return data, targets

    def reward_of(d, target):
        # d is BATCHED (N envs): site_xpos is (N, nsite, 3) -> index axis 1 for the ee site of
        # every env. (Indexing [ee] instead grabbed env #ee's whole site block -> garbage dist,
        # flat reward, no learning — the bug that made every run report 0% regardless of reward.)
        ee_pos = d.site_xpos[:, ee]
        dist = jp.linalg.norm(ee_pos - target, axis=-1)
        dist = jp.where(jp.isfinite(dist), dist, 5.0)   # diverged env -> far (penalty), not NaN
        success = (dist < 0.06).astype(jp.float32)
        # -dist gives a gradient EVERYWHERE (the prior tanh(dist/0.1) saturated to 0 across the
        # whole reachable range too). Bonus rewards actually reaching.
        return -dist + 5.0 * success, success, dist

    CHUNK = 5  # steps per fused kernel — short enough to stay under the WSL2 GPU watchdog (TDR).
    # (10 completed a 150-iter run but a 400-iter run tripped CUDA_ERROR_LAUNCH_FAILED; halving the
    # fused-kernel length cuts the per-launch duration so long unattended runs don't trip the watchdog.)

    @functools.partial(jax.jit, static_argnums=())
    def roll_chunk(params, data, targets, keys):
        std = jp.exp(params["logstd"])

        def body(data, key):
            obs = jax.vmap(obs_of)(data, targets)
            mean = mlp(params["pi"], obs)
            val = mlp(params["vf"], obs)[:, 0]
            act = mean + std * jax.random.normal(key, mean.shape)
            logp = (-0.5 * (((act - mean) / std) ** 2 + 2 * jp.log(std) + jp.log(2 * jp.pi))).sum(-1)
            ctrl = jp.clip(act, -1, 1) * act_scale
            data2 = step_v(mx, data.replace(ctrl=ctrl))
            rew, succ, _ = reward_of(data2, targets)
            return data2, (obs, act, logp, rew, val, succ)

        return jax.lax.scan(body, data, keys)  # over CHUNK steps only

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
        return obs, act, logp, advs, advs + val, rew.sum(0).mean(), succ.max(0).mean()

    def rollout(params, key):
        # Host loop over CHUNK-step kernels (like the working M0), NOT one fused L-step kernel —
        # this is what avoids the CUDA launch failure (TDR) on the WSL2 consumer GPU.
        rkey, key = jax.random.split(key)
        data, targets = reset(rkey)
        chunks = []
        for c in range(L // CHUNK):
            key, ck = jax.random.split(key)
            data, out = roll_chunk(params, data, targets, jax.random.split(ck, CHUNK))
            jax.block_until_ready(data.qpos)  # force a short kernel boundary each chunk
            chunks.append(out)
        traj = [jp.concatenate([c[i] for c in chunks], axis=0) for i in range(6)]
        return finish(params, data, targets, traj)

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
        B = obs.shape[0]
        mb = B // args.minibatches
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

    t0 = time.time()
    for it in range(args.iters):
        key, rk, uk = jax.random.split(key, 3)
        obs, act, logp, adv, ret, ep_rew, ep_succ = rollout(params, rk)
        flat = lambda a: a.reshape((-1,) + a.shape[2:])
        params, opt_state = update(params, opt_state, flat(obs), flat(act), flat(logp), flat(adv), flat(ret), uk)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  iter {it:>4}  ep_reward={float(ep_rew):8.2f}  reach_success={float(ep_succ):.0%}  "
                  f"({(time.time()-t0):.0f}s)", flush=True)
    print(f"done: {args.gene} reach, {args.iters} iters, final success {float(ep_succ):.0%}, {time.time()-t0:.0f}s on {jax.default_backend()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
