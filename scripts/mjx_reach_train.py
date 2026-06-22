"""Phase 3 Milestone 1 (reach): train a learned REACH policy on MJX for a gene robot (GPU).

A pure-physics first RL skill: drive the end-effector (`ee_site`) to a randomized target in
the workspace. No grasp physics (that needs an articulated gripper) — this validates the full
gene→MJCF→MJX→Brax-PPO chain on the 3060 and trains a real policy. Reward is bounded
(`1 - tanh(dist/σ)`) + a sparse success bonus, per the RewardSpec discipline.

    PYTHONPATH=src ~/jax/bin/python scripts/mjx_reach_train.py --gene arm --timesteps 2000000 --envs 1024
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def build_env(gene_name: str):
    import jax
    import jax.numpy as jp
    import mujoco
    from brax.base import State as PipelineState  # noqa: F401  (type only)
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf

    from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    gene = humanoid_upper_body_gene() if gene_name == "humanoid" else tabletop_arm_gene()
    mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, physics_only=True))  # MJX-safe
    ee_site = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    # Reachable target box (in front of the robot, on/above the table).
    lo = jp.array([0.20, -0.20, 0.05]); hi = jp.array([0.45, 0.20, 0.35])

    class ReachEnv(PipelineEnv):
        def __init__(self):
            sys = mjcf.load_model(mj)
            super().__init__(sys=sys, backend="mjx", n_frames=5)
            self._ee = ee_site

        def _ee_pos(self, ps):
            return ps.site_xpos[self._ee]

        def _obs(self, ps, target):
            return jp.concatenate([ps.qpos, ps.qvel, self._ee_pos(ps), target, self._ee_pos(ps) - target])

        def reset(self, rng):
            rng, tk = jax.random.split(rng)
            target = jax.random.uniform(tk, (3,), minval=lo, maxval=hi)
            ps = self.pipeline_init(self.sys.qpos0, jp.zeros(self.sys.nv))
            obs = self._obs(ps, target)
            info = {"target": target, "success": jp.float32(0)}
            return State(ps, obs, jp.float32(0), jp.float32(0), {}, info)

        def step(self, state, action):
            ps = self.pipeline_step(state.pipeline_state, action)
            target = state.info["target"]
            dist = jp.linalg.norm(self._ee_pos(ps) - target)
            success = (dist < 0.05).astype(jp.float32)
            reward = (1.0 - jp.tanh(dist / 0.1)) + 2.0 * success - 0.01 * jp.sum(action ** 2)
            obs = self._obs(ps, target)
            info = {**state.info, "success": success}
            return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=jp.float32(0), info=info)

    return ReachEnv()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", choices=["arm", "humanoid"], default="arm")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--episode", type=int, default=120)
    args = ap.parse_args(argv)

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    print("backend:", jax.default_backend(), jax.devices())
    env = build_env(args.gene)

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        s = metrics.get("eval/episode_success", metrics.get("eval/episode_reward", float("nan")))
        print(f"  step {step:>9,}  eval_reward={r:8.3f}  success~={s}", flush=True)

    net = functools.partial(ppo_networks.make_ppo_networks,
                            policy_hidden_layer_sizes=(128, 128), value_hidden_layer_sizes=(256, 256))
    t0 = time.time()
    make_inf, params, _ = ppo.train(
        environment=env, num_timesteps=args.timesteps, num_evals=6,
        episode_length=args.episode, num_envs=args.envs, batch_size=256, num_minibatches=8,
        unroll_length=15, num_updates_per_batch=8, learning_rate=3e-4, entropy_cost=1e-2,
        discounting=0.97, normalize_observations=True, network_factory=net, seed=0, progress_fn=progress)
    print(f"trained {args.gene} reach in {time.time()-t0:.0f}s on {jax.default_backend()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
