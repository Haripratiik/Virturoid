"""VirturoidLocomotionEnv — a standard Gym contract over the morph_graph control interface (plan §39), so
off-the-shelf RL trainers can learn a gait, not just our in-house ES. gymnasium is OPTIONAL: the env is
duck-typed (reset/step/terminated/truncated) and works + is testable without it. See [[task-effectiveness-loop]]."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class GymEnvTests(unittest.TestCase):
    def _env(self, horizon=40):
        from virturoid.services.gym_env import VirturoidLocomotionEnv
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a robot dog that walks", llm=None)
        return VirturoidLocomotionEnv(g, horizon=horizon)

    def test_reset_and_step_contract(self):
        import numpy as np
        env = self._env(horizon=20)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (env.n_tokens * env.feature_dim,))
        self.assertIsInstance(info, dict)
        self.assertEqual(env.action_space.shape, (env.n_tokens,))
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(np.zeros(env.n_tokens))
            self.assertEqual(obs.shape, (env.n_tokens * env.feature_dim,))
            self.assertTrue(np.isfinite(reward))
            steps += 1
            self.assertLessEqual(steps, 25)                  # truncates at the horizon, never runs forever
        self.assertTrue(terminated or truncated)

    def test_truncates_at_horizon_or_terminates_on_fall(self):
        import numpy as np
        env = self._env(horizon=15)
        env.reset(seed=1)
        last = None
        for _ in range(15):
            last = env.step(np.zeros(env.n_tokens))
            if last[2]:                                      # terminated (fell) is allowed before horizon
                break
        terminated, truncated = last[2], last[3]
        self.assertTrue(terminated or truncated)

    def test_a_morph_policy_can_drive_the_env(self):
        import numpy as np
        from virturoid.services.morph_policy import MorphPolicy
        env = self._env(horizon=30)
        obs, _ = env.reset(seed=0)
        pol = MorphPolicy(env.feature_dim, seed=0)
        total = 0.0
        for _ in range(30):
            action = env.act_with_morph_policy(pol, obs)     # reshape flat obs -> tokens -> per-token action
            self.assertEqual(np.asarray(action).shape, (env.n_tokens,))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        self.assertTrue(np.isfinite(total))


if __name__ == "__main__":
    unittest.main()
