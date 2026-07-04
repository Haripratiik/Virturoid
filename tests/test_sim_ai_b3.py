"""B3 simulation-AI hardening. Reward-as-code must (a) sandbox out every dangerous construct, (b) compile the safe
templates, (c) pick the winner by TRUSTED success and flag reward-gaming. Scene Author must realize a relational
scene deterministically and its fail-closed gates must reject interpenetration / out-of-reach placements."""

import unittest

import numpy as np

from virturoid.services.reward_dsl import (
    compile_reward, propose_rewards, select_reward, RewardSyntaxError, REWARD_FEATURES)
from virturoid.services.scene_author import (
    Asset, On, Near, Clear, WithinReach, InBounds, realize_scene, validate_scene, scene_to_mjcf, propose_scene,
    Scene, Placement)


class RewardDslSafetyTests(unittest.TestCase):
    def test_rejects_dangerous_constructs(self):
        for bad in ["__import__('os').system('x')", "open('f')", "forward_vel.__class__",
                    "eval('1')", "[x for x in range(3)]", "data[0]", "lambda: 1", "unknown_feature*2"]:
            with self.assertRaises(RewardSyntaxError, msg=bad):
                compile_reward(bad)

    def test_compiles_and_evaluates_whitelisted(self):
        fn = compile_reward("max(0.0, forward_vel)*alive + 0.3*upright - 0.2*slip")
        v = fn({"forward_vel": 0.5, "alive": 1.0, "upright": 0.9, "slip": 0.1})
        self.assertAlmostEqual(v, 0.5 + 0.27 - 0.02, places=6)
        # missing features default to 0.0, and a NaN-prone expr returns 0.0 not NaN
        self.assertEqual(compile_reward("sqrt(forward_vel)")({"forward_vel": -1.0}), 0.0)

    def test_templates_all_compile(self):
        cands = propose_rewards("walk forward", n=5)
        self.assertEqual(len(cands), 5)
        self.assertTrue(all(c.compiled is not None for c in cands))

    def test_select_ranks_by_trusted_success_and_flags_gaming(self):
        cands = propose_rewards("walk", n=4)
        gamed_fn = cands[0].compiled                               # candidate 0 games: huge reward, poor success
        def rollout(fn):
            if fn is gamed_fn:
                return (0.1, 100.0)                                # low success, outlier reward
            return (0.8, 1.0)                                      # honest: good success, modest reward
        out = select_reward(cands, rollout)
        self.assertEqual(out["n_gamed"], 1)                        # exactly the outlier is caught
        self.assertIs(out["best"].compiled, cands[1].compiled)     # an honest candidate wins
        self.assertFalse(out["best"].gamed)
        self.assertGreater(out["best"].trusted_success, 0.5)

    def test_feature_list_is_closed(self):
        # a reward may not smuggle in a new name even if it looks plausible
        with self.assertRaises(RewardSyntaxError):
            compile_reward("reward + forward_vel")


class SceneAuthorTests(unittest.TestCase):
    def _catalog(self):
        return [Asset("table", radius=0.25, height=0.02, kind="support"),
                Asset("cube", radius=0.03, height=0.05),
                Asset("can", radius=0.035, height=0.10),
                Asset("ball", radius=0.03, height=0.06)]

    def test_realizes_and_passes_gates(self):
        assets = self._catalog()
        cons = [On("cube", "table"), On("can", "table"),
                WithinReach("cube", 0.5), WithinReach("can", 0.5),
                Clear("cube", 0.08), Clear("can", 0.08), InBounds("table")]
        sc = realize_scene(assets, cons, seed=1)
        self.assertTrue(sc.solved, sc.report)
        self.assertTrue(sc.report["ok"], sc.report)
        # the 'on' objects rest on the table's surface
        tz = sc.placements["table"].pos[2] + 0.02 / 2
        self.assertAlmostEqual(sc.placements["cube"].pos[2], tz + 0.05 / 2, places=3)

    def test_gate_catches_interpenetration(self):
        a = [Asset("a", radius=0.1, height=0.05), Asset("b", radius=0.1, height=0.05)]
        # hand-craft an overlapping scene and validate: two 0.1m-radius blocks 0.05m apart interpenetrate
        sc = Scene(placements={"a": Placement(np.array([0.0, 0.0, 0.025]), a[0]),
                               "b": Placement(np.array([0.05, 0.0, 0.025]), a[1])},
                   assets={x.id: x for x in a}, constraints=[], base_pos=np.zeros(3),
                   bounds=np.array([[-1, -1, 0], [1, 1, 1]]))
        rep = validate_scene(sc)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("interpenetration" in v for v in rep["violations"]))

    def test_determinism_same_seed(self):
        assets = self._catalog()
        cons = [On("cube", "table"), WithinReach("cube", 0.5), InBounds("table")]
        p1 = realize_scene(assets, cons, seed=7).placements["cube"].pos
        p2 = realize_scene(assets, cons, seed=7).placements["cube"].pos
        self.assertTrue(np.allclose(p1, p2))

    def test_propose_scene_heuristic_is_solvable(self):
        assets, cons = propose_scene("stack the blocks", self._catalog())
        sc = realize_scene(assets, cons, seed=3)
        self.assertTrue(sc.solved, sc.report)
        self.assertIn("<body", scene_to_mjcf(sc))                  # emits MJCF for the free objects


if __name__ == "__main__":
    unittest.main()
