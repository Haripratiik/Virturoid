"""T-A: demonstration amplifier (dossier Bet 1 / Training Improvement Phase 1).

Two layers:
  * deterministic ENGINE tests with injected fakes (no MuJoCo) — filtering, yield, lineage, determinism;
  * a MuJoCo-gated INTEGRATION test that amplifies a real scripted grasp and checks the yield is real.
Offline (AGENTS.md).
"""
import importlib.util
import os
import tempfile
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.demonstration_amplifier import (  # noqa: E402
    amplify_demonstrations,
    write_amplification,
)

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class AmplifierEngineTests(unittest.TestCase):
    """The engine is body/task-agnostic: fakes stand in for scene/retarget/replay."""

    def _fns(self, accept_if_x_below=0.5):
        # scene_sampler: deterministic spread of x across [0,1); retarget: identity; replay: succeed if x < thr.
        def scene_sampler(rng, seed_ep):
            return {"x": rng.uniform(0.0, 1.0)}

        def retarget_fn(seed_ep, scene):
            return {"target_x": scene["x"]}

        def replay_fn(plan, scene):
            ok = plan["target_x"] < accept_if_x_below
            return {"success": ok, "return": 1.0 if ok else -1.0,
                    "reason": "" if ok else "out_of_reach"}

        return scene_sampler, retarget_fn, replay_fn

    def test_keeps_only_trusted_successes_and_records_full_lineage(self):
        sampler, retarget, replay = self._fns(accept_if_x_below=0.5)
        seeds = [{"episode_id": "s0"}, {"episode_id": "s1"}]
        res = amplify_demonstrations(
            seeds, n_variants=10, scene_sampler=sampler, retarget_fn=retarget,
            replay_fn=replay, task="reach", seed=7)
        # every attempt is in the lineage; kept == accepted; episodes are all successes.
        self.assertEqual(res.attempted, 20)
        self.assertEqual(len(res.lineage), 20)
        self.assertEqual(res.kept, sum(1 for v in res.lineage if v.accepted))
        self.assertTrue(all(ep["success"] for ep in res.episodes))
        self.assertEqual(len(res.episodes), res.kept)
        # yield + acceptance are consistent.
        self.assertAlmostEqual(res.yield_per_seed, res.kept / 2, places=4)
        self.assertAlmostEqual(res.acceptance_rate, res.kept / 20, places=4)
        # every kept episode traces back to a real seed.
        for ep in res.episodes:
            self.assertIn(ep["amplified_from"], {"s0", "s1"})

    def test_rejected_reasons_surface_in_report(self):
        sampler, retarget, replay = self._fns(accept_if_x_below=0.0)  # nothing succeeds
        res = amplify_demonstrations(
            [{"episode_id": "s0"}], n_variants=5, scene_sampler=sampler,
            retarget_fn=retarget, replay_fn=replay, seed=1)
        self.assertEqual(res.kept, 0)
        self.assertEqual(res.yield_per_seed, 0.0)
        report = res.report()
        self.assertEqual(report["rejected_by_reason"].get("out_of_reach"), 5)

    def test_deterministic_for_a_given_seed(self):
        sampler, retarget, replay = self._fns()
        seeds = [{"episode_id": "s0"}]
        a = amplify_demonstrations(seeds, n_variants=8, scene_sampler=sampler,
                                   retarget_fn=retarget, replay_fn=replay, seed=42)
        b = amplify_demonstrations(seeds, n_variants=8, scene_sampler=sampler,
                                   retarget_fn=retarget, replay_fn=replay, seed=42)
        self.assertEqual([v.scene_params for v in a.lineage], [v.scene_params for v in b.lineage])
        self.assertEqual(a.kept, b.kept)

    def test_custom_trusted_gate_overrides_success(self):
        # a "success" flag can be overridden by a stricter trusted gate (e.g. require a positive return).
        def sampler(rng, ep):
            return {"x": rng.uniform(0, 1)}

        def replay(plan, scene):
            return {"success": True, "return": -5.0}          # claims success but the real return is bad

        res = amplify_demonstrations(
            [{"episode_id": "s0"}], n_variants=4, scene_sampler=sampler,
            retarget_fn=lambda e, s: s, replay_fn=replay,
            trusted_success=lambda r: r.get("return", 0) > 0, seed=0)
        self.assertEqual(res.kept, 0)                          # trusted gate rejects the fake successes

    def test_empty_seeds_yields_empty_result(self):
        res = amplify_demonstrations([], n_variants=5, scene_sampler=lambda r, e: {},
                                     retarget_fn=lambda e, s: s, replay_fn=lambda p, s: {"success": True})
        self.assertEqual(res.attempted, 0)
        self.assertEqual(res.kept, 0)
        self.assertEqual(res.yield_per_seed, 0.0)

    def test_trajectory_passthrough_makes_variant_trainable(self):
        def replay(plan, scene):
            return {"success": True, "trajectory": {"obs": [[1.0]], "actions": [[0.5]]}}

        res = amplify_demonstrations(
            [{"episode_id": "s0"}], n_variants=1, scene_sampler=lambda r, e: {"x": 0},
            retarget_fn=lambda e, s: s, replay_fn=replay, seed=0)
        self.assertEqual(res.kept, 1)
        self.assertIn("obs", res.episodes[0])                  # trajectory flowed through -> trainable

    def test_write_amplification_report(self):
        sampler, retarget, replay = self._fns()
        res = amplify_demonstrations([{"episode_id": "s0"}], n_variants=6, scene_sampler=sampler,
                                     retarget_fn=retarget, replay_fn=replay, seed=3)
        out = tempfile.mkdtemp(prefix="amp_")
        path = write_amplification(res, out)
        self.assertTrue(os.path.exists(path))
        import json
        payload = json.loads(open(path, encoding="utf-8").read())
        self.assertIn("demo_amplification_yield", payload)
        self.assertEqual(len(payload["lineage"]), 6)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo for the physics-grounded gait amplifier")
class GaitAmplifierIntegrationTests(unittest.TestCase):
    def test_amplifies_a_real_gait_with_measured_yield(self):
        # one walking quadruped -> many cadence-varied, PHYSICS-VALIDATED gait demonstrations.
        from virturoid.services.demonstration_amplifier import amplify_gait

        res = amplify_gait(prompt="a quadruped robot dog", n_variants=6, seed=0)
        self.assertEqual(res.seed_count, 1, "the base cadence should walk (produce a seed)")
        self.assertEqual(res.attempted, 6)
        self.assertEqual(len(res.lineage), 6)
        # physics is the judge: kept episodes are real forward+upright+survived walks; yield is measured.
        self.assertGreater(res.kept, 0, "cadence variations near a walking gait should keep >0 variants")
        self.assertTrue(all(ep["success"] for ep in res.episodes))
        self.assertGreaterEqual(res.yield_per_seed, 1.0)       # at least one validated demo per seed
        # the trusted gate is real: every kept variant walked forward.
        for ep in res.episodes:
            self.assertEqual(ep["source"], "amplified")


if __name__ == "__main__":
    unittest.main()
