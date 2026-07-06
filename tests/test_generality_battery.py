"""GEN-7 (docs/generality_plan.md): the GENERALITY BATTERY as CI — the one discipline that keeps the product
GENERAL. Every future "make robot X walk" must improve a GENERAL mechanism and show these did not regress,
never add a robot-X-shaped code path. Guards the measured guarantees GEN-1 + GEN-2 established:

  * a diverse set of prompted bodies is discovered correctly from STRUCTURE (leg counts, wheels, spine);
  * EVERY legged body gets a firing wave gait (tokens oscillate) — not the name-gated 0/20-token scalar
    shuffle that made "hexapods don't work";
  * every legged body travels FORWARD and upright under the general scripted gait (honest: reported forward
    matches the actual displacement — no abs-masked backward walk), and the quad credibly walks.

Offline (llm=None) so it is deterministic in CI. The full LLM battery runs nightly (not here).
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class GeneralityBatteryTests(unittest.TestCase):
    def _build(self, prompt):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot(prompt, llm=None)

    def test_structural_discovery_across_morphologies(self):
        # GEN-1: leg/wheel/spine counts READ from structure, not names/labels — an 8-leg spider is 8, a snake
        # is a spine, a rover is wheels, however the builder named the links.
        from virturoid.services.appendage_map import build_appendage_map
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        def amap(p):
            return build_appendage_map(compiled_model(robot_mjcf(self._build(p))))
        self.assertEqual(amap("a quadruped robot dog").n_legs, 4)
        self.assertEqual(amap("a hexapod").n_legs, 6)
        self.assertEqual(amap("a spider").n_legs, 8)
        self.assertIsNotNone(amap("a snake robot").spine)
        self.assertGreaterEqual(amap("a wheeled rover").n_wheels, 2)

    def test_every_legged_body_gets_a_firing_gait(self):
        # GEN-2: the wave-gait engine synthesizes a gait for ANY leg count — the anti-regression for the
        # name-gated CPG (an LLM hexapod named front_leg_1_l_0 got 0/20 oscillating tokens -> scalar shuffle).
        import numpy as np

        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import (CPG_DEFAULT, _trot_cpg_tokens, compiled_model, robot_mjcf)
        for p in ("a hexapod", "an octopod", "a quadruped robot dog"):
            m = compiled_model(robot_mjcf(self._build(p))); gr = encode_robot(m)
            amp, _phase, gate = _trot_cpg_tokens(m, gr, CPG_DEFAULT)
            self.assertTrue(gate, f"{p}: a legged body must get a firing gait (not the scalar shuffle)")
            self.assertGreater(int(np.sum(np.abs(amp) > 0)), 0, f"{p}: gait tokens must oscillate")

    def test_legged_bodies_walk_forward_upright_and_honestly(self):
        # GEN-2/GEN-3: under the general scripted gait every legged body travels FORWARD (reported == actual,
        # no abs-masked backward walk) and stays upright; the quad credibly walks once its stance is fanned.
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.morph_policy import crawl_gait_rollout
        from scripts.verify_gait import classify
        walks = 0
        for p in ("a quadruped robot dog", "a hexapod", "an octopod", "a centipede"):
            g = ensure_walkable_quad(self._build(p), p)
            r = crawl_gait_rollout(g, steps=1200, record_qpos=True)
            net = r["qpos_frames"][-1][0] - r["qpos_frames"][0][0]
            self.assertLess(abs(r["forward"] - net), 0.06, f"{p}: reported forward must match actual displacement")
            self.assertGreater(r["forward"], 0.0, f"{p}: must travel FORWARD (+x), not backward/in-place")
            self.assertGreater(r["height_ratio"], 0.6, f"{p}: must stay upright, not collapse")
            if classify(r).startswith("CREDIBLE"):
                walks += 1
        self.assertGreaterEqual(walks, 1, "at least the fanned quad must be a CREDIBLE WALK (baseline was 0)")


if __name__ == "__main__":
    unittest.main()
