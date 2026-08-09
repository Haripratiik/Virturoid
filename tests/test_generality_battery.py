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
        # THIS TEST USED TO BE SATISFIED BY A SLIDE. It asserted `forward > 0.0` and `height_ratio > 0.6`, and
        # BOTH are cleared by a body dragged along the floor that never lifts a foot — which is how the #140
        # "many-leg gait SOLVED" claim was published off a centipede that travelled 0.037 m in 6000 steps at
        # cadence 0.0 and support_frac 0.000. The claim is retracted; this is the assertion that let it through.
        #
        # The fix is to judge with the ruler the PRODUCT judges with. `gait_quality.classify` returns SLIDE
        # exactly when `cadence < 1.0 or support_frac < 0.25` — feet barely lift — and CIRCLE/MILLS/TURNS OFF
        # COURSE when the body went round rather than anywhere. Both are asserted per body, so a regression to
        # dragging or to circling FAILS instead of scoring a walk.
        #
        # MEASURED 2026-08-09 on this checkout, so a future reader can see the margin rather than guess it:
        #   dog      forward 0.664  cadence 15.83  support 0.977  -> CREDIBLE WALK
        #   hexapod  forward 1.121  cadence 50.42  support 0.981  -> CREDIBLE WALK
        #   octopod  forward 0.119  cadence 42.50  support 0.978  -> FORWARD BUT SHORT
        #   centipede forward 0.248 cadence 62.08  support 1.000  -> FORWARD BUT SHORT
        # The octopod and the centipede are NOT credible walks, and this test does not claim they are. What it
        # claims — and what its name claims — is that every legged body STEPS, travels forward, and stays up.
        walks = 0
        for p in ("a quadruped robot dog", "a hexapod", "an octopod", "a centipede"):
            g = ensure_walkable_quad(self._build(p), p)
            r = crawl_gait_rollout(g, steps=1200, record_qpos=True)
            net = r["qpos_frames"][-1][0] - r["qpos_frames"][0][0]
            v = classify(r)
            self.assertLess(abs(r["forward"] - net), 0.06, f"{p}: reported forward must match actual displacement")
            self.assertGreater(r["forward"], 0.0, f"{p}: must travel FORWARD (+x), not backward/in-place")
            self.assertGreater(r["height_ratio"], 0.6, f"{p}: must stay upright, not collapse")
            # ...and it must have WALKED there. A slide clears every line above.
            self.assertNotIn("SLIDE", v, f"{p}: dragged along the floor, it did not step — {v}")
            self.assertGreaterEqual(r["cadence"], 1.0, f"{p}: no stepping rhythm (cadence {r['cadence']}) — {v}")
            self.assertGreaterEqual(r["support_frac"], 0.25, f"{p}: feet never planted ({r['support_frac']}) — {v}")
            # ...and it must have gone SOMEWHERE. A closed loop books its far side as +x delta (test_gait_course_gate).
            for bad in ("CIRCLE", "MILLS", "TURNS OFF COURSE"):
                self.assertNotIn(bad, v, f"{p}: went round rather than anywhere — {v}")
            if v.startswith("CREDIBLE"):
                walks += 1
        self.assertGreaterEqual(walks, 1, "at least the fanned quad must be a CREDIBLE WALK (baseline was 0)")


if __name__ == "__main__":
    unittest.main()
