"""Aquatic robots actually SWIM — the body/controller co-design that closed the swim frontier on CPU.

Before: a "fish robot" composed a round-bodied quadruped (a dog) that barely moved in water (~0.04 m, DOES NOT
SWIM). Now aquatic prompts compose a serial spine, laterally-compressed into a fish cross-section (drag anisotropy),
driven by a tuned travelling wave -> genuine undulatory thrust (measured ~0.20 m, SWIMS). Round vs flat is the whole
difference (measured 0.06 m round -> 0.25 m flat on the same spine).
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class AquaticHelperTests(unittest.TestCase):
    def test_is_aquatic_prompt(self):
        from virturoid.services.aquatic import is_aquatic_prompt
        for p in ("a fish robot", "an eel that swims", "a shark", "an underwater submarine"):
            self.assertTrue(is_aquatic_prompt(p), p)
        for p in ("a quadruped robot dog", "a robotic arm", "a wheeled rover"):
            self.assertFalse(is_aquatic_prompt(p), p)

    def test_cross_section_survives_serialization(self):
        from virturoid.schemas.gene import GeneSegment, RobotGene
        g = RobotGene(id="t", species="s", robot_class="aquatic", segments=[
            GeneSegment(name="b0", shape="box", cross_section=(0.02, 0.09))])
        g2 = RobotGene.from_dict(g.to_dict())
        self.assertEqual(tuple(g2.segments[0].cross_section), (0.02, 0.09))


@unittest.skipUnless(_MUJOCO, "swimming needs the MuJoCo fluid sim")
class SwimTests(unittest.TestCase):
    def test_fish_composes_an_aquatic_undulator_and_swims(self):
        from virturoid.services.ai_native_tools import _honest_swim
        from virturoid.services.morphology_composer import compose_robot
        for p in ("a fish robot that swims forward", "an eel robot", "a shark robot"):
            g = compose_robot(p)
            self.assertEqual(g.robot_class, "aquatic", f"{p} must compose an aquatic body, not a quadruped")
            self.assertTrue(any(s.cross_section for s in g.segments), "segments must be laterally compressed")
            r = _honest_swim(g, steps=1500)
            self.assertGreater(r["swim_m"], 0.15, f"{p} must SWIM (>0.15 m): {r}")
            self.assertIn("SWIMS", r["verdict"])

    def test_flat_body_out_swims_round_body(self):
        # the drag-anisotropy lever: the SAME spine swims further flat than round
        import copy

        from virturoid.services.ai_native_tools import _honest_swim
        from virturoid.services.morphology_composer import _compose_robot_impl
        round_g = _compose_robot_impl("a snake robot", llm=None)
        flat_g = copy.deepcopy(round_g)
        from virturoid.services.aquatic import ensure_aquatic_body
        ensure_aquatic_body(flat_g)
        rr = _honest_swim(round_g, steps=1500)["swim_m"]
        rf = _honest_swim(flat_g, steps=1500)["swim_m"]
        self.assertGreater(rf, rr + 0.05, f"a flat (fish) body must out-swim the round one: flat={rf} round={rr}")

    def test_non_aquatic_body_is_not_reshaped(self):
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped robot dog that walks")
        self.assertNotEqual(g.robot_class, "aquatic")
        self.assertFalse(any(s.cross_section for s in g.segments))

    def test_fish_is_classified_aquatic_not_legged(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_matched_eval import robot_capabilities, robot_kind
        g = compose_robot("a fish robot that swims")
        self.assertEqual(robot_kind(g), "aquatic", "a spine undulator must classify aquatic, not legged")
        self.assertIn("aquatic", robot_capabilities(g))

    def test_run_task_swims_a_fish(self):
        # the TASK layer: a fish must RUN a swim task (feasible + success), not route to a failing walk skill
        from virturoid.services import session_state as S
        from virturoid.services.agent_design_tools import run_task
        from virturoid.services.ai_native_tools import create_robot
        S.reset()
        rid = create_robot({"prompt": "a fish robot that swims forward"})["robot_id"]
        for goal in ("swim forward", "swim to the target"):
            r = run_task({"robot_id": rid, "goal": goal})
            self.assertTrue(r.get("feasible"), f"a fish must be able to attempt '{goal}': {r}")
            self.assertTrue(r.get("success"), f"a fish must complete '{goal}' via the swim skill: {r}")

    def test_evaluate_robot_scores_a_fish_on_swim(self):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_matched_eval import evaluate_robot
        ev = evaluate_robot(compose_robot("an eel robot"), prompt="an eel")
        self.assertEqual(ev["task"], "swim")
        self.assertGreater(ev["value"], 0.15, f"a real undulator swims past the threshold: {ev}")


if __name__ == "__main__":
    unittest.main()
