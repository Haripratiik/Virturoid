"""A snake robot CRAWLS on land by serpentine undulation — the limbless-spine analogue of the swim close.

Before: a snake (a limbless serial spine) was classified 'legged' and driven by the LEG-based crawl gait, which
can't drive a body with no legs -> it fell (FELL by ROLL-OVER, locomotion 0.0). Now a limbless serial spine is
driven by a lateral travelling wave down the spine (morph_policy.serpentine_rollout) -> it undulates forward
against ground friction (measured ~0.27-0.44 m). Wired through verify / evaluate / the locomote task so a snake
is scored on the locomotion it actually implies, not a leg gait it can't run.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "serpentine needs the MuJoCo sim")
class SerpentineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from virturoid.services.morphology_composer import _compose_robot_impl
        cls.snake = _compose_robot_impl("a snake robot", llm=None)

    def test_serpentine_rollout_moves_a_snake_forward(self):
        from virturoid.services.morph_policy import serpentine_rollout
        r = serpentine_rollout(self.snake, steps=2000)
        self.assertTrue(r["finite"])
        self.assertGreater(r["planar_m"], 0.15, f"a snake must undulate forward on land: {r}")

    def test_snake_is_a_serial_spine_not_a_walker(self):
        from virturoid.services.aquatic import _is_serial_spine
        self.assertTrue(_is_serial_spine(self.snake), "a snake is a single non-branching revolute chain")

    def test_verify_reports_serpentine_crawl_not_a_fall(self):
        from virturoid.services import session_state as S
        from virturoid.services.ai_native_tools import create_robot, verify_robot
        S.reset()
        rid = create_robot({"prompt": "a snake robot"})["robot_id"]
        v = verify_robot({"robot_id": rid, "mode": "quick"})
        self.assertEqual(v.get("gait_source"), "serpentine")
        self.assertIn("CRAWLS", v["verdict"])
        self.assertNotIn("FELL", v["verdict"])

    def test_run_task_move_forward_succeeds_for_a_snake(self):
        from virturoid.services import session_state as S
        from virturoid.services.agent_design_tools import run_task
        from virturoid.services.ai_native_tools import create_robot
        S.reset()
        rid = create_robot({"prompt": "a snake robot"})["robot_id"]
        r = run_task({"robot_id": rid, "goal": "move forward"})
        self.assertTrue(r.get("feasible"))
        self.assertTrue(r.get("success"), f"a snake must complete 'move forward' via serpentine: {r}")

    def test_evaluate_scores_a_snake_on_serpentine(self):
        from virturoid.services.task_matched_eval import evaluate_robot
        ev = evaluate_robot(self.snake, prompt="a snake robot")
        self.assertEqual(ev["task"], "serpentine_locomotion")
        self.assertGreater(ev["value"], 0.15)

    def test_a_quadruped_is_not_misrouted_to_serpentine(self):
        # a branching legged body (quad) must keep the leg gait, not be treated as a spine
        from virturoid.services.aquatic import _is_serial_spine
        from virturoid.services.morphology_composer import compose_robot
        quad = compose_robot("a quadruped robot dog", ensure_walkable=True)
        self.assertFalse(_is_serial_spine(quad), "a quad branches into legs — not a serial spine")


if __name__ == "__main__":
    unittest.main()
