"""A legged robot can be given a GO-TO-GOAL task and actually completes it.

Two defects fixed together (2026-07-09), both measured through run_task:
  1. ROUTING: the task proposer keyword-mapped "walk/navigate to the goal" to the `navigate` skill (mobile-only),
     so a legged body was wrongly rejected as INFEASIBLE even though a `locomote` skill (which also establishes
     reached_goal) exists. Now the go-to-goal intent picks the skill the body HAS (legged -> locomote).
  2. CONTROLLER: the locomote skill scored travel from locomotion_episode (the learned/trot path), which leaves a
     fresh quad STANDING (-0.02 m) while the open-loop CRAWL gait walks the same body ~0.5 m. Now it adopts the
     crawl walk when it is a genuine CREDIBLE WALK (un-gameable classify) that travels further -> the task
     succeeds on a body that demonstrably walks.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "composing + walking a body needs MuJoCo")
class LeggedGoToGoalTests(unittest.TestCase):
    def _gene(self, prompt):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot(prompt, ensure_walkable=True)

    def test_legged_goto_goal_routes_to_locomote(self):
        from virturoid.services.task_proposer import propose_task
        gene = self._gene("a quadruped robot dog that walks")
        for goal in ("walk to the goal", "navigate to the goal", "go to the target"):
            ts = propose_task(goal, gene, llm=None)
            self.assertEqual(ts.steps[0].skill, "locomote",
                             f"a legged body asked to '{goal}' must route to locomote, not the mobile navigate")

    def test_mobile_goto_goal_still_routes_to_navigate(self):
        from virturoid.services.task_proposer import propose_task
        ts = propose_task("drive to the goal", self._gene("a wheeled rover"), llm=None)
        self.assertEqual(ts.steps[0].skill, "navigate")

    def test_quad_completes_a_goto_goal_task(self):
        from virturoid.services.agent_design_tools import run_task
        from virturoid.services.ai_native_tools import create_robot
        rid = create_robot({"prompt": "a quadruped robot dog that walks"})["robot_id"]
        r = run_task({"robot_id": rid, "goal": "walk to the goal"})
        self.assertTrue(r["feasible"], f"a walking quad must be able to ATTEMPT a go-to-goal task: {r}")
        self.assertTrue(r["success"], f"a quad that credibly walks ~0.5 m must reach a 0.3 m goal: {r}")


if __name__ == "__main__":
    unittest.main()
