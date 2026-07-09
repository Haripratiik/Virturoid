"""The maze task family is SUPPORTED — and the planner must say so.

services/maze.py (A* over a generated maze) + the solve_maze skill drive a mobile base through the walls to the
goal; run_task('solve the maze') returns success. The intent_planner used to list 'maze' as UNSUPPORTED ("no maze
task family yet") — a stale UNDER-claim (telling users we can't do something we demonstrably can). Removed, guarded
here so it can't creep back.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class MazePlanningTests(unittest.TestCase):
    def test_astar_finds_a_path_to_the_goal(self):
        from virturoid.services.maze import astar, generate_maze
        path = astar(generate_maze(n=11, seed=0))
        self.assertTrue(path and len(path) > 5, "A* must route through the maze to the goal")

    def test_maze_prompt_is_not_falsely_flagged_unsupported(self):
        from virturoid.services.intent_planner import plan_build
        p = plan_build("a robot that solves a maze", llm=None).to_dict()
        self.assertTrue(p["buildable"])
        self.assertEqual(p["robot_class"], "mobile_base")
        self.assertFalse(any("maze" in g.lower() for g in p["gaps"]),
                         f"maze is a supported task family — it must not be gapped: {p['gaps']}")


@unittest.skipUnless(_MUJOCO, "run_task solve_maze needs MuJoCo")
class MazeTaskTests(unittest.TestCase):
    def test_mobile_robot_solves_a_maze_end_to_end(self):
        from virturoid.services.agent_design_tools import run_task
        from virturoid.services.ai_native_tools import create_robot
        rid = create_robot({"prompt": "a wheeled rover"})["robot_id"]
        r = run_task({"robot_id": rid, "goal": "solve the maze"})
        self.assertTrue(r["feasible"], f"a mobile robot must be able to attempt a maze: {r}")
        self.assertTrue(r["success"], f"solve_maze (A* plan + drive) must reach the goal: {r}")
        self.assertIn("solve_maze", [s.get("skill") for s in r.get("steps", [])])


if __name__ == "__main__":
    unittest.main()
