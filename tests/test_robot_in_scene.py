"""B2 (plan_v5 W-B): verify_robot honors a held scene_id — the robot is composed INTO the scene world and
its motion verdict runs among the real obstacles. Before this, scene_id was silently ignored (bare floor),
so 'change the scene to a house' was cosmetic. Now it is physical. Offline + NO_INTERNAL_LLM."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")  # offline (get_llm -> None); NO module-level
# NO_INTERNAL_LLM — it leaks process-wide during collection and breaks the llm-router tests
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class RobotInSceneTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def _cage_scene(self):
        objs = [{"name": f"wall_{d}", "object_type": "wall", "category": "obstacle",
                 "size_xyz": [0.6, 0.08, 0.4], "pose_xyz_rpy": p}
                for d, p in [("n", [0.3, 0, 0.2, 0, 0, 0]), ("s", [-0.3, 0, 0.2, 0, 0, 0]),
                             ("e", [0, 0.3, 0.2, 0, 0, 1.57]), ("w", [0, -0.3, 0.2, 0, 0, 1.57])]]
        return self._call("submit_scene_spec", {"objects": objs, "task": "navigation"})["scene_id"]

    def test_scene_obstacles_physically_constrain_the_robot(self):
        # a rover boxed in by walls travels far less than on a bare floor -> the scene is REAL, not cosmetic.
        rid = self._call("create_robot", {"prompt": "a six-wheeled rover"})["robot_id"]
        bare = self._call("verify_robot", {"robot_id": rid, "mode": "full"})["forward_m"]
        caged = self._call("verify_robot", {"robot_id": rid, "mode": "full", "scene_id": self._cage_scene()})
        self.assertIn("composed into scene", caged["scene_note"])
        self.assertLess(caged["forward_m"], 0.5 * bare + 0.05,
                        f"the cage must impede travel: bare={bare} caged={caged['forward_m']}")

    def test_legged_scene_note_is_honest(self):
        rid = self._call("create_robot", {"prompt": "a quadruped robot dog"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick", "scene_id": self._cage_scene()})
        self.assertEqual(v["kind"], "legged")
        self.assertIn("obstacle-free", v["scene_note"])          # honest: scripted gait isn't obstacle-aware yet

    def test_unknown_scene_id_is_honest_not_a_crash(self):
        rid = self._call("create_robot", {"prompt": "a six-wheeled rover"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick", "scene_id": "nope_zzz"})
        self.assertEqual(v["kind"], "mobile")
        self.assertIn("not found", v["scene_note"])


if __name__ == "__main__":
    unittest.main()
