"""B5 (plan_v5 W-B): the taught examples' out-of-box capability is REAL and honestly labeled. The quadruped
walks and the rover drives straight from submit_design (no training); the hexapod compiles but its scripted
gait is marginal (6+ legs = the learned-control frontier), and the schema SAYS so rather than pretending it
walks. Offline + NO_INTERNAL_LLM."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")  # offline (get_llm -> None); NO module-level
# NO_INTERNAL_LLM — it leaks process-wide during collection and breaks the llm-router tests
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class WalkableDefaultTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_quadruped_example_walks_out_of_the_box(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["quadruped"]})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "full"})
        self.assertTrue(v["credible_walk"], f"the taught quad must walk out of the box, got {v['verdict']}")
        self.assertGreater(v["forward_m"], 0.5)

    def test_rover_example_drives_out_of_the_box(self):
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["rover"]})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "full"})
        self.assertEqual(v["kind"], "mobile")
        self.assertTrue(v["verdict"].startswith("DRIVES"), f"the taught rover must drive, got {v['verdict']}")
        self.assertGreater(v["forward_m"], 0.1)

    def test_example_capability_is_honestly_labeled(self):
        # the schema must NOT pretend the hexapod walks — it tells the agent to train it.
        cap = self._call("get_design_schema")["example_capability"]
        self.assertIn("VERIFIED", cap["quadruped"])
        self.assertIn("VERIFIED", cap["rover"])
        self.assertIn("train_held", cap["hexapod"])            # honest: needs training for a credible gait


if __name__ == "__main__":
    unittest.main()
