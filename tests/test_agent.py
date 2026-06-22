import importlib.util
import tempfile
import unittest
from pathlib import Path

from virturoid.services.agent import VirturoidAgent, parse_intent

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class IntentParsingTests(unittest.TestCase):
    def test_intents_route_correctly(self):
        cases = {
            "build a tabletop arm that sorts red and blue blocks": "build",
            "evaluate it": "evaluate",
            "it keeps missing grasps, make it better": "iterate",
            "use the camera / add perception": "perceive",
            "train a policy": "train",
            "export the package": "export",
            "status": "status",
            "give it more reach, 80 cm": "adjust_and_rebuild",
        }
        for message, expected in cases.items():
            intent, _ = parse_intent(message)
            self.assertEqual(expected, intent, f"{message!r} -> {intent}")

    def test_reach_param_is_parsed_to_meters(self):
        intent, params = parse_intent("give it more reach, 80 cm")
        self.assertEqual("adjust_and_rebuild", intent)
        self.assertAlmostEqual(0.8, params["reach_m"], places=3)

    def test_actions_require_a_project_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = VirturoidAgent(Path(tmpdir))
            self.assertIn("build", agent.handle("evaluate it").message.lower())
            self.assertIn("build", agent.handle("status").message.lower())


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class AgentSessionTests(unittest.TestCase):
    def test_build_then_status_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = VirturoidAgent(Path(tmpdir), target_success=0.8)
            built = agent.handle("build a tabletop arm that sorts red and blue blocks into matching bins")
            self.assertEqual("build", built.intent)
            self.assertIn("success", built.message.lower())
            self.assertIn("succeeded", built.data)

            status = agent.handle("status")
            self.assertEqual("status", status.intent)
            self.assertIn("final_success_rate", status.data)


if __name__ == "__main__":
    unittest.main()
