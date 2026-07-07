"""T1/T2 (total_generality_plan): behavior is dispatched from STRUCTURE, not the robot_class label. The census
flagged ~24 class-word dispatch sites, but on the AGENT path (submit_design -> build_from_anatomy -> GEN-1
appendage discovery -> GEN-2 gait -> robot_kind -> B1 verify) the class label never gates behavior. These
tests lock that in: a mislabeled or nonsense-labeled body still routes + verifies by what it physically IS.
(The remaining class-word sites are the OFFLINE prompt->body generator, which must read the class from a
prompt, plus packaging/naming — not agent-path behavior.) Offline + NO_INTERNAL_LLM."""
import copy
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_NO_INTERNAL_LLM", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class StructuralDispatchTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_nonsense_class_label_still_routes_by_structure(self):
        g = copy.deepcopy(self._call("get_design_schema")["examples"]["quadruped"])
        g["robot_class"] = "totally_made_up_xyz"               # a label the enum has never seen
        r = self._call("submit_design", {"graph": g})
        self.assertTrue(r["ok"])
        self.assertEqual(r["appendages"]["legs"], 4)           # GEN-1 finds the legs from the body
        v = self._call("verify_robot", {"robot_id": r["robot_id"], "mode": "quick"})
        self.assertEqual(v["kind"], "legged")                  # robot_kind routes by structure, not the label

    def test_wrong_class_label_does_not_override_structure(self):
        g = copy.deepcopy(self._call("get_design_schema")["examples"]["rover"])
        g["robot_class"] = "legged"                            # wheels, but LABELED legged
        r = self._call("submit_design", {"graph": g})
        self.assertEqual(r["appendages"]["wheels"], 6)         # still detected as wheels
        v = self._call("verify_robot", {"robot_id": r["robot_id"], "mode": "quick"})
        self.assertEqual(v["kind"], "mobile")                  # driven, not walked, despite the 'legged' label

    def test_out_of_vocab_creature_gets_a_structural_body(self):
        # an animal absent from the ~90-word class list still yields a legged body with a real gait verdict.
        rid = self._call("create_robot", {"prompt": "a capybara robot"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v["kind"], "legged")
        self.assertIn("cadence", v)


if __name__ == "__main__":
    unittest.main()
