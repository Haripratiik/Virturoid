"""B1 (plan_v5 W-B): verify_robot gives an honest motion verdict FOR THE BODY'S KIND. Before this, a rover
or an arm got a legged crawl verdict (CROUCH/SLIDE) — physically meaningless. Now verify dispatches on the
structural robot_kind: legged->gait, mobile->drive, manipulator->reach. Offline + NO_INTERNAL_LLM so it's
deterministic and LLM-free."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")  # offline (get_llm -> None); NO module-level
# NO_INTERNAL_LLM — it leaks process-wide during collection and breaks the llm-router tests
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_GAIT_WORDS = ("CROUCH", "SLIDE", "CREDIBLE WALK", "FORWARD BUT SHORT", "ROLL-OVER")


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class VerifyByKindTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_mobile_gets_a_drive_verdict_not_a_gait_verdict(self):
        rid = self._call("create_robot", {"prompt": "a six-wheeled rover"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v["kind"], "mobile")
        self.assertFalse(v["credible_walk"])
        self.assertIn("forward_m", v)                              # a drive metric, not cadence
        # the verdict must be a DRIVE verdict, never a legged crawl verdict
        self.assertTrue(any(w in v["verdict"] for w in ("DRIVES", "STUCK", "TIPPED", "NO ACTUATORS")),
                        f"expected a drive verdict, got {v['verdict']!r}")
        self.assertFalse(any(w in v["verdict"] for w in ("CROUCH", "SLIDE", "CREDIBLE WALK")))

    def test_manipulator_gets_a_reach_verdict(self):
        rid = self._call("create_robot", {"prompt": "a robot arm that sorts blocks"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v["kind"], "manipulator")
        self.assertIn("reach_span_m", v)
        self.assertTrue(any(w in v["verdict"] for w in ("ARTICULATES", "STUCK", "NO ACTUATORS")))
        self.assertNotIn("CROUCH", v["verdict"])                  # never a gait verdict on an arm

    def test_legged_verdict_is_unchanged(self):
        sch = self._call("get_design_schema")
        rid = self._call("submit_design", {"graph": sch["examples"]["quadruped"]})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v["kind"], "legged")
        self.assertIn("cadence", v)                               # the gait verdict fields survive
        self.assertIn("credible_walk", v)
        self.assertNotIn("physics_envelope", v)                  # a terrestrial dog is in-tier, no flag

    def test_aquatic_body_is_simulated_in_water_not_land_proxied(self):
        # T7: an aquatic prompt runs the REAL MuJoCo fluid sim (kind=aquatic + an honest swim_m), NOT a land
        # proxy. The verdict is honest about thrust (never claims swimming falsely).
        rid = self._call("create_robot", {"prompt": "an eel robot that swims"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v["kind"], "aquatic")
        self.assertIn("swim_m", v)                              # a real fluid-sim measurement, not a gait verdict
        self.assertFalse(v["credible_walk"])
        self.assertTrue(v["verdict"].startswith("SWIMS") or v["verdict"].startswith("DOES NOT SWIM"))

    def test_aerial_intent_still_flagged_no_aerial_tier(self):
        # aerial has no physics tier yet -> still honestly flagged as an unsupported land proxy.
        rid = self._call("create_robot", {"prompt": "a quadcopter drone"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v.get("physics_envelope"), "aerial")
        self.assertFalse(v["credible_walk"])
        self.assertIn("proxy", v["envelope_note"].lower())


if __name__ == "__main__":
    unittest.main()
