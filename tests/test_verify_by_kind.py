"""B1 (plan_v5 W-B): verify_robot gives an honest motion verdict FOR THE BODY'S KIND. Before this, a rover
or an arm got a legged crawl verdict (CROUCH/SLIDE) — physically meaningless. Now verify dispatches on the
structural robot_kind: legged->gait, mobile->drive, manipulator->reach. Offline + NO_INTERNAL_LLM so it's
deterministic and LLM-free."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_NO_INTERNAL_LLM", "1")
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

    def test_swim_fly_intent_is_flagged_not_silently_land_verdicted(self):
        # T7-lite: an eel/drone routes terrestrial (no fluid tier) -> the verdict must be flagged a LAND PROXY,
        # never presented as its real swimming/flying capability.
        for prompt, env in (("an eel robot that swims", "aquatic"), ("a quadcopter drone", "aerial")):
            rid = self._call("create_robot", {"prompt": prompt})["robot_id"]
            v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
            self.assertEqual(v.get("physics_envelope"), env, f"{prompt} must flag {env}")
            self.assertFalse(v["credible_walk"], "an unsupported-envelope body is never a credible walker")
            self.assertIn("proxy", v["envelope_note"].lower())


if __name__ == "__main__":
    unittest.main()
