"""T3 (total_generality_plan): the design language gains a first-class WHEEL role and stops silently
mis-compiling unknown roles. Before: a 'wheel' part became a tapered spindle counted as a LEG (wheels=0,
legs=4); an unknown role compiled to a limb silently. After: wheel -> a rolling cylinder GEN-1 counts as a
wheel (robot_kind=mobile, a DRIVE verdict), and an unknown role returns a teaching error. Offline +
NO_INTERNAL_LLM -> deterministic + LLM-free."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_NO_INTERNAL_LLM", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class WheelRoleTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def test_wheel_role_compiles_to_wheels_not_legs(self):
        # the exact failure from the census: a wheel graph used to hold as legs=4, wheels=0.
        sch = self._call("get_design_schema")
        r = self._call("submit_design", {"graph": sch["examples"]["rover"]})
        self.assertTrue(r["ok"], r.get("error"))
        self.assertEqual(r["appendages"]["wheels"], 6, "wheel parts must compile as WHEELS")
        self.assertEqual(r["appendages"]["legs"], 0, "a rover has no legs")

    def test_wheeled_body_routes_mobile_with_a_drive_verdict(self):
        sch = self._call("get_design_schema")
        rid = self._call("submit_design", {"graph": sch["examples"]["rover"]})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})
        self.assertEqual(v["kind"], "mobile")
        # a drive verdict (DRIVES/STUCK/TIPPED), never a legged gait verdict
        self.assertTrue(any(w in v["verdict"] for w in ("DRIVES", "STUCK", "TIPPED", "NO ACTUATORS")))
        self.assertNotIn("CROUCH", v["verdict"])

    def test_a_proper_wheeled_body_actually_drives(self):
        # the drive MACHINERY works end-to-end: a well-formed wheeled body travels forward under torque.
        rid = self._call("create_robot", {"prompt": "a six-wheeled rover"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "full"})
        self.assertEqual(v["kind"], "mobile")
        self.assertGreater(v["forward_m"], 0.1, f"a rover should drive forward, got {v['verdict']}")

    def test_unknown_role_teaches_not_silently_compiles(self):
        g = {"robot_class": "x", "name": "y", "parts": [
            {"name": "b", "role": "body", "size": 0.4, "girth": 0.1},
            {"name": "r", "role": "rotor", "parent": "b", "attach": "front_top", "size": 0.2}]}
        r = self._call("submit_design", {"graph": g})
        self.assertFalse(r["ok"])
        self.assertIn("rotor", r["error"])                    # names the bad role
        self.assertIn("role", r["error"].lower())

    def test_wheel_is_in_the_taught_vocabulary(self):
        sch = self._call("get_design_schema")
        self.assertIn("wheel", sch["part_fields"]["role"] + str(sch.get("examples")))  # taught + exampled
        # a sane creature design still holds (no false rejection from role validation)
        self.assertTrue(self._call("submit_design", {"graph": sch["examples"]["quadruped"]})["ok"])


if __name__ == "__main__":
    unittest.main()
