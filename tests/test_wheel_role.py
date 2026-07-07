"""T3 (total_generality_plan): the design language gains a first-class WHEEL role and stops silently
mis-compiling unknown roles. Before: a 'wheel' part became a tapered spindle counted as a LEG (wheels=0,
legs=4); an unknown role compiled to a limb silently. After: wheel -> a rolling cylinder GEN-1 counts as a
wheel (robot_kind=mobile, a DRIVE verdict), and an unknown role returns a teaching error. Offline +
NO_INTERNAL_LLM -> deterministic + LLM-free."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")   # offline: skip .env, so get_llm returns None (no
# module-level NO_INTERNAL_LLM here — that leaks process-wide during collection and breaks the llm-router tests)
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

    def test_drive_verdict_is_verified_rolling_not_gameable(self):
        # a 'DRIVES' verdict must be backed by REAL rolling: wheels in ground contact AND spinning AND upright,
        # so a slide/tip/float can't earn it (the honesty fix after the floating-wheels regression).
        rid = self._call("create_robot", {"prompt": "a six-wheeled rover"})["robot_id"]
        v = self._call("verify_robot", {"robot_id": rid, "mode": "full"})
        if v["verdict"].startswith("DRIVES"):
            self.assertGreater(v["wheel_spin_radps"], 0.3, "DRIVES requires the wheels to actually turn")
            self.assertGreater(v["wheel_ground_contact_frac"], 0.2, "DRIVES requires wheels on the ground")
            self.assertGreater(v["upright_min"], 0.5, "DRIVES requires staying upright")

    def test_wheels_rest_on_the_ground_at_spawn_not_floating(self):
        # the render/spawn pose must show wheels ON the floor (the bug: a 30 mm legged-foot clearance floated
        # the wheels). Measure the lowest wheel point at the spawn pose (no hidden settle).
        import mujoco
        from virturoid.services import session_state as S
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        rid = self._call("submit_design", {"graph": self._call("get_design_schema")["examples"]["rover"]})["robot_id"]
        gene = S.get_robot(rid)
        m = compiled_model(robot_mjcf(gene)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        wheel_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
        bottom = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in wheel_geoms)
        self.assertLess(bottom, 0.01, f"wheels must rest on the floor at spawn, lowest={bottom:.3f} (floating)")

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
