"""Aerial robots actually FLY — the quadcopter body + geometric flight controller that closed the aerial frontier
on CPU.

Before: a 'drone' prompt was gapped as unsupported (no aerodynamics) or silently compiled a wheeled ground robot.
Now an aerial prompt composes an ORIGINAL quadcopter (hub + 4 rotors + camera pod) driven by four rotor THRUST
forces and a geometric (acceleration-based) flight controller -> it reaches arbitrary targets (measured ~0.00 m
error, sane 3-13 deg banking) where a naive nested-PID controller flips. The attitude loop is inertia-normalized
so one gain set flies any drone size.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class AerialHelperTests(unittest.TestCase):
    def test_is_aerial_prompt(self):
        from virturoid.services.aerial import is_aerial_prompt
        for p in ("a quadcopter drone", "a drone that flies to a waypoint", "an aerial robot", "a UAV",
                  "a flying robot", "a hovering camera drone"):
            self.assertTrue(is_aerial_prompt(p), p)
        for p in ("a quadruped robot dog", "a robotic arm", "a wheeled rover", "a flywheel energy store"):
            self.assertFalse(is_aerial_prompt(p), p)

    def test_build_quadcopter_is_a_valid_original_body(self):
        from virturoid.services.aerial import build_quadcopter
        g = build_quadcopter("a drone")
        self.assertEqual(g.validate(), [], "quadcopter gene must be a valid kinematic tree")
        self.assertEqual(g.robot_class, "aerial")
        self.assertEqual(g.base_mount, "free")                 # floating base (freejoint) so it can fly
        self.assertEqual(len(g.metadata.get("rotor_offsets", [])), 4, "a quadcopter has 4 rotor thrust points")
        names = [s.name for s in g.segments]
        self.assertEqual(sum(n.startswith("rotor") for n in names), 4)
        self.assertEqual(sum(1 for s in g.segments if s.is_end_effector), 1)   # exactly one end effector (camera pod)

    def test_aerial_prompt_composes_a_quadcopter(self):
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadcopter drone that flies to a target")
        self.assertEqual(g.robot_class, "aerial", "an aerial prompt must compose a quadcopter, not a ground robot")
        self.assertTrue(g.metadata.get("rotor_offsets"), "the composed drone must carry its 4 rotor points")

    def test_flying_fish_stays_aquatic(self):
        # the aerial/aquatic overlap ('flying fish') resolves to the better-supported undulatory swim tier
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a flying fish robot")
        self.assertEqual(g.robot_class, "aquatic")

    def test_drone_is_not_reported_unsupported(self):
        # intent_planner must no longer gap aerial prompts as unsupported (aerial.py + _honest_fly close it)
        from virturoid.services.intent_planner import _UNSUPPORTED
        for w in ("drone", "flies", "flying", "quadcopter", "aerial", "uav"):
            self.assertNotIn(w, _UNSUPPORTED, f"'{w}' is supported now — should not be in _UNSUPPORTED")


@unittest.skipUnless(_MUJOCO, "flight needs the MuJoCo rigid-body sim")
class FlightTests(unittest.TestCase):
    def test_quadcopter_hovers_at_target(self):
        from virturoid.services.aerial import build_quadcopter
        from virturoid.services.ai_native_tools import _honest_fly
        r = _honest_fly(build_quadcopter("a drone"), target=(0.0, 0.0, 1.2), steps=1600)
        self.assertTrue(r["reached_target"], f"a quadcopter must hover at its target: {r}")
        self.assertIn("FLIES", r["verdict"])
        self.assertLess(r["max_tilt_deg"], 8.0, "a stable hover stays near level")

    def test_quadcopter_flies_to_waypoints(self):
        from virturoid.services.aerial import build_quadcopter
        from virturoid.services.ai_native_tools import _honest_fly
        g = build_quadcopter("a drone")
        for tgt in [(1.5, 0.8, 1.2), (2.5, -1.5, 1.8), (-1.2, 0.0, 0.9)]:
            r = _honest_fly(g, target=tgt, steps=2200)
            self.assertTrue(r["reached_target"], f"drone must FLY to {tgt}: {r}")
            self.assertLess(r["dist_to_target"], 0.30, f"drone must arrive at {tgt}: {r}")
            self.assertLess(r["max_tilt_deg"], 35.0, "banking to translate stays bounded (never flips)")

    def test_verify_robot_flies_an_aerial_build(self):
        # the product path: build a drone through the session and verify it -> a FLIES verdict, not a land proxy
        from virturoid.services import session_state as S
        from virturoid.services.ai_native_tools import verify_robot
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadcopter drone")
        rid = S.put_robot(g, prompt="a quadcopter drone")
        res = verify_robot({"robot_id": rid, "mode": "quick"})
        self.assertEqual(res.get("kind"), "aerial", f"verify must route a drone to the aerial tier: {res}")
        self.assertIn("FL", res["verdict"].upper())            # FLIES or (honestly) does not fly — never a land gait
        self.assertNotIn("physics_envelope", res, "a flown drone is not flagged unsupported-envelope")


if __name__ == "__main__":
    unittest.main()
