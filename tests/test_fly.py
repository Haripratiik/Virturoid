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

    def test_drone_is_classified_aerial_by_structure(self):
        from virturoid.services.aerial import build_quadcopter
        from virturoid.services.task_matched_eval import robot_capabilities, robot_kind
        g = build_quadcopter("a drone")
        self.assertEqual(robot_kind(g), "aerial", "a rotor-thrust body must classify as aerial, not manipulator")
        self.assertIn("aerial", robot_capabilities(g))

    def test_run_task_flies_a_drone_to_a_target(self):
        # the TASK layer: a drone must be able to RUN a flight task (feasible + success), not just verify.
        from virturoid.services import session_state as S
        from virturoid.services.agent_design_tools import run_task
        from virturoid.services.ai_native_tools import create_robot
        S.reset()
        rid = create_robot({"prompt": "a quadcopter drone that flies to a target"})["robot_id"]
        for goal in ("fly to the target", "go to the goal", "hover in place"):
            r = run_task({"robot_id": rid, "goal": goal})
            self.assertTrue(r.get("feasible"), f"a drone must be able to attempt '{goal}': {r}")
            self.assertTrue(r.get("success"), f"a drone must complete '{goal}' (fly skill): {r}")

    def test_evaluate_robot_scores_a_drone_on_flight(self):
        from virturoid.services.aerial import build_quadcopter
        from virturoid.services.task_matched_eval import evaluate_robot
        ev = evaluate_robot(build_quadcopter("a drone"), prompt="a drone")
        self.assertEqual(ev["task"], "flight")
        self.assertGreater(ev["value"], 0.6, f"a working quadcopter reaches most waypoints: {ev}")

    def test_drone_ros2_export_is_flight_honest(self):
        # a jointless airframe must NOT ship a joint-trajectory "ReachController"/"trot" node as if it flies it;
        # the package must document the real deploy path (autopilot + MAVROS) + carry the flight-controller ref.
        import glob
        import json
        import tempfile
        from pathlib import Path

        from virturoid.services.aerial import build_quadcopter
        from virturoid.services.ros2_exporter import export_ros2_package
        pkg = Path(tempfile.mkdtemp()) / "pkg"
        (pkg / "robot").mkdir(parents=True)
        # write the export-format genome (links/joints) the ros2 exporter reads
        g = build_quadcopter("a drone")
        genome = {"id": g.id, "robot_class": "aerial",
                  "links": [s.name for s in g.segments], "joints": []}
        (pkg / "robot" / "robot_genome.json").write_text(json.dumps(genome), encoding="utf-8")
        root = export_ros2_package(pkg, "drone_pkg")
        fy = glob.glob(str(root) + "/**/flight.yaml", recursive=True)
        self.assertTrue(fy, "an aerial ROS2 package must carry a flight.yaml (the deploy reference)")
        flight = json.loads(Path(fy[0]).read_text())
        self.assertEqual(flight["airframe"], "quadcopter")
        self.assertEqual(flight["n_rotors"], 4)
        readme = (root / "README.md").read_text().lower()
        self.assertNotIn("reachcontroller", readme, "a drone must not claim a reach controller")
        self.assertNotIn("trot gait", readme, "a drone must not claim a trot gait")
        self.assertIn("mavros", readme)                        # the real ROS2 flight-deploy interface

    def test_drone_bom_has_propulsion_and_a_battery(self):
        # a buildable drone needs its DEFINING parts: a motor+ESC per rotor, a flight controller, a battery
        # (NOT a socketed wall PSU — a flying robot can't be tethered). Rotors are welded, so the joint-driven
        # actuator loop misses them without the aerial-propulsion path.
        from virturoid.services.aerial import build_quadcopter
        from virturoid.services.bom_builder import build_bom
        bom = build_bom(build_quadcopter("a drone"))
        cats = {l["category"] for l in bom["lines"]}
        self.assertIn("rotor", cats, f"a drone BOM must list rotor propulsion: {cats}")
        self.assertIn("flight_controller", cats)
        rotor = next(l for l in bom["lines"] if l["category"] == "rotor")
        self.assertEqual(rotor["qty"], 4, "a quadcopter has 4 rotor motors")
        esc = next(l for l in bom["lines"] if l["category"] == "esc")
        self.assertEqual(esc["qty"], 4, "one ESC per rotor")
        power = next(l for l in bom["lines"] if l["category"] == "power")
        self.assertNotIn("PSU", power["part"], f"a drone must be battery-powered, not a socketed PSU: {power['part']}")


if __name__ == "__main__":
    unittest.main()
