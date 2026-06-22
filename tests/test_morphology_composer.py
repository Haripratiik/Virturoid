"""Agent-driven composition: prompt/requirements -> block recipe -> validated robot (no catalog pick)."""

import importlib.util
import unittest

from virturoid.services.morphology_composer import (
    _articulation_issues, _morphology_quality_issues, compose_from_spec, compose_robot,
    morphology_from_requirements, propose_morphology_spec)

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class _MockLLM:
    """Returns a fixed recipe each call; records the user prompts so a test can assert the repair loop ran."""
    def __init__(self, spec):
        self.spec, self.calls = spec, []

    def complete_json(self, system, user, schema, max_tokens=None, reasoning_effort=None):
        self.calls.append(user)
        return self.spec


_VALID_ARM = {
    "robot_class": "manipulator", "base_mount": "table",
    "base": {"name": "base", "shape": "box", "length": 0.08, "radius": 0.05, "mass": 1.0},
    "links": [{"name": "l1", "length": 0.2, "axis": [0, 1, 0]}, {"name": "l2", "length": 0.2, "axis": [0, 1, 0]},
              {"name": "l3", "length": 0.1, "axis": [0, 1, 0]}],
    "end_effector": {"kind": "gripper", "span": 0.1},
}
_GAPPY_QUAD = {                                    # a limb floating 0.5 m off a 0.1 m-wide torso
    "robot_class": "quadruped", "base_mount": "free",
    "base": {"name": "torso", "shape": "box", "length": 0.1, "radius": 0.1, "mass": 2.0},
    "limbs": [{"prefix": "leg1", "parent": "torso", "mount_offset": [0, 0.5, 0],
               "links": [{"length": 0.1, "axis": [0, 1, 0]}, {"length": 0.1, "axis": [0, 1, 0]}]}],
    "end_effector": {"kind": "none"},
}
# A "beaded noodle" humanoid: the torso is a long articulated SPINE of many tiny stacked revolute links.
# It can satisfy coarse bounding-box checks but reads as a stack of beads, not a body — the gpt-5.2 failure.
_NOODLE_HUMANOID = {
    "robot_class": "humanoid", "base_mount": "free",
    "base": {"name": "torso", "shape": "capsule", "length": 0.08, "radius": 0.04, "mass": 0.3},
    "links": [{"name": f"v{i}", "length": 0.08, "radius": 0.04, "axis": [0, 1, 0], "joint": "revolute"}
              for i in range(16)],
    "end_effector": {"kind": "none"},
}


class MorphologyComposerTests(unittest.TestCase):
    def test_grasp_prompt_composes_a_gripper_arm(self):
        g = compose_robot("sort red and blue blocks into bins on a table")
        self.assertEqual(g.robot_class, "manipulator")
        self.assertEqual(g.end_effector_type, "gripper")
        self.assertEqual(g.validate(), [])
        self.assertGreaterEqual(len(g.actuated_joints()), 4)        # arm joints + 2 fingers

    def test_redundant_end_effector_is_dropped_when_fingers_are_limbs(self):
        # An LLM design that builds articulated finger LIMBS AND asks for a hand end-effector must not get
        # extra jaw stubs welded to a fingertip — the limbs are the hand. (The gripper-on-a-fingertip bug.)
        spec = {
            "robot_class": "manipulator", "base_mount": "table",
            "base": {"name": "palm", "shape": "box", "length": 0.08, "radius": 0.05},
            "limbs": [{"prefix": f"finger{i}", "parent": "palm", "mount_offset": (0.0, 0.02 * i, 0.0),
                       "links": [{"name": "p0", "length": 0.04, "joint": "revolute"},
                                 {"name": "p1", "length": 0.03, "joint": "fixed"}]} for i in range(3)],
            "end_effector": {"kind": "hand", "fingers": 3},
        }
        g = compose_from_spec(spec)
        self.assertEqual(g.validate(), [])
        self.assertFalse([s for s in g.segments if s.name.startswith("finger_")])   # no builder-added stubs
        self.assertEqual(sum(1 for s in g.segments if s.name.startswith("finger")), 6)  # only the 3 limb fingers

    def test_llm_designs_any_class_when_valid(self):
        # P2: the class-gate is gone — the LLM designs even known classes; a valid proposal is used as-is.
        llm = _MockLLM(_VALID_ARM)
        spec = propose_morphology_spec("a robot arm", llm=llm, robot_class="manipulator")
        self.assertEqual(spec["design_source"], "llm")
        self.assertEqual(len(llm.calls), 1)                  # accepted on the first try, no repair needed
        self.assertEqual(compose_from_spec(spec).validate(), [])

    def test_disconnected_llm_body_is_repaired_then_falls_back(self):
        # A floating-limb ("empty space") proposal is REJECTED, the errors are fed back to repair, and a
        # persistently-bad LLM falls back to the deterministic template — a broken body never ships.
        llm = _MockLLM(_GAPPY_QUAD)
        spec = propose_morphology_spec("a quadruped robot", llm=llm, robot_class="quadruped", repair_tries=2)
        self.assertEqual(spec["design_source"], "heuristic")     # fell back, did not ship the float
        self.assertEqual(len(llm.calls), 3)                      # initial + 2 repair attempts
        self.assertTrue(any("REJECTED" in c for c in llm.calls[1:]))   # repair errors fed back
        self.assertEqual(_morphology_quality_issues(compose_from_spec(spec)), [])  # the fallback is clean

    def test_quality_gate_flags_floating_limb(self):
        issues = _morphology_quality_issues(compose_from_spec(_GAPPY_QUAD))
        self.assertTrue(any("floats in empty space" in i for i in issues))

    def test_quality_gate_flags_over_articulated_noodle(self):
        # The over-articulation gate must catch a beaded-noodle humanoid (too many DOF + a spine torso) —
        # the failure where the LLM stacks dozens of tiny joints and the coarse critic still rubber-stamps it.
        g = compose_from_spec(_NOODLE_HUMANOID)
        issues = _morphology_quality_issues(g)
        self.assertGreater(len(g.actuated_joints()), 12)
        self.assertTrue(any("actuated joints" in i for i in issues))   # DOF cap
        self.assertTrue(any("articulated spine" in i for i in issues))  # torso-is-one-part

    def test_credible_bodies_pass_articulation_gate(self):
        # The caps are class-aware: credible templates AND open-ended animals must NOT be falsely flagged.
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        self.assertEqual(_articulation_issues(build_anthropometric_humanoid()), [])
        for prompt, rc in (("a humanoid robot", "humanoid"), ("a quadruped dog", "quadruped"),
                           ("a robot arm", "manipulator"), ("an octopus with 8 arms", "legged"),
                           ("a snake robot", "legged")):
            g = compose_from_spec(morphology_from_requirements(0.6, 0.25, prompt=prompt, robot_class=rc))
            self.assertEqual(_articulation_issues(g), [], f"{rc} falsely flagged")

    def test_noodle_llm_design_is_rejected_and_falls_back(self):
        # End-to-end: an LLM that emits a noodle must NOT ship it — it is rejected through the repair loop
        # and falls back to the deterministic template, so the product never outputs a beaded column.
        llm = _MockLLM(_NOODLE_HUMANOID)
        spec = propose_morphology_spec("a humanoid robot", llm=llm, robot_class="humanoid", repair_tries=2)
        self.assertEqual(spec["design_source"], "heuristic")
        self.assertEqual(len(llm.calls), 3)                            # initial + 2 repairs, all rejected
        self.assertTrue(any("noodle" in c or "actuated joints" in c for c in llm.calls[1:]))

    def test_no_llm_is_deterministic_template(self):
        spec = propose_morphology_spec("a quadruped robot", llm=None, robot_class="quadruped")
        self.assertEqual(spec["design_source"], "heuristic")

    def test_mobile_prompt_composes_a_wheeled_base(self):
        g = compose_robot("a mobile robot base that can drive and navigate indoors to deliver parts")
        self.assertEqual(g.robot_class, "mobile_base")
        self.assertEqual(g.end_effector_type, "none")
        self.assertEqual(g.validate(), [])
        # two driven wheels attached to the chassis
        wheels = [s for s in g.segments if s.joint_type == "revolute"]
        self.assertGreaterEqual(len(wheels), 2)

    def test_mobile_base_is_a_free_floating_rover(self):
        g = compose_robot("a mobile base to deliver parts indoors")
        self.assertEqual(g.base_mount, "free")                       # floating base, not welded
        self.assertEqual(sum(1 for s in g.segments if "wheel" in s.name), 4)
        self.assertTrue(all(s.shape == "cylinder" for s in g.segments if "wheel" in s.name))

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_composed_mobile_base_actually_drives(self):
        import numpy as np
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        g = compose_robot("a mobile base to deliver parts indoors")
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        d = mujoco.MjData(mj); mujoco.mj_forward(mj, d)
        for _ in range(400):
            mujoco.mj_step(mj, d)                                    # settle on its wheels
        x0, y0 = float(d.qpos[0]), float(d.qpos[1])
        for _ in range(800):
            for u in range(mj.nu):
                d.ctrl[u] = mj.actuator_forcerange[u, 1]             # drive all wheels forward
            mujoco.mj_step(mj, d)
        disp = ((float(d.qpos[0]) - x0) ** 2 + (float(d.qpos[1]) - y0) ** 2) ** 0.5
        self.assertTrue(np.all(np.isfinite(d.qpos)))                 # stable
        self.assertGreater(disp, 0.1)                                # it moved (rolls, not welded)

    def test_spray_prompt_sets_spray_end_effector(self):
        g = compose_robot("spray paint a car panel with 0.8 m reach")
        self.assertEqual(g.end_effector_type, "spray_nozzle")
        self.assertEqual(g.validate(), [])

    def test_precision_prompt_composes_a_high_dof_anthropomorphic_arm(self):
        g = compose_robot("a Franka-style 7-dof arm for precise insertion")
        arm = [s for s in g.segments if s.joint_type == "revolute" and "finger" not in s.name]
        self.assertGreaterEqual(len(arm), 6)                         # 6-7 DOF, not the 3-DOF default
        # anthropomorphic: a yaw base + pitch links + at least one wrist roll/yaw (orientation control)
        yaw = sum(1 for s in arm if abs(s.joint_axis[2]) > 0.5)
        pitch = sum(1 for s in arm if abs(s.joint_axis[1]) > 0.5)
        self.assertGreaterEqual(yaw, 2)
        self.assertGreaterEqual(pitch, 2)
        self.assertEqual(g.validate(), [])

    def test_dexterous_prompt_composes_a_multi_finger_hand(self):
        g = compose_robot("pick up delicate irregular fruit without crushing it")
        fingers = [s for s in g.segments if s.joint_type == "prismatic"]
        self.assertGreaterEqual(len(fingers), 3)                    # multi-finger hand, not 2-jaw gripper
        self.assertEqual(g.end_effector_type, "gripper")
        self.assertEqual(g.validate(), [])

    def test_reach_drives_link_sizing(self):
        short = compose_robot("pick small parts", reach_m=0.3, payload_kg=0.1)
        long = compose_robot("pick small parts", reach_m=1.0, payload_kg=0.1)
        reach_of = lambda g: sum(s.length_m for s in g.segments if s.joint_type == "revolute")
        self.assertGreater(reach_of(long), reach_of(short))         # longer reach -> longer arm

    def test_payload_drives_torque(self):
        light = morphology_from_requirements(0.6, 0.1, prompt="pick")
        heavy = morphology_from_requirements(0.6, 5.0, prompt="pick")
        t_light = light["links"][0]["torque"]
        t_heavy = heavy["links"][0]["torque"]
        self.assertGreater(t_heavy, t_light)                        # heavier payload -> more torque

    def test_invalid_spec_is_rejected(self):
        with self.assertRaises(ValueError):
            compose_from_spec({"robot_class": "manipulator",
                               "links": [{"name": "bad", "length": 0.0}]})   # zero-length link

    def test_propose_without_llm_is_deterministic_heuristic(self):
        a = propose_morphology_spec("sort blocks", reach_m=0.6, payload_kg=0.3, llm=None)
        b = propose_morphology_spec("sort blocks", reach_m=0.6, payload_kg=0.3, llm=None)
        self.assertEqual(a, b)
        self.assertEqual(a["end_effector"]["kind"], "gripper")

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_compose_to_working_robot_co_designs_against_physics(self):
        from virturoid.services.morphology_composer import compose_working_robot
        res = compose_working_robot("tabletop arm to sort blocks into bins",
                                    iterations=2, population=4, scene_count=3)
        self.assertIn("gene", res)
        self.assertEqual(res["gene"].validate(), [])                 # the tuned gene is still valid
        self.assertTrue(0.0 <= res["success_rate"] <= 1.0)
        self.assertGreaterEqual(res["success_rate"], res["baseline_success"])  # co-design never regresses
        self.assertIn("controller_params", res)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_composed_robots_compile_and_auto_place(self):
        import tempfile
        from pathlib import Path
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.species_discovery import auto_place_species

        arm = compose_robot("sort blocks on a table")
        mobile = compose_robot("mobile base to deliver parts indoors")
        for g in (arm, mobile):
            mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        with tempfile.TemporaryDirectory() as tmp, MemoryDB(Path(tmp) / "m.db") as db:
            auto_place_species(arm, db)
            placed = auto_place_species(mobile, db)                 # different class -> a new species
            self.assertEqual(placed["action"], "created")


if __name__ == "__main__":
    unittest.main()
