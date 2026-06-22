import importlib.util
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.schemas.scenes import SceneGraph, SceneObject
from virturoid.services.design_critic import (
    add_parallel_gripper,
    apply_operator,
    diagnose,
    improve_design,
    joint_hold_torques,
    kinematic_reach,
    propose_amendment,
    reach_feasibility,
    reach_required_from_scenes,
    reachable_workspace,
)
from virturoid.services.memory_db import MemoryDB

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def short_weak_arm(gene_id: str = "seed_arm") -> RobotGene:
    """A deliberately under-built manipulator: too short to reach AND too weak to hold a payload."""
    return RobotGene(
        id=gene_id, species="manipulator.test.seed", robot_class="manipulator",
        base_mount="table", end_effector_type="gripper",
        segments=[
            GeneSegment("base", None, "box", 0.10, 0.04, 1.0, joint_type="fixed"),
            GeneSegment("shoulder", "base", "box", 0.12, 0.03, 0.30, joint_type="revolute",
                        joint_lower=-1.0, joint_upper=1.0, actuator_torque_nm=3.0),
            GeneSegment("forearm", "shoulder", "box", 0.12, 0.025, 0.20, joint_type="revolute",
                        joint_lower=-1.5, joint_upper=1.5, actuator_torque_nm=2.0),
            GeneSegment("gripper", "forearm", "box", 0.04, 0.02, 0.05, joint_type="fixed",
                        is_end_effector=True),
        ],
    )


# A decoupled, monotone physics stub: each fix the diagnosis recommends crosses its threshold.
def stub_eval(gene: RobotGene) -> float:
    reach = kinematic_reach(gene)
    min_act = min((s.actuator_torque_nm or 0.0) for s in gene.actuated_joints())
    score = 0.1
    if reach >= 0.33:          # lengthen_reach (1.25x on 0.28 -> 0.34) clears this
        score += 0.4
    if min_act >= 2.9:         # upgrade_actuators (1.5x on 2.0 -> 3.0) clears this
        score += 0.4
    return round(score, 3)


class DiagnosisTests(unittest.TestCase):
    def test_reach_deficit_is_measured(self):
        self.assertAlmostEqual(kinematic_reach(short_weak_arm()), 0.28, places=5)

    def test_diagnose_flags_both_reach_and_torque(self):
        modes = diagnose(short_weak_arm(), reach_required_m=0.30, payload_kg=1.0)
        codes = {m.code for m in modes}
        self.assertIn("reach_limited", codes)
        self.assertIn("torque_limited", codes)
        # torque is the more severe limit here, so it ranks first
        self.assertEqual(modes[0].code, "torque_limited")
        # the root cause is stated in physical units, not a vague label
        self.assertIn("Nm", next(m for m in modes if m.code == "torque_limited").root_cause)

    def test_healthy_body_has_no_failure_modes(self):
        modes = diagnose(short_weak_arm(), reach_required_m=0.10, payload_kg=0.0)
        self.assertEqual(modes, [])

    def test_hold_torque_grows_toward_the_base(self):
        torques = {t["joint"]: t["required_nm"] for t in joint_hold_torques(short_weak_arm(), payload_kg=1.0)}
        self.assertGreater(torques["shoulder"], torques["forearm"])  # base joint carries more


class TaskRequirementTests(unittest.TestCase):
    def test_reach_required_is_farthest_manipulable_from_shoulder(self):
        scene = SceneGraph(
            id="s", name="s", backend_targets=["mujoco"],
            robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0),
            objects=[
                SceneObject("near", "cube", (0.20, 0.00, 0.165, 0, 0, 0)),
                SceneObject("far", "cube", (0.40, 0.30, 0.165, 0, 0, 0)),  # hypot(0.4,0.3)=0.5
                SceneObject("bin", "container", (0.90, 0.90, 0.0, 0, 0, 0)),  # not a manipulable
            ])
        # the requirement is set by the task (farthest object), not the current robot
        self.assertAlmostEqual(reach_required_from_scenes([scene], base_height_m=0.165), 0.5, places=4)


class OperatorTests(unittest.TestCase):
    def test_lengthen_increases_reach_and_stays_valid(self):
        out = apply_operator(short_weak_arm(), "lengthen_reach")
        self.assertEqual(out.validate(), [])
        self.assertGreater(kinematic_reach(out), kinematic_reach(short_weak_arm()))

    def test_upgrade_increases_actuator_torque(self):
        out = apply_operator(short_weak_arm(), "upgrade_actuators")
        before = min(s.actuator_torque_nm for s in short_weak_arm().actuated_joints())
        after = min(s.actuator_torque_nm for s in out.actuated_joints())
        self.assertGreater(after, before)

    def test_unknown_operator_raises(self):
        with self.assertRaises(KeyError):
            apply_operator(short_weak_arm(), "make_it_better")


class ImproveLoopTests(unittest.TestCase):
    def test_directed_loop_fixes_the_body_and_records_lessons(self):
        with tempfile.TemporaryDirectory() as d:
            mem = MemoryDB(Path(d) / "m.db")
            trace = improve_design(short_weak_arm(), stub_eval, reach_required_m=0.30,
                                   payload_kg=1.0, memory=mem, target_success=0.85, task_type="lift")
            self.assertGreater(trace.final_success, trace.baseline_success)
            self.assertGreaterEqual(trace.final_success, 0.85)
            self.assertGreaterEqual(trace.accepted, 2)  # both reach and torque got fixed
            # every accepted change is grounded in a diagnosis + a directed operator (not random)
            for rd in trace.rounds:
                self.assertIn(rd["diagnosis"], {"reach_limited", "torque_limited"})
                self.assertIn(rd["operator"], {"lengthen_reach", "upgrade_actuators",
                                               "widen_joint_ranges", "lighten_links"})
            # the proven fixes were written to the flywheel
            lessons = mem.lessons_for_class("manipulator")
            self.assertTrue(any(l["failure_code"] == "torque_limited" for l in lessons))
            self.assertTrue(any(l["failure_code"] == "reach_limited" for l in lessons))
            mem.close()

    def test_second_customer_reuses_first_customers_fix(self):
        """The flywheel: once a class+failure is solved, the same diagnosis retrieves the fix."""
        with tempfile.TemporaryDirectory() as d:
            mem = MemoryDB(Path(d) / "m.db")
            # Customer 1 discovers the fixes from scratch (heuristic reasoning).
            improve_design(short_weak_arm("cust1"), stub_eval, reach_required_m=0.30,
                           payload_kg=1.0, memory=mem, target_success=0.85)
            # Customer 2's same-class robot hits the same diagnosis...
            modes = diagnose(short_weak_arm("cust2"), reach_required_m=0.30, payload_kg=1.0)
            amd = propose_amendment(short_weak_arm("cust2"), modes, memory=mem)
            self.assertEqual(amd.source, "lesson")  # ...and reuses the proven fix, not a guess
            # ...and the full loop on customer 2 uses at least one reused lesson
            trace2 = improve_design(short_weak_arm("cust2"), stub_eval, reach_required_m=0.30,
                                    payload_kg=1.0, memory=mem, target_success=0.85)
            self.assertTrue(any(rd["source"] == "lesson" for rd in trace2.rounds))
            self.assertGreaterEqual(trace2.final_success, 0.85)
            mem.close()


class GripperGeneTests(unittest.TestCase):
    """M3 G1: parallel-jaw gripper gene (+ the mount_offset schema addition that enables side-by-side fingers)."""

    def _gripper(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        return add_parallel_gripper(tabletop_arm_gene())

    def test_gripper_is_valid_and_adds_two_prismatic_fingers(self):
        g = self._gripper()
        self.assertEqual(g.validate(), [])
        pris = [s for s in g.segments if s.joint_type == "prismatic" and s.name.startswith("finger")]
        self.assertEqual(len(pris), 2)
        ys = sorted(s.mount_offset[1] for s in pris)   # fingers offset to opposite sides in y
        self.assertLess(ys[0], 0.0)
        self.assertGreater(ys[-1], 0.0)
        self.assertEqual(g.end_effector_type, "gripper")

    def test_mount_offset_round_trips_through_serialization(self):
        g = self._gripper()
        g2 = RobotGene.from_dict(g.to_dict())
        self.assertEqual([tuple(s.mount_offset) for s in g2.segments],
                         [tuple(s.mount_offset) for s in g.segments])

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_gripper_gene_compiles_to_mjcf_with_finger_actuators(self):
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf

        g = self._gripper()
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=False))
        self.assertGreaterEqual(mj.nu, 5)   # arm revolute joints + 2 finger actuators

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_gripper_emits_grasp_tcp_site_beyond_the_palm(self):
        """The compiler emits a grasp_site (TCP) at the finger mid-plane — targeting the palm (ee_site,
        the finger base) overshoots the object by a finger-length, which is why early grasps missed."""
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf

        g = self._gripper()
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=False))
        ee = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        tcp = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        self.assertGreaterEqual(tcp, 0, "gripper must expose a grasp_site TCP")
        # the TCP sits deeper along the palm's local +z than the palm tip (between/at the finger tips)
        self.assertGreater(float(mj.site_pos[tcp][2]), float(mj.site_pos[ee][2]))


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class ReachabilityTests(unittest.TestCase):
    """M3 G3: the reach-feasibility capability that generalized the M2 push win into the platform."""

    def _arm(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        return tabletop_arm_gene()

    def test_reachable_workspace_is_a_subregion(self):
        ws = reachable_workspace(self._arm(), (0.24, -0.12), (0.42, 0.12), z=0.05, grid=5, margin=0.06)
        self.assertGreater(ws["feasibility"], 0.0)     # the arm CAN contact part of the table...
        self.assertLess(ws["feasibility"], 1.0)        # ...but not all of it (the M2 finding)
        self.assertIsNotNone(ws["bbox"]["x"])
        self.assertGreaterEqual(ws["bbox"]["x"][1], 0.38)   # the far region is reachable

    def test_reach_feasibility_far_reachable_near_not(self):
        arm = self._arm()
        self.assertEqual(reach_feasibility(arm, [(0.40, 0.0, 0.05)], margin=0.06), 1.0)   # far: contactable
        self.assertEqual(reach_feasibility(arm, [(0.24, 0.0, 0.05)], margin=0.06), 0.0)   # near-base low: not


class LessonStoreTests(unittest.TestCase):
    def test_record_recall_and_idempotent_best(self):
        with tempfile.TemporaryDirectory() as d:
            mem = MemoryDB(Path(d) / "m.db")
            mem.record_lesson("humanoid", "reach_limited", "lengthen_reach",
                              params={"factor": 1.25}, improvement=0.2, root_cause="short by 0.1 m")
            mem.record_lesson("humanoid", "reach_limited", "lengthen_reach",
                              params={"factor": 1.4}, improvement=0.5)  # better, on another robot
            best = mem.recall_lesson("humanoid", "reach_limited")
            self.assertEqual(best["operator"], "lengthen_reach")
            self.assertEqual(best["improvement"], 0.5)
            self.assertEqual(best["params"]["factor"], 1.4)   # kept the better magnitude
            self.assertEqual(best["times_applied"], 2)         # collapsed, not duplicated
            self.assertIsNone(mem.recall_lesson("humanoid", "never_seen"))
            mem.close()


if __name__ == "__main__":
    unittest.main()
