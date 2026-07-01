"""Robot serializer (Phase 0 of the robotics-native-AI plan): a gene made legible to an LLM.

Pure-Python, no MuJoCo. Verifies the token list matches the RL tokenizer's actuated-joint unit, the summary
is grounded, and the TopoPE topology path is **edit-invariant** (the property that lets knowledge transfer
across similar bodies — the whole point of the positional code).
"""

import unittest

from virturoid.schemas.gene import GeneSegment, RobotGene, amend_gene
from virturoid.services.robot_serializer import describe_robot, robot_tokens


def _two_dof_arm() -> RobotGene:
    return RobotGene(
        id="arm1", species="manipulator.arm.test", robot_class="manipulator", base_mount="table",
        end_effector_type="gripper",
        segments=[
            GeneSegment(name="base", parent=None, joint_type="fixed"),
            GeneSegment(name="shoulder", parent="base", joint_type="revolute", joint_axis=(0, 1, 0),
                        joint_lower=-1.5, joint_upper=1.5, actuator_torque_nm=20.0),
            GeneSegment(name="elbow", parent="shoulder", joint_type="revolute", joint_axis=(0, 1, 0),
                        joint_lower=-2.0, joint_upper=2.0, actuator_torque_nm=12.0),
            GeneSegment(name="hand", parent="elbow", joint_type="fixed", is_end_effector=True),
        ],
    )


def _two_chain_body() -> RobotGene:
    return RobotGene(
        id="q1", species="quadruped.test", robot_class="quadruped", base_mount="floor",
        end_effector_type="none",
        segments=[
            GeneSegment(name="torso", parent=None, joint_type="fixed"),
            GeneSegment(name="leg1_hip", parent="torso", joint_type="revolute", joint_axis=(0, 1, 0)),
            GeneSegment(name="leg1_knee", parent="leg1_hip", joint_type="revolute", joint_axis=(0, 1, 0)),
            GeneSegment(name="leg2_hip", parent="torso", joint_type="revolute", joint_axis=(0, 1, 0)),
            GeneSegment(name="leg2_knee", parent="leg2_hip", joint_type="revolute", joint_axis=(0, 1, 0),
                        is_end_effector=True),
        ],
    )


class RobotSerializerTests(unittest.TestCase):
    def test_describe_shape_and_grounded_summary(self):
        gene = _two_dof_arm()
        self.assertEqual(gene.validate(), [])
        d = describe_robot(gene)
        self.assertEqual(set(d), {"tokens", "summary_text", "stats"})
        self.assertEqual(d["stats"]["dof"], 2)
        self.assertEqual(d["stats"]["robot_class"], "manipulator")
        # summary is grounded: mentions the class and the real DOF count, is human-readable text
        self.assertIn("manipulator", d["summary_text"])
        self.assertIn("2-DOF", d["summary_text"])
        self.assertGreater(len(d["summary_text"]), 40)

    def test_tokens_are_the_actuated_joints(self):
        gene = _two_dof_arm()
        toks = robot_tokens(gene)
        self.assertEqual([t["name"] for t in toks], ["shoulder", "elbow"])  # fixed base + hand excluded
        for t in toks:
            self.assertIn(t["joint_type"], ("revolute", "prismatic"))
            self.assertIn("topo_path", t)
            self.assertIn(t["axis"], ("x", "y", "z"))
        self.assertEqual(toks[0]["axis"], "y")           # (0,1,0) dominant axis
        self.assertEqual(toks[0]["range"], [-1.5, 1.5])

    def test_stats_grounded_on_branched_body(self):
        d = describe_robot(_two_chain_body())
        st = d["stats"]
        self.assertEqual(st["dof"], 4)
        self.assertEqual(st["n_chains"], 2)              # two legs off the torso
        self.assertEqual(st["n_revolute"], 4)
        self.assertEqual(st["axis_mix"]["y"], 4)

    def test_topo_path_is_edit_invariant(self):
        """The keystone property: amending a body with a NEW chain must not renumber existing limbs."""
        gene = _two_chain_body()
        before = {t["name"]: t["topo_path"] for t in robot_tokens(gene)}
        self.assertEqual(before["leg1_knee"], [0, 0])
        self.assertEqual(before["leg2_knee"], [1, 0])
        # the real flywheel op: amend the gene with a third leg (appended)
        grown = amend_gene(
            gene, new_id="q2", species="quadruped.test.grown",
            add_segments=[GeneSegment(name="leg3_hip", parent="torso", joint_type="revolute",
                                      joint_axis=(0, 1, 0))],
        )
        after = {t["name"]: t["topo_path"] for t in robot_tokens(grown)}
        # existing limbs keep their positional code even though the body grew
        self.assertEqual(after["leg1_knee"], before["leg1_knee"])
        self.assertEqual(after["leg2_knee"], before["leg2_knee"])


if __name__ == "__main__":
    unittest.main()
