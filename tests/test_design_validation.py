import unittest

from virturoid.fixtures.components import curated_component_library
from virturoid.schemas.scenes import SceneGraph, SceneObject, SceneSet
from virturoid.services.design_validator import validate_design
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.robot_arm_builder import build_reference_robot_arm
from virturoid.services.scene_generator import generate_scene_set
from virturoid.services.task_builder import build_task_graph


def _far_scene_set(task_id: str) -> SceneSet:
    return SceneSet(
        id="scenes_far",
        task_graph_id=task_id,
        purpose="variation",
        scenes=[
            SceneGraph(
                id="scene_far_000",
                name="unreachable block",
                backend_targets=["mujoco"],
                robot_spawn_xyz_rpy=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                objects=[
                    SceneObject(name="red_block", object_type="cube", pose_xyz_rpy=(2.5, 0.0, 0.05, 0, 0, 0), mass_kg=0.05),
                ],
            )
        ],
    )


class DesignValidationTests(unittest.TestCase):
    def test_feasible_design_passes_all_hard_gates(self):
        requirements = build_requirements_from_prompt("Build a tabletop arm for sorting blocks.")
        build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        scenes = generate_scene_set(task, count=8, purpose="variation")

        report = validate_design(requirements, build.robot_genome, build.cad_models, build.components, [scenes])

        self.assertTrue(report.validate().ok)
        self.assertTrue(report.ok)
        self.assertEqual([], report.blocking_failures)
        gate_status = {g.name: g.status for g in report.gates}
        self.assertEqual("pass", gate_status["actuator_torque"])
        self.assertEqual("pass", gate_status["joint_limits"])

    def test_unreachable_task_fails_early(self):
        requirements = build_requirements_from_prompt("Build a tabletop arm for sorting blocks.")
        build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)

        report = validate_design(
            requirements, build.robot_genome, build.cad_models, build.components, [_far_scene_set(task.id)]
        )

        self.assertFalse(report.ok)
        self.assertIn("reach", report.blocking_failures)
        self.assertIn("scene_coverage", report.blocking_failures)

    def test_excessive_payload_fails_actuator_gate(self):
        requirements = build_requirements_from_prompt(
            "Build a tabletop arm to move heavy parts.", payload_kg=40.0
        )
        build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        scenes = generate_scene_set(task, count=4, purpose="variation")

        report = validate_design(requirements, build.robot_genome, build.cad_models, build.components, [scenes])

        gate_status = {g.name: g.status for g in report.gates}
        self.assertEqual("fail", gate_status["actuator_torque"])
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
