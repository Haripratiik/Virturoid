import tempfile
import unittest
from pathlib import Path

from virturoid.mvp import build_mvp_robot_arm_project, write_mvp_robot_arm_project
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.task_builder import build_task_graph


class CustomPromptMvpTests(unittest.TestCase):
    def test_requirements_builder_extracts_payload_reach_and_environment(self):
        requirements = build_requirements_from_prompt(
            "Build a warehouse robot arm to move 500 g boxes from a conveyor into a bin with 80 cm reach."
        )

        self.assertEqual("warehouse", requirements.environment)
        self.assertAlmostEqual(0.5, requirements.payload_kg)
        self.assertAlmostEqual(0.8, requirements.reach_m)
        self.assertEqual(["rgbd_camera"], requirements.sensor_requirements)

    def test_box_prompt_generates_box_task_and_scene(self):
        requirements = build_requirements_from_prompt(
            "Build a robot arm to pick 1 kg boxes from a conveyor and place them in a bin.",
            reach_m=0.9,
        )
        task = build_task_graph(requirements)
        project = build_mvp_robot_arm_project(requirements)

        self.assertEqual("pick_place_box", task.task_type)
        self.assertEqual("pick_place_box", project["task_graph"].task_type)
        self.assertEqual("box", project["scene_set"].scenes[0].objects[0].name)
        self.assertEqual("conveyor_zone", project["scene_set"].scenes[0].objects[2].name)
        self.assertEqual("cmp_actuator_cubemars_ak70_10", project["bom"].items[0].component_id)
        self.assertEqual("synchronize_pick", project["policy_plan"].steps[1].step_id)
        self.assertEqual("match_conveyor_then_grasp", project["policy_plan"].steps[2].command)
        self.assertTrue(any(check.check.startswith("actuator_torque") and check.status == "pass" for check in project["compatibility_report"].checks))

    def test_custom_prompt_export_validates(self):
        requirements = build_requirements_from_prompt(
            "Build a warehouse robot arm to move 500 g boxes from a conveyor into a bin.",
            requirement_id="req_test_custom_box_arm",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "custom_box_arm", requirements)

            self.assertTrue((output_dir / "reports" / "package_validation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
