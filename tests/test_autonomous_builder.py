import json
import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.services.autonomous_builder import build_robot_package_from_prompt
from virturoid.services.package_writer_registry import package_writer_statuses


class AutonomousBuilderTests(unittest.TestCase):
    def test_package_writer_registry_marks_implemented_and_planned_routes(self):
        statuses = package_writer_statuses(morphology_template_catalog())
        by_class = {item.robot_class: item for item in statuses}

        self.assertEqual("implemented", by_class["manipulator"].status)
        self.assertEqual("mvp_manipulator_training_package", by_class["manipulator"].package_type)
        self.assertEqual("implemented", by_class["mobile_base"].status)
        self.assertEqual("mobile_base_foundation_package", by_class["mobile_base"].package_type)
        self.assertEqual("planned", by_class["humanoid"].status)
        self.assertEqual("planned", by_class["quadruped"].status)

    def test_generic_builder_routes_manipulator_prompt_to_full_mvp_package(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_robot_package_from_prompt(
                "Build a tabletop robot arm that sorts red and blue blocks.",
                Path(tmpdir) / "arm_package",
            )
            output_dir = Path(result.output_dir)
            summary = json.loads((output_dir / result.summary_uri).read_text(encoding="utf-8"))
            contract = json.loads((output_dir / "reports" / "robot_package_contract.json").read_text(encoding="utf-8"))
            readiness = json.loads((output_dir / "reports" / "mvp_readiness_report.json").read_text(encoding="utf-8"))

            self.assertEqual("manipulator", result.selected_robot_class)
            self.assertEqual("mvp_manipulator_training_package", result.package_type)
            self.assertTrue(result.package_valid)
            self.assertTrue(result.readiness["ready"])
            self.assertEqual(100, result.readiness["score"])
            self.assertEqual(result.selected_morphology_template_id, summary["selected_morphology_template_id"])
            self.assertTrue(contract["ok"])
            self.assertEqual("manipulator", contract["robot_class"])
            self.assertTrue(readiness["ready"])
            self.assertEqual(100, readiness["score"])
            self.assertIn("generic_package_boundary", {gate["key"] for gate in readiness["gates"]})
            self.assertIn("training_run_config", contract["capabilities"])
            self.assertIn("cv_perception_contract", contract["capabilities"])
            self.assertIn("world_model_contract", contract["capabilities"])
            self.assertIn("synthetic_observation_dataset", contract["capabilities"])
            self.assertIn("generated_cv_annotations", contract["capabilities"])
            self.assertIn("generated_world_state_snapshots", contract["capabilities"])
            self.assertIn("robot_package_contract", result.artifacts)
            self.assertIn("mvp_readiness_report", result.artifacts)
            self.assertIn("workbench", result.artifacts)
            workbench = (output_dir / "reports" / "workbench.html").read_text(encoding="utf-8")
            self.assertIn("Virturoid Build Workbench", workbench)
            self.assertIn("mvp_manipulator_training_package", workbench)
            self.assertIn("Open Contract", workbench)
            self.assertIn("MVP Readiness", workbench)
            self.assertTrue((output_dir / "simulation" / "scene_set.json").exists())
            self.assertTrue((output_dir / "training" / "training_run_config.json").exists())
            self.assertTrue((output_dir / "reports" / "package_validation_report.json").exists())

    def test_generic_builder_routes_navigation_prompt_to_mobile_base_package(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_robot_package_from_prompt(
                "Build a mobile robot base that can drive and navigate indoors to deliver parts.",
                Path(tmpdir) / "mobile_package",
            )
            output_dir = Path(result.output_dir)
            summary = json.loads((output_dir / result.summary_uri).read_text(encoding="utf-8"))
            contract = json.loads((output_dir / "reports" / "robot_package_contract.json").read_text(encoding="utf-8"))
            readiness = json.loads((output_dir / "reports" / "mvp_readiness_report.json").read_text(encoding="utf-8"))

            self.assertEqual("mobile_base", result.selected_robot_class)
            self.assertEqual("navigation", result.task_type)
            self.assertEqual("mobile_base_foundation_package", result.package_type)
            self.assertTrue(result.package_valid)
            self.assertTrue(result.readiness["ready"])
            self.assertTrue(contract["ok"])
            self.assertEqual("mobile_base", contract["robot_class"])
            self.assertTrue(readiness["ready"])
            self.assertEqual(100, readiness["score"])
            by_gate = {gate["key"]: gate for gate in readiness["gates"]}
            self.assertEqual("skip", by_gate["training_config_present"]["status"])
            self.assertFalse(by_gate["training_config_present"]["required"])
            self.assertIn("scene_generation", contract["capabilities"])
            self.assertIn("compiled_simulation_scenes", contract["capabilities"])
            self.assertIn("cv_perception_contract", contract["capabilities"])
            self.assertIn("world_model_contract", contract["capabilities"])
            self.assertIn("synthetic_observation_dataset", contract["capabilities"])
            self.assertIn("generated_cv_annotations", contract["capabilities"])
            self.assertIn("generated_world_state_snapshots", contract["capabilities"])
            self.assertIn("robot_package_contract", result.artifacts)
            self.assertIn("mvp_readiness_report", result.artifacts)
            self.assertIn("workbench", result.artifacts)
            workbench = (output_dir / "reports" / "workbench.html").read_text(encoding="utf-8")
            self.assertIn("Virturoid Build Workbench", workbench)
            self.assertIn("mobile_base_foundation_package", workbench)
            self.assertIn("goal_zone", workbench)
            self.assertIn("MVP Readiness", workbench)
            self.assertEqual("morph_mobile_base_differential", result.selected_morphology_template_id)
            self.assertEqual(result.selected_species, summary["selected_species"])
            self.assertTrue((output_dir / "robot" / "robot_genome.json").exists())
            self.assertTrue((output_dir / "simulation" / "mujoco" / "mobile_base_scene.xml").exists())
            self.assertTrue((output_dir / "simulation" / "scene_set.json").exists())
            self.assertTrue((output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").exists())


if __name__ == "__main__":
    unittest.main()
