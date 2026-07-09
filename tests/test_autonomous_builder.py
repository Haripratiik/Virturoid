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
            ledger = json.loads((output_dir / "reports" / "product_readiness_ledger.json").read_text(encoding="utf-8"))

            self.assertEqual("manipulator", result.selected_robot_class)
            self.assertEqual("mvp_manipulator_training_package", result.package_type)
            self.assertTrue(result.package_valid)
            self.assertFalse(result.readiness["ready"])
            self.assertFalse(result.readiness["safe_to_export"])
            self.assertEqual("reports/product_readiness_ledger.json", result.readiness["uri"])
            self.assertTrue(result.readiness["legacy_mvp_readiness"]["ready"])
            self.assertEqual(result.selected_morphology_template_id, summary["selected_morphology_template_id"])
            self.assertTrue(contract["ok"])
            self.assertEqual("manipulator", contract["robot_class"])
            self.assertTrue(readiness["ready"])
            self.assertEqual(100, readiness["score"])
            self.assertFalse(readiness["safe_to_export"])
            self.assertFalse(ledger["safe_to_export"])
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

    def test_legged_build_reports_general_engine_not_the_arm_it_traced_through(self):
        # A quadruped isn't buildable-by-template yet, so the selector traces it to the ARM to route it to the
        # general engine. The reported builder must then name the GENERAL ENGINE, not the arm — otherwise the
        # package contradicts its own "real morphology, not an arm fallback" note (a tester-confusing mislabel).
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_robot_package_from_prompt(
                "a quadruped robot dog that walks", Path(tmpdir) / "quad_package")
            self.assertEqual(result.selected_robot_class, "quadruped")
            tid = result.selected_morphology_template_id
            self.assertTrue(tid.startswith("general_engine"), f"legged build must report the engine, got {tid}")
            self.assertNotIn("arm", tid)
            self.assertNotIn("manipulator", tid)

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
            ledger = json.loads((output_dir / "reports" / "product_readiness_ledger.json").read_text(encoding="utf-8"))

            self.assertEqual("mobile_base", result.selected_robot_class)
            self.assertEqual("navigation", result.task_type)
            # Mobile bases now build through the general engine (clean nav package: no meshless URDF and a
            # task_type-tagged scene set) instead of the old foundation-template writer, which rendered as
            # Robot-mode placeholders. See build_robot_package_from_prompt.
            self.assertEqual("general_engine_package", result.package_type)
            self.assertTrue(result.package_valid)
            self.assertTrue(result.readiness["ready"])
            self.assertTrue(result.readiness["safe_to_export"])
            self.assertEqual("reports/product_readiness_ledger.json", result.readiness["uri"])
            self.assertTrue(contract["ok"])
            self.assertEqual("mobile_base", contract["robot_class"])
            self.assertTrue(readiness["ready"])
            self.assertEqual(100, readiness["score"])
            self.assertTrue(ledger["safe_to_export"])
            by_gate = {gate["key"]: gate for gate in readiness["gates"]}
            self.assertEqual("skip", by_gate["training_config_present"]["status"])
            self.assertFalse(by_gate["training_config_present"]["required"])
            self.assertIn("scene_generation", contract["capabilities"])
            self.assertIn("compiled_simulation_scenes", contract["capabilities"])
            # The clean gene navigation package does NOT ship the CV/world-model scaffolding the old
            # foundation writer did. Perception is a roadmap item, and maze_nav -- the demo's wheeled base --
            # is likewise a gene package without it, so fresh mobile-base builds now match it.
            self.assertIn("robot_package_contract", result.artifacts)
            self.assertIn("mvp_readiness_report", result.artifacts)
            self.assertIn("workbench", result.artifacts)
            workbench = (output_dir / "reports" / "workbench.html").read_text(encoding="utf-8")
            self.assertIn("Virturoid Build Workbench", workbench)
            self.assertIn("general_engine_package", workbench)
            self.assertIn("goal_zone", workbench)
            self.assertIn("MVP Readiness", workbench)
            self.assertEqual("morph_mobile_base_differential", result.selected_morphology_template_id)
            self.assertEqual(result.selected_species, summary["selected_species"])
            self.assertTrue((output_dir / "robot" / "robot_genome.json").exists())
            # Gene navigation packages compile per-variation scenes (compiled_scene_index.json), not the
            # foundation writer's single mobile_base_scene.xml.
            self.assertTrue((output_dir / "simulation" / "scene_set.json").exists())
            self.assertTrue((output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").exists())


if __name__ == "__main__":
    unittest.main()
