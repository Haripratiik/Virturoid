from hashlib import sha256
import tempfile
import unittest
import json
from pathlib import Path

from virturoid.demo import build_demo_entities
from virturoid.fixtures.components import curated_component_library
from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.schemas.exports import ExportBundle
from virturoid.schemas.runs import PolicyRecord
from virturoid.services.compatibility import check_bom_compatibility
from virturoid.services.evaluator import run_mock_evaluation
from virturoid.services.export_writer import write_export_bundle
from virturoid.services.fabrication_plan_builder import build_fabrication_plan
from virturoid.services.mujoco_exporter import write_mujoco_scene_collection
from virturoid.services.morphology_selector import select_morphology_template
from virturoid.services.part_resolver import resolve_robot_arm_parts
from virturoid.services.perception_config_builder import build_perception_config
from virturoid.services.policy_plan_builder import build_policy_plan
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.redesign_revision import _revise_scene_object
from virturoid.services.robot_builder_registry import build_dispatch_record, build_robot_from_selection
from virturoid.services.robot_arm_builder import build_reference_robot_arm
from virturoid.services.scene_generator import _scene_seed, generate_scene_set
from virturoid.services.task_builder import build_task_graph
from virturoid.services.training_objective_builder import build_training_objective
from virturoid.services.world_model_builder import build_world_model_contract, write_world_model_artifacts


class ServicesPipelineTests(unittest.TestCase):
    def test_requirements_to_task_to_scene_set(self):
        entities = build_demo_entities()
        requirements = entities["requirements"]

        task = build_task_graph(requirements)
        scene_set = generate_scene_set(task, count=4, purpose="variation")

        self.assertTrue(task.validate().ok)
        self.assertEqual("pick_place_sort", task.task_type)
        self.assertEqual(4, len(scene_set.scenes))
        self.assertTrue(scene_set.validate().ok)
        self.assertTrue(any(scene.variation_parameters["clutter"] for scene in scene_set.scenes))
        expected_seed = int(sha256(f"{task.id}|variation|0".encode("utf-8")).hexdigest()[:8], 16) % (2**31)
        self.assertEqual(expected_seed, _scene_seed(task, "variation", 0))
        self.assertEqual(expected_seed, scene_set.scenes[0].variation_parameters["seed"])
        red_block = next(item for item in scene_set.scenes[0].objects if item.name == "red_block")
        self.assertIsNotNone(red_block.friction)
        self.assertIsNotNone(red_block.scale)

    def test_scene_revision_preserves_randomized_object_physics(self):
        source = generate_scene_set(build_task_graph(build_requirements_from_prompt("Build a tabletop arm.")), count=1, purpose="variation")
        red_block = next(item for item in source.scenes[0].objects if item.name == "red_block")

        revised = _revise_scene_object(red_block)

        self.assertEqual(red_block.friction, revised.friction)
        self.assertEqual(red_block.scale, revised.scale)

    def test_morphology_selector_chooses_template_without_hardcoding_arms(self):
        templates = morphology_template_catalog()
        arm_requirements = build_requirements_from_prompt("Build a tabletop robot arm to sort blocks.")
        arm_task = build_task_graph(arm_requirements)

        arm_selection = select_morphology_template(arm_requirements, arm_task, templates)

        self.assertEqual("morph_fixed_arm_three_dof_tabletop", arm_selection.selected_template.id)
        self.assertEqual("manipulator", arm_selection.selected_template.robot_class)
        self.assertIn("morph_mobile_base_differential", {template.id for template in templates})
        self.assertIn("morph_humanoid_upper_body", {template.id for template in templates})

    def test_morphology_selector_can_choose_future_mobile_robot_template(self):
        templates = morphology_template_catalog()
        requirements = build_requirements_from_prompt("Build a mobile robot to navigate an indoor warehouse.")
        task = build_task_graph(requirements)
        task.task_type = "navigation"

        selection = select_morphology_template(requirements, task, templates)

        self.assertEqual("morph_mobile_base_differential", selection.selected_template.id)
        self.assertEqual("mobile_base", selection.selected_template.robot_class)

    def test_robot_builder_registry_dispatches_implemented_arm_builder(self):
        templates = morphology_template_catalog()
        requirements = build_requirements_from_prompt("Build a tabletop robot arm to sort blocks.")
        task = build_task_graph(requirements)
        selection = select_morphology_template(requirements, task, templates)

        result = build_robot_from_selection(requirements, curated_component_library(), selection, templates)

        self.assertEqual("implemented", result.dispatch_record.dispatch_status)
        self.assertEqual("morph_fixed_arm_three_dof_tabletop", result.dispatch_record.selected_morphology_template_id)
        self.assertEqual("morph_fixed_arm_three_dof_tabletop", result.robot_build.robot_genome.morphology_template_id)
        self.assertGreaterEqual(len(result.dispatch_record.planned_templates), 2)

    def test_robot_builder_registry_dispatches_mobile_base_and_marks_future_templates_planned(self):
        templates = morphology_template_catalog()
        requirements = build_requirements_from_prompt("Build a mobile robot to drive and navigate an indoor warehouse.")
        task = build_task_graph(requirements)
        selection = select_morphology_template(requirements, task, templates)

        dispatch = build_dispatch_record(selection, templates)

        # The mobile base is now a second implemented robot class.
        self.assertEqual("implemented", dispatch.dispatch_status)
        self.assertEqual("morph_mobile_base_differential", dispatch.selected_morphology_template_id)
        implemented = {item.morphology_template_id for item in dispatch.implemented_templates}
        self.assertIn("morph_fixed_arm_three_dof_tabletop", implemented)
        self.assertIn("morph_mobile_base_differential", implemented)
        # Humanoid and quadruped remain planned, keeping the species tree extensible.
        planned = {item.morphology_template_id for item in dispatch.planned_templates}
        self.assertIn("morph_humanoid_upper_body", planned)
        self.assertIn("morph_quadruped_quadruped", planned)

    def test_bom_compatibility_checks_actuator_margin(self):
        entities = build_demo_entities()
        report = check_bom_compatibility(
            requirements=entities["requirements"],
            bom=entities["bom"],
            components=entities["components"],
        )

        self.assertTrue(report.ok)
        self.assertTrue(any(check.check.startswith("actuator_torque") for check in report.checks))
        self.assertTrue(any(check.check.startswith("camera_fov") for check in report.checks))
        self.assertTrue(any(check.check.startswith("camera_range") for check in report.checks))
        self.assertTrue(any(check.check == "power_peak_current" for check in report.checks))
        self.assertTrue(any(check.check == "power_voltage_overlap" for check in report.checks))

    def test_part_resolver_honors_named_parts_and_reports_limits(self):
        requirements = build_requirements_from_prompt(
            "Build a tabletop arm with a VX-20 servo and LiDAR for sorting blocks.",
            sensor="lidar",
        )
        report = resolve_robot_arm_parts(requirements, curated_component_library())

        self.assertTrue(report.validate().ok)
        self.assertIn("VX-20", report.requested_part_mentions)
        self.assertIn("LIDAR-L1", report.requested_part_mentions)
        parts_by_role = {part.role: part for part in report.resolved_parts}
        self.assertEqual("cmp_actuator_vx20", parts_by_role["joint_actuator"].component_id)
        self.assertEqual("explicit_user_request", parts_by_role["joint_actuator"].source)
        self.assertEqual("cmp_lidar_planar_l1", parts_by_role["wrist_sensor"].component_id)
        self.assertEqual("lidar", parts_by_role["wrist_sensor"].technical_limits["sensor_type"])
        self.assertEqual(8.0, parts_by_role["wrist_sensor"].technical_limits["max_range_m"])

    def test_policy_plan_uses_selected_wrist_sensor(self):
        requirements = build_requirements_from_prompt(
            "Build a tabletop arm with a VX-20 servo and LiDAR for sorting blocks.",
            sensor="lidar",
        )
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        policy = PolicyRecord(
            id="policy_lidar_test",
            name="LiDAR policy test",
            compatible_robot_ids=[robot_build.robot_genome.id],
            action_space="discrete_skill_sequence",
            observation_space=["object_poses", "joint_positions", robot_build.robot_genome.sensors[0].name],
        )
        plan = build_policy_plan(robot_build.robot_genome, task, policy)

        self.assertIn("wrist_lidar", plan.observation_keys)
        self.assertIn("wrist_lidar", plan.steps[0].expected_observations)
        self.assertNotIn("wrist_rgbd", plan.steps[0].expected_observations)

    def test_fabrication_plan_maps_parts_to_cad_operations(self):
        requirements = build_requirements_from_prompt("Build a tabletop arm with a VX-20 servo for sorting blocks.")
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        plan = build_fabrication_plan(
            requirements,
            robot_build.robot_genome,
            robot_build.bom,
            robot_build.components,
            robot_build.part_resolution,
            robot_build.cad_models,
            robot_build.cad_assembly,
        )

        self.assertTrue(plan.validate().ok)
        self.assertEqual("cad/parametric/robot_arm.py", plan.parametric_source.uri)
        self.assertGreater(plan.estimated_printed_mass_kg, 0)
        self.assertIn("cad_forearm_link", plan.generated_cad_parameters)
        self.assertTrue(any(binding.role == "joint_actuator" for binding in plan.part_cad_bindings))
        self.assertTrue(any(operation.operation == "align_joint_axis" for operation in plan.assembly_operations))
        self.assertTrue(any(check.check == "component_cad_asset_coverage" for check in plan.fabrication_checks))

    def test_training_objective_builds_reward_and_termination_contract(self):
        requirements = build_requirements_from_prompt(
            "Build a warehouse robot arm to move 500 g boxes from a conveyor into a bin.",
        )
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        objective = build_training_objective(robot_build.robot_genome, task)

        self.assertTrue(objective.validate().ok)
        self.assertEqual("pick_place_box", task.task_type)
        reward_terms = {term.name for term in objective.reward_terms}
        self.assertIn("place_in_bin", reward_terms)
        self.assertIn("collision_penalty", reward_terms)
        self.assertTrue(any(rule.terminal_status == "success" for rule in objective.termination_rules))
        self.assertTrue(any(rule.terminal_status == "failure" for rule in objective.termination_rules))
        self.assertIn("success_rate", objective.metrics)

    def test_navigation_prompt_builds_navigation_task_graph(self):
        requirements = build_requirements_from_prompt(
            "Build a mobile robot base that can drive and navigate indoors to deliver parts.",
        )

        task = build_task_graph(requirements)

        self.assertTrue(task.validate().ok)
        self.assertEqual("navigation", task.task_type)
        self.assertIn("drive", task.required_skills)
        self.assertIn("goal_reached", {criterion.name for criterion in task.success_criteria})
        self.assertIn("obstacle_collision", {criterion.name for criterion in task.failure_criteria})

    def test_navigation_scene_set_contains_route_objects(self):
        requirements = build_requirements_from_prompt(
            "Build a mobile robot base that can drive and navigate indoors to deliver parts.",
        )
        task = build_task_graph(requirements)

        scene_set = generate_scene_set(task, count=3, purpose="navigation")

        self.assertTrue(scene_set.validate().ok)
        self.assertEqual("navigation", scene_set.purpose)
        self.assertEqual(3, len(scene_set.scenes))
        first_objects = {item.name: item for item in scene_set.scenes[0].objects}
        self.assertIn("start_zone", first_objects)
        self.assertIn("goal_zone", first_objects)
        self.assertTrue(any(item.object_type == "obstacle" for item in scene_set.scenes[0].objects))
        self.assertGreater(scene_set.scenes[0].variation_parameters["route_length_m"], 0)

    def test_perception_config_uses_selected_lidar_stream(self):
        requirements = build_requirements_from_prompt(
            "Build a tabletop arm with LiDAR for sorting blocks.",
            sensor="lidar",
        )
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        scene_set = generate_scene_set(task, count=2, purpose="variation")

        config = build_perception_config(robot_build.robot_genome, task, robot_build.components, [scene_set])

        self.assertTrue(config.validate().ok)
        # Scene ids are FILENAMES, so they carry a short digest of the task id, not the prompt slug (a 150-char
        # tracked path broke `git clone` on Windows). Deterministic for the same task + purpose + index.
        from virturoid.services.scene_generator import _task_token
        tok = _task_token(task)
        self.assertEqual([f"scene_{tok}_variation_000", f"scene_{tok}_variation_001"],
                         [scene.id for scene in scene_set.scenes])
        self.assertLessEqual(max(len(s.id) for s in scene_set.scenes), 40)
        self.assertEqual("wrist_lidar", config.sensor_streams[0].name)
        self.assertEqual("lidar", config.sensor_streams[0].sensor_type)
        self.assertEqual(["range_scan", "point_cloud"], config.sensor_streams[0].output_modalities)
        self.assertIn("observations/wrist_lidar/points.pcd", config.sensor_streams[0].simulated_outputs)
        self.assertTrue(config.vision_annotations)
        self.assertEqual("range_geometry", config.vision_annotations[0].name)
        self.assertIn("point_cloud_cluster", config.vision_annotations[0].annotation_modalities)
        self.assertIsNotNone(config.synthetic_dataset)
        self.assertIn("worldlabs_or_equivalent_texture_generator", config.synthetic_dataset.renderer_backends)
        self.assertGreaterEqual(len(config.texture_randomization), 3)

    def test_world_model_contract_maps_perception_to_physical_ai_targets(self):
        requirements = build_requirements_from_prompt("Build a tabletop robot arm with RGBD vision to sort blocks.")
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        scene_set = generate_scene_set(task, count=2, purpose="variation")
        perception = build_perception_config(robot_build.robot_genome, task, robot_build.components, [scene_set])

        contract = build_world_model_contract(robot_build.robot_genome, task, perception, [scene_set])

        self.assertTrue(contract.validate().ok)
        self.assertEqual(perception.id, contract.perception_config_id)
        self.assertIn("robot.joint_positions", {item.key for item in contract.state_variables})
        self.assertIn("world.object_poses", {item.key for item in contract.state_variables})
        self.assertTrue(any(item.name == "link_geometry" for item in contract.physical_parameter_targets))
        self.assertTrue(any(entity.name == "red_block" for entity in contract.observable_entities))
        self.assertEqual("datasets/synthetic_observation_manifest.json", contract.synthetic_dataset_manifest_uri)

    def test_world_model_artifacts_generate_cv_annotations_and_state_snapshots(self):
        requirements = build_requirements_from_prompt("Build a tabletop robot arm with RGBD vision to sort blocks.")
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        scene_set = generate_scene_set(task, count=1, purpose="variation")
        perception = build_perception_config(robot_build.robot_genome, task, robot_build.components, [scene_set])

        with tempfile.TemporaryDirectory() as tmpdir:
            write_world_model_artifacts(Path(tmpdir), robot_build.robot_genome, task, perception, [scene_set])
            index = json.loads((Path(tmpdir) / "datasets" / "synthetic_observations" / "index.json").read_text(encoding="utf-8"))
            first_frame = index["frames"][0]
            annotation = json.loads((Path(tmpdir) / first_frame["annotation_uri"]).read_text(encoding="utf-8"))
            world_state = json.loads((Path(tmpdir) / first_frame["world_state_uri"]).read_text(encoding="utf-8"))

            self.assertEqual(1, index["frame_count"])
            self.assertTrue(annotation["annotations"])
            self.assertIn("bbox_xywh", annotation["annotations"][0])
            self.assertIn("physical_parameters", annotation["annotations"][0])
            self.assertEqual(annotation["scene_id"], world_state["scene_id"])
            self.assertTrue(world_state["objects"])

    def test_mujoco_scene_collection_writes_one_xml_per_scene(self):
        requirements = build_requirements_from_prompt("Build a tabletop arm for sorting blocks.")
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        baseline_scene_set = generate_scene_set(task, count=2, purpose="baseline")
        variation_scene_set = generate_scene_set(task, count=3, purpose="variation")

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = write_mujoco_scene_collection(
                robot_build.robot_genome,
                [baseline_scene_set, variation_scene_set],
                Path(tmpdir),
            )

            self.assertTrue(index_path.exists())
            self.assertTrue((Path(tmpdir) / "simulation" / "mujoco" / "scenes" / "baseline").exists())
            self.assertTrue((Path(tmpdir) / "simulation" / "mujoco" / "scenes" / "variation").exists())
            self.assertEqual(5, len(list((Path(tmpdir) / "simulation" / "mujoco" / "scenes").glob("*/*.xml"))))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual("mujoco", index["backend"])
            self.assertEqual(5, index["scene_count"])
            first_xml = (Path(tmpdir) / index["scenes"][0]["mujoco_xml"]).read_text(encoding="utf-8")
            self.assertIn('<camera name="wrist_rgbd_camera"', first_xml)
            self.assertIn('<sensor>', first_xml)

    def test_mujoco_scene_collection_reflects_lidar_sensor_choice(self):
        requirements = build_requirements_from_prompt("Build a tabletop arm with LiDAR for sorting blocks.", sensor="lidar")
        robot_build = build_reference_robot_arm(requirements, curated_component_library())
        task = build_task_graph(requirements)
        scene_set = generate_scene_set(task, count=1, purpose="baseline")

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = write_mujoco_scene_collection(robot_build.robot_genome, [scene_set], Path(tmpdir))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            scene_xml = (Path(tmpdir) / index["scenes"][0]["mujoco_xml"]).read_text(encoding="utf-8")

            self.assertIn('site name="wrist_lidar_ray_origin"', scene_xml)
            self.assertIn('<rangefinder name="wrist_lidar_range"', scene_xml)

    def test_mock_evaluation_generates_failures(self):
        entities = build_demo_entities()
        task = build_task_graph(entities["requirements"])
        scene_set = generate_scene_set(task, count=5, purpose="variation")
        policy = PolicyRecord(
            id="policy_test",
            name="Test policy",
            compatible_robot_ids=[entities["robot_genome"].id],
            action_space="discrete_skill_sequence",
            observation_space=["object_poses"],
        )

        run = run_mock_evaluation(entities["robot_genome"], task, scene_set, policy)

        self.assertTrue(run.validate().ok)
        self.assertEqual(5, len(run.episodes))
        self.assertGreaterEqual(len(run.failures), 1)
        self.assertEqual("collision", run.failures[0].failure_type)

    def test_export_writer_materializes_manifest_and_project(self):
        entities = build_demo_entities()
        bundle = entities["export_bundle"]
        self.assertIsInstance(bundle, ExportBundle)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_export_bundle(bundle, entities, Path(tmpdir) / "export")

            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "project.json").exists())


if __name__ == "__main__":
    unittest.main()
