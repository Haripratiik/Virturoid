import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from virturoid.mobile_base import build_mobile_base_project
from virturoid.fixtures.components import curated_component_library
from virturoid.fixtures.morphologies import morphology_template_catalog
from virturoid.services.mobile_base_exporter import mobile_base_to_mjcf, write_mobile_base_scene_collection
from virturoid.services.morphology_selector import select_morphology_template
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.robot_builder_registry import build_robot_from_selection
from virturoid.services.scene_generator import generate_scene_set
from virturoid.services.task_builder import build_task_graph

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MOBILE_PROMPT = "Build a mobile robot base that can drive and navigate indoors to deliver parts."


def _build_mobile():
    requirements = build_requirements_from_prompt(_MOBILE_PROMPT)
    task = build_task_graph(requirements)
    templates = morphology_template_catalog()
    selection = select_morphology_template(requirements, task, templates)
    result = build_robot_from_selection(requirements, curated_component_library(), selection, templates)
    return selection, result


class MobileBaseTests(unittest.TestCase):
    def test_selector_and_registry_dispatch_a_second_robot_class(self):
        selection, result = _build_mobile()

        self.assertEqual("mobile_base", selection.selected_template.robot_class)
        self.assertEqual("implemented", result.dispatch_record.dispatch_status)
        self.assertEqual(
            "virturoid.services.mobile_base_builder.build_differential_mobile_base",
            result.dispatch_record.selected_builder_service,
        )

    def test_mobile_genome_is_a_valid_wheeled_robot(self):
        _, result = _build_mobile()
        genome = result.robot_build.robot_genome

        self.assertTrue(genome.validate().ok, genome.validate().issues)
        self.assertEqual("mobile_base.differential.indoor", genome.species)
        self.assertEqual(["left_wheel", "right_wheel"], [j.name for j in genome.joints])
        self.assertTrue(all(j.joint_type == "continuous" for j in genome.joints))

    def test_mobile_base_package_includes_navigation_scenes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = build_mobile_base_project(_MOBILE_PROMPT, Path(tmpdir) / "mobile")
            output_dir = project["output_dir"]

            scene_set = json.loads((output_dir / "simulation" / "scene_set.json").read_text(encoding="utf-8"))
            perception = json.loads((output_dir / "simulation" / "perception_config.json").read_text(encoding="utf-8"))
            world_model = json.loads((output_dir / "simulation" / "world_model_contract.json").read_text(encoding="utf-8"))
            synthetic_manifest = json.loads((output_dir / "datasets" / "synthetic_observation_manifest.json").read_text(encoding="utf-8"))
            observation_index = json.loads((output_dir / "datasets" / "synthetic_observations" / "index.json").read_text(encoding="utf-8"))
            world_state_index = json.loads((output_dir / "simulation" / "world_state_index.json").read_text(encoding="utf-8"))
            compiled_index = json.loads(
                (output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8")
            )

            self.assertEqual("navigation", project["task_graph"].task_type)
            self.assertEqual("navigation", scene_set["purpose"])
            self.assertEqual(4, len(scene_set["scenes"]))
            self.assertEqual(4, compiled_index["scene_count"])
            self.assertIn("rgb_image", perception["sensor_streams"][0]["output_modalities"])
            self.assertIn("world.obstacle_map", {item["key"] for item in world_model["state_variables"]})
            self.assertEqual("simulation/world_model_contract.json", synthetic_manifest["world_model_contract_uri"])
            self.assertEqual(4, observation_index["frame_count"])
            self.assertEqual(4, world_state_index["scene_state_count"])
            self.assertTrue((output_dir / observation_index["frames"][0]["annotation_uri"]).exists())
            self.assertTrue((output_dir / world_state_index["scene_states"][0]["world_state_uri"]).exists())
            self.assertTrue((output_dir / compiled_index["scenes"][0]["mujoco_xml"]).exists())

    def test_mobile_scene_collection_compiles_route_objects_to_mjcf(self):
        _, result = _build_mobile()
        task = build_task_graph(build_requirements_from_prompt(_MOBILE_PROMPT))
        scene_set = generate_scene_set(task, count=1, purpose="navigation")

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = write_mobile_base_scene_collection(result.robot_build.robot_genome, scene_set, Path(tmpdir))
            index = json.loads(index_path.read_text(encoding="utf-8"))
            xml_path = Path(tmpdir) / index["scenes"][0]["mujoco_xml"]
            xml = xml_path.read_text(encoding="utf-8")

            self.assertEqual("mujoco", ElementTree.parse(xml_path).getroot().tag)
            self.assertIn('name="goal_zone"', xml)
            self.assertIn('name="obstacle_0_0"', xml)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_mobile_base_mjcf_drives_forward_and_turns(self):
        import mujoco

        from virturoid.services.mobile_base_exporter import drive_rollout

        _, result = _build_mobile()
        genome = result.robot_build.robot_genome
        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = Path(tmpdir) / "mobile.xml"
            mjcf_path.write_text(mobile_base_to_mjcf(genome), encoding="utf-8")
            mujoco.MjModel.from_xml_path(str(mjcf_path))  # compiles

            forward = drive_rollout(mjcf_path, left_torque=2.0, right_torque=2.0, steps=500)
            turn = drive_rollout(mjcf_path, left_torque=2.5, right_torque=-2.5, steps=500)

            self.assertTrue(forward["stable"])
            # Equal torque drives forward; the base should travel a meaningful distance.
            self.assertGreater(forward["forward_distance_m"], 0.1)
            # Opposite torques rotate the base in place without much translation.
            self.assertGreater(abs(turn["yaw_change_rad"]), 0.3)


if __name__ == "__main__":
    unittest.main()
