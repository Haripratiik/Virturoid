import unittest
from xml.etree import ElementTree

from virturoid.fixtures.components import curated_component_library
from virturoid.services.mujoco_exporter import robot_and_single_scene_to_mjcf
from virturoid.services.robot_arm_builder import build_reference_robot_arm
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.robot_kinematics import compute_arm_layout, iter_links
from virturoid.services.scene_generator import generate_scene_set
from virturoid.services.task_builder import build_task_graph
from virturoid.services.urdf_exporter import robot_genome_to_urdf

try:
    import mujoco  # noqa: F401

    _MUJOCO = True
except Exception:  # noqa: BLE001
    _MUJOCO = False


def _build():
    requirements = build_requirements_from_prompt("Build a tabletop arm for sorting blocks.")
    build = build_reference_robot_arm(requirements, curated_component_library())
    task = build_task_graph(requirements)
    scene = generate_scene_set(task, count=1, purpose="baseline").scenes[0]
    return build, scene


class RobotModelFidelityTests(unittest.TestCase):
    def test_layout_is_a_serial_chain_with_single_end_effector(self):
        build, _ = _build()
        root = compute_arm_layout(build.robot_genome, build.cad_models)
        links = list(iter_links(root))

        self.assertEqual("base_link", root.name)
        # Every genome joint's parent/child relationship is reflected in the tree.
        layout_by_name = {link.name: link for link in links}
        for joint in build.robot_genome.joints:
            child = layout_by_name[joint.child_link]
            self.assertEqual(joint.parent_link, child.parent)
        # Exactly one end-effector site holder, and it is the gripper tip.
        ee_links = [link for link in links if link.is_end_effector]
        self.assertEqual(1, len(ee_links))
        self.assertEqual("gripper_link", ee_links[0].name)

    def test_urdf_and_mjcf_share_link_lengths(self):
        build, scene = _build()
        root = compute_arm_layout(build.robot_genome, build.cad_models)
        layout_by_name = {link.name: link for link in iter_links(root)}

        urdf = robot_genome_to_urdf(build.robot_genome, cad_models=build.cad_models)
        urdf_root = ElementTree.fromstring(urdf)
        # The shoulder joint origin in URDF equals the upper_link length from the layout.
        shoulder = next(j for j in urdf_root.findall("joint") if j.get("name") == "shoulder_pitch")
        origin_z = float(shoulder.find("origin").get("xyz").split()[2])
        self.assertAlmostEqual(layout_by_name["upper_link"].length_m, origin_z, places=4)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_mjcf_compiles_as_serial_chain_with_real_limits(self):
        build, scene = _build()
        xml = robot_and_single_scene_to_mjcf(build.robot_genome, scene, "scenes_test", build.cad_models)
        model = mujoco.MjModel.from_xml_string(xml)

        bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        self.assertIn("gripper_link", bodies)
        # Manipulable blocks are dynamic free bodies (7 qpos each) on top of 3 hinges.
        self.assertIn("obj_red_block", bodies)
        # 3 arm hinges + 2 parallel-jaw finger actuators: the gripper is actuated for real contact grasp/lift.
        self.assertEqual(5, model.nu)
        self.assertEqual(2, model.nsite)  # ee_site + grasp_site (the gripper's grasp reference frame)
        self.assertEqual("ee_site", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, 0))

        # Joint ranges come from the genome limits, not a hardcoded default.
        genome_limits = {j.name: j.limit for j in build.robot_genome.joints}
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name in genome_limits and genome_limits[name] and genome_limits[name].lower is not None:
                self.assertTrue(bool(model.jnt_limited[i]))
                self.assertAlmostEqual(genome_limits[name].lower, float(model.jnt_range[i][0]), places=3)

        # Actuator force range reflects the actuator stall torque from the part spec.
        self.assertGreater(float(model.actuator_forcerange[0][1]), 0.0)


if __name__ == "__main__":
    unittest.main()
