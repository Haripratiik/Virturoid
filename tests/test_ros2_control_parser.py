"""Input Ingestion plan, Phase 3: ros2_control interface extraction (the highest-value enterprise truth).

Parses the <ros2_control> URDF/xacro tag + controller_manager YAML into a ControllerInterfaceSpec, and ties it to
the policy-import acceptance check (policy N actions vs M command interfaces). Pure/offline (AGENTS.md).
"""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.ros2_control_parser import (  # noqa: E402
    controller_interface_from_ros2_control,
    parse_controller_yaml,
    parse_ros2_control,
    parse_ros_package,
)

_URDF = """<?xml version="1.0"?>
<robot name="arm" xmlns:xacro="http://ros.org/wiki/xacro">
  <ros2_control name="arm_hw" type="system">
    <hardware><plugin>fake_components/GenericSystem</plugin></hardware>
    <joint name="shoulder">
      <command_interface name="position"><param name="min">-3.14</param><param name="max">3.14</param></command_interface>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="elbow">
      <command_interface name="position"/>
      <state_interface name="position"/>
    </joint>
    <sensor name="ft_sensor"><state_interface name="force.x"/></sensor>
  </ros2_control>
</robot>"""

_YAML = """
controller_manager:
  ros__parameters:
    update_rate: 100
    arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController
arm_controller:
  ros__parameters:
    joints: [shoulder, elbow]
    command_interfaces: [position]
    state_interfaces: [position, velocity]
"""


class Ros2ControlTests(unittest.TestCase):
    def test_extracts_joints_interfaces_limits_plugin(self):
        p = parse_ros2_control(_URDF)
        self.assertEqual(p["joints"], ["shoulder", "elbow"])
        self.assertEqual(p["command_interfaces"], ["shoulder/position", "elbow/position"])
        self.assertIn("shoulder/velocity", p["state_interfaces"])
        self.assertIn("ft_sensor/force.x", p["state_interfaces"])   # sensor state interface too
        self.assertEqual(p["safety_limits"]["shoulder/position"], {"min": "-3.14", "max": "3.14"})
        self.assertEqual(p["hardware_plugins"], ["fake_components/GenericSystem"])
        self.assertEqual(p["sensors"], ["ft_sensor"])

    def test_unresolved_xacro_joint_is_warned_not_faked(self):
        xml = ('<robot><ros2_control name="h" type="system">'
               '<joint name="${prefix}wrist"><command_interface name="position"/></joint>'
               '</ros2_control></robot>')
        p = parse_ros2_control(xml)
        self.assertEqual(p["joints"], [])                           # not treated as a literal joint
        self.assertTrue(any("xacro not expanded" in w for w in p["warnings"]))

    def test_controller_yaml(self):
        y = parse_controller_yaml(_YAML)
        self.assertEqual(y["update_rate_hz"], 100.0)
        self.assertIn("arm_controller", y["controllers"])
        self.assertEqual(y["controllers"]["arm_controller"]["joints"], ["shoulder", "elbow"])

    def test_controller_interface_spec_and_policy_acceptance(self):
        spec = controller_interface_from_ros2_control(_URDF, controller_yaml=_YAML)
        self.assertEqual(spec.action_keys, ["shoulder/position", "elbow/position"])
        self.assertEqual(spec.joint_order, ["shoulder", "elbow"])
        self.assertEqual(spec.control_frequency_hz, 100.0)
        self.assertTrue(spec.validate().ok)
        # the plan's acceptance: a 3-action policy mismatches the 2 command interfaces.
        from virturoid.schemas.policy_import import PolicyImportSpec
        from virturoid.services.policy_importer import check_action_dim
        pol = PolicyImportSpec(id="p", source_ref="p.py", action_dim=3)
        issues = check_action_dim(pol, len(spec.action_keys))
        self.assertTrue(issues and "3 actions" in issues[0])

    def test_package_xml(self):
        pkg = parse_ros_package(
            '<package><name>my_robot_bringup</name><depend>ros2_control</depend>'
            '<exec_depend>controller_manager</exec_depend></package>')
        self.assertEqual(pkg["name"], "my_robot_bringup")
        self.assertIn("ros2_control", pkg["dependencies"])
        self.assertIn("controller_manager", pkg["dependencies"])


if __name__ == "__main__":
    unittest.main()
