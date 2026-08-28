"""Virturoid evaluation node: runs the exported controller and publishes its joint targets.

If a controller bundle was exported with this package, the node loads the real ReachController and,
each tick, infers joint position targets for the next target position and publishes them as a
JointTrajectory. With no controller it publishes a neutral pose. This is a runnable harness for the
exported policy (plan §24), not a stub.
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_PKG = Path(__file__).resolve().parent


class EvaluationNode(Node):
    def __init__(self):
        super().__init__("virturoid_evaluation_node")
        config = json.loads((_PKG.parents[1] / "config" / "robot.yaml").read_text())
        self.joints = config["joints"]
        self.targets = config.get("target_positions") or [[0.4, 0.0]]
        self.policy_type = config.get("policy_type", "reach")
        self.i = 0
        self.t = 0.0
        self.dt = 1.0 / config.get("control_frequency_hz", 20.0)
        self.controller = None
        if config.get("has_controller") and (_PKG / "controller.py").exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("vq_controller", _PKG / "controller.py")
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            cls = mod.GaitController if self.policy_type == "trot_cpg_gait" else mod.ReachController
            self.controller = cls.from_file(str(_PKG / "policy_params.json"))
            self.joints = self.controller.joint_names
        self.pub = self.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
        self.create_timer(self.dt, self.tick)

    def tick(self):
        msg = JointTrajectory()
        msg.joint_names = self.joints
        point = JointTrajectoryPoint()
        if self.controller is not None:
            if self.policy_type == "trot_cpg_gait":
                targets = self.controller.infer(self.t); self.t += self.dt
            else:
                target = self.targets[self.i % len(self.targets)]; self.i += 1
                targets = self.controller.infer(target)
            point.positions = [float(targets[j]) for j in self.joints]
        else:
            point.positions = [0.0 for _ in self.joints]
        msg.points = [point]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = EvaluationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
