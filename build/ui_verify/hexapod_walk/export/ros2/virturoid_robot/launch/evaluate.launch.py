from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="virturoid_robot", executable="evaluation_node", name="virturoid_evaluation_node", output="screen"),
    ])
