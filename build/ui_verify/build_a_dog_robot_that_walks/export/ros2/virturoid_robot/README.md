# virturoid_robot

Generated ROS2 package for `genome_anatomy_creature` (no controller bundle; node publishes a neutral pose).

```
colcon build --packages-select virturoid_robot
ros2 launch virturoid_robot evaluate.launch.py
```

## Deploy to hardware (§4.7)
`config/ros2_control.yaml` (controller manager) + `config/hardware_interface.yaml` (each joint -> its
real BOM actuator) wire the controller to ros2_control; set `hardware_plugin` to your motor-bus driver
(Dynamixel / ODrive / CAN / EtherCAT). `virturoid_robot/safety_filter.py` clamps every command to
joint + rate limits before a motor sees it. Validate the closed loop in sim first via
`services/sim_ros_bridge` (MuJoCo as virtual hardware behind the same command/state interface).
