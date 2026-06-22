from __future__ import annotations

from virturoid.schemas.morphology import MorphologyModule, MorphologyTemplate


def morphology_template_catalog() -> list[MorphologyTemplate]:
    return [
        MorphologyTemplate(
            id="morph_fixed_arm_three_dof_tabletop",
            robot_class="manipulator",
            species_pattern="fixed_arm.three_dof.tabletop",
            parent_species="fixed_arm",
            morphology_tags=["fixed_base", "serial_chain", "manipulator", "tabletop"],
            supported_task_types=["pick_place_sort", "pick_place_box", "generic_manipulation"],
            required_modules=[
                MorphologyModule("base", "Fixed base or mounting plate.", compatible_component_categories=["structural"]),
                MorphologyModule("joint_actuator", "Repeated rotary actuator used for arm joints.", compatible_component_categories=["actuator"]),
                MorphologyModule("links", "Rigid upper and forearm links sized from reach requirements.", compatible_component_categories=["structural"]),
                MorphologyModule("wrist_sensor", "Task-facing wrist perception module.", compatible_component_categories=["sensor"]),
                MorphologyModule("end_effector", "Task-appropriate gripper or tool.", compatible_component_categories=["gripper"]),
                MorphologyModule("main_compute", "Onboard or nearby controller computer.", compatible_component_categories=["compute"]),
            ],
            default_links=["base_link", "upper_link", "forearm_link", "wrist_link", "gripper_link"],
            default_skills=["reach", "grasp", "lift", "place", "sort"],
            builder_service="virturoid.services.robot_arm_builder.build_reference_robot_arm",
            notes=[
                "Current MVP template.",
                "Exercises the generic build pipeline through a bounded manipulation robot.",
            ],
        ),
        MorphologyTemplate(
            id="morph_mobile_base_differential",
            robot_class="mobile_base",
            species_pattern="mobile_base.differential.indoor",
            parent_species="mobile_robot",
            morphology_tags=["mobile", "differential_drive", "indoor"],
            supported_task_types=["navigation", "delivery", "inspection"],
            required_modules=[
                MorphologyModule("chassis", "Mobile chassis with protected electronics bay.", compatible_component_categories=["structural"]),
                MorphologyModule("drive_actuator", "Left/right wheel drive actuators.", compatible_component_categories=["actuator"]),
                MorphologyModule("localization_sensor", "Camera, LiDAR, or depth sensor for navigation.", compatible_component_categories=["sensor"]),
                MorphologyModule("main_compute", "Navigation and control compute.", compatible_component_categories=["compute"]),
                MorphologyModule("power_pack", "Battery and regulator stack.", compatible_component_categories=["power"]),
            ],
            default_links=["base_link", "left_wheel_link", "right_wheel_link", "sensor_mast_link"],
            default_skills=["localize", "plan_path", "avoid_obstacle", "dock"],
            builder_service="virturoid.services.mobile_base_builder.build_differential_mobile_base",
            notes=[
                "Second implemented robot class; proves the build pipeline is not arm-specific.",
                "Built via `python -m virturoid.mobile_base`.",
            ],
        ),
        MorphologyTemplate(
            id="morph_humanoid_upper_body",
            robot_class="humanoid",
            species_pattern="humanoid.upper_body.bimanual",
            parent_species="humanoid",
            morphology_tags=["humanoid", "bimanual", "torso", "manipulator"],
            supported_task_types=["bimanual_manipulation", "tool_use", "human_environment_interaction"],
            required_modules=[
                MorphologyModule("torso", "Stable torso structure with shoulder mounts.", compatible_component_categories=["structural"]),
                MorphologyModule("arm_pair", "Two multi-DOF arm chains.", compatible_component_categories=["actuator", "structural"]),
                MorphologyModule("head_sensor", "Head-mounted scene perception.", compatible_component_categories=["sensor"]),
                MorphologyModule("hand_end_effectors", "Dexterous hands or grippers.", compatible_component_categories=["gripper"]),
                MorphologyModule("main_compute", "Whole-body policy compute.", compatible_component_categories=["compute"]),
            ],
            default_links=["torso_link", "left_upper_arm_link", "right_upper_arm_link", "head_link"],
            default_skills=["reach", "grasp", "handover", "coordinate_two_arms"],
            builder_service="planned.humanoid_builder",
            notes=["Catalog entry to keep the species tree extensible beyond the arm MVP."],
        ),
        MorphologyTemplate(
            id="morph_quadruped_quadruped",
            robot_class="quadruped",
            species_pattern="quadruped.four_leg.indoor",
            parent_species="legged_robot",
            morphology_tags=["legged", "quadruped", "dynamic"],
            supported_task_types=["locomotion", "inspection", "rough_terrain"],
            required_modules=[
                MorphologyModule("torso", "Quadruped torso with leg mounts.", compatible_component_categories=["structural"]),
                MorphologyModule("leg_actuators", "Three actuators per leg.", compatible_component_categories=["actuator"]),
                MorphologyModule("perception", "Body-mounted terrain perception.", compatible_component_categories=["sensor"]),
                MorphologyModule("main_compute", "Whole-body locomotion compute.", compatible_component_categories=["compute"]),
            ],
            default_links=["torso_link", "front_left_leg", "front_right_leg", "rear_left_leg", "rear_right_leg"],
            default_skills=["walk", "trot", "balance", "climb_step"],
            builder_service="planned.quadruped_builder",
            notes=["Catalog entry to keep the species tree extensible beyond arm and wheeled bases."],
        ),
    ]
