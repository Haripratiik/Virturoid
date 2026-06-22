from __future__ import annotations

from virturoid.schemas.policy_plan import PolicyPlan, PolicyStep
from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.runs import PolicyRecord
from virturoid.schemas.tasks import TaskGraph


def build_policy_plan(robot: RobotGenome, task: TaskGraph, policy: PolicyRecord) -> PolicyPlan:
    primary_sensor = robot.sensors[0].name if robot.sensors else "wrist_sensor"
    if task.task_type == "pick_place_box":
        steps = _box_handling_steps(primary_sensor)
    else:
        steps = _sorting_steps(primary_sensor)

    return PolicyPlan(
        id=f"plan_{policy.id}",
        policy_id=policy.id,
        task_graph_id=task.id,
        robot_genome_id=robot.id,
        observation_keys=list(policy.observation_space),
        action_space=policy.action_space,
        steps=steps,
        notes=[
            "MVP scripted policy plan generated from the task graph.",
            "A learned controller can replace or refine these skill steps after simulator rollouts exist.",
        ],
    )


def _sorting_steps(primary_sensor: str) -> list[PolicyStep]:
    return [
        PolicyStep(
            step_id="observe_workspace",
            skill="perception",
            command="detect_sortable_objects_and_bins",
            target_selector="all_visible_objects",
            expected_observations=["object_poses", "bin_poses", primary_sensor],
            success_condition="at least one object and matching bin are localized",
            fallback="request wider camera sweep",
        ),
        PolicyStep(
            step_id="select_target",
            skill="task_planning",
            command="select_next_object_by_color",
            target_selector="nearest_unsorted_object",
            expected_observations=["object_poses"],
            success_condition="target object has a matching destination bin",
            fallback="skip ambiguous object",
        ),
        PolicyStep(
            step_id="plan_grasp",
            skill="motion_planning",
            command="compute_top_down_grasp",
            target_selector="selected_object",
            expected_observations=["object_pose", "joint_positions"],
            success_condition="grasp pose is reachable and collision free",
            fallback="try side grasp",
        ),
        PolicyStep(
            step_id="execute_pick",
            skill="manipulation",
            command="move_close_gripper_lift",
            target_selector="selected_object",
            expected_observations=["joint_positions", "gripper_state"],
            success_condition="object is retained by gripper",
            fallback="retry with lower approach speed",
        ),
        PolicyStep(
            step_id="execute_place",
            skill="manipulation",
            command="move_to_matching_bin_open_gripper",
            target_selector="matching_bin",
            expected_observations=["bin_pose", "joint_positions"],
            success_condition="object rests inside matching bin",
            fallback="drop at safe staging zone",
        ),
    ]


def _box_handling_steps(primary_sensor: str) -> list[PolicyStep]:
    return [
        PolicyStep(
            step_id="observe_conveyor",
            skill="perception",
            command="detect_box_and_destination_bin",
            target_selector="moving_payload",
            expected_observations=["object_poses", "conveyor_pose", primary_sensor],
            success_condition="box and destination are localized",
            fallback="wait for stable detection",
        ),
        PolicyStep(
            step_id="synchronize_pick",
            skill="task_planning",
            command="estimate_intercept_pose",
            target_selector="box_on_conveyor",
            expected_observations=["object_velocity", "joint_positions"],
            success_condition="intercept pose is reachable before box leaves pickup zone",
            fallback="advance conveyor to next box",
        ),
        PolicyStep(
            step_id="execute_pick",
            skill="manipulation",
            command="match_conveyor_then_grasp",
            target_selector="box_on_conveyor",
            expected_observations=["joint_positions", "gripper_state"],
            success_condition="box is retained by gripper",
            fallback="release and retry on next pass",
        ),
        PolicyStep(
            step_id="stabilize_payload",
            skill="control",
            command="limit_acceleration_while_lifting",
            target_selector="held_box",
            expected_observations=["joint_positions", "payload_estimate"],
            success_condition="payload remains stable during lift",
            fallback="lower to recovery zone",
        ),
        PolicyStep(
            step_id="execute_place",
            skill="manipulation",
            command="place_box_in_bin_and_release",
            target_selector="destination_bin",
            expected_observations=["bin_pose", "joint_positions"],
            success_condition="box rests inside destination bin",
            fallback="place at safe staging zone",
        ),
    ]
