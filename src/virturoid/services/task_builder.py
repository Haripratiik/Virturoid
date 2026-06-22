from __future__ import annotations

from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.tasks import Criterion, TaskGraph


def build_task_graph(requirements: RequirementsRecord) -> TaskGraph:
    """Create a first-pass task graph from structured requirements.

    This is intentionally deterministic. Later, an AI agent can propose richer
    fields, but this service should remain the validation-friendly fallback.
    """
    prompt = requirements.prompt.lower()

    if "sort" in prompt and "block" in prompt:
        return TaskGraph(
            id=f"task_{requirements.id}",
            name="Sort colored blocks",
            prompt=requirements.prompt,
            task_type="pick_place_sort",
            required_skills=["reach", "grasp", "lift", "place"],
            objects=["red_block", "blue_block", "red_bin", "blue_bin"],
            success_criteria=[
                Criterion(name="red_sorted", expression="red_block inside red_bin"),
                Criterion(name="blue_sorted", expression="blue_block inside blue_bin"),
            ],
            failure_criteria=[
                Criterion(name="dropped_object", expression="any block outside workspace"),
                Criterion(name="timeout", expression="episode_time > 60"),
                Criterion(name="collision", expression="robot collides with forbidden object"),
            ],
            safety_constraints=["no_joint_limit_violation", "no_forbidden_collision"],
            metrics=["success_rate", "collision_count", "timeout_rate", "drop_rate"],
        )

    if "box" in prompt or "conveyor" in prompt:
        return TaskGraph(
            id=f"task_{requirements.id}",
            name="Move boxes to target bin",
            prompt=requirements.prompt,
            task_type="pick_place_box",
            required_skills=["reach", "grasp", "lift", "place"],
            objects=["box", "target_bin", "conveyor_zone"],
            success_criteria=[
                Criterion(name="box_placed", expression="box inside target_bin"),
                Criterion(name="box_stable", expression="box stable for 2 seconds"),
            ],
            failure_criteria=[
                Criterion(name="dropped_box", expression="box outside workspace"),
                Criterion(name="conveyor_collision", expression="robot collides with conveyor_zone"),
                Criterion(name="timeout", expression="episode_time > 60"),
            ],
            safety_constraints=["no_joint_limit_violation", "no_conveyor_collision"],
            metrics=["success_rate", "collision_count", "drop_rate", "timeout_rate"],
        )

    if any(token in prompt for token in ["mobile", "drive", "navigate", "navigation", "deliver", "delivery", "inspect"]):
        return TaskGraph(
            id=f"task_{requirements.id}",
            name="Navigate indoor route",
            prompt=requirements.prompt,
            task_type="navigation",
            required_skills=["localize", "plan_path", "drive", "avoid_obstacle", "dock"],
            objects=["start_zone", "goal_zone", "obstacle"],
            success_criteria=[
                Criterion(name="goal_reached", expression="base_link inside goal_zone"),
                Criterion(name="route_stable", expression="robot remains upright for full route"),
            ],
            failure_criteria=[
                Criterion(name="obstacle_collision", expression="robot collides with obstacle"),
                Criterion(name="timeout", expression="episode_time > 120"),
                Criterion(name="tip_over", expression="base roll or pitch exceeds safety limit"),
            ],
            safety_constraints=["no_obstacle_collision", "stay_inside_route_bounds", "max_speed_limit"],
            metrics=["success_rate", "collision_count", "route_time", "path_error_m", "tip_over_rate"],
        )

    return TaskGraph(
        id=f"task_{requirements.id}",
        name=requirements.task_summary or "User task",
        prompt=requirements.prompt,
        task_type="generic_manipulation",
        required_skills=["reach"],
        objects=[],
        success_criteria=[Criterion(name="task_completed", expression="user-defined success state reached")],
        failure_criteria=[Criterion(name="timeout", expression="episode_time > 60")],
        safety_constraints=["no_joint_limit_violation"],
        metrics=["success_rate", "timeout_rate"],
    )
