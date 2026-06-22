from __future__ import annotations

from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.tasks import TaskGraph
from virturoid.schemas.training_objective import RewardTerm, TerminationRule, TrainingObjective


def build_training_objective(robot: RobotGenome, task: TaskGraph) -> TrainingObjective:
    reward_terms = _reward_terms_for_task(task)
    termination_rules = [
        *[
            TerminationRule(
                name=f"success_{criterion.name}",
                expression=criterion.expression,
                terminal_status="success",
            )
            for criterion in task.success_criteria
        ],
        *[
            TerminationRule(
                name=f"failure_{criterion.name}",
                expression=criterion.expression,
                terminal_status="failure",
            )
            for criterion in task.failure_criteria
        ],
    ]
    objective = TrainingObjective(
        id=f"objective_{task.id}",
        task_graph_id=task.id,
        robot_genome_id=robot.id,
        objective_type="scripted_rl_contract",
        success_gates=[criterion.expression for criterion in task.success_criteria],
        failure_gates=[criterion.expression for criterion in task.failure_criteria],
        reward_terms=reward_terms,
        termination_rules=termination_rules,
        metrics=list(task.metrics),
        notes=[
            "MVP objective translates task criteria into simulator-consumable reward and termination terms.",
            "A learned controller can use this contract for rollouts once a real simulator backend is connected.",
        ],
    )
    validation = objective.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{objective.id} failed validation: {issues}")
    return objective


def _reward_terms_for_task(task: TaskGraph) -> list[RewardTerm]:
    if task.task_type == "pick_place_box":
        return [
            RewardTerm("reach_box", "negative distance from gripper to box", 0.15, "dense"),
            RewardTerm("secure_grasp", "box retained by gripper", 0.25, "sparse", "box_stable"),
            RewardTerm("place_in_bin", "box inside target_bin", 1.0, "sparse", "box_placed"),
            RewardTerm("payload_stability", "box stable for 2 seconds", 0.4, "sparse", "box_stable"),
            RewardTerm("collision_penalty", "robot collides with conveyor_zone", -1.0, "sparse", "conveyor_collision"),
            RewardTerm("timeout_penalty", "episode_time > 60", -0.5, "sparse", "timeout"),
        ]
    if task.task_type == "pick_place_sort":
        return [
            RewardTerm("reach_selected_object", "negative distance from gripper to selected block", 0.15, "dense"),
            RewardTerm("grasp_selected_object", "selected block retained by gripper", 0.25, "sparse"),
            RewardTerm("red_sorted", "red_block inside red_bin", 0.75, "sparse", "red_sorted"),
            RewardTerm("blue_sorted", "blue_block inside blue_bin", 0.75, "sparse", "blue_sorted"),
            RewardTerm("all_sorted_bonus", "red_block inside red_bin and blue_block inside blue_bin", 1.0, "sparse"),
            RewardTerm("collision_penalty", "robot collides with forbidden object", -1.0, "sparse", "collision"),
            RewardTerm("drop_penalty", "any block outside workspace", -0.8, "sparse", "dropped_object"),
            RewardTerm("timeout_penalty", "episode_time > 60", -0.5, "sparse", "timeout"),
        ]
    return [
        RewardTerm("task_completed", "user-defined success state reached", 1.0, "sparse", "task_completed"),
        RewardTerm("timeout_penalty", "episode_time > 60", -0.5, "sparse", "timeout"),
    ]
