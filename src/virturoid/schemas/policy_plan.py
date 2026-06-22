from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class PolicyStep:
    step_id: str
    skill: str
    command: str
    target_selector: str
    expected_observations: list[str] = field(default_factory=list)
    success_condition: str = ""
    fallback: str = ""


@dataclass
class PolicyPlan(VersionedEntity):
    policy_id: str = ""
    task_graph_id: str = ""
    robot_genome_id: str = ""
    policy_type: str = "scripted_skill_sequence"
    observation_keys: list[str] = field(default_factory=list)
    action_space: str = "discrete_skill_sequence"
    steps: list[PolicyStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.policy_id, "policy_id", "Policy plan must reference a policy.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Policy plan must reference a task graph.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Policy plan must reference a robot genome.")
        require_non_empty(result, self.observation_keys, "observation_keys", "Policy plan must define observations.")
        require_non_empty(result, self.steps, "steps", "Policy plan must include executable steps.")
        for index, step in enumerate(self.steps):
            if not step.step_id:
                result.add("missing_step_id", "Policy step must have an id.", f"steps[{index}].step_id")
            if not step.skill:
                result.add("missing_step_skill", "Policy step must name a skill.", f"steps[{index}].skill")
            if not step.command:
                result.add("missing_step_command", "Policy step must name a command.", f"steps[{index}].command")
        return result
