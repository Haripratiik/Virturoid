from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class RewardTerm:
    name: str
    expression: str
    weight: float
    kind: str = "dense"
    source_criterion: str | None = None


@dataclass
class TerminationRule:
    name: str
    expression: str
    terminal_status: str


@dataclass
class TrainingObjective(VersionedEntity):
    task_graph_id: str = ""
    robot_genome_id: str = ""
    objective_type: str = "reinforcement_learning"
    success_gates: list[str] = field(default_factory=list)
    failure_gates: list[str] = field(default_factory=list)
    reward_terms: list[RewardTerm] = field(default_factory=list)
    termination_rules: list[TerminationRule] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Training objective must reference a task graph.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Training objective must reference a robot.")
        require_non_empty(result, self.objective_type, "objective_type", "Training objective type is required.")
        require_non_empty(result, self.success_gates, "success_gates", "Training objective must include success gates.")
        require_non_empty(result, self.failure_gates, "failure_gates", "Training objective must include failure gates.")
        require_non_empty(result, self.reward_terms, "reward_terms", "Training objective must include reward terms.")
        require_non_empty(result, self.termination_rules, "termination_rules", "Training objective must include termination rules.")
        for index, reward in enumerate(self.reward_terms):
            if not reward.name:
                result.add("missing_reward_name", "Reward term name is required.", f"reward_terms[{index}].name")
            if not reward.expression:
                result.add("missing_reward_expression", "Reward expression is required.", f"reward_terms[{index}].expression")
        return result
