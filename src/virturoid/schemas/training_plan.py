"""TrainingPlan schema — the "missing narrow schema" from the Training dossier (R1) and Training Improvement Plan.

Today the trainer routing is fragmented across locomotion / navigation / grasp / reach-CEM / vision scripts.
The dossier's first concrete recommendation is a schema that represents the *plan* (task family, skill
decomposition, trainer ladder, teacher sources, budget, checkpoint/banking rules) rather than the trainer
implementation — so a compile step chooses the ladder deterministically and the UI/agents never call
simulator/RL libraries directly (AGENTS.md layering). Backend-agnostic, standard-library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


class TaskFamily(str, Enum):
    LOCOMOTION = "locomotion"
    NAVIGATION = "navigation"
    GRASP = "grasp"
    REACH = "reach"
    PUSH = "push"
    PICK_PLACE = "pick_place"
    SORT = "sort"
    TOOL_USE = "tool_use"
    INSPECT = "inspect"
    COMPOSITE = "composite"
    UNKNOWN = "unknown"


# The three-phase training ladder (dossier R2): cheap reuse -> distill/adapt -> expensive RL, plus a CPU fallback.
TRAINER_RUNGS = ("reuse_evaluate", "distill_adapt", "optimize_rl")


@dataclass
class TrainingPlan(VersionedEntity):
    task_family: str = TaskFamily.UNKNOWN.value
    robot_genome_id: str = ""
    task_graph_id: str = ""
    scene_set_refs: list[str] = field(default_factory=list)
    observation_contract_ref: str | None = None
    perception_training_plan_ref: str | None = None
    skill_sequence: list[str] = field(default_factory=list)
    candidate_warm_starts: list[str] = field(default_factory=list)
    objective_ref: str | None = None
    reward_candidates: list[str] = field(default_factory=list)
    trainer_ladder: list[str] = field(default_factory=list)
    teacher_sources: list[str] = field(default_factory=list)
    backend_budget: dict = field(default_factory=dict)
    domain_randomization_profile: dict = field(default_factory=dict)
    checkpoint_selection: dict = field(default_factory=dict)
    evaluation_protocol_ref: str | None = None
    banking_rules: dict = field(default_factory=dict)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.task_family, "task_family", "Training plan needs a task family.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Training plan must reference a robot.")
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Training plan must reference a task.")
        require_non_empty(result, self.trainer_ladder, "trainer_ladder", "Training plan needs a trainer ladder.")
        for rung in self.trainer_ladder:
            if rung not in TRAINER_RUNGS and rung != "cpu_fallback":
                result.add("unknown_trainer_rung", f"'{rung}' is not a known trainer rung.",
                           "trainer_ladder", severity="warning")
        return result
