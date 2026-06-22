"""Policy Card (startup plan §12.4 / §12.3) — the honest, portable summary of an evaluated policy.

Every trained or imported policy gets a card: what robots/tasks it is for, the scenes it was trained
and evaluated on, its held-out success (broken out by the §32.5 evaluation categories — fixed baseline,
randomized, known-failure/out-of-region), its known failures and limitations, required sensors, and
runtime requirements. The card is what flows into Memory and Export so a later build can decide whether
to reuse the policy without re-reading the training logs — the §32.5 antidote to "we trained something
and hope it works".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class CategoryResult:
    """§32.5 evaluation broken out per scene category (baseline / randomized / known_failure / ...)."""
    category: str
    scene_count: int = 0
    success_rate: float = 0.0
    mean_metric: float = 0.0          # task-natural metric (e.g. lift height for grasp)
    notes: str = ""


@dataclass
class PolicyCard(VersionedEntity):
    name: str = ""
    policy_type: str = ""                              # e.g. residual_rl_grasp
    task_type: str = ""
    compatible_robots: list[str] = field(default_factory=list)
    compatible_tasks: list[str] = field(default_factory=list)
    observation_space: list[str] = field(default_factory=list)
    action_space: str = ""
    valid_region: dict = field(default_factory=dict)   # the body↔task region the policy is valid in
    training_scenes: str = ""                          # provenance of the scenes it trained on
    success_rate: float = 0.0                          # headline = in-distribution (baseline+randomized)
    category_results: list[CategoryResult] = field(default_factory=list)  # §32.5 breakdown
    reproducible: bool = False                         # same seed -> same result (§21.4 / §32.5)
    failure_clusters: list[dict] = field(default_factory=list)   # {reason, count, example_scenes} (§8.5/§26)
    known_failures: list[str] = field(default_factory=list)
    safety_constraints: list[str] = field(default_factory=list)
    required_sensors: list[str] = field(default_factory=list)
    runtime_requirements: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    params_uri: str = ""
    backend: str = ""

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "Policy card must have a name.")
        require_non_empty(result, self.task_type, "task_type", "Policy card must name a task type.")
        require_non_empty(result, self.compatible_robots, "compatible_robots",
                          "Policy card must list at least one compatible robot.")
        if not 0.0 <= self.success_rate <= 1.0:
            result.add("invalid_success_rate", "success_rate must be in [0,1].", "success_rate")
        return result
