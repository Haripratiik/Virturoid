from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class DesignCandidate:
    forearm_mm: float
    wrist_mm: float
    actuator_torque_nm: float
    success_rate: float
    blocks_placed: int
    blocks_total: int
    # Controller ("brain") parameters; populated by the joint body+brain optimizer.
    controller: dict = field(default_factory=dict)


@dataclass
class CodesignReport(VersionedEntity):
    """Result of co-optimizing robot hardware against real task performance.

    The optimizer searches hardware parameters (link lengths, actuator torque),
    rebuilds the real robot and MJCF for each candidate, evaluates pick-and-place
    in real physics, and keeps the design with the best task success. This is the
    hardware half of the physical-AI co-design loop (policy/scene are the others).
    """

    task_prompt: str = ""
    objective: str = "maximize_pick_place_success"
    eval_scene_count: int = 0
    iterations: int = 0
    candidates_evaluated: int = 0
    baseline: DesignCandidate | None = None
    best: DesignCandidate | None = None
    success_improvement: float = 0.0
    search_space: dict = field(default_factory=dict)
    history: list[DesignCandidate] = field(default_factory=list)
    changed_parameters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.best, "best", "Co-design report must include a best design.")
        require_non_empty(result, self.baseline, "baseline", "Co-design report must include a baseline design.")
        require_non_empty(result, self.history, "history", "Co-design report must include the search history.")
        if self.best and (self.best.success_rate < 0 or self.best.success_rate > 1):
            result.add("invalid_success_rate", "Best success rate must be between 0 and 1.", "best")
        if self.baseline and self.best and self.best.success_rate < self.baseline.success_rate:
            result.add("codesign_regressed", "Best design is worse than the baseline.", "best")
        return result
