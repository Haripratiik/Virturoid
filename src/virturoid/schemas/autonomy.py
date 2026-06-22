from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class AutonomyDecision:
    iteration: int
    stage: str
    action: str
    detail: str
    success_before: float
    success_after: float


@dataclass
class AutonomyReport(VersionedEntity):
    """Verdict of an autonomous, end-to-end robot build.

    Records the whole closed loop the software ran without human tuning:
    design -> feasibility gates -> real-physics evaluation -> improve hardware
    from real failures -> converge -> finalize and export. ``succeeded`` is the
    honest claim that the software built a robot that performs the task.
    """

    prompt: str = ""
    robot_class: str = ""
    species: str = ""
    # When the requested robot class/species had no implemented builder, these record
    # what the user actually asked for; robot_class/species above are what was built
    # (the nearest buildable relative). Equal to the built values on an exact match.
    requested_robot_class: str = ""
    requested_species: str = ""
    task_type: str = ""
    target_success_rate: float = 0.0
    initial_success_rate: float = 0.0
    final_success_rate: float = 0.0
    feasible: bool = False
    succeeded: bool = False
    iterations: int = 0
    converged_design: dict = field(default_factory=dict)
    decisions: list[AutonomyDecision] = field(default_factory=list)
    exported_artifacts: dict = field(default_factory=dict)
    memory_reused: bool = False
    compute: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.prompt, "prompt", "Autonomy report must record the prompt.")
        require_non_empty(result, self.decisions, "decisions", "Autonomy report must record decisions.")
        for name in ("target_success_rate", "initial_success_rate", "final_success_rate"):
            value = getattr(self, name)
            if value < 0 or value > 1:
                result.add("invalid_rate", f"{name} must be between 0 and 1.", name)
        return result
