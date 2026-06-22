from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class DesignGate:
    name: str
    status: str  # "pass" | "warn" | "fail" | "skipped"
    detail: str = ""
    measured: dict = field(default_factory=dict)


@dataclass
class DesignValidationReport(VersionedEntity):
    """Physical-feasibility gates evaluated before committing to a design.

    These gates answer "can this robot actually do this task" — reachability,
    actuator torque, joint limits, mass plausibility, scene coverage, and (when
    MuJoCo is available) a collision-free spawn. A failing hard gate means the
    design is impossible as specified and should be revised before training.
    """

    robot_genome_id: str = ""
    task_graph_id: str = ""
    ok: bool = True
    gates: list[DesignGate] = field(default_factory=list)
    blocking_failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Report must reference a robot.")
        require_non_empty(result, self.gates, "gates", "Report must include at least one gate.")
        return result

    def recompute_ok(self) -> "DesignValidationReport":
        self.blocking_failures = [gate.name for gate in self.gates if gate.status == "fail"]
        self.ok = not self.blocking_failures
        return self
