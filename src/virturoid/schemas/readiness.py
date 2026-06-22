from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ReadinessGate:
    """One product-readiness check for a generated package."""

    key: str
    label: str
    status: str
    required: bool = True
    detail: str = ""
    evidence_uri: str | None = None


@dataclass
class MvpReadinessReport(VersionedEntity):
    """Audit report for the current autonomous robot-build MVP goal."""

    package_dir: str = ""
    robot_class: str = ""
    package_type: str = ""
    goal: str = ""
    ready: bool = False
    score: int = 0
    gates: list[ReadinessGate] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.package_dir, "package_dir", "Package directory is required.")
        require_non_empty(result, self.robot_class, "robot_class", "Robot class is required.")
        require_non_empty(result, self.package_type, "package_type", "Package type is required.")
        require_non_empty(result, self.goal, "goal", "MVP goal is required.")
        require_non_empty(result, self.gates, "gates", "Readiness gates are required.")
        failing_required = [gate.key for gate in self.gates if gate.required and gate.status != "pass"]
        if failing_required:
            result.add(
                "mvp_not_ready",
                f"Required readiness gates are not passing: {', '.join(failing_required)}.",
                "gates",
            )
        return result
