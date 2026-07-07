"""RobotModelImportReport — the durable enterprise import report (Input Ingestion plan, Phase 1).

Today ``model_import``/``robot_import`` produce ephemeral dicts. The plan's Phase-1 deliverable is a durable,
typed report that runs BOTH import lanes side by side — the *faithful native* lane (normalized MJCF that runs
as-is for accuracy) and the *inferred RobotGene* lane (editable, lossy, for Virturoid iteration) — groups every
warning by severity and a concrete fix action, and scores import readiness on independent axes. It is written to
``input/import_report.json`` so a robotics-company engineer sees exactly what was found, what is lossy, and what
to fix before training. Backend-agnostic, standard-library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty

_SEVERITIES = ("error", "warning", "info")


@dataclass
class ImportWarning:
    code: str
    message: str
    severity: str = "warning"
    fix_action: str = ""

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "severity": self.severity,
                "fix_action": self.fix_action}


@dataclass
class ImportLane:
    """One import lane's outcome: ``faithful_native`` (accuracy) or ``inferred_robot_gene`` (iteration)."""

    name: str
    ok: bool
    summary: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "summary": self.summary, "detail": self.detail}


@dataclass
class RobotModelImportReport(VersionedEntity):
    source: str = ""
    source_format: str = ""
    faithful_lane: ImportLane | None = None
    gene_lane: ImportLane | None = None
    warnings: list[ImportWarning] = field(default_factory=list)
    body_count: int = 0
    actuated_joint_count: int = 0
    free_base: bool = False
    backend_support: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    first_runnable_sim: str | None = None

    def errors(self) -> list[ImportWarning]:
        return [w for w in self.warnings if w.severity == "error"]

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.source, "source", "An import report must reference a source model.")
        for w in self.warnings:
            if w.severity not in _SEVERITIES:
                result.add("bad_severity", f"'{w.severity}' is not a known severity.", "warnings",
                           severity="warning")
        return result

    def to_dict(self) -> dict:
        data = super().to_dict()
        data["faithful_lane"] = self.faithful_lane.to_dict() if self.faithful_lane else None
        data["gene_lane"] = self.gene_lane.to_dict() if self.gene_lane else None
        data["warnings"] = [w.to_dict() for w in self.warnings]
        return data
