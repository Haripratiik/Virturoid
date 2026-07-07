"""Input Compiler schemas — Phase 0 of the Input Ingestion Research Plan.

The plan's thesis is that every user input (vague prompt, constrained prompt, form, file, folder, zip,
ROS bag, CAD, policy) should compile into typed, validated artifacts with *field-level provenance*: the
system must be able to say "the user explicitly supplied this" vs "I inferred this" vs "a default filled
a gap", with a confidence and any conflicts. These records are the durable, backend-agnostic substrate for
that. They live in ``schemas/`` and depend only on the standard library (AGENTS.md: no backend code in
schemas; explicit dataclasses with validation).

Phase 0 (this file) covers the prompt path: ``InputBundle`` / ``InputArtifact`` describe an intake event and
its recognized items, ``InputEvidence`` is the field-level source trace, and ``InputInterpretation`` is the
unified, provenance-rich interpretation of a build request that is written to ``input/interpretation.json``.
Later phases reuse the same records for enterprise folder/ROS/policy/log/CAD/BOM ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from virturoid.schemas.base import (
    ValidationResult,
    VersionedEntity,
    require_non_empty,
)


class InputSourceType(str, Enum):
    """Where a normalized field's value came from — the provenance axis the plan requires.

    Ordered loosely from most to least authoritative for conflict resolution.
    """

    EXPLICIT = "explicit"            # user typed / passed it directly (e.g. --payload 3)
    USER_CONFIRMED = "user_confirmed"  # user reviewed and accepted an inferred/defaulted value
    CALIBRATED = "calibrated"        # measured from logs / system identification
    PARSED = "parsed"                # deterministically extracted from the prompt/file (regex/format parser)
    INFERRED = "inferred"            # derived by heuristic from other signals (may carry conflicts)
    DEFAULTED = "defaulted"          # a system default filled a gap the user never addressed


class ParseStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    UNRECOGNIZED = "unrecognized"
    IGNORED = "ignored"


@dataclass
class InputEvidence:
    """Field-level source trace: one interpreted field + how/where/how-confidently it was obtained.

    This is what lets a downstream compliance report point back to input ("payload honored — value was
    parsed from the prompt at confidence 0.9") instead of silently pretending every value was requested.
    """

    field_path: str
    value: Any = None
    source_type: InputSourceType = InputSourceType.INFERRED
    unit: str | None = None
    source_artifact: str | None = None
    confidence: float = 0.5
    conflicts: list[str] = field(default_factory=list)
    note: str | None = None

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        require_non_empty(result, self.field_path, "field_path", "Evidence must name the field it traces.")
        try:
            conf = float(self.confidence)
        except (TypeError, ValueError):
            result.add("confidence_type", "Confidence must be numeric.", "confidence")
            return result
        if not 0.0 <= conf <= 1.0:
            result.add("confidence_range", "Confidence must be in [0, 1].", "confidence")
        return result


@dataclass
class IntakeQuestion:
    """A missing or low-confidence field that would improve the build/simulation if resolved.

    The plan's rule is "ask fewer questions up front, better questions after reading" — so a question should
    only be raised when it is genuinely high value (a conflict, or a blocking gap), never as a generic form.
    """

    id: str
    field_path: str
    question: str
    reason: str = ""
    blocks_simulation: bool = False
    blocks_training: bool = False
    candidates: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        require_non_empty(result, self.id, "id", "Intake question needs an id.")
        require_non_empty(result, self.question, "question", "Intake question text is required.")
        return result


@dataclass
class InputArtifact(VersionedEntity):
    """One recognized item inside an intake event (a file, a prompt, a form, a policy, a log)."""

    artifact_type: str = ""          # prompt | urdf | mjcf | mesh | bom | policy | log | ros_package | ...
    media_type: str | None = None
    checksum: str | None = None
    size_bytes: int | None = None
    parser: str | None = None
    parse_status: ParseStatus = ParseStatus.OK
    extracted_refs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.artifact_type, "artifact_type", "Artifact must declare its type.")
        if self.size_bytes is not None and self.size_bytes < 0:
            result.add("invalid_size", "Artifact size cannot be negative.", "size_bytes")
        return result


@dataclass
class InputBundle(VersionedEntity):
    """One intake event: the top-level record grouping every artifact a user supplied at once."""

    source_path: str = ""
    upload_id: str | None = None
    workspace_id: str | None = None
    created_at: str | None = None
    root_folder: str | None = None
    artifacts: list[InputArtifact] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.source_path, "source_path", "Input bundle needs a source path.")
        for index, artifact in enumerate(self.artifacts):
            issues = artifact.validate()
            for issue in issues.issues:
                result.add(issue.code, issue.message, f"artifacts[{index}].{issue.field or ''}", issue.severity)
        return result


@dataclass
class InputInterpretation(VersionedEntity):
    """The unified, provenance-rich interpretation of a build request.

    This is the Phase 0 deliverable written to ``input/interpretation.json`` for every prompt build. It merges
    the previously-split prompt parsers (``requirements_builder`` + ``spec_parser``) into one record where each
    interpreted field carries its own :class:`InputEvidence`, so a user (or an agent) can inspect and edit the
    interpretation *before* the build runs, and a compliance report can trace every honored constraint back to
    what the user actually asked for.
    """

    prompt: str = ""
    requirement_id: str | None = None
    bundle_id: str | None = None
    evidence: list[InputEvidence] = field(default_factory=list)
    intake_questions: list[IntakeQuestion] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------------
    def field(self, field_path: str) -> InputEvidence | None:
        for item in self.evidence:
            if item.field_path == field_path:
                return item
        return None

    def confidence_map(self) -> dict[str, float]:
        return {item.field_path: float(item.confidence) for item in self.evidence}

    def by_source(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for item in self.evidence:
            grouped.setdefault(item.source_type.value, []).append(item.field_path)
        return grouped

    def has_conflicts(self) -> bool:
        return any(item.conflicts for item in self.evidence)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.prompt, "prompt", "Interpretation must retain the source prompt.")
        require_non_empty(result, self.evidence, "evidence", "Interpretation must carry field-level evidence.")
        seen: set[str] = set()
        for index, item in enumerate(self.evidence):
            for issue in item.validate().issues:
                result.add(issue.code, issue.message, f"evidence[{index}].{issue.field or ''}", issue.severity)
            if item.field_path in seen:
                result.add("duplicate_evidence", f"Duplicate evidence for '{item.field_path}'.",
                           f"evidence[{index}].field_path")
            seen.add(item.field_path)
        for index, question in enumerate(self.intake_questions):
            for issue in question.validate().issues:
                result.add(issue.code, issue.message, f"intake_questions[{index}].{issue.field or ''}", issue.severity)
        return result
