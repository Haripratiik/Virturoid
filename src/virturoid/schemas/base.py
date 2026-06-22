from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class Visibility(str, Enum):
    PRIVATE = "private"
    TEAM = "team"
    ENTERPRISE = "enterprise"
    COMMUNITY = "community"
    PUBLIC = "public"


class TrainingPermission(str, Enum):
    NO_TRAINING = "no_training"
    ABSTRACTED_LEARNING_ONLY = "abstracted_learning_only"
    FULL_OPT_IN = "full_opt_in"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    field: str | None = None
    severity: str = "error"


@dataclass
class ValidationResult:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def add(self, code: str, message: str, field: str | None = None, severity: str = "error") -> None:
        self.issues.append(ValidationIssue(code=code, message=message, field=field, severity=severity))

    def extend(self, other: "ValidationResult") -> None:
        self.issues.extend(other.issues)


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    media_type: str | None = None
    sha256: str | None = None
    description: str | None = None


@dataclass
class Ownership:
    owner: str
    visibility: Visibility = Visibility.PRIVATE
    training_permission: TrainingPermission = TrainingPermission.NO_TRAINING


@dataclass
class VersionedEntity:
    id: str
    version: str = "0.1.0"

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if not self.id:
            result.add("missing_id", "Entity id is required.", "id")
        if not self.version:
            result.add("missing_version", "Entity version is required.", "version")
        return result

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


def require_non_empty(result: ValidationResult, value: Any, field_name: str, message: str) -> None:
    if value is None or value == "" or value == [] or value == {}:
        result.add("required", message, field_name)


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    return value

