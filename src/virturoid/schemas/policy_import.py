"""Policy / controller import schemas — Phase 4 of the Input Ingestion Research Plan.

The plan flags a real gap: there is no inbound ``PolicyImporter``. A user's controller (an ``.npz`` / ``.onnx`` /
``.pt`` checkpoint, a ROS controller YAML, or a black-box Python script) is only useful once Virturoid can answer
what it observes, what it outputs, in what joint order, at what rate, with what units and safety limits. These
records capture that contract. Phase-0-style tiers keep it safe: metadata-only (P0) and static parse (P1) never
execute user code; sandbox (P2) and native adapter (P3) come later. Backend-agnostic, standard-library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


class PolicyTier(str, Enum):
    P0_METADATA = "p0_metadata"          # config/README: extract interface assumptions, no execution
    P1_STATIC_PARSE = "p1_static_parse"  # Python/launch/package: AST parse, no execution
    P2_SANDBOX = "p2_sandbox"            # run in a restricted process against simulated observations
    P3_NATIVE_ADAPTER = "p3_native_adapter"  # ONNX/Torch/JAX/LeRobot via a known adapter


class PolicyFramework(str, Enum):
    NUMPY_NPZ = "numpy_npz"
    ONNX = "onnx"
    TORCH = "torch"
    JAX = "jax"
    STABLE_BASELINES = "stable_baselines"
    YAML_CONFIG = "yaml_config"
    PYTHON_SCRIPT = "python_script"
    UNKNOWN = "unknown"


@dataclass
class ControllerInterfaceSpec(VersionedEntity):
    """What a controller expects/produces — the contract that must be known before any policy can run."""

    observation_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    joint_order: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    control_frequency_hz: float | None = None
    normalization: dict = field(default_factory=dict)
    safety_limits: dict = field(default_factory=dict)
    state_estimator: str | None = None

    def validate(self) -> ValidationResult:
        result = super().validate()
        if self.control_frequency_hz is not None and self.control_frequency_hz <= 0:
            result.add("invalid_frequency", "Control frequency must be positive.", "control_frequency_hz")
        return result


@dataclass
class PolicyImportSpec(VersionedEntity):
    """A recognized inbound controller/policy + the safety tier and interface facts extracted so far."""

    source_ref: str = ""
    framework: PolicyFramework = PolicyFramework.UNKNOWN
    tier: PolicyTier = PolicyTier.P0_METADATA
    entrypoint: str | None = None
    dependencies: list[str] = field(default_factory=list)
    ros_topics: list[str] = field(default_factory=list)
    action_dim: int | None = None
    expected_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    interface: ControllerInterfaceSpec | None = None
    validation_status: str = "unvalidated"  # unvalidated | metadata_only | static_parsed | sandbox_passed | rejected
    warnings: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.source_ref, "source_ref", "Policy import must reference a source.")
        if self.action_dim is not None and self.action_dim < 0:
            result.add("invalid_action_dim", "Action dimension cannot be negative.", "action_dim")
        if self.interface is not None:
            for issue in self.interface.validate().issues:
                result.add(issue.code, issue.message, f"interface.{issue.field or ''}", issue.severity)
        return result
