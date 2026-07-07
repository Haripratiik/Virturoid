"""Perception / observation schemas — Phase 0 of the Training Improvement Plan.

The plan's core rule is that training quality depends on *what the robot can actually observe* and whether the
trainer secretly cheats with privileged simulator state. So before any perception code, it asks for typed,
backend-agnostic records: an :class:`ObservationContract` that pins the policy's real observation keys vs the
labels it may only use for supervision, a :class:`SensorNoiseProfile` for realistic sensor degradation, a
:class:`PerceptionTrainingPlan` for the ablation ladder, and a :class:`SensorFailureRecord` taxonomy so
perception failures are first-class. Each declares a *perception rung* (privileged state ... real sensors).

These live in ``schemas/`` and depend only on the standard library (AGENTS.md: no backend/simulator code here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


class PerceptionRung(str, Enum):
    """The perception fidelity ladder. Rung 0 (privileged state) is for debugging/teacher/upper-bound only."""

    RUNG_0_PRIVILEGED = "rung_0_privileged_state"
    RUNG_1_SYNTHETIC_ADAPTER = "rung_1_synthetic_adapter"
    RUNG_2_NATIVE_SENSORS = "rung_2_native_sensors"
    RUNG_3_SYNTHETIC_DATASET = "rung_3_synthetic_dataset"
    RUNG_4_PERCEPTION_MODEL = "rung_4_perception_model"
    RUNG_5_HIFI_ADAPTER = "rung_5_hifi_adapter"
    RUNG_6_REAL_SENSOR = "rung_6_real_sensor"


# Modality vocabularies used by the leakage gate and rung selector.
CAMERA_MODALITIES = frozenset({"rgb", "rgbd", "depth", "segmentation", "point_cloud"})
LOWDIM_MODALITIES = frozenset({"range", "contact", "joint_state", "proprioception", "imu", "force_torque", "wrench"})

# Perception-specific failure taxonomy from the plan (§Sensor Failure Taxonomy).
SENSOR_FAILURE_TYPES = frozenset({
    "missed_detection", "false_positive_detection", "bad_depth_at_grasp", "occlusion_failure",
    "out_of_fov_failure", "latency_instability", "segmentation_leak", "pose_estimate_drift",
    "sensor_saturation", "contact_false_positive", "proprioception_bias_failure", "privileged_state_leakage",
})


@dataclass
class SensorNoiseProfile(VersionedEntity):
    """Realistic sensor degradation applied at training time so a policy cannot assume perfect observations."""

    modalities: list[str] = field(default_factory=list)
    gaussian_sigma: dict[str, float] = field(default_factory=dict)
    bias: dict[str, float] = field(default_factory=dict)
    quantization: dict[str, float] = field(default_factory=dict)
    dropout_probability: float = 0.0
    hole_mask_model: str | None = None
    min_range_m: float | None = None
    max_range_m: float | None = None
    lens_distortion: dict | None = None
    rolling_shutter: dict | None = None
    motion_blur: dict | None = None
    seed: int = 0

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.modalities, "modalities", "Noise profile must target at least one modality.")
        if not 0.0 <= float(self.dropout_probability) <= 1.0:
            result.add("dropout_range", "Dropout probability must be in [0, 1].", "dropout_probability")
        if (self.min_range_m is not None and self.max_range_m is not None
                and self.min_range_m > self.max_range_m):
            result.add("range_inverted", "min_range_m cannot exceed max_range_m.", "min_range_m")
        return result


@dataclass
class ObservationContract(VersionedEntity):
    """The contract that separates what a policy may *observe* from what it may only use as a *label*.

    ``policy_observation_keys`` are the tensors the deployed policy will actually receive; ``privileged_label_keys``
    are simulator-truth signals allowed only for teachers/detectors/supervision. ``deploy_modalities`` is what the
    real robot will have at deployment — the leakage gate uses it to reject e.g. segmentation-in-policy when the
    robot ships RGB-only, or a stated held-out seed set that overlaps the training seeds.
    """

    task_graph_id: str = ""
    scene_set_id: str = ""
    robot_genome_id: str = ""
    policy_observation_keys: list[str] = field(default_factory=list)
    privileged_label_keys: list[str] = field(default_factory=list)
    required_modalities: list[str] = field(default_factory=list)
    deploy_modalities: list[str] = field(default_factory=list)
    sensor_specs: list[str] = field(default_factory=list)
    synchronization_policy: str = "nearest"
    leakage_policy: str = "strict"          # strict | permissive
    perception_rung: PerceptionRung = PerceptionRung.RUNG_0_PRIVILEGED
    train_scene_seeds: list[int] = field(default_factory=list)
    heldout_scene_seeds: list[int] = field(default_factory=list)
    randomization_logged: bool = False

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.task_graph_id, "task_graph_id", "Observation contract must reference a task.")
        require_non_empty(result, self.scene_set_id, "scene_set_id", "Observation contract must reference a scene set.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Observation contract must reference a robot.")
        require_non_empty(result, self.policy_observation_keys, "policy_observation_keys",
                          "A policy must declare at least one observation key.")
        if self.leakage_policy not in {"strict", "permissive"}:
            result.add("invalid_leakage_policy", "leakage_policy must be 'strict' or 'permissive'.", "leakage_policy")
        return result


@dataclass
class PerceptionTrainingPlan(VersionedEntity):
    """The perception ablation ladder for one observation contract (privileged -> sensor-only, etc.)."""

    observation_contract_ref: str = ""
    dataset_refs: list[str] = field(default_factory=list)
    perception_model_refs: list[str] = field(default_factory=list)
    teacher_label_sources: list[str] = field(default_factory=list)
    ablation_matrix: list[dict] = field(default_factory=list)
    evaluation_protocol_ref: str | None = None

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.observation_contract_ref, "observation_contract_ref",
                          "Perception training plan must reference an observation contract.")
        return result


@dataclass
class SensorFailureRecord(VersionedEntity):
    """A first-class perception failure: what broke, in which sensor/scene/episode, and how to repair it."""

    failure_type: str = ""
    sensor_ref: str = ""
    scene_ref: str | None = None
    episode_ref: str | None = None
    sampled_noise: dict = field(default_factory=dict)
    observed_symptoms: list[str] = field(default_factory=list)
    downstream_task_effect: dict = field(default_factory=dict)
    proposed_repairs: list[str] = field(default_factory=list)
    root_cause: str | None = None  # physics | controller | perception | latency | distribution_shift | reward

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.failure_type, "failure_type", "Sensor failure needs a type.")
        require_non_empty(result, self.sensor_ref, "sensor_ref", "Sensor failure must reference a sensor.")
        if self.failure_type and self.failure_type not in SENSOR_FAILURE_TYPES:
            result.add("unknown_failure_type",
                       f"'{self.failure_type}' is not in the known sensor-failure taxonomy.",
                       "failure_type", severity="warning")
        return result
