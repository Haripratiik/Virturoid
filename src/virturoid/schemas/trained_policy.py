from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class ActionDimension:
    name: str
    lower: float
    upper: float
    velocity_limit: float | None = None
    effort_limit: float | None = None


@dataclass
class PolicyTrainingMetrics:
    method: str = ""
    iterations: int = 0
    candidates_per_iteration: int = 0
    training_scene_count: int = 0
    reward_history: list[float] = field(default_factory=list)
    best_reward: float = 0.0
    initial_reward: float = 0.0


@dataclass
class PolicyEvalMetrics:
    eval_scene_count: int = 0
    mean_reach_distance_m: float = 0.0
    best_reach_distance_m: float = 0.0
    success_rate: float = 0.0
    stable_rate: float = 0.0
    success_threshold_m: float = 0.0


@dataclass
class TrainedPolicy(VersionedEntity):
    """A controller trained against real MuJoCo rollouts.

    The policy is a linear map from the observed target-block position to joint
    position targets, tracked by a PD low-level controller. Parameters, the
    observation/action spaces, control rate, and safety clamps are all stored so
    the controller can run outside the trainer.
    """

    name: str = ""
    policy_type: str = "learned_linear_reach"
    robot_genome_id: str = ""
    task_graph_id: str = ""
    observation_space: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    action_dimensions: list[ActionDimension] = field(default_factory=list)
    control_frequency_hz: float = 0.0
    low_level_control_frequency_hz: float = 0.0
    weights: list[list[float]] = field(default_factory=list)
    input_features: list[str] = field(default_factory=list)
    safety_clamps: dict = field(default_factory=dict)
    training: PolicyTrainingMetrics = field(default_factory=PolicyTrainingMetrics)
    evaluation: PolicyEvalMetrics = field(default_factory=PolicyEvalMetrics)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.name, "name", "Trained policy must have a name.")
        require_non_empty(result, self.robot_genome_id, "robot_genome_id", "Trained policy must reference a robot.")
        require_non_empty(result, self.joint_names, "joint_names", "Trained policy must list joint names.")
        require_non_empty(result, self.action_dimensions, "action_dimensions", "Trained policy must define an action space.")
        require_non_empty(result, self.weights, "weights", "Trained policy must include learned weights.")
        if self.control_frequency_hz <= 0:
            result.add("invalid_control_frequency", "Control frequency must be positive.", "control_frequency_hz")
        if len(self.weights) != len(self.action_dimensions):
            result.add("weight_action_mismatch", "Weight rows must match action dimensions.", "weights")
        if self.training.reward_history and self.training.best_reward < self.training.initial_reward:
            result.add("training_regressed", "Best reward is below the initial reward.", "training")
        return result
