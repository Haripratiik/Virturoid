"""Minimal typed training-tip contract for safe cross-body warm starts.

A tip is not prose and it is not an absolute magic constant. It is a scoped,
validated multiplier that may be applied only after its physics verdict and
morphology predicate match, and it states an effect that later runs can falsify.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty

_STAGES = {"locomotion_search"}
_VERDICTS = {"barely_moves", "fell_over", "upright_but_slow"}
_BUDGET_FIELDS = {"generations", "pop", "steps"}
_EFFECT_METRICS = {"forward_m", "time_to_first_credible_s"}


@dataclass
class TipTrigger:
    stage: str = ""
    verdict: str = ""
    min_tokens: int = 1
    max_tokens: int = 512

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if self.stage not in _STAGES:
            result.add("invalid_tip_stage", f"stage must be one of {sorted(_STAGES)}", "stage")
        if self.verdict not in _VERDICTS:
            result.add("invalid_tip_verdict", f"verdict must be one of {sorted(_VERDICTS)}", "verdict")
        if self.min_tokens < 1 or self.max_tokens < self.min_tokens:
            result.add("invalid_token_region", "token bounds must define a positive ordered range", "min_tokens")
        return result


@dataclass
class ConfigScale:
    field: str = ""
    operation: str = "multiply"
    factor: float = 1.0

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if self.field not in _BUDGET_FIELDS:
            result.add("invalid_tip_field", f"field must be one of {sorted(_BUDGET_FIELDS)}", "field")
        if self.operation != "multiply":
            result.add("invalid_tip_operation", "typed tips may scale but never replace an absolute value", "operation")
        if not 1.0 < float(self.factor) <= 3.0:
            result.add("invalid_tip_factor", "factor must be in (1, 3]", "factor")
        return result


@dataclass
class ExpectedEffect:
    metric: str = "forward_m"
    direction: str = "increase"
    minimum_delta: float = 0.01
    observed_delta: float | None = None

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        if self.metric not in _EFFECT_METRICS:
            result.add("invalid_effect_metric", f"metric must be one of {sorted(_EFFECT_METRICS)}", "metric")
        if self.direction not in {"increase", "decrease"}:
            result.add("invalid_effect_direction", "direction must be increase or decrease", "direction")
        if self.minimum_delta <= 0:
            result.add("invalid_effect_delta", "minimum_delta must be positive and falsifiable", "minimum_delta")
        return result


@dataclass
class TrainingTip(VersionedEntity):
    trigger: TipTrigger = field(default_factory=TipTrigger)
    deltas: list[ConfigScale] = field(default_factory=list)
    expected_effect: ExpectedEffect = field(default_factory=ExpectedEffect)
    source_gene: str = ""

    def validate(self) -> ValidationResult:
        result = super().validate()
        result.extend(self.trigger.validate())
        result.extend(self.expected_effect.validate())
        require_non_empty(result, self.deltas, "deltas", "A tip must contain at least one typed config delta.")
        require_non_empty(result, self.source_gene, "source_gene", "A tip must cite the gene that produced it.")
        seen: set[str] = set()
        for delta in self.deltas:
            result.extend(delta.validate())
            if delta.field in seen:
                result.add("duplicate_tip_field", "Each config field may be scaled once per tip.", "deltas")
            seen.add(delta.field)
        return result

    def applies(self, *, stage: str, verdict: str, n_tokens: int) -> bool:
        return (
            self.validate().ok
            and stage == self.trigger.stage
            and verdict == self.trigger.verdict
            and self.trigger.min_tokens <= int(n_tokens) <= self.trigger.max_tokens
        )

    def apply(self, config: dict[str, int]) -> dict[str, int]:
        if not self.validate().ok:
            raise ValueError("Cannot apply an invalid training tip")
        updated = dict(config)
        for delta in self.deltas:
            updated[delta.field] = max(1, int(round(int(updated[delta.field]) * delta.factor)))
        return updated

    @classmethod
    def from_dict(cls, value: dict) -> "TrainingTip":
        return cls(
            id=str(value.get("id", "")),
            version=str(value.get("version", "0.1.0")),
            trigger=TipTrigger(**dict(value.get("trigger") or {})),
            deltas=[ConfigScale(**dict(item)) for item in value.get("deltas", [])],
            expected_effect=ExpectedEffect(**dict(value.get("expected_effect") or {})),
            source_gene=str(value.get("source_gene", "")),
        )


def tip_from_budget_change(*, verdict: str, before: dict[str, int], after: dict[str, int],
                           n_tokens: int, source_gene: str) -> TrainingTip:
    """Create the smallest scaling-law tip that exactly describes an observed escalation."""
    deltas = [
        ConfigScale(field=key, factor=round(float(after[key]) / max(1.0, float(before[key])), 4))
        for key in sorted(_BUDGET_FIELDS)
        if int(after[key]) > int(before[key])
    ]
    return TrainingTip(
        id=f"locomotion-{verdict}-{source_gene}",
        trigger=TipTrigger(stage="locomotion_search", verdict=verdict,
                           min_tokens=max(1, int(n_tokens * 0.7)), max_tokens=max(1, int(n_tokens * 1.3))),
        deltas=deltas,
        expected_effect=ExpectedEffect(),
        source_gene=source_gene,
    )
