"""DataDividendRecord — the data-flywheel ledger required by BOTH training plans.

"Software gets better as people use it" is only real if every run records *which reusable prior it improved*
and by *how much*. The Training Improvement plan and the Training Pipeline dossier both specify the same record:
run id, the improved prior type/ref, before/after metrics, the measured delta, permission scope, and whether the
prior becomes reusable by default. This is the schema; :mod:`virturoid.services.data_dividend` computes and banks it.

Backend-agnostic, standard-library only (AGENTS.md layering).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty

# The reusable priors a run can improve (dossier "Data Dividend Ledger" + Training Improvement "Data Dividend").
IMPROVED_PRIOR_TYPES = frozenset({
    "body", "skill", "reward", "sensor_model", "perception_dataset", "success_detector",
    "failure_repair", "scene_curriculum", "transfer_rule", "export_readiness_rule", "demonstration_dataset",
})

# Permission scopes control whether a dividend may be reused, and how widely (never leak enterprise data globally).
PERMISSION_SCOPES = frozenset({"no_reuse", "workspace_private", "org_shared", "global_anonymized"})


@dataclass
class DataDividendRecord(VersionedEntity):
    run_id: str = ""
    improved_prior_type: str = ""
    improved_prior_ref: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    before_metrics: dict = field(default_factory=dict)
    after_metrics: dict = field(default_factory=dict)
    measured_delta: dict = field(default_factory=dict)
    key_metric: str | None = None
    permission_scope: str = "workspace_private"
    reusable_by_default: bool = False

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.run_id, "run_id", "A data dividend must reference the run that produced it.")
        require_non_empty(result, self.improved_prior_type, "improved_prior_type",
                          "A data dividend must name the reusable prior it improved.")
        if self.improved_prior_type and self.improved_prior_type not in IMPROVED_PRIOR_TYPES:
            result.add("unknown_prior_type",
                       f"'{self.improved_prior_type}' is not a known reusable-prior type.",
                       "improved_prior_type", severity="warning")
        if self.permission_scope not in PERMISSION_SCOPES:
            result.add("unknown_permission_scope",
                       f"'{self.permission_scope}' is not a known permission scope.",
                       "permission_scope", severity="warning")
        if self.reusable_by_default and self.permission_scope == "no_reuse":
            result.add("reuse_conflicts_permission",
                       "reusable_by_default is True but the permission scope forbids reuse.",
                       "permission_scope")
        return result
