"""TrainingEvidenceBundle — audit-ready memory retrieval (dossier R4 + "Retrieval Evidence Bundle").

The dossier's sharpest memory recommendation: recall should return EVIDENCE, not a nearest neighbor. A candidate
prior skill should carry its held-out success (not training reward), valid region, failure clusters, reward
provenance, IO compatibility, and negative-transfer history — so the training compiler can rank and REJECT
candidates on measured evidence. This is that record; :mod:`virturoid.services.evidence_retrieval` builds it.
Standard-library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.base import ValidationResult, VersionedEntity, require_non_empty


@dataclass
class TrainingEvidenceBundle(VersionedEntity):
    candidate_skill_id: str = ""
    robot_class: str = ""
    task_type: str = ""
    heldout_success: float | None = None
    valid_region: dict = field(default_factory=dict)
    failure_clusters: list[str] = field(default_factory=list)
    negative_transfer_records: list[str] = field(default_factory=list)
    reward_spec_ref: dict | None = None
    dataset_lineage: list[str] = field(default_factory=list)
    io_compatible: bool = False
    params_path: str | None = None
    permissions: dict = field(default_factory=dict)
    provenance_edge: dict = field(default_factory=dict)
    match_kind: str = "class_task"          # exact_io | class_task | same_class | cross_class
    evidence_score: float = 0.0

    def validate(self) -> ValidationResult:
        result = super().validate()
        require_non_empty(result, self.candidate_skill_id, "candidate_skill_id",
                          "An evidence bundle must name the candidate skill it describes.")
        if self.heldout_success is not None and not 0.0 <= float(self.heldout_success) <= 1.0:
            result.add("bad_success", "heldout_success must be in [0, 1].", "heldout_success")
        return result
