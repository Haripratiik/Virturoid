"""Product Readiness Ledger — the honest truth-gate the product audit asked for.

Every package producer emits ONE ``reports/product_readiness_ledger.json`` that states, per stage, what is
PROVABLY real vs scaffolded/placeholder/dry-run. Unlike the old soft scorecard (which only failed when a gate
said 'fail'), this is designed so ``safe_to_export`` can be True ONLY when every REQUIRED stage attained its
real status — a fake-but-present artifact (a 479-byte placeholder STEP, a CV manifest with no pixels, a
dry-run "evaluation") is a FAILURE, not a pass. This schema is the data; ``services/readiness_ledger.py`` runs
the probes and ``validate()`` is the enforcement hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The ordered readiness ladder. A build climbs as far as it can prove.
LEDGER_STAGES = (
    "schema_valid",          # core artifacts exist + parse + cross-refs resolve
    "sim_compiled",          # scenes compiled to loadable MJCF
    "real_cad_exported",     # cad/**/*.step are real B-rep (is_real_cad), not placeholders
    "rendered_cv_data",      # real rgb/depth/segmentation/point-cloud files on disk, not just manifests
    "physics_evaluated",     # a real (non-dry-run) physics evaluation + a passed spawn-penetration check
    "controller_exported",   # a trained/exported controller bundle (required only when training requested)
)

# Per-stage status vocabulary — the honest words.
ATTAINED = "attained"            # proven real
SCAFFOLDED = "scaffolded"        # nothing written for this stage yet
PLACEHOLDER = "placeholder"      # a fake/stub artifact is present (the dishonest case)
DRY_RUN_ONLY = "dry_run_only"    # only a dry-run/synthetic result, not real physics
COLLISION = "collision"          # body penetrates the floor at spawn
BELOW_GATE = "below_gate"        # REAL physics ran, but task success is below the §15.3 export gate
NOT_RUN = "not_run"              # the stage did not run
NOT_REQUIRED = "not_required"    # this stage does not apply to this build (and isn't gating)

_BAD_STATUSES = {SCAFFOLDED, PLACEHOLDER, DRY_RUN_ONLY, COLLISION, BELOW_GATE, NOT_RUN}


@dataclass
class StageRecord:
    stage: str
    status: str
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def real(self) -> bool:
        return self.status in (ATTAINED, NOT_REQUIRED)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "status": self.status, "detail": self.detail, "evidence": self.evidence}


@dataclass
class ProductReadinessLedger:
    package_dir: str
    robot_class: str
    stages: list[StageRecord]
    required: list[str]
    enforce: bool = False        # when True, validate() raising blocks the build (hard-fail teeth)

    @property
    def by_stage(self) -> dict:
        return {s.stage: s for s in self.stages}

    @property
    def highest_attained(self) -> str:
        """The furthest LADDER stage reached with a REAL status (stages climb in LEDGER_STAGES order)."""
        reached = "none"
        bs = self.by_stage
        for stage in LEDGER_STAGES:
            rec = bs.get(stage)
            if rec is not None and rec.real:
                reached = stage
            elif rec is not None and stage in self.required and not rec.real:
                break          # a required stage failed -> can't have honestly climbed past it
        return reached

    @property
    def safe_to_export(self) -> bool:
        """True ONLY when every REQUIRED stage attained a real status. A placeholder/dry-run/collision in any
        required stage forces this False — the product can't look complete over a fake stage."""
        bs = self.by_stage
        return all(bs.get(stage) is not None and bs[stage].real for stage in self.required)

    def validate(self) -> list[str]:
        """Return honesty violations (empty = consistent). A violation = a required stage is provably fake
        while the build would otherwise be presented as ready. ``write_product_readiness_ledger`` raises on
        these when ``enforce`` is set (the audit's hard-fail chokepoint)."""
        issues: list[str] = []
        bs = self.by_stage
        for stage in self.required:
            rec = bs.get(stage)
            if rec is None:
                issues.append(f"required stage '{stage}' was never probed")
            elif not rec.real:
                issues.append(f"required stage '{stage}' is '{rec.status}' (not real): {rec.detail}")
        return issues

    def to_dict(self) -> dict:
        return {
            "package_dir": self.package_dir,
            "robot_class": self.robot_class,
            "safe_to_export": self.safe_to_export,
            "highest_attained": self.highest_attained,
            "required": list(self.required),
            "enforced": self.enforce,
            "issues": self.validate(),
            "stages": [s.to_dict() for s in self.stages],
            "disclaimer": "Honest readiness screening: a stage is 'attained' only with a provably-real "
                          "artifact (real B-rep CAD, rendered pixels, non-dry-run physics). NOT a guarantee "
                          "of real-world transfer.",
        }
