"""Step 3 — the gated EvaluationRun + export gate (plan §15.3): "physics ran" must NOT equal "task passed".

A robot that runs REAL physics but fails the task (baseline/randomized below threshold) must be blocked from
export. The gated coordinator (`evaluation_loop.gated_evaluation_run`) writes an explicit pass/fail to
`reports/evaluation_run.json`, and the Product Readiness Ledger's `physics_evaluated` stage reads it:
export_ready True -> ATTAINED, False -> BELOW_GATE (a bad status -> not safe_to_export). See
[[task-effectiveness-loop]], [[product-truth-ledger]]."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.readiness_ledger import (
    ATTAINED, BELOW_GATE, ProductReadinessLedger, StageRecord,
)
from virturoid.services.readiness_ledger import build_product_readiness_ledger

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _write_eval_run(pkg: Path, *, export_ready: bool, blockers=None, tier_success=None):
    rep = pkg / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "evaluation_run.json").write_text(json.dumps({
        "schema": "evaluation_run/v1", "robot": "g", "task": "sort",
        "export_ready": export_ready, "export_blockers": blockers or [],
        "export_threshold": 0.8, "gate_tiers": ["baseline", "randomized"],
        "tier_success": tier_success or {"baseline": 1.0, "randomized": 1.0},
        "regression_cause_confirmed_rate": 1.0,
    }, indent=2), encoding="utf-8")


class ExportGateProbeTests(unittest.TestCase):
    def test_passing_eval_run_attains_physics_stage(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            _write_eval_run(pkg, export_ready=True, tier_success={"baseline": 1.0, "randomized": 0.83})
            led = build_product_readiness_ledger(pkg, robot_class="manipulator", gene=None)
            self.assertEqual(led.by_stage["physics_evaluated"].status, ATTAINED)

    def test_failing_eval_run_blocks_export(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            _write_eval_run(pkg, export_ready=False, blockers=["baseline success 0.25 < 0.8"],
                            tier_success={"baseline": 0.25, "randomized": 0.5})
            led = build_product_readiness_ledger(pkg, robot_class="manipulator", gene=None)
            rec = led.by_stage["physics_evaluated"]
            self.assertEqual(rec.status, BELOW_GATE)              # real physics, but the task FAILED
            self.assertFalse(rec.real)
            self.assertIn("0.25", rec.detail)

    def test_below_gate_forces_not_safe_to_export(self):
        # schema-level: every other required stage real, but a below-gate physics stage must veto export.
        stages = [StageRecord("schema_valid", ATTAINED), StageRecord("sim_compiled", ATTAINED),
                  StageRecord("real_cad_exported", ATTAINED),
                  StageRecord("physics_evaluated", BELOW_GATE, "task success below gate")]
        led = ProductReadinessLedger(package_dir="x", robot_class="manipulator", stages=stages,
                                     required=["schema_valid", "sim_compiled", "real_cad_exported",
                                               "physics_evaluated"])
        self.assertFalse(led.safe_to_export)
        self.assertTrue(any("physics_evaluated" in i and "below_gate" in i for i in led.validate()))


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class GatedEvaluationRunTests(unittest.TestCase):
    def test_coordinator_runs_and_persists_decision(self):
        from virturoid.services.evaluation_loop import gated_evaluation_run
        from virturoid.services.morphology_composer import compose_robot
        arm = compose_robot("a tabletop robot arm", llm=None)
        with tempfile.TemporaryDirectory() as d:
            out = gated_evaluation_run(arm, seed=0, package_dir=d,
                                       tier_counts={"smoke": 1, "baseline": 2, "randomized": 2, "stress": 2})
            dec = out["decision"]
            self.assertIn("export_ready", dec)
            self.assertIn("baseline", dec["tier_success"])
            # the gate decision is persisted where the ledger reads it
            f = Path(d) / "reports" / "evaluation_run.json"
            self.assertTrue(f.exists())
            data = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(data["export_ready"], dec["export_ready"])
            # and the ledger's physics stage reflects the SAME decision (attained iff export_ready)
            led = build_product_readiness_ledger(d, robot_class="manipulator", gene=None)
            self.assertEqual(led.by_stage["physics_evaluated"].status,
                             ATTAINED if dec["export_ready"] else BELOW_GATE)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class BuildGenePackageGateWiringTests(unittest.TestCase):
    """build_gene_package wires the §15.3 gate on the REAL gene (no genome->gene reconstruction). It is OPT-IN
    (gated_export=False default), so a fast iterative build is unchanged and only an explicit export pass runs
    the heavier tiered loop — and when it does, the ledger ENFORCES the persisted decision."""

    def _arm(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a tabletop robot arm that sorts blocks", llm=None)

    def test_opt_in_default_off_writes_no_eval_run(self):
        from virturoid.services.gene_build import build_gene_package
        with tempfile.TemporaryDirectory() as d:
            build_gene_package(self._arm(), "sort blocks", Path(d), scene_count=2)   # gated_export defaults False
            self.assertFalse((Path(d) / "reports" / "evaluation_run.json").exists())

    def test_gated_export_runs_gate_on_real_gene_and_ledger_enforces_it(self):
        from virturoid.services.gene_build import build_gene_package
        with tempfile.TemporaryDirectory() as d:
            summary = build_gene_package(self._arm(), "sort blocks", Path(d), scene_count=2, gated_export=True,
                                         export_tier_counts={"smoke": 1, "baseline": 2, "randomized": 2, "stress": 1})
            f = Path(d) / "reports" / "evaluation_run.json"
            self.assertTrue(f.exists(), "gated_export must persist the §15.3 decision the ledger reads")
            self.assertIn("export_gate", summary)
            dec = json.loads(f.read_text(encoding="utf-8"))
            led = build_product_readiness_ledger(d, robot_class="manipulator", gene=None)
            # the ledger's physics stage reflects the SAME real decision (no fabricated genome->gene robot)
            self.assertEqual(led.by_stage["physics_evaluated"].status,
                             ATTAINED if dec["export_ready"] else BELOW_GATE)


if __name__ == "__main__":
    unittest.main()
