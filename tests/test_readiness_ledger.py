"""Product Readiness Ledger: honest truth-gate. A fake-but-present artifact (placeholder STEP, CV manifest
with no pixels, dry-run eval) must NOT let safe_to_export be True — the audit's #1 ask."""

import json
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.readiness_ledger import ATTAINED, BELOW_GATE, PLACEHOLDER, ProductReadinessLedger, StageRecord
from virturoid.services.cad_geometry import is_real_cad
from virturoid.services.readiness_ledger import build_product_readiness_ledger

_PLACEHOLDER_STEP = ("ISO-10303-21;\nHEADER;\n/* Virturoid MVP placeholder STEP for x */\nENDSEC;\nDATA;\n"
                     "/* model_id=x */\nENDSEC;\nEND-ISO-10303-21;\n")
# A minimally "real"-looking STEP: >2KB, no placeholder marker, many #NNN = entity lines + a B-rep rep.
_REAL_STEP = ("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#2),#3);\n"
              + "".join(f"#{i} = CARTESIAN_POINT('',({i}.0,0.0,0.0));\n" for i in range(2, 120))
              + "ENDSEC;\nEND-ISO-10303-21;\n")


class IsRealCadTests(unittest.TestCase):
    def test_detects_placeholder_vs_real(self):
        with tempfile.TemporaryDirectory() as d:
            ph = Path(d) / "ph.step"; ph.write_text(_PLACEHOLDER_STEP, encoding="utf-8")
            real = Path(d) / "real.step"; real.write_text(_REAL_STEP, encoding="utf-8")
            self.assertFalse(is_real_cad(ph))               # 479-byte comment-only fake -> rejected
            self.assertTrue(is_real_cad(real))              # real B-rep with entity instances -> accepted
            self.assertFalse(is_real_cad(Path(d) / "missing.step"))


class BomPropulsionGateTests(unittest.TestCase):
    """The bom_present gate must recognize EVERY morphology's propulsion, not just joint actuators: a drone is
    propelled by ROTORS and a rover by DRIVE MOTORS — both are complete, buildable BOMs, not 'below gate'."""

    def _probe(self, category):
        from virturoid.services.readiness_ledger import _probe_bom
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            (pkg / "robot").mkdir()
            (pkg / "robot" / "bill_of_materials.json").write_text(json.dumps(
                {"lines": [{"part": "frame", "category": "material"},
                           {"part": "prop", "category": category, "qty": 4}]}), encoding="utf-8")
            return _probe_bom(pkg)

    def test_rotor_propulsion_attains_the_gate(self):
        self.assertEqual(self._probe("rotor").status, ATTAINED, "a drone (rotors) has a real, buildable BOM")

    def test_drive_motor_propulsion_attains_the_gate(self):
        self.assertEqual(self._probe("drive_motor").status, ATTAINED, "a rover (drive motors) has a real BOM")

    def test_joint_actuator_still_attains_the_gate(self):
        self.assertEqual(self._probe("actuator").status, ATTAINED)

    def test_no_propulsion_still_below_gate(self):
        self.assertEqual(self._probe("camera").status, BELOW_GATE, "a body with nothing that moves it is not real")


class ReadinessLedgerTests(unittest.TestCase):
    def test_placeholder_cad_blocks_safe_to_export(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            (pkg / "cad" / "exact").mkdir(parents=True)
            (pkg / "cad" / "exact" / "robot_assembly.step").write_text(_PLACEHOLDER_STEP, encoding="utf-8")
            (pkg / "robot").mkdir()
            (pkg / "robot" / "robot_genome.json").write_text(json.dumps({"id": "g"}), encoding="utf-8")
            ledger = build_product_readiness_ledger(pkg, robot_class="manipulator")
            self.assertFalse(ledger.safe_to_export)         # a placeholder STEP can't read as ready
            self.assertEqual(ledger.by_stage["real_cad_exported"].status, PLACEHOLDER)

    def test_real_cad_attains_that_stage(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            (pkg / "cad").mkdir()
            (pkg / "cad" / "robot_assembly.step").write_text(_REAL_STEP, encoding="utf-8")
            ledger = build_product_readiness_ledger(pkg, robot_class="manipulator")
            self.assertEqual(ledger.by_stage["real_cad_exported"].status, ATTAINED)

    def test_safe_to_export_requires_all_required_stages(self):
        # Pure schema-level check of the honesty logic: one fake required stage -> not safe + validate() flags.
        stages = [StageRecord("schema_valid", ATTAINED), StageRecord("sim_compiled", ATTAINED),
                  StageRecord("real_cad_exported", PLACEHOLDER, "479-byte fake"),
                  StageRecord("physics_evaluated", ATTAINED)]
        led = ProductReadinessLedger(package_dir="x", robot_class="manipulator", stages=stages,
                                     required=["schema_valid", "sim_compiled", "real_cad_exported",
                                               "physics_evaluated"])
        self.assertFalse(led.safe_to_export)
        self.assertTrue(any("real_cad_exported" in i for i in led.validate()))

    def test_navigation_report_gates_physics_stage(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d)
            (pkg / "reports").mkdir()
            (pkg / "robot").mkdir()
            (pkg / "robot" / "robot_genome.json").write_text(json.dumps({"id": "g"}), encoding="utf-8")
            (pkg / "reports" / "navigation_evaluation_report.json").write_text(
                json.dumps({"success_rate": 0.333, "reached": 2, "total_episodes": 6}),
                encoding="utf-8",
            )

            ledger = build_product_readiness_ledger(pkg, robot_class="mobile_base", require=["physics_evaluated"])
            self.assertEqual(BELOW_GATE, ledger.by_stage["physics_evaluated"].status)
            self.assertFalse(ledger.safe_to_export)

            (pkg / "reports" / "navigation_evaluation_report.json").write_text(
                json.dumps({"success_rate": 1.0, "reached": 6, "total_episodes": 6}),
                encoding="utf-8",
            )
            ledger = build_product_readiness_ledger(pkg, robot_class="mobile_base", require=["physics_evaluated"])
            self.assertEqual(ATTAINED, ledger.by_stage["physics_evaluated"].status)
            self.assertTrue(ledger.safe_to_export)


if __name__ == "__main__":
    unittest.main()
