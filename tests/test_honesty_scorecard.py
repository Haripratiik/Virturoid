"""§4.1 — the honesty scorecard surfaces every claim next to the gate's verdict (weak results shown openly)."""

import unittest

from virturoid.schemas.readiness_ledger import ATTAINED, BELOW_GATE
from virturoid.services.honesty_scorecard import format_scorecard_md, honesty_scorecard


class HonestyScorecardTests(unittest.TestCase):
    def test_unifies_signals_and_counts_honest_vs_flagged(self):
        readiness = {"stages": [
            {"stage": "real_cad_exported", "status": ATTAINED, "detail": "real STEP"},
            {"stage": "locomotion_walked", "status": BELOW_GATE, "detail": "did not walk (0.03 m)"},
        ]}
        spec = {"constraints": [
            {"constraint": "height_m", "honored": True},
            {"constraint": "self_weight_kg", "honored": False},
        ]}
        sim2sim = {"verdict": "strong — predicts perturbed", "pearson_r": 0.9, "mmrv": 0.0}
        sc = honesty_scorecard(readiness=readiness, spec_compliance=spec, sim2sim=sim2sim)
        self.assertEqual(sc["n_claims"], 5)
        self.assertEqual(sc["n_honest"], 3)              # cad attained + height honored + sim2sim
        self.assertEqual(sc["n_flagged"], 2)             # walk below gate + weight over budget
        claims = {r["claim"]: r for r in sc["rows"]}
        self.assertFalse(claims["locomotion_walked"]["honest"])      # the WEAK result is present, not hidden
        self.assertEqual(claims["locomotion_walked"]["verdict"], BELOW_GATE)
        self.assertFalse(claims["spec:self_weight_kg"]["honest"])
        self.assertIn("flagged", sc["headline"])

    def test_fidelity_row_flags_optimistic_sim(self):
        sc = honesty_scorecard(fidelity={"mass_fidelity_ratio": 2.26, "flags": ["optimistic"], "faithful": False})
        row = next(r for r in sc["rows"] if r["claim"] == "bom_sim_fidelity")
        self.assertFalse(row["honest"])                  # the optimistic-sim gap is surfaced as flagged
        self.assertIn("lighter", row["verdict"])

    def test_empty_scorecard(self):
        sc = honesty_scorecard()
        self.assertEqual(sc["n_claims"], 0)
        self.assertIn("no claims", sc["headline"])

    def test_markdown_render(self):
        sc = honesty_scorecard(readiness={"stages": [{"stage": "physics_evaluated", "status": ATTAINED}]})
        md = format_scorecard_md(sc)
        self.assertIn("Honesty Scorecard", md)
        self.assertIn("physics_evaluated", md)
        self.assertIn("| Claim |", md)


class ScorecardFromPackageTests(unittest.TestCase):
    def test_reads_all_honest_reports_from_disk(self):
        import json
        import tempfile
        from pathlib import Path

        from virturoid.services.honesty_scorecard import scorecard_from_package
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp) / "reports"
            r.mkdir()
            (r / "product_readiness_ledger.json").write_text(json.dumps(
                {"stages": [{"stage": "real_cad_exported", "status": ATTAINED},
                            {"stage": "rendered_cv_data", "status": BELOW_GATE}]}), encoding="utf-8")
            (r / "spec_compliance.json").write_text(json.dumps(
                {"constraints": [{"constraint": "height_m", "honored": True}]}), encoding="utf-8")
            (r / "bom_sim_fidelity.json").write_text(json.dumps(
                {"mass_fidelity_ratio": 2.2, "flags": ["optimistic"], "faithful": False}), encoding="utf-8")
            (r / "sim2sim_report.json").write_text(json.dumps(
                {"verdict": "transfer STRONG", "pearson_r": 0.9, "mmrv": 0.0}), encoding="utf-8")
            sc = scorecard_from_package(Path(tmp))
            claims = {x["claim"] for x in sc["rows"]}
            self.assertEqual(claims, {"real_cad_exported", "rendered_cv_data", "spec:height_m",
                                      "bom_sim_fidelity", "sim2sim transfer"})
            self.assertEqual(sc["n_flagged"], 2)          # CV below-gate + optimistic fidelity, shown openly


if __name__ == "__main__":
    unittest.main()
