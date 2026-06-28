"""The package API surfaces the build's honest signals (§4.1/§4.8A/§4.8E) so the Build Console can show
requested-vs-achieved + the BOM<->sim fidelity gap, not just the spec sheet."""

import json
import tempfile
import unittest
from pathlib import Path

from virturoid.ui_server import package_honesty_summary


class UIHonestyTests(unittest.TestCase):
    def test_reads_fidelity_and_compliance(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp)
            (child / "reports").mkdir()
            (child / "reports" / "bom_sim_fidelity.json").write_text(
                json.dumps({"mass_fidelity_ratio": 2.26, "flags": ["optimistic"]}), encoding="utf-8")
            (child / "reports" / "spec_compliance.json").write_text(
                json.dumps({"all_honored": True, "constraints": [{"constraint": "height_m"}]}), encoding="utf-8")
            h = package_honesty_summary(child)
            self.assertEqual(h["mass_fidelity_ratio"], 2.26)
            self.assertEqual(h["fidelity_flags"], 1)
            self.assertTrue(h["spec_all_honored"])
            self.assertEqual(h["spec_constraints"], 1)

    def test_none_when_no_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(package_honesty_summary(Path(tmp)))

    def test_resolve_build_root_keeps_a_root_that_has_packages(self):
        from virturoid.ui_server import _resolve_build_root
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "myroot"
            (root / "pkg1" / "robot").mkdir(parents=True)              # a built package -> no demo fallback
            self.assertEqual(_resolve_build_root(root), root)


if __name__ == "__main__":
    unittest.main()
