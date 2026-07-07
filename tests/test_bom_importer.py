"""I3: inbound BOM importer (Input Ingestion plan, Phase 6).

Column classification, unit normalization (g->kg), duplicate merge, file parsing (CSV/JSON), and
provenance-tracked reconciliation against a generated BOM. Pure/offline (AGENTS.md).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.bom_importer import (  # noqa: E402
    classify_columns,
    parse_bom_file,
    parse_bom_rows,
    reconcile_with_generated,
)


class ColumnClassifierTests(unittest.TestCase):
    def test_maps_messy_headers_to_canonical_fields(self):
        mapping = classify_columns(["Part Number", "Component", "Qty", "Mass (g)", "Unit Price USD", "Torque"])
        self.assertEqual(mapping["Part Number"], "part_number")
        self.assertEqual(mapping["Component"], "part")
        self.assertEqual(mapping["Qty"], "qty")
        self.assertEqual(mapping["Mass (g)"], "unit_mass_kg")
        self.assertEqual(mapping["Unit Price USD"], "unit_price_usd")
        self.assertEqual(mapping["Torque"], "torque_nm")


class ParseRowsTests(unittest.TestCase):
    def test_unit_normalization_grams_to_kg(self):
        rows = [{"part": "motor", "mass (g)": "340", "qty": "2", "price": "300"}]
        res = parse_bom_rows(rows)
        self.assertAlmostEqual(res.items[0].unit_mass_kg, 0.340, places=6)   # grams -> kg
        self.assertEqual(res.items[0].qty, 2)
        self.assertAlmostEqual(res.totals["mass_kg"], 0.68, places=4)        # 0.34 * 2

    def test_missing_unit_assumed_kg_with_warning(self):
        res = parse_bom_rows([{"name": "link", "mass": "0.5"}])
        self.assertAlmostEqual(res.items[0].unit_mass_kg, 0.5, places=6)
        self.assertTrue(any("assuming kilograms" in w for w in res.warnings))

    def test_duplicate_parts_merge_quantity(self):
        rows = [{"part": "bolt", "qty": "4"}, {"part": "bolt", "qty": "6"}]
        res = parse_bom_rows(rows)
        self.assertEqual(len(res.items), 1)
        self.assertEqual(res.items[0].qty, 10)
        self.assertTrue(any("merged duplicate" in w for w in res.warnings))

    def test_row_without_part_is_skipped(self):
        res = parse_bom_rows([{"name": "", "mass_kg": "1.0"}, {"name": "arm", "mass_kg": "0.5"}])
        self.assertEqual(len(res.items), 1)
        self.assertEqual(res.items[0].part, "arm")


class FileParseTests(unittest.TestCase):
    def test_csv(self):
        d = tempfile.mkdtemp(prefix="bom_")
        p = os.path.join(d, "parts.csv")
        Path(p).write_text("Component,Qty,Mass_kg,Unit Price\nservo,3,0.05,45\nbracket,6,0.02,3\n",
                           encoding="utf-8")
        res = parse_bom_file(p)
        self.assertEqual(len(res.items), 2)
        self.assertEqual(res.totals["line_items"], 2)
        self.assertTrue(res.validate().ok)

    def test_json_items_key(self):
        d = tempfile.mkdtemp(prefix="bom_")
        p = os.path.join(d, "parts.json")
        Path(p).write_text(json.dumps({"items": [{"part": "camera", "qty": 1, "mass_kg": 0.1, "price": 120}]}),
                           encoding="utf-8")
        res = parse_bom_file(p)
        self.assertEqual(res.items[0].part, "camera")


class ReconcileTests(unittest.TestCase):
    def test_override_and_augment_with_provenance(self):
        generated = {
            "lines": [
                {"part": "Unitree GO-M8010-6", "category": "actuator", "qty": 2,
                 "unit_mass_kg": 0.34, "unit_price_usd": 300.0, "mass_kg": 0.68, "price_usd": 600.0},
            ],
            "totals": {"mass_kg": 0.68, "price_usd": 600.0},
        }
        imported = parse_bom_rows([
            {"part": "Unitree GO-M8010-6", "mass_kg": "0.30", "price": "250"},   # override the actuator
            {"part": "LiDAR Puck", "mass_kg": "0.5", "price": "800", "qty": "1"},  # augment a new line
        ])
        out = reconcile_with_generated(generated, imported)
        by_part = {ln["part"]: ln for ln in out["lines"]}
        # overridden line takes the user's numbers + provenance.
        self.assertAlmostEqual(by_part["Unitree GO-M8010-6"]["unit_mass_kg"], 0.30, places=6)
        self.assertEqual(by_part["Unitree GO-M8010-6"]["provenance"], "user_bom")
        # augmented line is added with its own provenance.
        self.assertIn("LiDAR Puck", by_part)
        self.assertEqual(by_part["LiDAR Puck"]["provenance"], "user_bom_added")
        # reconciliation summary records both.
        self.assertIn("Unitree GO-M8010-6", out["bom_reconciliation"]["overridden"])
        self.assertIn("LiDAR Puck", out["bom_reconciliation"]["added"])


if __name__ == "__main__":
    unittest.main()
