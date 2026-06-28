"""§4.8A — the BOM<->sim fidelity report honestly surfaces the optimistic-sim mass gap + the per-joint torque
picture (rather than silently shipping a body lighter than its real parts imply)."""

import unittest

from virturoid.services.fidelity_report import bom_sim_fidelity, format_fidelity_md
from virturoid.services.morphology_composer import compose_robot


class FidelityReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gene = compose_robot("a quadruped walking robot", llm=None)

    def test_reports_mass_gap_and_per_joint_motors(self):
        r = bom_sim_fidelity(self.gene)
        self.assertGreater(r["sim_mass_kg"], 0)
        self.assertIsNotNone(r["grounded_mass_kg"])
        self.assertGreater(r["mass_fidelity_ratio"], 1.15)             # sim is optimistically light -> flagged
        self.assertTrue(r["flags"])
        self.assertFalse(r["faithful"])
        self.assertEqual(r["n_joints"], len(r["joints"]))
        for j in r["joints"]:
            self.assertIn("selected_motor", j)
            self.assertGreaterEqual(j["motor_peak_nm"], j["sim_torque_limit_nm"])   # real motor clears the req

    def test_markdown_render(self):
        md = format_fidelity_md(bom_sim_fidelity(self.gene))
        self.assertIn("BOM", md)
        self.assertIn("sim mass", md)


if __name__ == "__main__":
    unittest.main()
