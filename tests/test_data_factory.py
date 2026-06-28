"""§4.3 — the MimicGen-style demo factory: replay one scripted grasp under randomization, rejection-sample the
successes into a verified demonstration dataset (the in-sim data engine)."""

import json
import tempfile
import unittest
from pathlib import Path

from virturoid.services.data_factory import generate_grasp_demos, write_grasp_dataset
from virturoid.services.morphology_composer import compose_robot


class DataFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gene = compose_robot("grasp and lift a box on a table", llm=None)

    def test_factory_produces_verified_demos(self):
        ds = generate_grasp_demos(self.gene, n=6, seed=0)
        self.assertEqual(ds["n_attempts"], 6)
        self.assertGreaterEqual(ds["n_success"], 3)          # high-yield engine (probed ~1.0); robust to seed
        self.assertEqual(ds["augmentation_x"], ds["n_success"])   # demos produced from the ONE scripted skill
        self.assertGreater(ds["yield"], 0.0)
        for d in ds["demos"]:                                # every demo is a REAL, verified success
            self.assertIn("object_xy", d["scene"])
            self.assertIn("fclose", d["action_params"])
            self.assertGreater(d["outcome"]["lifted_m"], 0.0)

    def test_write_dataset_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "grasp_demos.json"
            summary = write_grasp_dataset(self.gene, out, n=4, seed=2)
            self.assertTrue(out.exists())
            self.assertEqual(summary["n_demos"], summary["n_success"])
            data = json.loads(out.read_text())
            self.assertIn("demos", data)
            self.assertIn("yield", data)


if __name__ == "__main__":
    unittest.main()
