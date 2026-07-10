"""R2' with a LIVE LLM designer (Claude) — the quantified Thesis A measurement, made reproducible. A real LLM
authored both arms; the verdict is un-gameable geometry. Asserts the measured lift holds and is auditable (the
two OFF failures are genuine from-scratch mistakes the retrieved verified exemplar avoids).
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_B123D = importlib.util.find_spec("build123d") is not None


@unittest.skipUnless(_B123D, "the design verdict realizes shape programs with build123d")
class R2PrimeLLMDesignerTests(unittest.TestCase):
    def test_corpus_grounding_lifts_a_live_llm_designer(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.r2prime_design import run_llm_designer_battery
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="r2llm_")) / "m.db")
        res = run_llm_designer_battery(db)
        # the corpus (retrieved verified exemplars) beats the LLM's from-scratch authoring — the Thesis A claim
        self.assertGreater(res["corpus"], res["off"], f"corpus grounding must lift verified-solve: {res}")
        self.assertEqual(res["corpus"], res["n"], "every retrieved verified exemplar solves its role by construction")
        self.assertGreaterEqual(res["lift"], 0.2, f"the measured lift held at ~+25%: {res}")
        # AUDITABLE: the OFF failures are the fin (a round 'tapered fin', not a flat panel) + the mantle
        # (a full-length loft that reads tubular, not bulbous) — genuine authoring mistakes, not sandbagging
        fails = {r["role"] for r in res["rows"] if not r["off"]}
        self.assertIn("fin", fails)
        self.assertIn("mantle", fails)
        for r in res["rows"]:
            self.assertTrue(r["corpus"], f"the retrieved exemplar for '{r['role']}' must solve: {r}")


if __name__ == "__main__":
    unittest.main()
