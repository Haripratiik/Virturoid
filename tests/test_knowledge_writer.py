"""Phase 4 of the robotics-native-AI plan: the VERIFIED write-back loop (the flywheel turning).

On an honest PASS, the tip-writer banks a grounded, retrievable tip keyed to the species; a non-pass records
nothing (verified-only, anti-Goodhart). Closes the loop with Phase 1: write -> index -> recall. No MuJoCo.
"""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.services.design_critic import add_parallel_gripper
from virturoid.services.memory_db import MemoryDB
from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
from virturoid.services.knowledge_writer import record_verified_knowledge
from virturoid.services.species_discovery import auto_place_species


class _TipLLM:
    name = "fake"

    def complete_json(self, system, user, schema, max_tokens=None):
        return {"tip": "AUTHORED: keep the gripper light and the wrist compliant", "audience": "trainer"}


class WriteBackTests(unittest.TestCase):
    def _mem(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_verified_pass_banks_a_grounded_tip(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            gene = tabletop_arm_gene()
            sp = auto_place_species(gene, db)["species_pattern"]
            wb = record_verified_knowledge(db, gene, sp, task_type="grasp", success=0.9, target=0.8)
            self.assertTrue(wb["wrote_tip"])
            self.assertIn("90%", wb["tip"])                 # grounded in the real outcome
            self.assertIn("grasp", wb["tip"])

    def test_non_pass_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            gene = tabletop_arm_gene()
            sp = auto_place_species(gene, db)["species_pattern"]
            wb = record_verified_knowledge(db, gene, sp, task_type="grasp", success=0.4, target=0.8)
            self.assertFalse(wb["wrote_tip"])               # below its bar -> not recorded (verified-only)
            self.assertIn("verified-only", wb["reason"])

    def test_llm_authors_the_tip_when_available(self):
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            gene = tabletop_arm_gene()
            sp = auto_place_species(gene, db)["species_pattern"]
            wb = record_verified_knowledge(db, gene, sp, task_type="grasp", success=0.9, target=0.8,
                                           llm=_TipLLM())
            self.assertIn("AUTHORED", wb["tip"])
            self.assertEqual(wb["audience"], "trainer")

    def test_write_then_recall_closes_the_loop(self):
        """The whole point: a verified tip banked on one body is recalled for a NEAR body."""
        with tempfile.TemporaryDirectory() as tmp, self._mem(tmp) as db:
            gene = tabletop_arm_gene()
            sp = auto_place_species(gene, db)["species_pattern"]
            record_verified_knowledge(db, gene, sp, task_type="grasp", success=0.95, target=0.8)
            know = RoboticsVectorMemory(db).recall_knowledge(add_parallel_gripper(tabletop_arm_gene()), "grasp")
            self.assertTrue(know["tips"])
            self.assertIn("reuse this body", know["tips"][0]["meta"]["tip"])


if __name__ == "__main__":
    unittest.main()
