"""Open-world request concepts are remembered without becoming unearned routes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ConceptMemoryTests(unittest.TestCase):
    def test_candidate_only_becomes_recallable_after_target_attainment(self):
        from virturoid.services.concept_memory import (
            observe_request,
            promote_after_evaluation,
            recall_verified_route,
        )

        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp) / "memory"
            candidate = observe_request(memory, "trilobite-like rover", "build a trilobite-like rover")
            self.assertEqual("candidate", candidate["state"])
            self.assertIsNone(recall_verified_route(memory, "trilobite-like rover"))

            evaluated = promote_after_evaluation(
                memory, "trilobite-like rover", execution_family="quadruped", task_type="locomotion",
                species_pattern="legged6.composed", success_rate=0.35, target_success_rate=0.8,
            )
            self.assertEqual("evaluated", evaluated["state"])
            self.assertIsNone(recall_verified_route(memory, "trilobite-like rover"))

            verified = promote_after_evaluation(
                memory, "trilobite-like rover", execution_family="quadruped", task_type="locomotion",
                species_pattern="legged6.composed", success_rate=0.85, target_success_rate=0.8,
            )
            self.assertEqual("verified", verified["state"])
            recalled = recall_verified_route(memory, "trilobite-like rover")
            self.assertEqual("quadruped", recalled["execution_family"])
            self.assertEqual("locomotion", recalled["task_type"])
            self.assertEqual(2, len(recalled["evidence"]))

    def test_zero_target_diagnostics_do_not_create_a_verified_route(self):
        from virturoid.services.concept_memory import promote_after_evaluation, recall_verified_route

        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp) / "memory"
            record = promote_after_evaluation(
                memory, "unproven crawler", execution_family="quadruped", task_type="locomotion",
                species_pattern="crawler.v1", success_rate=0.0, target_success_rate=0.0,
            )
            self.assertEqual("evaluated", record["state"])
            self.assertFalse(record["evidence"][-1]["verified"])
            self.assertIsNone(recall_verified_route(memory, "unproven crawler"))

    def test_llm_alias_recalls_only_a_verified_concept(self):
        from virturoid.services.concept_memory import (
            observe_request,
            promote_after_evaluation,
            recall_verified_route,
        )

        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp) / "memory"
            canonical = "trilobite-like rover"
            alias = "armored six-legged crawler"
            candidate = observe_request(
                memory, canonical, "build a trilobite-like rover", aliases=[alias],
            )
            self.assertIsNone(recall_verified_route(memory, alias))

            promoted = promote_after_evaluation(
                memory, canonical, execution_family="quadruped", task_type="locomotion",
                species_pattern="legged6.composed", success_rate=0.85, target_success_rate=0.8,
                aliases=[alias],
            )
            recalled = recall_verified_route(memory, alias)
            self.assertEqual("verified", promoted["state"])
            self.assertEqual(candidate["concept_id"], recalled["concept_id"])
            self.assertEqual("quadruped", recalled["execution_family"])

    def test_autobuild_records_an_unroutable_named_concept_before_clarifying(self):
        from virturoid.services.autonomous_build import autonomous_build
        from virturoid.services.memory_db import MemoryDB, _concept_id

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = autonomous_build("build a blorptron robot", root / "output", memory_dir=root / "memory")
            self.assertFalse(report.succeeded)
            self.assertFalse(report.feasible)
            with MemoryDB(root / "memory" / "virturoid_memory.db") as db:
                concept = db.concept(_concept_id("blorptron"))
            self.assertIsNotNone(concept)
            self.assertEqual("candidate", concept["state"])

    def test_ai_first_autobuild_never_substitutes_a_template_when_llm_is_off(self):
        from virturoid.services.autonomous_build import autonomous_build

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"VIRTUROID_LLM_BACKEND": "off", "VIRTUROID_NO_INTERNAL_LLM": "1"}
        ):
            root = Path(tmp)
            report = autonomous_build(
                "build a six-legged trilobite-like robot that walks",
                root / "output",
                memory_dir=root / "memory",
                allow_heuristic_fallback=False,
            )
            self.assertFalse(report.feasible)
            self.assertFalse(report.succeeded)
            self.assertEqual("clarify_intent", report.decisions[0].stage)
            self.assertIn("LLM planner", report.notes[0])
            self.assertFalse((root / "output" / "robot").exists())


if __name__ == "__main__":
    unittest.main()
