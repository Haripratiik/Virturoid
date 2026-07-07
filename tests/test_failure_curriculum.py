"""Failure -> curriculum repair mapping (dossier R7 + perception-failure taxonomy)."""

import unittest

from virturoid.services.failure_curriculum import (
    curriculum_from_failure_clusters,
    curriculum_updates_for_failure,
    is_mapped,
)


class FailureCurriculumTests(unittest.TestCase):
    def test_known_failures_map_to_repairs(self):
        self.assertIn("lift_phase_reward", curriculum_updates_for_failure("gripped_no_lift"))
        self.assertIn("stability_prior", curriculum_updates_for_failure("fell_over"))
        self.assertIn("depth_dropout_curriculum", curriculum_updates_for_failure("bad_depth_at_grasp"))

    def test_unknown_failure_is_empty(self):
        self.assertEqual(curriculum_updates_for_failure("banana_peel"), [])
        self.assertFalse(is_mapped("banana_peel"))

    def test_clusters_ordered_by_frequency_and_deduped(self):
        result = curriculum_from_failure_clusters({"no_contact": 3, "fell_over": 1, "banana_peel": 2})
        self.assertTrue(result["updates"])
        # most frequent mapped failure's repairs come first
        self.assertLess(result["updates"].index("closer_approach_scenes"),
                        result["updates"].index("stability_prior"))
        self.assertIn("banana_peel", result["unmapped"])
        self.assertIn("no_contact", result["by_failure"])
        self.assertEqual(len(result["updates"]), len(set(result["updates"])))  # de-duplicated

    def test_reward_hack_sets_rejection_flag(self):
        result = curriculum_from_failure_clusters(["reward_high_success_low"])
        self.assertTrue(result["reject_reward_candidates"])
        self.assertIn("reject_reward_candidate", result["updates"])

    def test_list_input_counts_occurrences(self):
        result = curriculum_from_failure_clusters(["no_contact", "no_contact", "fell_over"])
        self.assertLess(result["updates"].index("closer_approach_scenes"),
                        result["updates"].index("stability_prior"))


if __name__ == "__main__":
    unittest.main()
