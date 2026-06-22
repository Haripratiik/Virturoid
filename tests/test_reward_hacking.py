"""The reward must reward the PROCESS, not just the outcome (the kick-vs-place problem).

A pure outcome metric ('object_in_zone') is gamed equally by a box KICKED into the zone and one
PLACED under control — only the latter is the real, transferable skill. These tests lock in that
(a) the vocabulary can express process, (b) the system flags an outcome-only (gameable) reward,
and (c) the reward agent self-repairs a gameable reward into a process-gated one. All offline.
"""

import unittest

from virturoid.schemas.reward import (
    ALLOWED_QUANTITIES,
    PROCESS_QUANTITIES,
    Criterion,
    RewardSpec,
    RewardTerm,
)
from virturoid.services.llm_client import MockLLM
from virturoid.services.reward_agent import translate_task_to_reward

_TIMEOUT = Criterion("timeout", "time > 200")


class RewardHackingTests(unittest.TestCase):
    def test_process_quantities_in_vocabulary(self):
        for q in ("object_speed", "contact_sustained", "object_settled"):
            self.assertIn(q, ALLOWED_QUANTITIES)
            self.assertIn(q, PROCESS_QUANTITIES)

    def test_outcome_only_success_is_flagged_as_gameable(self):
        spec = RewardSpec(
            task_type="place", sparse_success=Criterion("placed", "object_in_zone == 1"),
            failure=[_TIMEOUT], end_effector="gripper",
            dense_terms=[RewardTerm("object_to_target", "attract", 1.0)])
        self.assertEqual(spec.validate(), [])             # schema-valid...
        risks = spec.hacking_risks()
        self.assertTrue(any("OUTCOME-ONLY" in r for r in risks))   # ...but gameable

    def test_process_gated_success_is_clean(self):
        spec = RewardSpec(
            task_type="place",
            sparse_success=Criterion("placed", "object_in_zone == 1 and object_settled == 1"),
            failure=[_TIMEOUT], end_effector="gripper",
            dense_terms=[RewardTerm("object_to_target", "attract", 1.0),
                         RewardTerm("object_speed", "penalty", 0.5)])
        self.assertEqual(spec.hacking_risks(), [])

    def test_agent_self_repairs_a_gameable_reward(self):
        # First proposal is outcome-only (gameable); the agent should reject it and harden it.
        calls = {"n": 0}

        def responder(system, user, schema):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"task_type": "place", "sparse_success": {"name": "placed", "expression": "object_in_zone == 1"},
                        "failure": [{"name": "timeout", "expression": "time > 200"}],
                        "dense_terms": [{"quantity": "object_to_target", "kind": "attract", "weight": 1.0}],
                        "end_effector": "gripper"}
            return {"task_type": "place",
                    "sparse_success": {"name": "placed", "expression": "object_in_zone == 1 and object_settled == 1"},
                    "failure": [{"name": "timeout", "expression": "time > 200"}],
                    "dense_terms": [{"quantity": "object_to_target", "kind": "attract", "weight": 1.0},
                                    {"quantity": "object_speed", "kind": "penalty", "weight": 0.5}],
                    "end_effector": "gripper"}

        out = translate_task_to_reward("place the box in the bin", None, MockLLM(responder=responder))
        self.assertTrue(out["valid"])
        self.assertEqual(out["hacking_risks"], [])     # accepted only the hardened, non-gameable spec
        self.assertGreaterEqual(calls["n"], 2)         # it iterated to fix the gameable one


if __name__ == "__main__":
    unittest.main()
