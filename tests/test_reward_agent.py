import unittest

from virturoid.services.llm_client import MockLLM
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.reward_agent import translate_task_to_reward


def _req():
    return build_requirements_from_prompt("Build a humanoid that lifts boxes onto a shelf.")


_GOOD = {
    "task_type": "lift_to_shelf",
    "sparse_success": {"name": "box_on_shelf", "expression": "object_in_zone == 1"},
    "failure": [
        {"name": "dropped", "expression": "object_height < 0.02"},
        {"name": "timeout", "expression": "episode_time > 60"},
    ],
    "dense_terms": [
        {"quantity": "object_to_target", "kind": "attract", "weight": 2.0, "target": 0.0},
        {"quantity": "grasp_contact", "kind": "bonus", "weight": 1.0, "target": None},
    ],
    "scene_requirements": ["box", "shelf_zone"],
    "end_effector": "gripper",
    "domain_randomization": {"box_mass_kg": [0.5, 2.0], "friction": [0.8, 1.3]},
}


class RewardAgentTests(unittest.TestCase):
    def test_no_backend_returns_none(self):
        self.assertIsNone(translate_task_to_reward("x", _req(), None))

    def test_valid_spec_passes_and_parses(self):
        out = translate_task_to_reward("lift boxes onto a shelf", _req(), MockLLM(fixed=_GOOD), max_repairs=0)
        self.assertTrue(out["valid"], out.get("issues"))
        spec = out["spec"]
        self.assertEqual("lift_to_shelf", spec.task_type)
        self.assertEqual("object_in_zone == 1", spec.sparse_success.expression)
        self.assertEqual(2, len(spec.dense_terms))

    def test_missing_timeout_is_rejected(self):
        bad = dict(_GOOD, failure=[{"name": "dropped", "expression": "object_height < 0.02"}])
        out = translate_task_to_reward("x", _req(), MockLLM(fixed=bad), max_repairs=0)
        self.assertFalse(out["valid"])
        self.assertTrue(any("timeout" in i for i in out["issues"]))

    def test_unknown_quantity_or_overweight_term_is_rejected(self):
        bad = dict(_GOOD, dense_terms=[{"quantity": "make_it_win", "kind": "bonus", "weight": 999.0}])
        out = translate_task_to_reward("x", _req(), MockLLM(fixed=bad), max_repairs=0)
        self.assertFalse(out["valid"])
        self.assertTrue(any("vocabulary" in i for i in out["issues"]))
        self.assertTrue(any("weight" in i for i in out["issues"]))

    def test_self_repair_recovers(self):
        calls = {"n": 0}
        # The recovered spec is process-gated (object_settled), so it is both schema-valid AND
        # non-gameable — no extra anti-hacking round is needed (recovers in 2 attempts).
        good_process = dict(_GOOD, sparse_success={
            "name": "box_on_shelf", "expression": "object_in_zone == 1 and object_settled == 1"})

        def responder(system, user, schema):
            calls["n"] += 1
            if calls["n"] == 1:
                return dict(_GOOD, dense_terms=[{"quantity": "bogus", "kind": "bonus", "weight": 1.0}])
            return good_process

        out = translate_task_to_reward("lift boxes", _req(), MockLLM(responder=responder))
        self.assertTrue(out["valid"])
        self.assertEqual(2, out["attempts"])
        self.assertEqual([], out.get("hacking_risks", []))   # recovered to a non-gameable spec

    def test_outcome_only_reward_triggers_anti_hacking_repair(self):
        # An outcome-only (gameable) but schema-valid spec should prompt a hardening round.
        calls = {"n": 0}
        good_process = dict(_GOOD, sparse_success={
            "name": "box_on_shelf", "expression": "object_in_zone == 1 and object_settled == 1"})

        def responder(system, user, schema):
            calls["n"] += 1
            return _GOOD if calls["n"] == 1 else good_process   # gameable first, hardened second

        out = translate_task_to_reward("lift boxes", _req(), MockLLM(responder=responder))
        self.assertTrue(out["valid"])
        self.assertEqual(2, out["attempts"])                 # one extra round to de-game it
        self.assertEqual([], out["hacking_risks"])


if __name__ == "__main__":
    unittest.main()
