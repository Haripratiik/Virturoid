"""H3 — the LLM search operators (Proposer w/ diversity reflection, Critic, Diagnostician) + their adapter
into the harness. Fake-LLM tested (records prompts). Includes an end-to-end harness run driven by the LLM
proposer that climbs a fake evaluator to a gate pass. No MuJoCo/real LLM."""

import unittest

from virturoid.services.design_search import run_design_search
from virturoid.services.search_operators import (
    critique_design, diagnose_next_edit, make_llm_proposer, propose_designs)


class _FakeLLM:
    """Returns scripted proposal batches; approves critic; records every user prompt."""
    name = "fake"

    def __init__(self, proposal_batches=None):
        self.batches = proposal_batches or [[{"edit_kind": "cpg", "params": {"calf_phase": 0.0}}]]
        self._i = 0
        self.calls = []

    def complete_json(self, system, user, schema, max_tokens=None):
        props = schema.get("properties", {})
        if "proposals" in props:
            self.calls.append(("proposer", user))
            batch = self.batches[min(self._i, len(self.batches) - 1)]
            self._i += 1
            return {"proposals": batch}
        if "viable" in props:
            self.calls.append(("critic", user))
            return {"viable": True, "reason": "ok"}
        self.calls.append(("diag", user))
        return {"edit_kind": "cpg", "params": {"calf_phase": 0.0}}


_ART = {"summary_text": "VERDICT: FAIL — walks_backward", "next_actions": ["flip direction"]}


class OperatorTests(unittest.TestCase):
    def test_proposer_uses_diversity_reflection_and_shows_priors(self):
        llm = _FakeLLM([[{"edit_kind": "cpg", "params": {"calf_phase": 0.3}}]])
        out = propose_designs("make it walk", _ART, [{"edit_kind": "cpg", "params": {"calf_phase": 1.57}}], llm)
        self.assertEqual(out[0]["edit_kind"], "cpg")
        user = llm.calls[0][1]
        self.assertIn("maximally different", user.lower())        # the ~38x lever
        self.assertIn("1.57", user)                               # the prior candidate is shown

    def test_proposer_offline_returns_empty(self):
        self.assertEqual(propose_designs("walk", _ART, [], None), [])

    def test_critic_screens_and_fails_open_offline(self):
        self.assertTrue(critique_design({"edit_kind": "cpg", "params": {}}, "walk", _ART, None)["viable"])
        llm = _FakeLLM()
        self.assertTrue(critique_design({"edit_kind": "cpg", "params": {}}, "walk", _ART, llm)["viable"])

    def test_diagnostician_names_an_edit(self):
        self.assertEqual(diagnose_next_edit("walk", _ART, _FakeLLM())["edit_kind"], "cpg")
        self.assertIsNone(diagnose_next_edit("walk", _ART, None))

    def test_make_llm_proposer_offline_uses_heuristic(self):
        prop = make_llm_proposer("walk", None, heuristic=lambda p, h: {"edit_kind": "cpg", "params": {"calf_phase": 0.0}})
        self.assertEqual(prop(None, [])["params"]["calf_phase"], 0.0)

    def test_harness_driven_by_llm_proposer_solves(self):
        # fake physics: forward = 0.5 - 0.4*|calf_phase|; gate 0.30 -> passes once calf_phase small enough
        def evaluate(spec):
            cp = abs(float(spec.get("params", {}).get("calf_phase", 1.5)))
            return {"forward": round(0.5 - 0.4 * cp, 3), "cadence": 8.0, "upright_frac": 0.9, "survived": True}
        llm = _FakeLLM([[{"edit_kind": "cpg", "params": {"calf_phase": 0.8}}],   # 0.18 fail
                        [{"edit_kind": "cpg", "params": {"calf_phase": 0.2}}],   # 0.42 pass
                        [{"edit_kind": "cpg", "params": {"calf_phase": 0.0}}]])
        prop = make_llm_proposer("make it walk forward", llm, k=1, screen=True)
        rep = run_design_search(propose=prop, evaluate=evaluate, max_evals=6)
        self.assertTrue(rep.solved)
        self.assertEqual(rep.best.artifact["failure_mode"], "walking")


class RationaleLineageTests(unittest.TestCase):
    def test_priors_block_surfaces_rationale(self):
        # EoH (plan v2 §4.5): the diversity-reflection prompt shows each prior's REASONING, not just its params
        from virturoid.services.search_operators import _priors_block
        block = _priors_block([{"edit_kind": "cpg", "params": {"calf_phase": 0.0},
                                "rationale": "flip the gait phase to face forward"}])
        self.assertIn("flip the gait phase to face forward", block)   # lineage of ideas threaded
        self.assertIn("cpg", block)
        # a spec with no rationale still lists cleanly (back-compat)
        self.assertIn("gains", _priors_block([{"edit_kind": "gains", "params": {"kp": 40}}]))


if __name__ == "__main__":
    unittest.main()
