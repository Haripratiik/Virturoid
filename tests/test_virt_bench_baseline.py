"""Plan v3 M1 — Arm 0, the literal Claude+MCP baseline. A scripted agent drives the primitive MCP tools and
submits a controller; the SAME independent verifier scores it. No real LLM/GPU."""

import unittest

from virturoid.services.virt_bench_baseline import run_arm_0


class _ScriptedAgent:
    """Returns scripted ReAct actions in order; records the user prompts it saw."""
    name = "scripted"

    def __init__(self, actions):
        self.actions = actions
        self.i = 0
        self.prompts = []

    def complete_json(self, system, user, schema, max_tokens=2048, reasoning_effort=None):
        self.prompts.append(user)
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return dict(a)


class Arm0Tests(unittest.TestCase):
    def test_no_backend_is_honest(self):
        # there is no "Claude+MCP" without a model -> honest no_llm_backend, not a silent pass
        res = run_arm_0("L1_quad_walk", llm=None, auto_llm=False)
        self.assertEqual(res["arm"], "A0")
        self.assertFalse(res["verified_pass"])
        self.assertEqual(res["failure_mode"], "no_llm_backend")
        self.assertFalse(res["claimed_pass"])

    def test_locomotion_agent_inspects_then_submits_cpg(self):
        agent = _ScriptedAgent([
            {"action": "tool", "tool": "diagnose_body", "args": {"prompt": "a quadruped that walks"}},
            {"action": "submit", "controller": {"calf_phase": 1.57, "freq": 1.5}, "expect_pass": True},
        ])
        res = run_arm_0("L1_quad_walk", llm=agent, seed=20260701)
        self.assertEqual(res["arm"], "A0")
        self.assertIsInstance(res["verified_pass"], bool)          # a REAL verdict from the verifier re-run
        self.assertIn("forward_m", res["metrics"])                 # verified locomotion metrics
        self.assertEqual(res["budget"]["tool_calls"], 1)           # one primitive tool used
        self.assertEqual(res["budget"]["llm_calls"], 2)            # inspect + submit
        self.assertTrue(res["claimed_pass"])                       # the agent expected a pass -> honesty axis
        self.assertEqual(res["transcript"][0]["tool"], "diagnose_body")

    def test_only_primitive_tools_allowed(self):
        # the baseline may NOT call our verified harness (design_search) or memory tools — that would collapse
        # the delta we are measuring. Such a call is rejected and the agent falls through to submit.
        agent = _ScriptedAgent([
            {"action": "tool", "tool": "design_search", "args": {"prompt": "x"}},   # blocked
            {"action": "submit", "controller": {}, "expect_pass": False},
        ])
        res = run_arm_0("L1_quad_walk", llm=agent, seed=20260701)
        self.assertEqual(res["budget"]["tool_calls"], 0)           # the blocked call was NOT dispatched
        self.assertIn("not available", res["transcript"][0]["error"])
        self.assertFalse(res["claimed_pass"])

    def test_manipulation_default_controller(self):
        agent = _ScriptedAgent([{"action": "submit", "controller": {}, "expect_pass": True}])
        res = run_arm_0("M1_arm_grasp", llm=agent, seed=20260701)
        self.assertEqual(res["arm"], "A0")
        self.assertIn("success_rate", res["metrics"])


class HeadToHeadBaselineTests(unittest.TestCase):
    def test_head_to_head_with_baseline_reports_delta(self):
        from virturoid.services.virt_bench_arms import run_head_to_head
        agent = _ScriptedAgent([{"action": "submit", "controller": {}, "expect_pass": True}])
        r = run_head_to_head(split="dev", families=("manipulation",), max_evals=1, with_baseline=True,
                             baseline_llm=agent)
        self.assertIn("A0_solved", r)
        self.assertIn("baseline_delta", r)                         # B_solved - A0_solved (the headline)
        self.assertEqual(r["baseline_delta"], r["B_solved"] - r["A0_solved"])
        self.assertIn("A0", r["honesty"])                          # over-claim tracked for the baseline too
        self.assertTrue(all("A0_pass" in row for row in r["rows"]))


if __name__ == "__main__":
    unittest.main()
