"""Plan v2 §4.2 layer-1 spend guard: per-run HARD CAPS on LLM calls/$ that fail OPEN to the heuristic. The guard
is a transparent LLMClient wrapper, so operators need no changes."""

import unittest

from virturoid.services.llm_client import MockLLM, SpendCapExceeded, SpendGuard


class SpendGuardTests(unittest.TestCase):
    def _client(self):
        return MockLLM(fixed={"ok": True})

    def test_call_cap_raises_after_budget(self):
        g = SpendGuard(self._client(), max_calls=2)
        self.assertEqual(g.complete_json("s", "u", {}), {"ok": True})   # call 1
        self.assertEqual(g.complete_json("s", "u", {}), {"ok": True})   # call 2
        with self.assertRaises(SpendCapExceeded):
            g.complete_json("s", "u", {})                              # call 3 -> capped
        self.assertEqual(g.calls, 2)                                   # the capped call didn't reach the client

    def test_usd_cap_raises_when_over_budget(self):
        # price_out huge so one call blows a tiny budget; the NEXT call is refused
        g = SpendGuard(self._client(), max_usd=1e-6, price_out=1000.0)
        g.complete_json("system", "user", {})                          # first call allowed, accrues cost
        self.assertGreater(g.usd, 0.0)
        with self.assertRaises(SpendCapExceeded):
            g.complete_json("system", "user", {})                      # now over budget -> refused

    def test_transparent_passthrough_and_tracking(self):
        g = SpendGuard(self._client(), price_in=5.0, price_out=25.0, role="diagnostician")
        out = g.complete_json("a long system prompt here", "user question", {})
        self.assertEqual(out, {"ok": True})                            # returns the client's output unchanged
        rep = g.report()
        self.assertEqual(rep["calls"], 1)
        self.assertEqual(rep["role"], "diagnostician")
        self.assertGreater(rep["tokens_in"], 0)
        self.assertGreater(rep["usd"], 0.0)

    def test_fail_open_lets_operators_fall_back(self):
        # the design-search Proposer treats ANY LLM error as 'no backend' -> heuristic. A cap=0 guard triggers that.
        from virturoid.services.search_operators import propose_designs
        g = SpendGuard(self._client(), max_calls=0)                    # every call is immediately capped
        proposals = propose_designs("walk", {"failure_mode": "fell"}, [], g)
        self.assertEqual(proposals, [])                                # fails OPEN -> [] -> caller uses heuristic


if __name__ == "__main__":
    unittest.main()
