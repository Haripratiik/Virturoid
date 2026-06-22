"""Intent router: an LLM picks the best-fit real model or routes novel/compositional requests to the
procedural path — replacing brittle keyword lookup. Deterministic keyword fallback when no LLM."""

import unittest


class BuildPlannerTests(unittest.TestCase):
    def test_llm_plan_picks_chosen_catalogue_model(self):
        from virturoid.services.build_planner import plan_build
        from virturoid.services.llm_client import MockLLM
        llm = MockLLM(fixed={"base": "real", "model_key": "ur10e", "attachments": ["arm"], "reason": "long reach"})
        plan = plan_build("a long-reach welding arm", llm)
        self.assertEqual(plan["base"], "real")
        self.assertEqual(plan["model_key"], "ur10e")          # honored the qualifier, not just 'arm'
        self.assertEqual(plan["source"], "llm")

    def test_llm_routes_novel_to_procedural(self):
        from virturoid.services.build_planner import plan_build
        from virturoid.services.llm_client import MockLLM
        llm = MockLLM(fixed={"base": "procedural", "model_key": "", "reason": "no real snake exists"})
        plan = plan_build("a snake robot", llm)
        self.assertEqual(plan["base"], "procedural")
        self.assertIsNone(plan["model_key"])

    def test_llm_invalid_key_falls_back_to_keyword(self):
        from virturoid.services.build_planner import plan_build
        from virturoid.services.llm_client import MockLLM
        llm = MockLLM(fixed={"base": "real", "model_key": "NOPE", "reason": "x"})
        plan = plan_build("a quadruped robot", llm)            # bad key -> deterministic keyword fallback
        self.assertEqual(plan["model_key"], "quadruped")

    def test_no_llm_uses_keyword_fallback(self):
        from virturoid.services.build_planner import plan_build
        self.assertEqual(plan_build("a humanoid robot", None)["model_key"], "humanoid")
        self.assertEqual(plan_build("a tentacle thing", None)["base"], "procedural")


if __name__ == "__main__":
    unittest.main()
