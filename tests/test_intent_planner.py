"""LLM build planner (startup plan §31.1): discern intent from a free prompt + honestly check feasibility
against what we can actually build and run. Hermetic — uses the heuristic and a MockLLM, never the network."""

import unittest

from virturoid.services.intent_planner import BuildPlan, plan_build


class HeuristicRoutingTests(unittest.TestCase):
    def test_frog_maze_is_routed_legged_and_flagged_not_buildable(self):
        # the exact failure case: keyword composer built an ARM; the planner must read it as legged +
        # navigation AND flag the real gaps (maze task, frog morphology) instead of mis-building.
        plan = plan_build("a frog-like robot to solve and run through a maze", llm=None)
        self.assertEqual(plan.robot_class, "quadruped")        # nearest buildable to a frog
        self.assertFalse(plan.buildable)                       # honest: legged navigation isn't wired yet
        # the maze TASK is supported now (solve_maze); the honest remaining gap is cross-morphology —
        # navigation is wired for a mobile base, not a legged body (a real gap, flagged not mis-built).
        self.assertTrue(plan.gaps, "an infeasible plan must carry an honest gap")
        self.assertTrue(any("navigation" in g or "quadruped" in g for g in plan.gaps), plan.gaps)

    def test_sort_blocks_is_buildable_manipulator(self):
        plan = plan_build("sort red and blue blocks into matching bins", llm=None)
        self.assertEqual(plan.robot_class, "manipulator")
        self.assertEqual(plan.task_family, "pick_place_sort")
        self.assertTrue(plan.buildable, plan.gaps)

    def test_mobile_navigation_is_buildable(self):
        plan = plan_build("a mobile rover that delivers parts to a goal", llm=None)
        self.assertEqual(plan.robot_class, "mobile_base")
        self.assertTrue(plan.buildable, plan.gaps)

    def test_cloth_task_flagged_as_needing_deformable_backend(self):
        plan = plan_build("a humanoid that folds and irons clothes", llm=None)
        self.assertFalse(plan.buildable)
        self.assertTrue(any("deformable" in g or "Isaac" in g for g in plan.gaps))

    def test_spray_routes_to_coverage(self):
        plan = plan_build("spray paint a panel with 0.7 m reach", llm=None)
        self.assertEqual(plan.task_family, "spray_coverage")
        self.assertTrue(plan.buildable, plan.gaps)

    def test_unrecognised_prompt_requires_clarification_not_a_silent_default(self):
        plan = plan_build("build a blorptron", llm=None)
        # A nearest class remains available to repair/inspection tools, but it is not
        # reported as a buildable request until the person specifies their intent.
        self.assertEqual(plan.routing_confidence, "uncertain")
        self.assertFalse(plan.buildable)
        self.assertTrue(any("clarify" in gap.lower() for gap in plan.gaps), plan.gaps)
        self.assertFalse(plan.to_dict()["buildable"])


class LLMPlannerTests(unittest.TestCase):
    def test_llm_proposal_is_used_but_feasibility_is_ours(self):
        from virturoid.services.llm_client import MockLLM

        # The LLM optimistically proposes a quadruped maze-runner; our registry still flags the gap.
        llm = MockLLM(fixed={"robot_class": "quadruped", "task_family": "navigation",
                             "environment": "maze", "morphology": "frog-like hopper",
                             "objects": ["walls"], "action_verbs": ["hop", "navigate"],
                             "reasoning": "a maze needs a mobile, legged body"})
        plan = plan_build("frog robot maze runner", llm=llm)
        self.assertEqual(plan.source, "llm")
        self.assertEqual(plan.robot_class, "quadruped")
        self.assertFalse(plan.buildable)                       # registry disposes, not the LLM
        # honest gap is cross-morphology (navigation wired for a mobile base, not a legged one), not the maze task
        self.assertTrue(plan.gaps, "an infeasible plan must carry an honest gap")
        self.assertTrue(any("navigation" in g or "quadruped" in g for g in plan.gaps), plan.gaps)

    def test_bad_llm_output_falls_back_to_heuristic(self):
        from virturoid.services.llm_client import MockLLM

        plan = plan_build("a tabletop arm that grasps a box", llm=MockLLM(fixed={"junk": 1}))
        self.assertIsInstance(plan, BuildPlan)
        self.assertEqual(plan.robot_class, "manipulator")      # heuristic recovered it
        self.assertEqual(plan.source, "heuristic")

    def test_uncertain_llm_route_is_also_blocked(self):
        from virturoid.services.llm_client import MockLLM

        plan = plan_build(
            "build a blorptron",
            llm=MockLLM(fixed={"robot_class": "quadruped", "task_family": "locomotion",
                               "morphology": "generic walker", "routing_confidence": "uncertain"}),
        )
        self.assertFalse(plan.buildable)
        self.assertTrue(any("clarify" in gap.lower() for gap in plan.gaps), plan.gaps)


if __name__ == "__main__":
    unittest.main()
