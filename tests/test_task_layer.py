"""General task layer: an LLM-proposed, verifier-checked, executor-run TaskSpec over composable skills —
so any prompt maps to a feasibility-checked task instead of a hard-coded handler. Verifier/proposer are
hermetic (no network); the executor uses MuJoCo."""

import importlib.util
import unittest

from virturoid.schemas.task_spec import Entity, Predicate, PredicateOp, SkillCall, TaskSpec
from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
from virturoid.services.task_proposer import _fallback_task, propose_task
from virturoid.services.task_verifier import verify_task

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _gene(robot_class, prompt="x"):
    return compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=prompt, robot_class=robot_class))


class VerifierTests(unittest.TestCase):
    def test_rejects_wheeled_base_asked_to_grasp(self):
        rover = _gene("mobile_base")
        spec = TaskSpec(name="bad", steps=[SkillCall("s0", "grasp_lift")],
                        goal=[Predicate(PredicateOp.OBJECT_LIFTED)])
        v = verify_task(spec, rover)
        self.assertFalse(v["ok"])
        self.assertTrue(any("morphology" in i for i in v["issues"]), v["issues"])

    def test_accepts_rover_navigation(self):
        rover = _gene("mobile_base")
        spec = TaskSpec(name="nav", steps=[SkillCall("s0", "navigate")],
                        goal=[Predicate(PredicateOp.REACHED_GOAL)])
        self.assertTrue(verify_task(rover and spec and spec, rover)["ok"])

    def test_rejects_unknown_skill(self):
        arm = _gene("manipulator", "grasp")
        spec = TaskSpec(steps=[SkillCall("s0", "teleport")], goal=[Predicate(PredicateOp.OBJECT_LIFTED)])
        self.assertFalse(verify_task(spec, arm)["ok"])

    def test_rejects_unreachable_goal(self):
        arm = _gene("manipulator", "grasp")
        # a grasp step can't establish SURFACE_COVERED
        spec = TaskSpec(steps=[SkillCall("s0", "grasp_lift")], goal=[Predicate(PredicateOp.SURFACE_COVERED)])
        self.assertFalse(verify_task(spec, arm)["ok"])


class ProposerTests(unittest.TestCase):
    def test_fallback_routes_by_morphology(self):
        self.assertEqual(_fallback_task("walk around", _gene("quadruped")).steps[0].skill, "locomote")
        self.assertEqual(_fallback_task("drive to goal", _gene("mobile_base")).steps[0].skill, "navigate")
        self.assertEqual(_fallback_task("run through a maze", _gene("mobile_base")).steps[0].skill, "solve_maze")
        self.assertEqual(_fallback_task("grasp and lift a box", _gene("manipulator", "grasp")).steps[0].skill, "grasp_lift")
        self.assertEqual(_fallback_task("sort blocks into bins", _gene("manipulator", "grasp")).steps[0].skill, "pick_place")

    def test_proposer_offline_returns_verifiable_task(self):
        arm = _gene("manipulator", "grasp")
        spec = propose_task("grasp and lift a box", arm, llm=None)
        self.assertTrue(verify_task(spec, arm)["ok"], spec.to_dict())

    def test_llm_infeasible_proposal_falls_back(self):
        from virturoid.services.llm_client import MockLLM
        rover = _gene("mobile_base")
        # LLM optimistically tells a rover to grasp; verifier rejects -> fallback to a feasible nav task
        bad = MockLLM(fixed={"name": "x", "steps": [{"step_id": "s0", "skill": "grasp_lift"}],
                             "goal": [{"op": "object_lifted", "args": []}]})
        spec = propose_task("go somewhere", rover, llm=bad)
        self.assertTrue(verify_task(spec, rover)["ok"])
        self.assertNotEqual(spec.steps[0].skill, "grasp_lift")


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class ExecutorTests(unittest.TestCase):
    def test_executes_locomotion_task(self):
        from virturoid.services.task_executor import run_task
        gene = _gene("quadruped")
        spec = propose_task("walk forward", gene, llm=None)
        r = run_task(spec, gene)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["steps"][0]["skill"], "locomote")
        self.assertEqual(r["goal_total"], 1)

    def test_infeasible_task_is_not_run(self):
        from virturoid.services.task_executor import run_task
        rover = _gene("mobile_base")
        spec = TaskSpec(steps=[SkillCall("s0", "grasp_lift")], goal=[Predicate(PredicateOp.OBJECT_LIFTED)])
        r = run_task(spec, rover)
        self.assertFalse(r["feasible"])
        self.assertTrue(r["issues"])

    def test_evaluate_task_end_to_end(self):
        from virturoid.services.task_executor import evaluate_task
        r = evaluate_task("walk forward", _gene("quadruped"), llm=None)
        self.assertTrue(r["feasible"])
        self.assertEqual(r["steps_planned"], ["locomote"])
        self.assertIn("score", r)


if __name__ == "__main__":
    unittest.main()
