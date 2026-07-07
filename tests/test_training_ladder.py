"""Training ladder planner (dossier R1/R2/R6/R8) — deterministic task->TrainingPlan compilation."""

import unittest

from virturoid.schemas.training_plan import TRAINER_RUNGS, TaskFamily, TrainingPlan
from virturoid.services.training_ladder import (
    classify_task_family,
    format_ladder_report,
    is_manipulation,
    ladder_report,
    plan_training,
)


class ClassifyTaskFamilyTests(unittest.TestCase):
    def test_single_families(self):
        cases = {
            "make the robot walk forward": TaskFamily.LOCOMOTION,
            "navigate around obstacles to the goal": TaskFamily.NAVIGATION,
            "grasp the red cup": TaskFamily.GRASP,
            "pick and place the box in the bin": TaskFamily.PICK_PLACE,
            "sort blocks by color": TaskFamily.SORT,
            "push the block to the target": TaskFamily.PUSH,
            "insert the peg with the tool": TaskFamily.TOOL_USE,
            "inspect the pipe for cracks": TaskFamily.INSPECT,
            "do something vague": TaskFamily.UNKNOWN,
        }
        for text, expected in cases.items():
            self.assertEqual(classify_task_family(text), expected, text)

    def test_mobile_manipulation_is_composite(self):
        self.assertEqual(classify_task_family("walk to the shelf and grasp the box"), TaskFamily.COMPOSITE)

    def test_enum_passthrough(self):
        self.assertEqual(classify_task_family(TaskFamily.PUSH), TaskFamily.PUSH)


class PlanTrainingTests(unittest.TestCase):
    def test_valid_three_phase_plan(self):
        plan = plan_training("grasp the cup", robot_genome_id="g1", task_graph_id="t1",
                             observation_contract_ref="oc1")
        self.assertTrue(plan.validate().ok)
        self.assertEqual(plan.trainer_ladder, list(TRAINER_RUNGS))
        self.assertEqual(plan.task_family, "grasp")
        self.assertTrue(plan.teacher_sources)
        self.assertEqual(plan.observation_contract_ref, "oc1")
        self.assertTrue(plan.banking_rules["require_heldout_eval"])
        self.assertEqual(plan.checkpoint_selection["by"], "deploy_sim")
        self.assertEqual(plan.backend_budget["optimize_rl"], "gpu_mjx_ppo")

    def test_manipulation_leads_with_demonstrations(self):
        self.assertTrue(is_manipulation(TaskFamily.SORT))
        plan = plan_training("sort blocks by color", robot_genome_id="g1", task_graph_id="t1")
        self.assertEqual(plan.teacher_sources[0], "demonstration_amplifier")  # dossier R6

    def test_no_gpu_adds_cpu_fallback(self):
        plan = plan_training("walk", robot_genome_id="g1", task_graph_id="t1", gpu_available=False)
        self.assertIn("cpu_fallback", plan.trainer_ladder)
        self.assertEqual(plan.backend_budget["optimize_rl"], "cpu_es")

    def test_non_deployable_relaxes_gates(self):
        plan = plan_training("reach the target", robot_genome_id="g1", task_graph_id="t1", deployable=False)
        self.assertFalse(plan.banking_rules["require_actuator_cert_if_deployable"])
        self.assertEqual(plan.domain_randomization_profile, {})


class TrainingPlanSchemaTests(unittest.TestCase):
    def test_missing_refs_invalid(self):
        self.assertFalse(TrainingPlan(id="tp1").validate().ok)

    def test_unknown_rung_warns_only(self):
        plan = TrainingPlan(id="tp1", task_family="grasp", robot_genome_id="g1", task_graph_id="t1",
                            trainer_ladder=["reuse_evaluate", "warp_drive"])
        result = plan.validate()
        self.assertTrue(result.ok)  # warning, not error
        self.assertTrue(any(i.code == "unknown_trainer_rung" for i in result.issues))


class ReportTests(unittest.TestCase):
    def test_ladder_report_and_format(self):
        plan = plan_training("pick and place the box", robot_genome_id="g1", task_graph_id="t1")
        report = ladder_report(plan)
        self.assertEqual(report["task_family"], "pick_place")
        self.assertIn("demonstration", " ".join(report["notes"]).lower())
        text = format_ladder_report(report)
        self.assertIn("ladder:", text)
        self.assertIn("reuse_evaluate", text)


if __name__ == "__main__":
    unittest.main()
