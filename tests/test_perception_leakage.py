"""Privileged-state leakage gate + rung report (Training Improvement Plan, Phase 0).

The plan requires rejecting a deployable training plan that lets the policy cheat with privileged simulator
state, and names the concrete case: ``policy_trainer`` still feeds privileged ``target_block_xy`` to the policy.
"""

import unittest

from virturoid.schemas.observation_contract import ObservationContract, PerceptionRung
from virturoid.services.perception_leakage import (
    check_privileged_leakage,
    format_report,
    select_perception_rung,
    training_plan_report,
)


def clean_contract(**overrides) -> ObservationContract:
    base = dict(
        id="oc_clean", task_graph_id="t1", scene_set_id="s1", robot_genome_id="g1",
        policy_observation_keys=["rgb", "depth", "joint_state", "gripper_contact"],
        required_modalities=["rgb", "depth", "joint_state", "contact"],
        deploy_modalities=["rgb", "depth", "joint_state", "contact"],
        train_scene_seeds=[1, 2, 3], heldout_scene_seeds=[10, 11],
        randomization_logged=True, perception_rung=PerceptionRung.RUNG_2_NATIVE_SENSORS,
    )
    base.update(overrides)
    return ObservationContract(**base)


class LeakageGateTests(unittest.TestCase):
    def test_clean_sensor_only_contract_passes(self):
        self.assertTrue(check_privileged_leakage(clean_contract()).ok)

    def test_target_block_xy_is_flagged(self):
        result = check_privileged_leakage(clean_contract(policy_observation_keys=["rgb", "target_block_xy"]))
        self.assertFalse(result.ok)
        self.assertTrue(any(i.code == "privileged_state_in_policy" for i in result.issues))

    def test_object_pose_and_truth_markers_flagged(self):
        for key in ("object_pose", "gt_goal", "true_contact", "oracle_state"):
            result = check_privileged_leakage(clean_contract(policy_observation_keys=["rgb", key]))
            self.assertFalse(result.ok, key)

    def test_privileged_label_used_as_policy_input(self):
        contract = clean_contract(policy_observation_keys=["object_pose", "rgb"],
                                  privileged_label_keys=["object_pose"])
        codes = {i.code for i in check_privileged_leakage(contract).issues}
        self.assertIn("privileged_label_in_policy", codes)

    def test_segmentation_leak_when_deploy_is_rgb_only(self):
        contract = clean_contract(policy_observation_keys=["rgb", "segmentation_mask"],
                                  deploy_modalities=["rgb", "depth"])
        self.assertTrue(any(i.code == "segmentation_leak" for i in check_privileged_leakage(contract).issues))
        # ...but segmentation IS allowed when the deployed robot actually has it.
        ok_contract = clean_contract(policy_observation_keys=["rgb", "segmentation_mask"],
                                     deploy_modalities=["rgb", "segmentation"])
        self.assertFalse(any(i.code == "segmentation_leak" for i in check_privileged_leakage(ok_contract).issues))

    def test_train_heldout_seed_overlap(self):
        contract = clean_contract(train_scene_seeds=[1, 2, 3], heldout_scene_seeds=[3, 4])
        self.assertTrue(any(i.code == "seed_overlap" for i in check_privileged_leakage(contract).issues))

    def test_unlogged_randomization_blocks_under_strict_but_warns_under_permissive(self):
        strict = check_privileged_leakage(clean_contract(randomization_logged=False, leakage_policy="strict"))
        self.assertFalse(strict.ok)
        self.assertTrue(any(i.code == "randomization_unlogged" and i.severity == "error" for i in strict.issues))
        permissive = check_privileged_leakage(clean_contract(randomization_logged=False, leakage_policy="permissive"))
        self.assertTrue(permissive.ok)
        self.assertTrue(any(i.code == "randomization_unlogged" and i.severity == "warning"
                            for i in permissive.issues))


class RungAndReportTests(unittest.TestCase):
    def test_rung_selection_from_modalities(self):
        self.assertEqual(select_perception_rung(clean_contract(required_modalities=["rgb"])),
                         PerceptionRung.RUNG_2_NATIVE_SENSORS)
        self.assertEqual(select_perception_rung(clean_contract(required_modalities=["range", "joint_state"])),
                         PerceptionRung.RUNG_1_SYNTHETIC_ADAPTER)
        self.assertEqual(select_perception_rung(clean_contract(required_modalities=[])),
                         PerceptionRung.RUNG_0_PRIVILEGED)

    def test_report_passes_and_surfaces_demos(self):
        report = training_plan_report(clean_contract())
        self.assertTrue(report["leakage"]["ok"])
        self.assertTrue(report["visible_demos"])  # existing range/vision demos are visible
        self.assertTrue(report["recommendation"].startswith("PASS"))
        self.assertIn("leakage gate", format_report(report))

    def test_report_blocks_on_leak(self):
        report = training_plan_report(clean_contract(policy_observation_keys=["target_block_xy"]))
        self.assertFalse(report["leakage"]["ok"])
        self.assertTrue(report["recommendation"].startswith("BLOCKED"))
        self.assertTrue(report["leakage"]["errors"])


if __name__ == "__main__":
    unittest.main()
