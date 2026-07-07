"""Perception / observation schemas (Training Improvement Plan, Phase 0) — validation + serialization."""

import json
import unittest

from virturoid.schemas.observation_contract import (
    ObservationContract,
    PerceptionRung,
    PerceptionTrainingPlan,
    SensorFailureRecord,
    SensorNoiseProfile,
)


class SensorNoiseProfileTests(unittest.TestCase):
    def test_requires_modalities(self):
        self.assertFalse(SensorNoiseProfile(id="n1").validate().ok)

    def test_dropout_probability_bounds(self):
        issues = SensorNoiseProfile(id="n1", modalities=["depth"], dropout_probability=1.5).validate().issues
        self.assertTrue(any(i.code == "dropout_range" for i in issues))

    def test_inverted_range_flagged(self):
        issues = SensorNoiseProfile(id="n1", modalities=["range"], min_range_m=5.0, max_range_m=1.0).validate().issues
        self.assertTrue(any(i.code == "range_inverted" for i in issues))

    def test_good_profile_ok(self):
        profile = SensorNoiseProfile(id="n1", modalities=["depth"], dropout_probability=0.1,
                                     min_range_m=0.2, max_range_m=8.0, seed=7)
        self.assertTrue(profile.validate().ok)


class ObservationContractTests(unittest.TestCase):
    def _contract(self, **overrides):
        base = dict(id="oc1", task_graph_id="t1", scene_set_id="s1", robot_genome_id="g1",
                    policy_observation_keys=["rgb"])
        base.update(overrides)
        return ObservationContract(**base)

    def test_requires_core_refs_and_policy_keys(self):
        self.assertFalse(ObservationContract(id="oc1").validate().ok)
        result = ObservationContract(id="oc1", task_graph_id="t1", scene_set_id="s1",
                                     robot_genome_id="g1").validate()
        self.assertTrue(any(i.field == "policy_observation_keys" for i in result.issues))

    def test_invalid_leakage_policy(self):
        issues = self._contract(leakage_policy="loose").validate().issues
        self.assertTrue(any(i.code == "invalid_leakage_policy" for i in issues))

    def test_good_contract_ok_and_serializes_rung(self):
        contract = self._contract(perception_rung=PerceptionRung.RUNG_2_NATIVE_SENSORS)
        self.assertTrue(contract.validate().ok)
        data = json.loads(json.dumps(contract.to_dict()))
        self.assertEqual(data["perception_rung"], "rung_2_native_sensors")


class PerceptionTrainingPlanTests(unittest.TestCase):
    def test_requires_contract_ref(self):
        self.assertFalse(PerceptionTrainingPlan(id="p1").validate().ok)
        self.assertTrue(PerceptionTrainingPlan(id="p1", observation_contract_ref="oc1").validate().ok)


class SensorFailureRecordTests(unittest.TestCase):
    def test_requires_type_and_sensor(self):
        self.assertFalse(SensorFailureRecord(id="f1").validate().ok)
        result = SensorFailureRecord(id="f1", failure_type="bad_depth_at_grasp").validate()
        self.assertTrue(any(i.field == "sensor_ref" for i in result.issues))

    def test_known_type_ok_unknown_only_warns(self):
        known = SensorFailureRecord(id="f1", failure_type="bad_depth_at_grasp", sensor_ref="wrist_rgbd")
        self.assertTrue(known.validate().ok)
        unknown = SensorFailureRecord(id="f2", failure_type="banana_peel", sensor_ref="wrist_rgbd")
        self.assertTrue(unknown.validate().ok)  # unknown type is a warning, not an error
        self.assertTrue(any(i.code == "unknown_failure_type" for i in unknown.validate().issues))


if __name__ == "__main__":
    unittest.main()
