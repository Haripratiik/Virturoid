"""Inbound PolicyImporter, Tiers P0/P1 (Input Ingestion Plan, Phase 4) — no user code is ever executed."""

import unittest

from virturoid.schemas.policy_import import (
    ControllerInterfaceSpec,
    PolicyFramework,
    PolicyImportSpec,
    PolicyTier,
)
from virturoid.services.policy_importer import (
    check_action_dim,
    negotiate_observations,
    sniff_policy_metadata,
    static_parse_python,
)

SAMPLE_CONTROLLER = '''
import numpy as np
import torch

JOINT_COMMAND_TOPIC = "/joint_command"
action_dim = 6

def act(obs):
    rgb = obs["rgb"]
    depth = obs["depth"]
    return np.zeros(action_dim)
'''


class SniffMetadataTests(unittest.TestCase):
    def test_extension_frameworks(self):
        self.assertEqual(sniff_policy_metadata("model.onnx").framework, PolicyFramework.ONNX)
        self.assertEqual(sniff_policy_metadata("weights.npz").framework, PolicyFramework.NUMPY_NPZ)
        self.assertEqual(sniff_policy_metadata("checkpoint.pt").framework, PolicyFramework.TORCH)
        self.assertEqual(sniff_policy_metadata("controller.yaml").framework, PolicyFramework.YAML_CONFIG)
        self.assertEqual(sniff_policy_metadata("policy.py").framework, PolicyFramework.PYTHON_SCRIPT)
        self.assertEqual(sniff_policy_metadata("weights.bin").framework, PolicyFramework.UNKNOWN)

    def test_tier_and_status(self):
        spec = sniff_policy_metadata("model.onnx")
        self.assertEqual(spec.tier, PolicyTier.P0_METADATA)
        self.assertEqual(spec.validation_status, "metadata_only")
        self.assertTrue(spec.validate().ok)


class StaticParseTests(unittest.TestCase):
    def test_extracts_full_interface(self):
        spec = static_parse_python(SAMPLE_CONTROLLER, source_ref="ctrl.py")
        self.assertEqual(spec.tier, PolicyTier.P1_STATIC_PARSE)
        self.assertEqual(spec.entrypoint, "act")
        self.assertIn("numpy", spec.dependencies)
        self.assertIn("torch", spec.dependencies)
        self.assertEqual(spec.framework, PolicyFramework.TORCH)
        self.assertEqual(spec.action_dim, 6)
        self.assertIn("/joint_command", spec.ros_topics)
        self.assertIn("rgb", spec.expected_inputs)
        self.assertIn("depth", spec.expected_inputs)
        self.assertTrue(spec.validate().ok)

    def test_syntax_error_is_rejected_not_executed(self):
        spec = static_parse_python("def broken(:\n    pass\n", source_ref="bad.py")
        self.assertEqual(spec.validation_status, "rejected")
        self.assertTrue(spec.warnings)

    def test_zeros_hint_when_no_explicit_dim(self):
        spec = static_parse_python("import numpy as np\ndef policy(o):\n    return np.zeros(28)\n")
        self.assertEqual(spec.action_dim, 28)
        self.assertEqual(spec.entrypoint, "policy")


class NegotiationTests(unittest.TestCase):
    def test_action_dim_mismatch_flagged(self):
        spec = static_parse_python("action_dim = 30\ndef act(o):\n    return o\n")
        self.assertTrue(check_action_dim(spec, 28))   # 30 vs 28 -> problem (the plan's example)
        self.assertFalse(check_action_dim(spec, 30))  # matches -> no problem

    def test_unknown_action_dim_flagged(self):
        spec = static_parse_python("def act(o):\n    return o\n")
        self.assertTrue(check_action_dim(spec, 28))

    def test_missing_observations_reported(self):
        spec = static_parse_python("def act(obs):\n    return obs['wrist_cam']\n")
        missing = negotiate_observations(spec, ["rgb", "depth"])
        self.assertTrue(any("wrist_cam" in message for message in missing))
        self.assertFalse(negotiate_observations(spec, ["wrist_cam", "rgb"]))


class SchemaTests(unittest.TestCase):
    def test_policy_import_requires_source(self):
        self.assertFalse(PolicyImportSpec(id="p1").validate().ok)

    def test_controller_interface_frequency_positive(self):
        self.assertFalse(ControllerInterfaceSpec(id="c1", control_frequency_hz=0).validate().ok)
        self.assertTrue(ControllerInterfaceSpec(id="c1", control_frequency_hz=50.0).validate().ok)


if __name__ == "__main__":
    unittest.main()
