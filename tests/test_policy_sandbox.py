"""PolicyImporter Tier P2: sandboxed policy execution (Input plan Phase 4 + safety contract).

The first tier that runs user code — so it must stay fail-closed and isolated. Verifies the acceptance test
(one obs -> action, dim-correct, finite, within limits) AND the rejections (bad dim, unsafe value, banned import,
torch->P3, crash, timeout). Offline (AGENTS.md).
"""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.policy_sandbox import sandbox_policy_step  # noqa: E402


class PolicySandboxTests(unittest.TestCase):
    def test_valid_numpy_policy_passes(self):
        src = "import numpy as np\ndef act(obs):\n    return np.array(obs)[:2] * 0.1\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[1.0, 2.0, 3.0], action_dim=2)
        self.assertTrue(out["ran"])
        self.assertEqual(out["validation_status"], "sandbox_passed")
        self.assertTrue(out["action_dim_ok"])
        self.assertTrue(out["finite"])
        self.assertEqual(out["action_len"], 2)

    def test_action_dim_mismatch_rejected(self):
        src = "def act(obs):\n    return [0.1, 0.2]\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0], action_dim=3)
        self.assertEqual(out["validation_status"], "rejected")
        self.assertFalse(out["action_dim_ok"])
        self.assertIn("2 actions", out["reason"])
        self.assertIn("3 actuators", out["reason"])

    def test_safety_limit_violation_rejected(self):
        src = "def act(obs):\n    return [5.0]\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0], action_dim=1,
                                  safety_limits={"j1/position": {"min": "-1.0", "max": "1.0"}})
        self.assertEqual(out["validation_status"], "rejected")
        self.assertFalse(out["within_limits"])
        self.assertTrue(any("outside" in w for w in out["warnings"]))

    def test_within_limits_passes(self):
        src = "def act(obs):\n    return [0.5]\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0], action_dim=1,
                                  safety_limits={"j1/position": {"min": "-1.0", "max": "1.0"}})
        self.assertEqual(out["validation_status"], "sandbox_passed")
        self.assertTrue(out["within_limits"])

    def test_non_finite_action_rejected(self):
        src = "def act(obs):\n    return [float('inf')]\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0])
        self.assertFalse(out["finite"])
        self.assertEqual(out["validation_status"], "rejected")

    def test_banned_import_rejected_by_ast_gate(self):
        src = "import os\ndef act(obs):\n    return [0.1]\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0])
        self.assertEqual(out["validation_status"], "rejected")
        self.assertIn("AST gate", out["reason"])

    def test_torch_routed_to_native_adapter(self):
        src = "import torch\ndef act(obs):\n    return [0.1]\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0])
        self.assertEqual(out["validation_status"], "rejected")
        self.assertTrue(any("P3 native adapter" in w for w in out["warnings"]))

    def test_crash_is_caught(self):
        src = "def act(obs):\n    raise ValueError('boom')\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0])
        self.assertEqual(out["validation_status"], "rejected")
        self.assertIn("crashed", out["reason"])

    def test_infinite_loop_times_out(self):
        src = "def act(obs):\n    while True:\n        pass\n"
        out = sandbox_policy_step(src, entrypoint="act", observation=[0], timeout=2.0)
        self.assertEqual(out["validation_status"], "rejected")
        self.assertIn("timed out", out["reason"])


if __name__ == "__main__":
    unittest.main()
