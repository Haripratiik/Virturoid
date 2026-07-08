"""PolicyImporter Tier P3: native ONNX adapter (Input plan Phase 4).

Builds a tiny ONNX policy (torch export, controlled weights so the action is deterministic) and checks the P3
acceptance contract: run one inference, validate dimension/finiteness/limits. Offline (AGENTS.md).
"""
import importlib.util
import os
import tempfile
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

_ORT = importlib.util.find_spec("onnxruntime") is not None
_TORCH = importlib.util.find_spec("torch") is not None


def _export_linear_onnx(path, in_dim=6, out_dim=3, bias_value=0.5):
    import torch
    import torch.nn as nn

    lin = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        lin.weight.zero_()                                      # zero weights -> action == bias (deterministic)
        lin.bias.fill_(bias_value)
    lin.eval()
    dummy = torch.zeros(1, in_dim)
    torch.onnx.export(lin, dummy, path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}})


@unittest.skipUnless(_ORT and _TORCH, "needs onnxruntime + torch to build/run an ONNX policy")
class OnnxAdapterTests(unittest.TestCase):
    def _model(self, **kw):
        d = tempfile.mkdtemp(prefix="onnx_")
        p = os.path.join(d, "policy.onnx")
        _export_linear_onnx(p, **kw)
        return p

    def test_inspect_reports_io(self):
        from virturoid.services.policy_native_adapter import inspect_onnx
        info = inspect_onnx(self._model())
        self.assertTrue(info["ok"])
        self.assertEqual(info["inputs"][0]["name"], "obs")
        self.assertEqual(info["outputs"][0]["name"], "action")

    def test_valid_policy_passes(self):
        from virturoid.services.policy_native_adapter import run_onnx_policy
        out = run_onnx_policy(self._model(out_dim=3, bias_value=0.2),
                              observation=[0.0] * 6, action_dim=3)
        self.assertTrue(out["ran"])
        self.assertEqual(out["validation_status"], "native_validated")
        self.assertEqual(out["action_len"], 3)
        self.assertTrue(out["finite"])
        for a in out["action"]:
            self.assertAlmostEqual(a, 0.2, places=4)            # zero weights -> action == bias

    def test_action_dim_mismatch_rejected(self):
        from virturoid.services.policy_native_adapter import run_onnx_policy
        out = run_onnx_policy(self._model(out_dim=3), observation=[0.0] * 6, action_dim=2)
        self.assertEqual(out["validation_status"], "rejected")
        self.assertFalse(out["action_dim_ok"])
        self.assertIn("3 actions", out["reason"])

    def test_safety_limit_violation_rejected(self):
        from virturoid.services.policy_native_adapter import run_onnx_policy
        out = run_onnx_policy(self._model(out_dim=2, bias_value=5.0), observation=[0.0] * 6, action_dim=2,
                              safety_limits={"j1": {"min": "-1", "max": "1"}, "j2": {"min": "-1", "max": "1"}})
        self.assertEqual(out["validation_status"], "rejected")
        self.assertFalse(out["within_limits"])

    def test_bad_path_is_rejected_not_fatal(self):
        from virturoid.services.policy_native_adapter import run_onnx_policy
        out = run_onnx_policy("/nonexistent/model.onnx", observation=[0.0] * 6)
        self.assertEqual(out["validation_status"], "rejected")
        self.assertIn("could not load", out["reason"])


if __name__ == "__main__":
    unittest.main()
