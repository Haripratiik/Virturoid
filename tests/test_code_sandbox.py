"""Plan v2 §4.4 / G2+G9: sandboxed execution of LLM-generated success-detector code. The AST gate + isolated
subprocess stop stray imports, infinite loops, and key exfil; the fail-closed calibration gate rejects a
detector that can't separate known pass/fail episodes."""

import unittest

from virturoid.services.code_sandbox import (SandboxError, calibrate_detector, generate_detector, run_detector,
                                             validate_ast)
from virturoid.services.llm_client import MockLLM

# a well-behaved detector: "success" iff forward travel clears a threshold
_GOOD = "def detect(ep):\n    f = float(ep.get('forward', 0.0))\n    return {'ok': f >= 0.4, 'score': f}\n"


class CodeSandboxTests(unittest.TestCase):
    def test_good_detector_runs_in_subprocess(self):
        self.assertEqual(run_detector(_GOOD, {"forward": 0.6})["ok"], True)
        self.assertEqual(run_detector(_GOOD, {"forward": 0.1})["ok"], False)
        self.assertAlmostEqual(run_detector(_GOOD, {"forward": 0.6})["score"], 0.6)

    def test_ast_gate_rejects_imports_and_dunder_and_builtins(self):
        with self.assertRaises(SandboxError):
            validate_ast("import os\ndef detect(ep):\n    return {'ok': True}\n")     # disallowed import
        with self.assertRaises(SandboxError):
            validate_ast("def detect(ep):\n    return {'ok': open('x')}\n")           # banned builtin
        with self.assertRaises(SandboxError):
            validate_ast("def detect(ep):\n    return {'ok': (1).__class__.__bases__}\n")  # introspection dunder
        validate_ast(_GOOD)                                                            # the good one passes

    def test_infinite_loop_times_out(self):
        loop = "def detect(ep):\n    x = 0\n    while True:\n        x += 1\n    return {'ok': True}\n"
        with self.assertRaises(SandboxError):
            run_detector(loop, {}, timeout=3.0)                                        # killed, not hung forever

    def test_non_finite_score_rejected(self):
        bad = "def detect(ep):\n    return {'ok': True, 'score': float('inf')}\n"
        with self.assertRaises(SandboxError):
            run_detector(bad, {})

    def test_calibration_gate_fails_closed(self):
        pass_fx = [{"forward": 0.6}, {"forward": 0.9}]
        fail_fx = [{"forward": 0.1}, {"forward": -0.2}]
        good = calibrate_detector(_GOOD, pass_fx, fail_fx)
        self.assertTrue(good["trusted"])                                               # separates pass/fail
        # a detector that always says ok=True misclassifies the fails -> NOT trusted
        always = "def detect(ep):\n    return {'ok': True}\n"
        bad = calibrate_detector(always, pass_fx, fail_fx)
        self.assertFalse(bad["trusted"])
        self.assertEqual(bad["fail_ok"], False)


class GenerateDetectorTests(unittest.TestCase):
    _PASS = [{"forward": 0.6}, {"forward": 0.9}]
    _FAIL = [{"forward": 0.1}, {"forward": -0.2}]

    def test_generated_detector_that_calibrates_is_trusted(self):
        llm = MockLLM(fixed={"code": _GOOD, "rationale": "threshold forward at 0.4"})
        r = generate_detector("walk forward >= 0.4 m", self._PASS, self._FAIL, llm)
        self.assertTrue(r["trusted"])
        self.assertIsNotNone(r["code"])

    def test_generated_detector_that_fails_calibration_is_rejected(self):
        # LLM emits an always-True detector -> misclassifies the fails -> NOT trusted, code withheld (fail-closed)
        llm = MockLLM(fixed={"code": "def detect(ep):\n    return {'ok': True}\n"})
        r = generate_detector("walk forward", self._PASS, self._FAIL, llm)
        self.assertFalse(r["trusted"])
        self.assertIsNone(r["code"])

    def test_no_llm_is_not_trusted(self):
        r = generate_detector("walk", self._PASS, self._FAIL, None)
        self.assertFalse(r["trusted"])


if __name__ == "__main__":
    unittest.main()
