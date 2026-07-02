"""H1 — the Diagnosis Artifact: structured, prompt-ready eval reads with gate margins + actionable failure
modes. The headline test is the live hexapod case (real gait, wrong direction) — it must be named, not buried.
Pure-Python, no MuJoCo."""

import unittest

from virturoid.services.diagnosis_artifact import build_diagnosis_artifact


class LocomotionArtifactTests(unittest.TestCase):
    def test_backward_gait_is_named_not_read_as_weak_forward(self):
        # tonight's bilateral hexapod: cadence 10, upright, survived, but forward -0.43 m
        art = build_diagnosis_artifact({"forward": -0.431, "cadence": 10.0, "upright_frac": 0.86,
                                        "survived": True}, task_type="locomotion")
        self.assertEqual(art["verdict"], "fail")
        self.assertEqual(art["failure_mode"], "walks_backward")     # NOT "weak_forward"/"shuffle"
        self.assertIn("BACKWARD", art["explanation"])
        self.assertIn("parity", art["explanation"].lower())          # points at G4
        # the actionable next step is flip-direction, explicitly NOT redesign-the-body
        self.assertTrue(any("calf_phase" in a or "velocity" in a for a in art["next_actions"]))
        self.assertIn("Do NOT redesign the body", art["explanation"])

    def test_real_walk_passes(self):
        art = build_diagnosis_artifact({"forward": 0.55, "cadence": 8.0, "upright_frac": 0.95, "survived": True})
        self.assertEqual(art["verdict"], "pass")
        self.assertEqual(art["failure_mode"], "walking")

    def test_fall_is_fatal_mode(self):
        art = build_diagnosis_artifact({"forward": 0.2, "cadence": 5.0, "upright_frac": 0.3, "survived": False})
        self.assertEqual(art["failure_mode"], "fell")

    def test_shuffle_vs_weak_forward(self):
        shuffle = build_diagnosis_artifact({"forward": 0.15, "cadence": 1.0, "upright_frac": 0.9, "survived": True})
        self.assertEqual(shuffle["failure_mode"], "shuffle")
        weak = build_diagnosis_artifact({"forward": 0.2, "cadence": 6.0, "upright_frac": 0.9, "survived": True})
        self.assertEqual(weak["failure_mode"], "weak_forward")

    def test_gate_report_has_signed_margins(self):
        art = build_diagnosis_artifact({"forward": 0.5, "cadence": 8.0, "upright_frac": 0.9, "survived": True})
        fwd = next(r for r in art["gate_report"] if r["gate"] == "forward_m")
        self.assertAlmostEqual(fwd["margin"], 0.5 - 0.30, places=4)   # value - threshold
        self.assertTrue(fwd["pass"])
        self.assertIn("VERDICT", art["summary_text"])
        self.assertIn("forward_m", art["summary_text"])

    def test_reads_gait_diagnostics_keys(self):
        # the richer gait_diagnostics dict uses different field names — must be read transparently
        art = build_diagnosis_artifact({"forward_m": -0.4, "cadence_steps_per_s": 10.0, "upright_mean": 0.8,
                                        "fell": False}, task_type="locomotion")
        self.assertEqual(art["failure_mode"], "walks_backward")

    def test_trend_vs_history(self):
        h = [{"forward_m": 0.2}]
        art = build_diagnosis_artifact({"forward": 0.35, "cadence": 6.0, "upright_frac": 0.9, "survived": True},
                                       history=h)
        self.assertEqual(art["trend"]["direction"], "up")
        self.assertAlmostEqual(art["trend"]["delta_vs_prev"], 0.15, places=4)

    def test_ws3_support_gate_only_when_task_opts_in(self):
        # WS3: a many-legged task adds a ``support`` gate. A flat slide (low support_frac) MISSES it even though
        # upright/cadence pass; a proper tripod (high support) passes. A task WITHOUT the gate ignores support.
        gates = {"forward_m": 0.4, "cadence": 3.0, "upright": 0.6, "support": 0.3}
        slide = build_diagnosis_artifact({"forward": 0.5, "cadence": 4.0, "upright_frac": 0.9,
                                          "support_frac": 0.05, "survived": True}, gates=gates)
        self.assertFalse(next(r for r in slide["gate_report"] if r["gate"] == "support")["pass"])
        self.assertEqual(slide["verdict"], "fail")             # a low, all-feet-planted slide is not a walk
        step = build_diagnosis_artifact({"forward": 0.5, "cadence": 4.0, "upright_frac": 0.9,
                                         "support_frac": 0.6, "survived": True}, gates=gates)
        self.assertEqual(step["verdict"], "pass")              # real stepping support -> passes
        nogate = build_diagnosis_artifact({"forward": 0.5, "cadence": 4.0, "upright_frac": 0.9, "survived": True})
        self.assertFalse(any(r["gate"] == "support" for r in nogate["gate_report"]))   # quad L1 untouched
        self.assertEqual(nogate["verdict"], "pass")


class ManipulationArtifactTests(unittest.TestCase):
    def test_miss_vs_slip_vs_unreachable(self):
        miss = build_diagnosis_artifact({"success_rate": 0.1, "contacted": False}, task_type="grasp")
        self.assertEqual(miss["failure_mode"], "miss")
        slip = build_diagnosis_artifact({"success_rate": 0.2, "contacted": True, "lifted": False}, task_type="grasp")
        self.assertEqual(slip["failure_mode"], "slip")
        unr = build_diagnosis_artifact({"success_rate": 0.0, "reached": False}, task_type="grasp")
        self.assertEqual(unr["failure_mode"], "unreachable")

    def test_grasped_passes(self):
        art = build_diagnosis_artifact({"success_rate": 0.85}, task_type="grasp")
        self.assertEqual(art["verdict"], "pass")
        self.assertEqual(art["failure_mode"], "grasped")


if __name__ == "__main__":
    unittest.main()
