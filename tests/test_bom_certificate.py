"""B1 executable-on-BOM certificate (wedge #2). The pure grader must (a) CERTIFY a physically-modest trajectory
(low torque at low speed) on all six datasheet gates, (b) FAIL the torque-speed envelope gate (G4) when the
motion commands peak torque at high joint speed -- the exact operating point a constant-clamp sim hides and real
servos cannot reach, (c) report every gate as an auditable ratio, and (d) mark an empty (fell-immediately) trace
UNCERTIFIABLE rather than silently passing."""

import unittest

import numpy as np

from virturoid.services.bom_certificate import grade_actuation
from virturoid.services.component_catalog import select_actuator
from virturoid.services.actuator_model import knee_speed


class BomCertificateTests(unittest.TestCase):
    def test_feasible_trajectory_certifies(self):
        # a small servo (peak ~1.9 Nm), operated well within its envelope: modest torque, well below knee speed.
        # pass clamps so the grader is pinned to THIS shipped servo (the real path certify() uses)
        servo = select_actuator(1.0)
        clamps = np.full(3, servo.peak_torque_nm)
        tau = np.full((200, 3), 0.4 * servo.peak_torque_nm)             # 40% peak, constant, all joints
        qv = np.full((200, 3), 0.2 * servo.max_speed_radps)            # 20% no-load, below the knee -> full torque
        cert = grade_actuation(tau, qv, clamps=clamps)
        self.assertTrue(cert["pass"], cert["summary"])
        self.assertEqual(cert["n_gates_pass"], 6)
        # G4 envelope violation is exactly zero when everything is below the knee
        g4 = next(g for g in cert["gates"] if g["name"] == "G4_torque_speed_envelope")
        self.assertEqual(g4["measured"], 0.0)

    def test_peak_torque_at_high_speed_fails_envelope(self):
        # command near-peak torque WHILE spinning past the knee toward no-load speed: impossible on real hardware.
        # clamps pins the SHIPPED servo (else speed-aware demand sizing would just buy a faster motor)
        servo = select_actuator(1.0)
        pk, qd = servo.peak_torque_nm, servo.max_speed_radps
        knee = knee_speed(qd)
        clamps = np.full(2, pk)
        # DRIVING quadrant (torque & speed same sign), speed at 90% no-load where the envelope has collapsed to ~10%
        tau = np.full((200, 2), 0.9 * pk)
        qv = np.full((200, 2), 0.9 * qd)
        self.assertGreater(0.9 * qd, knee)                             # we really are past the knee
        cert = grade_actuation(tau, qv, clamps=clamps, envelope_tol=0.02)
        g4 = next(g for g in cert["gates"] if g["name"] == "G4_torque_speed_envelope")
        self.assertFalse(g4["passed"])
        self.assertGreater(g4["measured"], 0.5)                        # a large fraction of samples are infeasible
        self.assertFalse(cert["pass"])

    def test_braking_torque_is_always_feasible(self):
        # same high speed + high torque, but BRAKING (opposite signs): a motor can always resist -> G4 must pass
        servo = select_actuator(1.0)
        pk, qd = servo.peak_torque_nm, servo.max_speed_radps
        clamps = np.full(2, pk)
        tau = np.full((200, 2), -0.9 * pk)                            # torque opposes...
        qv = np.full((200, 2), 0.9 * qd)                             # ...the motion
        cert = grade_actuation(tau, qv, clamps=clamps)
        g4 = next(g for g in cert["gates"] if g["name"] == "G4_torque_speed_envelope")
        self.assertEqual(g4["measured"], 0.0)

    def test_gates_are_ratios_and_bom_is_real(self):
        tau = np.full((100, 4), 0.5); qv = np.full((100, 4), 0.3)
        cert = grade_actuation(tau, qv)
        self.assertEqual(len(cert["gates"]), 6)
        for g in cert["gates"]:
            self.assertIn("measured", g); self.assertIn("limit", g)
            self.assertIsInstance(g["measured"], float)
        # BOM lists real catalog parts with a price and vendor, one line per SKU, qty summing to n_joints
        self.assertTrue(cert["bom"])
        self.assertEqual(sum(b["qty"] for b in cert["bom"]), 4)
        for b in cert["bom"]:
            self.assertGreater(b["price_usd"], 0.0)
            self.assertTrue(b["vendor"])

    def test_speed_aware_selection_picks_a_faster_motor(self):
        # a real servo cannot be sized on torque alone: a fast joint needs a higher no-load speed even at equal
        # torque. Requiring a high speed must select a motor with a strictly higher max_speed than the torque-only
        # pick (or fall back to the highest-capability motor).
        torque_only = select_actuator(2.0)
        speed_needed = select_actuator(2.0, required_speed_radps=torque_only.max_speed_radps + 5.0)
        self.assertGreater(speed_needed.max_speed_radps, torque_only.max_speed_radps)

    def test_empty_trace_is_uncertifiable(self):
        from virturoid.services.bom_certificate import certify_policy_on_bom

        class _Fell:  # a stand-in gene whose rollout would yield no trace is simulated via monkeypatch below
            pass
        # directly exercise the grader's contract: certify_policy_on_bom returns pass=False when no trace survives.
        # Simulate by patching recipe_rollout_morph to always return an empty trace.
        import virturoid.services.bom_certificate as bc
        orig = bc.recipe_rollout_morph
        try:
            bc.recipe_rollout_morph = lambda *a, **k: {"tau_cmd": None, "qvel": None, "alive": 0, "survived": False}
            cert = certify_policy_on_bom(_Fell(), None, steps=10, n_seeds=2)
        finally:
            bc.recipe_rollout_morph = orig
        self.assertFalse(cert["pass"])
        self.assertIn("UNCERTIFIABLE", cert["summary"])


if __name__ == "__main__":
    unittest.main()
