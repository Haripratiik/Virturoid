"""P5 — the verdict certificate is the moat's honesty travelling with the export ('arrives in Isaac already
verified'). It must faithfully reflect the physics verdict (a CREDIBLE walk certifies credible; a FELL does not)
and never invent a pass. Pure formatting over a measured verdict, so testable without a rollout."""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class VerdictCertificateTests(unittest.TestCase):
    def _gene(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a small quadruped robot dog")

    def test_credible_walk_certifies_credible_with_checks(self):
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CREDIBLE (trot, 0.83 m)", "survived": True, "forward_m": 0.83,
             "cadence": 2.1, "height_ratio": 0.9, "gait_source": "flywheel_hint"}
        cert = build_certificate(self._gene(), v, task="walk forward", robot_id="r1")
        self.assertEqual(cert["artifact"], "virturoid_verification_certificate")
        self.assertTrue(cert["credible"])
        self.assertEqual(cert["gait_source"], "flywheel_hint")
        self.assertEqual(cert["checks"]["forward_m"], 0.83)
        self.assertIn("deploy==measure", cert["verified_with"])
        self.assertIn("integrator", cert["disclaimer"])          # honest sim-to-real caveat, not a deploy claim

    def test_a_fall_is_never_certified_credible(self):
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "FELL (toppled at 0.3 s)", "survived": False, "forward_m": 0.02, "height_ratio": 0.15}
        cert = build_certificate(self._gene(), v, robot_id="r2")
        self.assertFalse(cert["credible"])                       # un-gameable: a fall cannot be certified a walk
        self.assertEqual(cert["verdict"], "FELL (toppled at 0.3 s)")

    def test_certificate_is_a_registered_export_format(self):
        from virturoid.services.agent_design_tools import _EXPORT_FORMATS
        self.assertIn("certificate", _EXPORT_FORMATS)


if __name__ == "__main__":
    unittest.main()
