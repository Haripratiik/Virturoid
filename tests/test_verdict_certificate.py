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

    #: what the export writer stamps on the gene after it has MEASURED the control program it wrote. The
    #: operating point defaults to the SHIPPED DEFAULT crawl point, so it matches a ``default_crawl`` verdict.
    def _shipped(self, law="crawl_wave_gait", fwd=0.71, **op):
        return {"policy_type": law, "entrypoint": "software/gait_controller.py",
                "parameters_file": "software/control_program.json", "program_fingerprint": "abc123",
                "control_frequency_hz": 50.0, "pd_gains": {"kp": 32.0, "kd": 1.5},
                "frequency_hz": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5,
                "verified_walk": True, "sim_forward_m": fwd, "sim_verdict": "CREDIBLE WALK", **op}

    def test_credible_walk_certifies_credible_with_checks(self):
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CREDIBLE (trot, 0.83 m)", "survived": True, "forward_m": 0.83,
             "cadence": 2.1, "height_ratio": 0.9, "gait_source": "default_crawl"}
        cert = build_certificate(self._gene(), v, task="walk forward", robot_id="r1",
                                 shipped_controller=self._shipped())
        self.assertEqual(cert["artifact"], "virturoid_verification_certificate")
        self.assertTrue(cert["credible"])
        self.assertEqual(cert["gait_source"], "default_crawl")
        self.assertEqual(cert["checks"]["forward_m"], 0.83)
        self.assertIn("deploy==measure", cert["verified_with"])
        self.assertIn("integrator", cert["disclaimer"])          # honest sim-to-real caveat, not a deploy claim

    # ------------------------------------------------------------------ the OPERATING POINT is part of "same"
    def test_same_control_law_at_a_different_operating_point_is_a_different_controller(self):
        """MEASURED on a generated quadruped: the held body deploys freq 2.056 / hip 0.746 / knee 0.770 while the
        exported package ships freq 2.55 / hip 0.9 / knee 1.0, because ``extract_crawl_gait_params`` runs its OWN
        search at export time. Same law, different machine — and matching only the law would have called it
        deploy==measure."""
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CREDIBLE WALK", "survived": True, "forward_m": 0.67, "gait_source": "default_crawl"}
        cert = build_certificate(self._gene(), v, robot_id="r7",
                                 shipped_controller=self._shipped(frequency_hz=2.55))   # default is 1.5
        self.assertTrue(cert["controller_parity"]["control_law_match"])
        self.assertIs(cert["controller_parity"]["operating_point_match"], False)
        self.assertIs(cert["controller_parity"]["same"], False)
        self.assertIs(cert["deploy_is_measure"], False)
        self.assertNotIn("deploy==measure", cert["verified_with"])
        self.assertIn("DIFFERENT OPERATING POINT", cert["controller_parity"]["reason"])

    def test_an_unrecoverable_operating_point_reads_unknown_not_matching(self):
        """A mined ``flywheel_hint`` leaves no record of the parameters it deployed, so the two cannot be
        compared. Unknown, not equal."""
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CREDIBLE WALK", "survived": True, "forward_m": 0.7, "gait_source": "flywheel_hint"}
        cert = build_certificate(self._gene(), v, robot_id="r8", shipped_controller=self._shipped())
        self.assertTrue(cert["controller_parity"]["control_law_match"])
        self.assertIsNone(cert["controller_parity"]["operating_point_match"])
        self.assertIsNone(cert["controller_parity"]["same"])
        self.assertIsNone(cert["deploy_is_measure"])
        self.assertNotIn("deploy==measure", cert["verified_with"])

    # ------------------------------------------------------------------ deploy==measure is about a ROLLOUT
    # A rollout is a body AND a controller. `body_parity` has been checked since an exported Go2 quoted a body
    # 8 kg lighter; the controller half went unasked, and on a real Menagerie Go2 the answer was no. These three
    # pin the tri-state so a package can never again print the claim over two different machines.
    def test_deploy_is_measure_needs_the_shipped_controller_too(self):
        """The signed rollout ran a crawl gait; the package deploys a trot CPG. Not the same rollout."""
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CROUCH (low/unstable stance)", "survived": True, "forward_m": 0.119,
             "cadence": 10.0, "gait_source": "default_crawl"}
        cert = build_certificate(self._gene(), v, robot_id="r3",
                                 shipped_controller=self._shipped(law="trot_cpg_gait", fwd=0.311))
        self.assertIs(cert["deploy_is_measure"], False)
        self.assertEqual(cert["deploy_is_measure_parts"], {"same_body": True, "same_controller": False})
        self.assertIs(cert["controller_parity"]["same"], False)
        self.assertNotIn("deploy==measure", cert["verified_with"])
        self.assertIn("THE CONTROLLER IS NOT THE ONE THIS PACKAGE DEPLOYS", cert["verified_with"])
        # BOTH numbers survive, each attached to the controller that produced it — the package must not have to
        # choose which measurement to hide.
        self.assertEqual(cert["controller_parity"]["measured_controller"]["forward_m"], 0.119)
        self.assertEqual(cert["controller_parity"]["shipped_controller"]["forward_m"], 0.311)

    def test_unknown_shipped_controller_is_not_a_match(self):
        """No stamp = we did not check. That must read as unknown, never as agreement."""
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CREDIBLE WALK", "survived": True, "forward_m": 0.8, "gait_source": "default_crawl"}
        cert = build_certificate(self._gene(), v, robot_id="r4")            # no shipped_controller
        self.assertIsNone(cert["deploy_is_measure"])
        self.assertIsNone(cert["controller_parity"]["same"])
        self.assertIsNone(cert["controller_parity"]["control_law_match"])
        self.assertNotIn("deploy==measure", cert["verified_with"])
        self.assertIn("NO DEPLOYABLE CONTROL PROGRAM IS RECORDED", cert["verified_with"])

    def test_a_learned_policy_verdict_is_not_the_scripted_program_the_package_ships(self):
        from virturoid.services.verdict_certificate import build_certificate
        v = {"verdict": "CREDIBLE WALK", "survived": True, "forward_m": 1.2, "gait_source": "learned_policy"}
        cert = build_certificate(self._gene(), v, robot_id="r5",
                                 shipped_controller=self._shipped(law="crawl_wave_gait"))
        self.assertIs(cert["controller_parity"]["same"], False)
        self.assertIs(cert["deploy_is_measure"], False)

    def test_never_measured_still_beats_the_controller_check(self):
        """No rollout at all -> None, and the controller half must not resurrect a claim."""
        from virturoid.services.verdict_certificate import build_certificate
        cert = build_certificate(self._gene(), {"verdict": "could not simulate (ValueError)"}, robot_id="r6",
                                 shipped_controller=self._shipped())
        self.assertIsNone(cert["deploy_is_measure"])
        self.assertIs(cert["rollout_ran"], False)
        self.assertNotIn("deploy==measure", cert["verified_with"])
        self.assertIn("NOTHING WAS MEASURED", cert["verified_with"])

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
