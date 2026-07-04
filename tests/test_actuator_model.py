"""B1 real-actuator model: the four-quadrant torque-speed clamp must (a) allow full torque near zero speed,
(b) taper to zero driving torque at the no-load speed, (c) always allow braking torque, (d) be numpy==jax."""

import unittest

import numpy as np

from virturoid.services.actuator_model import available_torque, clamp_torque, knee_speed


class ActuatorModelTests(unittest.TestCase):
    def test_torque_speed_envelope(self):
        tau_max, qd_knee, qd_max = 4.0, 5.0, 10.0
        # near zero speed -> full torque; at knee -> full torque; past knee -> tapered; at/over no-load -> 0
        self.assertAlmostEqual(float(available_torque(0.0, tau_max, qd_knee, qd_max, xp=np)), 4.0, places=5)
        self.assertAlmostEqual(float(available_torque(5.0, tau_max, qd_knee, qd_max, xp=np)), 4.0, places=5)
        self.assertAlmostEqual(float(available_torque(7.5, tau_max, qd_knee, qd_max, xp=np)), 2.0, places=5)
        self.assertAlmostEqual(float(available_torque(10.0, tau_max, qd_knee, qd_max, xp=np)), 0.0, places=5)
        self.assertAlmostEqual(float(available_torque(99.0, tau_max, qd_knee, qd_max, xp=np)), 0.0, places=5)

    def test_four_quadrant_clamp(self):
        tau_max, qd_knee, qd_max = 4.0, 5.0, 10.0
        # DRIVING at high speed: a peak-torque command is throttled to the tapered envelope (the transfer killer)
        self.assertAlmostEqual(float(clamp_torque(4.0, 7.5, tau_max, qd_knee, qd_max, xp=np)), 2.0, places=5)
        # BRAKING at the same high speed (torque opposes motion): full tau_max still available
        self.assertAlmostEqual(float(clamp_torque(-4.0, 7.5, tau_max, qd_knee, qd_max, xp=np)), -4.0, places=5)
        # low speed: unthrottled either way
        self.assertAlmostEqual(float(clamp_torque(3.0, 0.5, tau_max, qd_knee, qd_max, xp=np)), 3.0, places=5)

    def test_per_joint_vectorized(self):
        tau = np.array([4.0, 4.0, 4.0]); qv = np.array([0.0, 7.5, 10.0])
        tmax = np.array([4.0, 4.0, 4.0]); knee = np.array([5.0, 5.0, 5.0]); qmax = np.array([10.0, 10.0, 10.0])
        out = clamp_torque(tau, qv, tmax, knee, qmax, xp=np)
        self.assertTrue(np.allclose(out, [4.0, 2.0, 0.0], atol=1e-5))

    def test_numpy_jax_parity(self):
        try:
            import jax.numpy as jnp
        except Exception:  # noqa: BLE001
            self.skipTest("jax not installed")
        rng = np.random.default_rng(0)
        tau = rng.normal(0, 5, 8); qv = rng.normal(0, 8, 8)
        a = clamp_torque(tau, qv, 4.0, 5.0, 10.0, xp=np)
        b = clamp_torque(jnp.asarray(tau), jnp.asarray(qv), 4.0, 5.0, 10.0, xp=jnp)
        self.assertTrue(np.allclose(a, np.asarray(b), atol=1e-5))
        self.assertAlmostEqual(knee_speed(10.0), 5.0)


if __name__ == "__main__":
    unittest.main()
