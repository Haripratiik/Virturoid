"""Gait-QUALITY diagnostics + the AI gait-critic — the AI-assisted training brain that distinguishes WALKING
from a forward slide/drag (forward velocity alone is a gameable proxy). These tests pin: (a) a dragging gait
(feet planted, moving forward) reads is_walking=False with cadence~0, while an alternating-stepping gait reads
is_walking=True; (b) a fall caps quality; (c) the critic is offline-safe and clamps proposals to safe bounds."""

import math
import unittest

import numpy as np

from virturoid.services.gait_diagnostics import diagnose_gait
from virturoid.services.gait_critic import GAIT_REWARD_KNOBS, critique_gait


def _base_traj(T, *, z=0.5, vx=0.4, dt=0.01):
    """A (T, nq) qpos with a free base (x,y,z, wxyz quat) + 2 dummy joints; upright, moving forward at vx."""
    q = np.zeros((T, 9))
    q[:, 0] = vx * dt * np.arange(T)        # x advances
    q[:, 2] = z                              # constant height (upright, doesn't fall)
    q[:, 3] = 1.0                            # quat w=1 -> upright
    return q


class GaitDiagnosticsTests(unittest.TestCase):
    def test_drag_is_not_walking(self):
        T, dt = 400, 0.01
        q = _base_traj(T, dt=dt)
        foot_z = np.full((T, 2), 0.02)       # both feet PLANTED the whole time (never lift)
        r = diagnose_gait(q, foot_z, 0, dt, foot_z0=np.array([0.02, 0.02]))
        self.assertFalse(r["is_walking"])
        self.assertEqual(r["n_steps"], 0)
        self.assertAlmostEqual(r["cadence_steps_per_s"], 0.0)
        self.assertAlmostEqual(r["mean_duty_factor"], 1.0)
        self.assertEqual(r["max_foot_clearance_m"], 0.0)
        self.assertIn("NOT", r["summary"].upper())

    def test_alternating_steps_is_walking(self):
        T, dt, P = 600, 0.01, 40
        q = _base_traj(T, dt=dt)
        z0 = 0.02
        f0 = z0 + 0.06 * np.clip(np.sin(2 * math.pi * np.arange(T) / P), 0, 1)          # foot0 swings
        f1 = z0 + 0.06 * np.clip(np.sin(2 * math.pi * np.arange(T) / P + math.pi), 0, 1)  # foot1 anti-phase
        foot_z = np.stack([f0, f1], axis=1)
        r = diagnose_gait(q, foot_z, 0, dt, foot_z0=np.array([z0, z0]))
        self.assertTrue(r["is_walking"], r["summary"])
        self.assertGreater(r["cadence_steps_per_s"], 0.6)
        self.assertGreater(r["alternation"], 0.3)            # feet alternate (anti-phase)
        self.assertGreater(r["max_foot_clearance_m"], 0.02)  # feet actually lift

    def test_fall_caps_quality(self):
        T, dt = 300, 0.01
        q = _base_traj(T, dt=dt)
        q[150:, 2] = 0.1                     # base height collapses partway through
        foot_z = np.full((T, 2), 0.02)
        r = diagnose_gait(q, foot_z, 0, dt, foot_z0=np.array([0.02, 0.02]), stand_z=0.5)
        self.assertTrue(r["fell"])
        self.assertLess(r["gait_quality"], 0.3)

    def test_lunge_then_collapse_is_not_walking(self):
        """Regression for the bug a reviewer caught by EYE: a crouch-lunge that travels a bit, then sinks to
        half-height and FREEZES, with flailing feet faking cadence — the old metric (fall referenced to the
        trajectory MEAN) called it is_walking=True. The fix references the STANDING height + sustained progress."""
        T, dt = 1100, 0.01
        q = _base_traj(T, dt=dt, z=0.5)
        # sink from 0.5 -> 0.15 over the first 300 steps, then hold low (collapsed); x lunges to 0.3 then freezes
        for t in range(T):
            frac = min(t / 300.0, 1.0)
            q[t, 2] = 0.5 - 0.35 * frac                 # height collapses to 0.15
            q[t, 0] = 0.30 * frac                       # forward lunge that then freezes at 0.30 m
        # feet flail (fake cadence/clearance) — must NOT be mistaken for stepping while the body is collapsed
        f0 = 0.02 + 0.05 * np.abs(np.sin(np.arange(T) / 7.0))
        foot_z = np.stack([f0, f0[::-1]], axis=1)
        r = diagnose_gait(q, foot_z, 0, dt, foot_z0=np.array([0.02, 0.02]), stand_z=0.5)
        self.assertFalse(r["is_walking"])               # the headline bug: this must read NOT walking
        self.assertTrue(r["fell"])                      # collapse detected via standing-height reference
        self.assertLess(r["height_held"], 0.5)          # spends most of the episode below standing height
        self.assertLess(r["gait_quality"], 0.3)


class _MockLLM:
    """Returns deliberately OUT-OF-BOUNDS weights to test clamping + a valid-shape response."""
    name = "mock"

    def complete_json(self, system, user, schema, max_tokens=700):
        return {"failure_diagnosis": "drag", "key_change": "slip", "rationale": "r",
                "weights": {"vtgt": 99.0, "clear_w": -5.0, "swing_w": 4.0, "slip_w": 7.5,
                            "alt_w": 3.0, "smooth_w": 0.5}}


class GaitCriticTests(unittest.TestCase):
    def test_offline_safe_returns_none(self):
        self.assertIsNone(critique_gait({"is_walking": False}, {}, None))

    def test_clamps_proposals_to_bounds(self):
        out = critique_gait({"is_walking": False, "cadence_steps_per_s": 0.0}, {}, _MockLLM())
        self.assertTrue(out["valid"])
        w = out["weights"]
        for k, (lo, hi, _d) in GAIT_REWARD_KNOBS.items():
            self.assertGreaterEqual(w[k], lo)
            self.assertLessEqual(w[k], hi)
        self.assertEqual(w["vtgt"], GAIT_REWARD_KNOBS["vtgt"][1])   # 99 -> clamped to max
        self.assertEqual(w["clear_w"], GAIT_REWARD_KNOBS["clear_w"][0])  # -5 -> clamped to min


if __name__ == "__main__":
    unittest.main()
