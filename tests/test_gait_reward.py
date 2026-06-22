"""Gait-reward DESIGN guard (the anti-Goodhart property).

The whole point of the locomotion reward is that a real, deliberate WALK must out-score a foot-dragging SLIDE, a
micro-lift SHUFFLE, and a STAND — otherwise PPO converges to the slide that games forward velocity (the failure the
user caught by eye: "slid at 0.55 m/s, cadence 0"). This test does NOT run MJX/PPO; it replicates the recipe step
reward from `scripts/mjx_morph_attention.py` (the RECIPE branch of `roll_chunk.body`, including the canonical
legged_gym `feet_air_time` term ported per docs/virturoid_research_dossier.md) on synthetic foot/base trajectories and
asserts the ordering holds. If you change the reward in the trainer, mirror it here.

Key claim under test: the `feet_air_time` term (rewarding committed SWING TIME, paid on touchdown) is what separates a
real step from a micro-lift that games the instantaneous clearance reward — so adding it must WIDEN the walk-vs-shuffle
margin. That is the dossier's "single most important anti-slide term", verified at the reward-design level.
"""

import unittest

import numpy as np

# default weights — mirror mjx_morph_attention.py argparse defaults (RECIPE reward)
DEFAULTS = dict(prog_w=4.0, clear_w=2.5, swing_w=0.0, slip_w=0.0, alt_w=0.0, smooth_w=0.3,
                air_w=2.0, air_tgt=0.3, vz_w=0.4, wxy_w=0.04, vtgt=0.3)
DT = 0.005
FOOT_Z0 = 0.05
Z_STAND = 0.30


def recipe_reward(traj, w=DEFAULTS, dt=DT, foot_z0=FOOT_Z0, z_stand=Z_STAND):
    """Sum the RECIPE step reward over a trajectory. `traj` keys are per-step arrays:
    fz (T, n_feet) foot heights, z (T,) base height, upr (T,), fwd (T,), vz (T,), wxy (T, 2), off (T, n_act).
    Mirrors scripts/mjx_morph_attention.py roll_chunk.body (recipe branch) EXACTLY, including feet_air_time."""
    fz = np.asarray(traj["fz"], float)
    T, n_feet = fz.shape
    z, upr, fwd = (np.asarray(traj[k], float) for k in ("z", "upr", "fwd"))
    vz = np.asarray(traj.get("vz", np.zeros(T)), float)
    wxy = np.asarray(traj.get("wxy", np.zeros((T, 2))), float)
    off = np.asarray(traj.get("off", np.zeros((T, n_feet))), float)
    air_time = np.zeros(n_feet)
    last_contact = np.ones(n_feet)        # feet start grounded
    a_prev = off[0]
    total = 0.0
    for t in range(T):
        progress = np.clip(fwd[t], 0.0, w["vtgt"])
        up = float(z[t] > 0.7 * z_stand)
        sm = float(np.mean((off[t] - a_prev) ** 2))
        lift = np.clip(fz[t] - foot_z0, 0.0, 0.08)
        clearance = float(np.mean(lift))
        swing = float(np.max(np.clip(fz[t] - foot_z0, 0.0, 0.12)))
        stance = (fz[t] < foot_z0 + 0.02).astype(float)
        n_stance = stance.sum()
        both_planted = float(n_stance >= n_feet)
        swing_phase = float((n_stance > 0.5) and (n_stance < n_feet - 0.5))
        slip = both_planted * abs(fwd[t])
        # feet_air_time (legged_gym): reward swing time, paid on touchdown, target air_tgt
        contact = stance
        contact_filt = np.maximum(contact, last_contact)
        first_contact = (air_time > 0.0).astype(float) * contact_filt
        air_acc = air_time + dt
        air_reward = float(np.sum((air_acc - w["air_tgt"]) * first_contact))
        air_time = air_acc * (1.0 - contact_filt)
        last_contact = contact
        stab = w["vz_w"] * vz[t] ** 2 + w["wxy_w"] * float(np.sum(wxy[t] ** 2))
        step_r = max(0.0, w["prog_w"] * progress * up + 0.2 * up + 0.15 * max(0.0, upr[t])
                     + w["clear_w"] * clearance * up + w["swing_w"] * swing * up + w["alt_w"] * swing_phase * up
                     + w["air_w"] * air_reward * up
                     - w["slip_w"] * slip - w["smooth_w"] * sm - stab)
        total += step_r
        a_prev = off[t]
    return total


def _const(T, n_feet, *, fwd, lift_feet=None):
    fz = np.full((T, n_feet), FOOT_Z0)
    if lift_feet is not None:
        fz = lift_feet
    return dict(fz=fz, z=np.full(T, Z_STAND), upr=np.ones(T), fwd=np.full(T, fwd),
                vz=np.full(T, 0.01), wxy=np.full((T, 2), 0.01), off=np.zeros((T, n_feet)))


def _walk(T=400, n_feet=4, swing_steps=80, lift=0.06):
    """Diagonal trot: pairs (0,3) and (1,2) alternate swing; committed ~0.4 s swings that fully clear the ground."""
    fz = np.full((T, n_feet), FOOT_Z0)
    period = 2 * swing_steps
    for t in range(T):
        phase = (t % period) < swing_steps
        swinging = (0, 3) if phase else (1, 2)
        for f in swinging:
            fz[t, f] = FOOT_Z0 + lift          # airborne (> foot_z0 + 0.02)
    tr = _const(T, n_feet, fwd=0.3)
    tr["fz"] = fz
    return tr


def _slide(T=400, n_feet=4):
    """Feet never leave the ground; body translates forward at target speed — the gameable slide."""
    return _const(T, n_feet, fwd=0.3)


def _shuffle(T=400, n_feet=4, on=2, off_=2, lift=0.03):
    """Micro-lifts: feet hop up for ~2 steps then down — gains a little instantaneous clearance but the swing TIME
    is tiny, so feet_air_time barely fires (and is penalized below target). The clearance-gaming failure mode."""
    fz = np.full((T, n_feet), FOOT_Z0)
    period = on + off_
    for t in range(T):
        if (t % period) < on:
            fz[t, :] = FOOT_Z0 + lift          # barely airborne, very briefly
    tr = _const(T, n_feet, fwd=0.15)
    tr["fz"] = fz
    return tr


def _stand(T=400, n_feet=4):
    return _const(T, n_feet, fwd=0.0)


class GaitRewardOrderingTests(unittest.TestCase):
    def test_walk_beats_slide_shuffle_and_stand(self):
        walk, slide, shuffle, stand = (recipe_reward(f()) for f in (_walk, _slide, _shuffle, _stand))
        self.assertGreater(walk, slide, f"a real walk must out-score the gameable slide (walk {walk:.1f} vs slide {slide:.1f})")
        self.assertGreater(walk, shuffle, f"a real walk must out-score a micro-lift shuffle (walk {walk:.1f} vs shuffle {shuffle:.1f})")
        self.assertGreater(walk, stand, f"a walk must out-score standing still (walk {walk:.1f} vs stand {stand:.1f})")
        self.assertGreater(slide, stand, "forward progress is still rewarded over standing (sanity)")

    def test_feet_air_time_widens_the_walk_vs_shuffle_margin(self):
        # The dossier's core claim: feet_air_time is what distinguishes a committed step from a clearance-gaming
        # micro-lift. Turning it OFF must SHRINK the walk-vs-shuffle gap; turning it ON must widen it.
        with_air = dict(DEFAULTS)
        without_air = dict(DEFAULTS, air_w=0.0)
        gap_with = recipe_reward(_walk(), with_air) - recipe_reward(_shuffle(), with_air)
        gap_without = recipe_reward(_walk(), without_air) - recipe_reward(_shuffle(), without_air)
        self.assertGreater(gap_with, gap_without,
                           f"feet_air_time must widen the walk-vs-shuffle margin (with {gap_with:.1f} vs without {gap_without:.1f})")

    def test_a_fallen_body_earns_nothing_after_the_up_gate(self):
        # If the body is below the upright gate (fallen/crouched), the positive terms are gated off -> ~0 reward,
        # which is what the alive-mask relies on. A collapsed trajectory must not out-score a walk.
        fallen = _walk()
        fallen["z"] = np.full(fallen["fz"].shape[0], 0.3 * Z_STAND)   # below the 0.7*z_stand up-gate
        self.assertLess(recipe_reward(fallen), recipe_reward(_walk()))


if __name__ == "__main__":
    unittest.main()
