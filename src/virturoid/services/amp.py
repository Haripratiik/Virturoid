"""Adversarial Motion Priors (AMP) — the physical-AI control keystone (docs/physical_ai_control_research.md).

The measured failure: locomotion control learned by reward-shaping is reward-HACKED (the policy slides instead of
stepping — 4 reward configs + a slide-proof --wtw-w term all cap out). The fix (Peng 2021; Walk-These-Ways;
Siekmann): stop scoring the trajectory with a hand-authored scalar and instead score it by SIMILARITY to a
distribution of GOOD trajectories (the CPG reference, which we generate for free for any body). A discriminator D
is trained adversarially to tell reference motion from policy motion; the policy is rewarded for FOOLING it. A
slide's motion-style is far from the CPG's high-cadence stepping, so D penalizes it automatically — no clearance
term to hand-tune.

MORPHOLOGY-INVARIANT by construction: D operates on a GLOBAL motion-STYLE vector (not per-joint), so ONE
discriminator + ONE CPG reference generalize across quadrupeds, hexapods, bipeds. These functions are
``xp``-generic (numpy or jax.numpy) exactly like ``morph_forward.attention_forward``, so the CPU test proves the
same math the GPU trainer runs.
"""

from __future__ import annotations

# Global motion-STYLE features (body-size-invariant). The trainer assembles these from the sim state each step:
#   [ base_fwd_vel, upright, mean_foot_clearance, foot_contact_frac, mean_grounded_foot_speed, base_vert_vel ].
# A SLIDE and a STEP separate cleanly here: a slide has ~full contact_frac, ~0 clearance, high grounded-foot-speed;
# the CPG step has ~0.5 contact_frac (alternating), high clearance, ~0 grounded-foot-speed.
STYLE_DIM = 6


def init_discriminator(rng, *, style_dim: int = STYLE_DIM, hidden: int = 64, scale: float = 0.3) -> dict:
    """A small 2-layer MLP discriminator D: (style_dim,) -> scalar logit. ``rng`` is a numpy Generator (CPU) or a
    jax PRNGKey handled by the trainer's own init; here we accept a numpy Generator for portability/tests."""
    import numpy as np
    return {
        "W1": np.asarray(rng.normal(0, scale, (style_dim, hidden)), dtype=float),
        "b1": np.zeros(hidden),
        "W2": np.asarray(rng.normal(0, scale, (hidden, 1)), dtype=float),
        "b2": np.zeros(1),
    }


def discriminator(D: dict, phi, *, xp):
    """D(phi) -> logit. ``phi`` is (..., style_dim); returns (...,). LSGAN convention: reference ≈ +1, policy ≈ -1."""
    h = xp.tanh(phi @ D["W1"] + D["b1"])
    return (h @ D["W2"] + D["b2"])[..., 0]


def style_reward(D: dict, phi, *, xp):
    """AMP style reward (Peng 2021): r_style = max(0, 1 - 0.25 (D(phi) - 1)^2). Peaks when D thinks phi is
    reference-like (D≈1); 0 once the motion looks clearly non-reference. This REPLACES hand-crafted gait terms."""
    d = discriminator(D, phi, xp=xp)
    return xp.maximum(0.0, 1.0 - 0.25 * (d - 1.0) ** 2)


def lsgan_loss(D: dict, phi_ref, phi_pol, *, xp):
    """Least-squares GAN discriminator loss (AMP's recipe): push D(reference)->+1 and D(policy)->-1. The gradient
    penalty (GP=10 on reference samples) + spectral norm are added by the trainer via jax.grad; this is the core
    data term, xp-generic so the CPU test exercises the exact objective the GPU minimizes."""
    d_ref = discriminator(D, phi_ref, xp=xp)
    d_pol = discriminator(D, phi_pol, xp=xp)
    return xp.mean((d_ref - 1.0) ** 2) + xp.mean((d_pol + 1.0) ** 2)
