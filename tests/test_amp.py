"""AMP core mechanism test (physical-AI keystone). Proves — in pure CPU/jax, the SAME math the GPU trainer runs —
that the adversarial discriminator learns to separate a REFERENCE motion-style (CPG step: alternating contact,
high foot-clearance, planted/low-speed feet) from a POLICY SLIDE style (near-full contact, ~0 clearance, dragging
high-speed feet), and that the resulting style reward genuinely pays reference-like motion more than a slide.
That is exactly the signal reward-shaping could not supply (the slide out-scored stepping under every hand term)."""

import unittest

import numpy as np

from virturoid.services.amp import (STYLE_DIM, discriminator, init_discriminator, lsgan_loss, style_reward)

# style = [ fwd, upright, mean_clearance, contact_frac, grounded_foot_speed, vert_vel ]
_REF = np.array([0.30, 0.90, 0.050, 0.50, 0.02, 0.0])   # CPG STEP: alternating contact, feet clear + planted
_SLIDE = np.array([0.30, 0.90, 0.005, 0.95, 0.30, 0.0])  # SLIDE: feet down + dragging, no clearance (same fwd!)


class AmpCoreTests(unittest.TestCase):
    def test_numpy_jax_parity(self):
        try:
            import jax.numpy as jnp
        except Exception:  # noqa: BLE001
            self.skipTest("jax not installed")
        rng = np.random.default_rng(1)
        D = init_discriminator(rng)
        phi = rng.normal(0, 1, (5, STYLE_DIM))
        d_np = discriminator(D, phi, xp=np)
        d_j = discriminator({k: jnp.asarray(v) for k, v in D.items()}, jnp.asarray(phi), xp=jnp)
        self.assertTrue(np.allclose(d_np, np.asarray(d_j), atol=1e-5))   # CPU test == GPU math

    def test_discriminator_learns_to_reward_stepping_over_sliding(self):
        try:
            import jax
            import jax.numpy as jnp
        except Exception:  # noqa: BLE001
            self.skipTest("jax not installed")
        rng = np.random.default_rng(0)
        D = {k: jnp.asarray(v) for k, v in init_discriminator(rng).items()}

        def ref(n):
            return jnp.asarray(_REF + rng.normal(0, 0.03, (n, STYLE_DIM)))

        def pol(n):
            return jnp.asarray(_SLIDE + rng.normal(0, 0.03, (n, STYLE_DIM)))

        loss_fn = lambda Dp, r, p: lsgan_loss(Dp, r, p, xp=jnp)   # noqa: E731
        for _ in range(400):                                     # adversarial fit (manual SGD; no optax dep)
            g = jax.grad(loss_fn)(D, ref(64), pol(64))
            D = {k: D[k] - 3e-2 * g[k] for k in D}

        d_ref = float(jnp.mean(discriminator(D, ref(256), xp=jnp)))
        d_pol = float(jnp.mean(discriminator(D, pol(256), xp=jnp)))
        sr_ref = float(jnp.mean(style_reward(D, ref(256), xp=jnp)))
        sr_pol = float(jnp.mean(style_reward(D, pol(256), xp=jnp)))
        self.assertGreater(d_ref, d_pol + 0.5, f"D failed to separate step({d_ref:.2f}) from slide({d_pol:.2f})")
        self.assertGreater(sr_ref, sr_pol + 0.1,
                           f"style reward must pay stepping({sr_ref:.2f}) > sliding({sr_pol:.2f})")


if __name__ == "__main__":
    unittest.main()
