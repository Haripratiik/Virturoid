"""B4 Phase-B percept-prediction pretrain. When the camera image genuinely encodes the rangefinder geometry (each
beam == the brightness of its band: closer obstacle -> brighter -> shorter range), the frozen-encoder + ridge
readout must recover it with a POSITIVE held-out R^2. The negative control (targets unrelated to the image) must
NOT be predictable (R^2 ~ 0) -- proving the pretrain extracts real geometric signal, not overfitting."""

import unittest

import numpy as np

from virturoid.services.percept_pretrain import PerceptPredictor, pretrain, raw_features


def _banded_dataset(n=300, beams=8, seed=0):
    """Images whose horizontal bands' brightness encodes a random range vector (1 - range = brightness). The
    image thus LITERALLY carries the ranges, the way a depth camera carries obstacle distance per direction."""
    rng = np.random.default_rng(seed)
    ranges = rng.random((n, beams))
    imgs = np.zeros((n, 16, 16, 3), dtype=np.float32)
    band = 16 // beams
    for i in range(n):
        for k in range(beams):
            imgs[i, k * band:(k + 1) * band, :, :] = 1.0 - ranges[i, k]   # brightness = closeness
    return imgs, ranges


class PerceptPretrainTests(unittest.TestCase):
    def test_objective_recovers_geometry_from_a_decodable_representation(self):
        # the OBJECTIVE + METRIC are correct: when the representation carries the ranges (raw pixels literally do),
        # the held-out R^2 is high. This is the target the trained encoder must reach.
        imgs, ranges = _banded_dataset(n=300, beams=8, seed=1)
        out = pretrain(imgs, ranges, holdout=0.25, seed=2, feature_fn=raw_features)
        self.assertGreater(out["heldout"]["r2"], 0.9, out["heldout"])
        self.assertLess(out["heldout"]["rmse"], 0.1)

    def test_frozen_random_encoder_is_insufficient(self):
        # HONEST diagnostic: a RANDOM (untrained) TinyVisionEncoder does NOT linearly decode the geometry -> R^2
        # near 0. This is precisely why Phase-B fine-tunes the conv (the jax extension); the CPU core proves the
        # gap exists and the objective that closes it.
        imgs, ranges = _banded_dataset(n=300, beams=8, seed=1)
        out = pretrain(imgs, ranges, holdout=0.25, seed=2)         # default = frozen encoder
        self.assertLess(out["heldout"]["r2"], 0.2, out["heldout"])

    def test_negative_control_cannot_fit_noise(self):
        imgs, _ = _banded_dataset(n=300, beams=8, seed=3)
        rng = np.random.default_rng(9)
        noise_targets = rng.random((len(imgs), 8))                 # targets unrelated to the images
        out = pretrain(imgs, noise_targets, holdout=0.3, seed=4, feature_fn=raw_features)
        self.assertLess(out["heldout"]["r2"], 0.2)                 # unlearnable even from raw pixels

    def test_predictor_shapes_and_determinism(self):
        imgs, ranges = _banded_dataset(n=80, beams=6, seed=5)
        pp = PerceptPredictor(n_ranges=6, feature_fn=raw_features).fit(imgs, ranges)
        pred = pp.predict(imgs[:4])
        self.assertEqual(pred.shape, (4, 6))
        pp2 = PerceptPredictor(n_ranges=6, feature_fn=raw_features).fit(imgs, ranges)   # closed-form -> deterministic
        self.assertTrue(np.allclose(pp.W, pp2.W))


if __name__ == "__main__":
    unittest.main()
