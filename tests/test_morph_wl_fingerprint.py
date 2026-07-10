"""The WL fingerprint must add TOPOLOGY the aggregate counts miss: a serial snake, a star-shaped spider and a
branched quadruped must get distinguishable fingerprints, and the fingerprint must be deterministic + fixed-length."""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class WLFingerprintTests(unittest.TestCase):
    def test_deterministic_fixed_length_normalized(self):
        from virturoid.services.morph_wl_fingerprint import wl_fingerprint
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a small quadruped robot dog")
        a = wl_fingerprint(g)
        b = wl_fingerprint(g)
        self.assertEqual(a, b)                                    # deterministic
        self.assertEqual(len(a), 32)
        self.assertAlmostEqual(sum(x * x for x in a) ** 0.5, 1.0, places=5)   # L2-normalized

    def test_topology_is_discriminated(self):
        from virturoid.services.morph_wl_fingerprint import wl_fingerprint
        from virturoid.services.morphology_composer import compose_robot

        def cos(p, q):
            return sum(x * y for x, y in zip(p, q))

        quad_a = wl_fingerprint(compose_robot("a small quadruped robot dog"))
        quad_b = wl_fingerprint(compose_robot("a medium quadruped robot"))
        snake = wl_fingerprint(compose_robot("a snake robot that slithers"))
        spider = wl_fingerprint(compose_robot("an eight-legged spider robot"))
        # two quadrupeds share topology -> more similar than a quad is to a serial snake
        self.assertGreater(cos(quad_a, quad_b), cos(quad_a, snake))
        # the snake (deep serial chain) is topologically distinct from the star-shaped spider
        self.assertLess(cos(snake, spider), 0.999)


if __name__ == "__main__":
    unittest.main()
