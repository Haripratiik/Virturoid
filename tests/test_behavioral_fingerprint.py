"""The behavioral fingerprint (z_dyn) embeds a body by its RESPONSE — so it must be deterministic, fixed-order,
robust to un-probeable bodies (zeros, never a crash), and it must actually TRACK the physics: bodies with
different leg lengths get different resonance/stance features (the signal geometry can't express)."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "probes run short MuJoCo rollouts")
class BehavioralFingerprintTests(unittest.TestCase):
    def test_deterministic_fixed_order_and_cached(self):
        from virturoid.services.behavioral_fingerprint import DYN_FEATURE_NAMES, z_dyn
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped robot")
        a = z_dyn(g)
        b = z_dyn(g)                                             # second call hits the cache
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(DYN_FEATURE_NAMES))

    def test_tracks_the_physics_across_leg_scales(self):
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.behavioral_fingerprint import dyn_fingerprint
        from virturoid.services.morphology_composer import compose_robot
        base = compose_robot("a quadruped robot")

        def scaled(ls, vid):
            g = RobotGene.from_dict(base.to_dict())
            g.id = vid
            for s in g.segments:
                if s.name.startswith("leg") and (s.name.endswith("_1") or s.name.endswith("_2")):
                    s.length_m = round(s.length_m * ls, 4)
            return g

        short = dyn_fingerprint(scaled(0.5, "short"))
        tall = dyn_fingerprint(scaled(2.0, "tall"))
        # taller legs -> taller stance; and the fingerprints must NOT be identical (they carry dynamic signal)
        self.assertGreater(tall["stance_h"], short["stance_h"])
        self.assertNotEqual([short[k] for k in short], [tall[k] for k in tall])

    def test_unprobeable_body_yields_zeros_not_a_crash(self):
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.behavioral_fingerprint import DYN_FEATURE_NAMES, z_dyn
        broken = RobotGene(id="x", species="x", robot_class="x", segments=[])
        self.assertEqual(z_dyn(broken), [0.0] * len(DYN_FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
