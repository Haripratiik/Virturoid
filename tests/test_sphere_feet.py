"""Sphere-feet + feet-only collision transform (breakthrough plan v2 T1.4 / gap N12).

Pins the manifold-invariant foot transform: (a) the foot geoms are found and turned into SPHERES with FEET-ONLY
collision (the floor + non-foot body geoms handled correctly), (b) it is idempotent, (c) the transform is opt-in
(default off -> byte-identical rollout), (d) the PHYSICS still works — a body with sphere feet stands and rolls
out finite (doesn't fall through the floor or eject), and (e) it round-trips through the policy meta so a banked
sphere-feet policy auto-deploys with sphere feet (deploy==train). These are the CPU-side guarantees; the MJX
training compile applies the SAME ``apply_sphere_feet`` before ``mjx.put_model`` so train and deploy match."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a quadruped walking robot", llm=None)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class SphereFeetTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        cls.g = _quad()
        cls.fd = encode_robot(compiled_model(robot_mjcf(cls.g))).feature_dim

    def _model(self):
        # deepcopy: compiled_model returns a SHARED lru_cached MjModel — the direct apply_sphere_feet tests below
        # mutate it, so hand each test a private copy or they'd pollute the cache for the rollout tests.
        import copy
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        return copy.deepcopy(compiled_model(robot_mjcf(self.g)))

    def test_finds_the_feet(self):
        # A parametric quad has 4 feet; the foot-finder should locate exactly the leaf leg tips (>=4, not the torso).
        from virturoid.services.sphere_feet import find_foot_geoms
        feet = find_foot_geoms(self._model())
        self.assertGreaterEqual(len(feet), 4)

    def test_feet_become_spheres_with_feet_only_collision(self):
        import mujoco
        from virturoid.services.sphere_feet import apply_sphere_feet
        m = self._model()
        feet = set(apply_sphere_feet(m))
        self.assertTrue(feet)
        sphere = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        for gi in feet:                                       # every foot is now a sphere with a positive radius
            self.assertEqual(int(m.geom_type[gi]), sphere)
            self.assertGreater(float(m.geom_size[gi][0]), 0.0)
        for gi in range(m.ngeom):                             # feet-only collision: only feet collide among body geoms
            if int(m.geom_bodyid[gi]) == 0:
                continue                                      # (floor/worldbody geoms keep their collision — untouched)
            want = 1 if gi in feet else 0
            self.assertEqual(int(m.geom_contype[gi]), want)
            self.assertEqual(int(m.geom_conaffinity[gi]), want)

    def test_idempotent(self):
        from virturoid.services.sphere_feet import apply_sphere_feet
        m = self._model()
        a = apply_sphere_feet(m)
        b = apply_sphere_feet(m)                              # applying twice changes nothing (already spheres)
        self.assertEqual(sorted(a), sorted(b))

    def test_floor_still_collides(self):
        # The feet-only mask must NOT disable the floor, or the body falls through into the void.
        from virturoid.services.sphere_feet import apply_sphere_feet
        m = self._model()
        apply_sphere_feet(m)
        floor = [gi for gi in range(m.ngeom) if int(m.geom_bodyid[gi]) == 0]
        self.assertTrue(any(int(m.geom_contype[gi]) or int(m.geom_conaffinity[gi]) for gi in floor))

    def test_rollout_default_off_is_byte_identical(self):
        # Opt-in: sphere_feet=False (default) must reproduce the untouched rollout exactly (no silent behavior change).
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        pol = MorphPolicy(self.fd, seed=3)
        base = recipe_rollout_morph(self.g, pol, steps=200)["forward"]
        off = recipe_rollout_morph(self.g, pol, steps=200, sphere_feet=False)["forward"]
        self.assertEqual(base, off)

    def test_rollout_with_sphere_feet_is_finite_and_stands(self):
        # THE physics guarantee: with sphere feet the body still stands and rolls out finite (doesn't fall through
        # the floor or eject to NaN) AND the trajectory actually differs from the capsule-foot rollout.
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        pol = MorphPolicy(self.fd, seed=5)
        base = recipe_rollout_morph(self.g, pol, steps=200)
        sph = recipe_rollout_morph(self.g, pol, steps=200, sphere_feet=True)
        self.assertTrue(sph["finite"])
        self.assertGreater(sph["height_ratio"], 0.3)         # still standing (not collapsed / not fallen through)
        self.assertGreater(sph["alive"], 50)                 # survived a meaningful stretch
        self.assertNotEqual(base["forward"], sph["forward"])  # the contact model changed -> the trajectory changed

    def test_banked_policy_round_trips_sphere_feet_and_auto_deploys(self):
        # deploy==train: a policy banked with sphere_feet=True reloads with it set, and rolling out the RELOADED
        # policy (without passing sphere_feet) still applies the transform (matches how it was trained).
        import tempfile
        from pathlib import Path
        from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
        pol = MorphPolicy(self.fd, seed=5)
        pol.sphere_feet = True
        with tempfile.TemporaryDirectory() as d:
            p = pol.to_npz(Path(d) / "sf.npz")
            re = MorphPolicy.from_npz(p)
            self.assertTrue(re.sphere_feet)                  # meta[8] round-tripped
            # rolling out the reloaded policy WITHOUT the kwarg must match the explicit sphere-feet rollout
            auto = recipe_rollout_morph(self.g, re, steps=150)["forward"]
            explicit = recipe_rollout_morph(self.g, pol, steps=150, sphere_feet=True)["forward"]
            self.assertEqual(auto, explicit)


if __name__ == "__main__":
    unittest.main()
