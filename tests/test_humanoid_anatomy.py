"""The ORIGINAL anthropometric humanoid: generated from head-canon proportions + elliptical-loft parts, with
NO copied real-robot meshes. Locks the proportions + the new 'loft'/'sole' geometry DSL families."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_BD = importlib.util.find_spec("build123d") is not None


class HumanoidDispatchTests(unittest.TestCase):
    def test_humanoid_prompt_builds_an_original_anthropometric_gene(self):
        import os
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a humanoid robot")
        self.assertEqual("humanoid.anthropometric", g.species)      # our own design, not a real model
        self.assertEqual("anthropometric", getattr(g, "design_source", None))
        self.assertEqual([], g.validate())
        self.assertEqual(8, len(g.actuated_joints()))               # hips+knees+shoulders+elbows, welded head/hands/feet
        # NONE of its segment geometry may reference a copied real-mesh role (kit-bash); all ORIGINAL DSL
        # families. The humanoid is now a MECHANICAL robot (chamfered extrude limbs, a compound torso, a sensor
        # head) — still our own geometry, never a real-robot mesh.
        fams = {(s.geometry or {}).get("family") for s in g.segments if s.geometry}
        self.assertTrue(fams <= {"loft", "revolve", "sole", "extrude", "compound"}, f"unexpected families: {fams}")
        self.assertNotIn("role", fams)                              # no kit-bashed real-robot part meshes

    def test_biped_synonyms_also_route_to_the_humanoid(self):
        from virturoid.services.morphology_composer import compose_robot
        for p in ("a bipedal robot", "an android", "a human-like robot"):
            self.assertEqual("humanoid.anthropometric", compose_robot(p).species, p)

    def test_balance_dof_variant_adds_hip_roll_and_ankle(self):
        # plan v4 P2: opt-in balance DOF (hip-ROLL + ankle-PITCH) -> 12 DOF for the walk ladder; default stays 8
        # so H1 + every existing humanoid path is byte-identical.
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        base = build_anthropometric_humanoid()
        wide = build_anthropometric_humanoid(balance_dof=True)
        self.assertEqual(8, len(base.actuated_joints()))            # default unchanged
        self.assertEqual(12, len(wide.actuated_joints()))          # +2 hip-roll +2 ankle-pitch
        names = {s.name for s in wide.segments}
        self.assertTrue({"l_hip", "r_hip"} <= names)               # the hip-roll blocks exist
        self.assertEqual([], wide.validate())                      # still a valid gene


@unittest.skipUnless(_BD, "needs build123d")
class LoftDslTests(unittest.TestCase):
    def test_loft_family_realizes_a_broad_front_shallow_side_solid(self):
        from virturoid.services.cad_geometry import realize_shape
        # an elliptical loft wider in y than deep in x -> the human cross-section a revolve cannot make
        part = realize_shape({"family": "loft", "sections": [[0.0, 0.05, 0.03], [0.2, 0.08, 0.04], [0.4, 0.04, 0.025]]})
        self.assertTrue(part.is_valid)
        self.assertGreater(part.volume, 0)
        bb = part.bounding_box()
        self.assertGreater(bb.max.Y - bb.min.Y, bb.max.X - bb.min.X)   # broader (y) than deep (x)
        self.assertAlmostEqual(bb.min.Z, 0.0, delta=1.0)               # floored to z=0 (mm)

    def test_sole_family_realizes_a_forward_footprint(self):
        from virturoid.services.cad_geometry import realize_shape
        part = realize_shape({"family": "sole", "length": 0.20, "width": 0.07, "height": 0.04, "heel": 0.04})
        self.assertTrue(part.is_valid)
        self.assertGreater(part.volume, 0)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class HumanoidProportionTests(unittest.TestCase):
    def test_proportions_and_silhouette_read_as_human(self):
        import os
        os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
        from virturoid.services.anatomy_critic import critique_gene
        from virturoid.services.morphology_composer import compose_robot
        c = critique_gene(compose_robot("a humanoid robot"))
        m = c["measures"]
        self.assertTrue(1.4 <= m["height_m"] <= 2.0, m["height_m"])       # human stature
        self.assertTrue(0.3 <= m["aspect_w_h"] <= 0.6, m["aspect_w_h"])   # not a stick/box; arms abduct slightly
        self.assertGreater(m["limb_y_spread_m"], 0.1)                     # limbs spread, not a column
        self.assertLess(m["biggest_part_frac"], 0.45)                    # no single part dominates
        self.assertNotIn("silhouette", {i["check"] for i in c["issues"]})


if __name__ == "__main__":
    unittest.main()
