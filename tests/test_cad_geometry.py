"""STAGE 3: real CAD geometry — shaped link solids exported as manufacturable STEP + STL with mass and
inertia from the solids. Skipped if the build123d/OpenCascade kernel isn't installed."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_B123D = importlib.util.find_spec("build123d") is not None
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_B123D, "build123d (OpenCascade) not installed.")
class CadGeometryTests(unittest.TestCase):
    def test_link_solid_is_a_valid_shaped_volume(self):
        from virturoid.services.cad_geometry import build_link_solid
        solid = build_link_solid("capsule", 0.2, 0.03, actuated=True)
        self.assertTrue(solid.is_valid)
        self.assertGreater(solid.volume, 0.0)            # real, non-degenerate geometry

    def test_visual_meshes_are_cached_and_skip_fingers(self):
        from virturoid.services.cad_geometry import build_visual_meshes
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a tabletop robot arm that picks and places")
        with tempfile.TemporaryDirectory() as tmp:
            meshes = build_visual_meshes(g, tmp)
            fingers = [s for s in g.segments if s.joint_type == "prismatic"]
            self.assertEqual(len(meshes), len(g.segments) - len(fingers))   # prismatic fingers skipped
            self.assertTrue(all(Path(p).exists() for p in meshes.values()))
            # second call is served from cache: no new files, same mapping
            again = build_visual_meshes(g, tmp)
            self.assertEqual(meshes, again)

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_meshed_model_has_identical_physics_to_primitive(self):
        import mujoco

        from virturoid.services.cad_geometry import build_visual_meshes
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a tabletop robot arm that picks and places")
        with tempfile.TemporaryDirectory() as tmp:
            prim = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
            mesh = mujoco.MjModel.from_xml_string(
                compile_gene_to_mjcf(g, meshes=build_visual_meshes(g, tmp)))
        self.assertGreater(mesh.nmesh, 0)                                   # the visual mesh layer exists
        self.assertEqual(float(sum(prim.body_mass)), float(sum(mesh.body_mass)))  # mass untouched
        coll = lambda m: sum(1 for i in range(m.ngeom) if m.geom_contype[i] != 0)
        self.assertEqual(coll(prim), coll(mesh))                            # same colliders
        # meshes=None must reproduce the default model exactly (the sim/training path is unchanged)
        self.assertEqual(compile_gene_to_mjcf(g, meshes=None), compile_gene_to_mjcf(g))

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_geometry_segments_render_with_identical_physics(self):
        # Regression for the revived detail path: segments carrying a `geometry`/role spec used to be
        # SKIPPED by build_visual_meshes (the detailed shapes silently fell back to bare primitives — a
        # root cause of the toy look). They must now get a visual mesh AND leave the MuJoCo physics model
        # byte-identical (geometry is visual/CAD only; the compiler reads shape/length/radius/mass).
        import numpy as np
        import mujoco

        from virturoid.services.cad_geometry import build_visual_meshes
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")          # its recipe emits geometry= specs
        geom_segs = [s for s in g.segments if getattr(s, "geometry", None)]
        self.assertGreater(len(geom_segs), 0)                   # the body actually has detailed shapes
        with tempfile.TemporaryDirectory() as tmp:
            meshes = build_visual_meshes(g, tmp)
            for s in geom_segs:                                 # every geometry segment is now rendered
                self.assertIn(s.name, meshes)
                self.assertTrue(Path(meshes[s.name]).exists())
            prim = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
            mesh = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, meshes=meshes))
        self.assertEqual(float(sum(prim.body_mass)), float(sum(mesh.body_mass)))   # mass untouched
        self.assertTrue(np.allclose(prim.body_inertia, mesh.body_inertia))         # inertia untouched
        coll = lambda m: sum(1 for i in range(m.ngeom) if m.geom_contype[i] != 0)
        self.assertEqual(coll(prim), coll(mesh))                                   # same colliders

    def test_export_gene_cad_writes_real_files_and_grounded_mass(self):
        from virturoid.services.cad_geometry import export_gene_cad
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        with tempfile.TemporaryDirectory() as tmp:
            man = export_gene_cad(g, tmp)
            self.assertEqual(man["part_count"], len(g.segments))
            self.assertGreater(man["total_mass_kg"], 0.0)
            steps = list(Path(tmp, "step").glob("*.step"))
            stls = list(Path(tmp, "stl").glob("*.stl"))
            self.assertEqual(len(steps), len(g.segments))    # a manufacturable STEP per link
            self.assertEqual(len(stls), len(g.segments))     # a mesh per link
            for p in steps:                                  # real B-rep STEP, not a text placeholder
                self.assertGreater(p.stat().st_size, 1000)
                self.assertIn("ISO-10303", p.read_text(encoding="utf-8", errors="ignore")[:64])
            for part in man["parts"]:                        # mass + inertia grounded in the solid
                self.assertGreater(part["mass_kg"], 0.0)
                self.assertGreater(part["volume_cm3"], 0.0)


if __name__ == "__main__":
    unittest.main()
