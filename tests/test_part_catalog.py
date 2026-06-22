"""P3 kit-bashing: real, license-clean Menagerie link meshes fitted onto a generated body as a VISUAL-ONLY
layer. Skipped if build123d / MuJoCo / robot_descriptions (the Menagerie cache) are unavailable."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_B123D = importlib.util.find_spec("build123d") is not None
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_DESCR = importlib.util.find_spec("robot_descriptions") is not None


@unittest.skipUnless(_B123D and _MUJOCO and _DESCR, "build123d / MuJoCo / robot_descriptions required.")
class PartCatalogTests(unittest.TestCase):
    def test_catalog_is_license_clean(self):
        from virturoid.services.part_catalog import LICENSES, attribution
        # Every source model used for redistribution must carry a permissive license.
        for entry in attribution():
            self.assertIn(entry["license"].split()[0], ("BSD-3-Clause", "Apache-2.0", "MIT"))
        self.assertTrue(LICENSES)

    def test_fitted_part_lands_in_the_link_frame(self):
        # A baked part is normalized to [0, length] +z (floor at z=0, length along z) so it drops onto the
        # compiler's primitive with no per-geom offset — same convention as build_anatomy.
        from virturoid.services.part_catalog import fitted_part_stl, has_part, source_stl
        if not has_part("thigh"):
            self.skipTest("Menagerie meshes not cached.")
        self.assertTrue(source_stl("thigh").exists())
        import mujoco
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "thigh.stl")
            self.assertEqual(fitted_part_stl("thigh", 0.18, 0.05, out), out)
            m = mujoco.MjModel.from_xml_string(
                f'<mujoco><asset><mesh name="t" file="{out}" scale="0.001 0.001 0.001"/></asset>'
                f'<worldbody><geom type="mesh" mesh="t"/></worldbody></mujoco>')
            self.assertGreater(m.mesh_vertnum[0], 100)               # a real, detailed mesh
            # MuJoCo recenters mesh verts to the CoM and compensates via geom_pos; the NET body-frame span
            # (geom_pos.z + vert range) is the link's [0, length] frame the compiler relies on.
            gz = float(m.geom_pos[0][2])
            zlo = gz + float(m.mesh_vert[:, 2].min())
            zhi = gz + float(m.mesh_vert[:, 2].max())
            self.assertAlmostEqual(zlo, 0.0, places=2)               # floored to z=0 (meters after 0.001)
            self.assertAlmostEqual(zhi, 0.18, places=2)              # spans the link length along +z

    def test_kitbash_keeps_physics_identical(self):
        # Kit-bashing is VISUAL ONLY: swapping in real meshes must not move a single simulated number.
        import mujoco

        from virturoid.services.cad_geometry import build_visual_meshes
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a humanoid robot that can walk")
        if not any(__import__("virturoid.services.part_catalog", fromlist=["has_part"]).has_part(
                (s.geometry or {}).get("role")) for s in g.segments if isinstance(s.geometry, dict)):
            self.skipTest("Menagerie meshes not cached.")
        prim = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        with tempfile.TemporaryDirectory() as tmp:
            kit = mujoco.MjModel.from_xml_string(
                compile_gene_to_mjcf(g, meshes=build_visual_meshes(g, tmp, kitbash=True)))
        self.assertEqual(float(sum(prim.body_mass)), float(sum(kit.body_mass)))
        coll = lambda m: sum(1 for i in range(m.ngeom) if m.geom_contype[i] != 0)
        self.assertEqual(coll(prim), coll(kit))
        self.assertGreater(kit.nmesh, 0)


if __name__ == "__main__":
    unittest.main()
