"""Input Ingestion plan, Phase 6 (CAD lane): STL/OBJ mesh import — bbox, unit detection, inertia estimate.

Builds a real binary STL cube and checks dimensions/volume/mass; checks mm-unit detection, OBJ, and the honest
STEP deferral. Pure/offline (AGENTS.md).
"""
import importlib.util
import os
import struct
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.cad_importer import import_cad  # noqa: E402


def _cube_tris(s: float):
    p = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0), (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris.append([p[a], p[b], p[c]])
        tris.append([p[a], p[c], p[d]])
    return tris


def _write_binary_stl(path: str, tris):
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(tris)))
        for tri in tris:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))          # normal (ignored)
            for v in tri:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


class CadImportTests(unittest.TestCase):
    def test_binary_stl_cube_metres(self):
        d = tempfile.mkdtemp(prefix="cad_")
        p = os.path.join(d, "cube.stl")
        _write_binary_stl(p, _cube_tris(0.2))                    # 0.2 m cube
        res = import_cad(p, material="abs")
        self.assertEqual(res.format, "stl_binary")
        self.assertEqual(res.triangles, 12)
        self.assertEqual(res.unit_guess, "m")
        for i in range(3):
            self.assertAlmostEqual(res.size_m[i], 0.2, places=5)
        self.assertAlmostEqual(res.volume_m3, 0.2 ** 3, places=5)  # 0.008 m^3
        self.assertAlmostEqual(res.estimated_mass_kg, 0.008 * 1040.0, places=2)  # abs density
        self.assertTrue(any("ESTIMATED" in w for w in res.warnings))

    def test_millimetre_units_detected_and_scaled(self):
        d = tempfile.mkdtemp(prefix="cad_")
        p = os.path.join(d, "big.stl")
        _write_binary_stl(p, _cube_tris(200.0))                  # 200 units == 0.2 m if it's mm
        res = import_cad(p)
        self.assertEqual(res.unit_guess, "mm")
        self.assertEqual(res.suggested_scale, 0.001)
        self.assertAlmostEqual(res.size_m[0], 0.2, places=4)
        self.assertTrue(any("millimet" in w for w in res.warnings))

    def test_ascii_stl(self):
        d = tempfile.mkdtemp(prefix="cad_")
        p = os.path.join(d, "tri.stl")
        Path(p).write_text(
            "solid t\nfacet normal 0 0 0\nouter loop\n"
            "vertex 0 0 0\nvertex 0.1 0 0\nvertex 0 0.1 0\n"
            "endloop\nendfacet\nendsolid t\n", encoding="utf-8")
        res = import_cad(p)
        self.assertEqual(res.format, "stl_ascii")
        self.assertEqual(res.vertices, 3)
        self.assertAlmostEqual(res.size_m[0], 0.1, places=5)

    def test_obj_has_geometry_no_mass(self):
        d = tempfile.mkdtemp(prefix="cad_")
        p = os.path.join(d, "m.obj")
        Path(p).write_text("v 0 0 0\nv 0.5 0 0\nv 0 0.3 0\nf 1 2 3\n", encoding="utf-8")
        res = import_cad(p)
        self.assertEqual(res.format, "obj")
        self.assertEqual(res.vertices, 3)
        self.assertIsNone(res.estimated_mass_kg)                 # OBJ: no reliable closed volume
        self.assertAlmostEqual(res.size_m[0], 0.5, places=5)

    def test_malformed_step_is_rejected_not_fatal(self):
        d = tempfile.mkdtemp(prefix="cad_")
        p = os.path.join(d, "part.step")
        Path(p).write_text("ISO-10303-21;", encoding="utf-8")   # not a real STEP body
        res = import_cad(p)
        self.assertEqual(res.format, "step")
        self.assertTrue(res.warnings)                            # honest failure, never crashes


@unittest.skipUnless(importlib.util.find_spec("build123d") is not None, "needs build123d for STEP")
class StepImportTests(unittest.TestCase):
    def test_step_roundtrip_dimensions_and_mass(self):
        import build123d as b3d
        d = tempfile.mkdtemp(prefix="cad_step_")
        p = os.path.join(d, "box.step")
        b3d.export_step(b3d.Box(100, 200, 300), p)              # 100x200x300 mm == 0.1x0.2x0.3 m
        res = import_cad(p, material="aluminum")
        self.assertEqual(res.format, "step")
        self.assertEqual(res.unit_guess, "mm")
        self.assertAlmostEqual(res.size_m[0], 0.1, places=4)
        self.assertAlmostEqual(res.size_m[1], 0.2, places=4)
        self.assertAlmostEqual(res.size_m[2], 0.3, places=4)
        self.assertAlmostEqual(res.volume_m3, 0.006, places=5)   # 0.1*0.2*0.3
        self.assertAlmostEqual(res.estimated_mass_kg, 0.006 * 2700.0, places=2)  # aluminum density


if __name__ == "__main__":
    unittest.main()
