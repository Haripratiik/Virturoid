"""Generative-3D part synthesis: AI-authored CAD (the safe box/cyl/sphere/cone DSL over build123d) baked
into a fitted, visual-only mesh. The cloud mesh-diffusion backends are exercised only when a key is set;
here we test the offline LLM-CAD path with a mock LLM. Skipped without build123d."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class _MockLLM:
    """Returns fixed DSL code (as the {code} JSON the synthesizer expects)."""
    def __init__(self, code):
        self.code = code

    def complete_json(self, system, user, schema, max_tokens=None):
        return {"code": self.code, "notes": ""}


class MeshSynthTests(unittest.TestCase):
    def test_dsl_exec_builds_a_floored_mesh(self):
        import numpy as np

        from virturoid.services.mesh_synth import _exec_cad
        code = ("part = cyl(R, L*0.6, at=(0,0,L*0.3)) + sphere(R, at=(0,0,L*0.6)) "
                "+ box(0.6*R, 2*R, 0.4*R, at=(0.6*R,0,0.6*L)) + cyl(0.5*R, 0.2*L, at=(0,0,0.08*L))")
        tris = _exec_cad(code, 0.17, 0.09)
        self.assertGreater(tris.shape[0], 20)                                  # a real multi-feature mesh
        self.assertTrue(np.isfinite(tris).all())
        self.assertAlmostEqual(float(tris.reshape(-1, 3)[:, 2].min()), 0.0, places=2)   # floored to z=0

    def test_dsl_blocks_imports_and_io(self):
        # The sandbox must reject import / file / network access in LLM-authored code.
        from virturoid.services.mesh_synth import _exec_cad
        for bad in ("import os\npart = cyl(R, L)", "part = open('x','w')", "part = __import__('os')"):
            with self.assertRaises(Exception):
                _exec_cad(bad, 0.1, 0.03)

    def test_synthesize_with_mock_llm_writes_loadable_mesh(self):
        from virturoid.services.mesh_synth import synthesize_part
        llm = _MockLLM("part = box(2*R,1.6*R,L*0.5, at=(0,0,L*0.4)) + cyl(0.5*R,0.2*L, at=(0,0,0.1*L))")
        with tempfile.TemporaryDirectory() as tmp:
            out = synthesize_part("a robot torso shell", 0.3, 0.12, str(Path(tmp) / "p.stl"), llm=llm)
            self.assertIsNotNone(out)
            self.assertTrue(Path(out).exists())
            if _MUJOCO:
                import mujoco
                m = mujoco.MjModel.from_xml_string(
                    f'<mujoco><asset><mesh name="m" file="{out}" scale="0.001 0.001 0.001"/></asset>'
                    f'<worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>')
                self.assertGreater(m.mesh_vertnum[0], 50)

    def test_bad_llm_code_falls_back_to_none(self):
        # Persistently invalid code (no valid `part`) must yield None so the caller falls back to anatomy.
        from virturoid.services.mesh_synth import synthesize_part
        llm = _MockLLM("part = None")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(synthesize_part("x", 0.1, 0.03, str(Path(tmp) / "p.stl"), llm=llm))

    def test_no_backend_returns_none(self):
        import os

        from virturoid.services.mesh_synth import available_backend, synthesize_part
        # No key + no llm + LLM backend off -> no synthesis (offline-safe).
        if os.environ.get("VIRTUROID_NO_LOCAL_ENV") == "1" and available_backend() is None:
            with tempfile.TemporaryDirectory() as tmp:
                self.assertIsNone(synthesize_part("x", 0.1, 0.03, str(Path(tmp) / "p.stl")))


if __name__ == "__main__":
    unittest.main()
