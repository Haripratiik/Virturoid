"""Smoke test for the one-command demo gallery (plan #4): a 2-robot mini battery produces a self-contained HTML.

Needs MuJoCo (create/render/verify compile + simulate). Kept to the mini battery in quick mode so it stays a
smoke test, not a slow render farm.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))


@unittest.skipUnless(_MUJOCO, "the gallery renders + simulates real robots")
class DemoGallerySmoke(unittest.TestCase):
    def test_mini_battery_writes_self_contained_html(self):
        os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="demo_sess_")
        import run_mvp_demo as demo

        out = tempfile.mkdtemp(prefix="demo_out_")
        path = demo.run_gallery(demo.MINI_BATTERY, out, quick=True, learn=False)
        self.assertTrue(os.path.exists(path))
        h = Path(path).read_text(encoding="utf-8")
        self.assertTrue(h.strip().startswith("<!doctype html>"))
        self.assertEqual(h.count("<article"), len(demo.MINI_BATTERY), "one card per robot")
        self.assertGreaterEqual(h.count("data:image/"), 1, "renders must be embedded (self-contained)")
        self.assertEqual(h.count("http://") + h.count("https://"), 0, "no external references — opens offline")
        self.assertIn("credible", h.lower())               # the honest verdict summary is present


if __name__ == "__main__":
    unittest.main()
