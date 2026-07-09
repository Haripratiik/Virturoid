"""Deployment integrity: the installable ``virturoid`` package must NEVER depend on the repo-root ``scripts/``
directory. ``scripts/`` is not part of the wheel (``packages.find where=["src"]``), so any ``import scripts`` in a
product code path breaks the moment Virturoid runs as an installed package or from an MCP server whose cwd isn't
the repo root — which is exactly how it ships.

Regression caught 2026-07-09: ``ai_native_tools._honest_gait`` imported ``from scripts.verify_gait import
classify`` — so EVERY legged robot's verify verdict silently became "could not simulate (ModuleNotFoundError)" in
any real deployment. Two guards below: (1) a static check that no ``src/virturoid`` module imports ``scripts``, and
(2) a behavioral check that the legged verdict still works with ``scripts`` import BLOCKED.
"""
import importlib.abc
import importlib.util
import os
import pathlib
import sys
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "virturoid"


class _BlockScripts(importlib.abc.MetaPathFinder):
    """A meta-path finder that makes any ``import scripts[...]`` fail — simulating the installed-package env."""
    def find_spec(self, name, path, target=None):
        if name == "scripts" or name.startswith("scripts."):
            raise ModuleNotFoundError(f"blocked '{name}' to simulate an installed-package deployment")
        return None


class StaticImportGuardTests(unittest.TestCase):
    def test_no_src_module_imports_scripts(self):
        offenders = []
        for py in _SRC.rglob("*.py"):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                s = line.strip()
                if s.startswith("from scripts") or s.startswith("import scripts"):
                    offenders.append(f"{py.relative_to(_SRC.parent.parent)}:{i}: {s}")
        self.assertEqual(offenders, [], "the installable package must not import the repo-root scripts/ dir:\n"
                         + "\n".join(offenders))


@unittest.skipUnless(_MUJOCO, "the legged verdict needs MuJoCo")
class DeploymentBehaviorTests(unittest.TestCase):
    def setUp(self):
        # drop any cached scripts modules, then block future scripts imports
        for m in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
            del sys.modules[m]
        self._blocker = _BlockScripts()
        sys.meta_path.insert(0, self._blocker)

    def tearDown(self):
        try:
            sys.meta_path.remove(self._blocker)
        except ValueError:
            pass

    def test_scripts_import_is_actually_blocked(self):
        with self.assertRaises(ModuleNotFoundError):
            __import__("scripts.verify_gait")

    def test_legged_verify_works_without_scripts(self):
        # the exact deployment failure mode: a legged robot verified while scripts/ is unavailable
        from virturoid.services import session_state as S
        from virturoid.services.ai_native_tools import verify_robot
        from virturoid.services.morphology_composer import compose_robot
        rid = S.put_robot(compose_robot("a quadruped robot dog that walks"), label="deploy")
        r = verify_robot({"robot_id": rid, "mode": "quick"})       # quick still runs the gait classifier
        self.assertEqual(r.get("kind"), "legged")
        self.assertNotIn("could not simulate", r.get("verdict", "").lower(),
                         f"legged verify must not die on a missing scripts/ import: {r}")
        # the verdict must be one of the real gait classifications
        self.assertRegex(r["verdict"], r"CREDIBLE WALK|CROUCH|SLIDE|FORWARD BUT SHORT|FELL")


if __name__ == "__main__":
    unittest.main()
