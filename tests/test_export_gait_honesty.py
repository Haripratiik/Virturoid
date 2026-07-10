"""The exported gait control program states HONESTLY whether the bare feed-forward gait walks the body.

Before: _write_gait_control_program unconditionally claimed "the bare gait already walks" for ANY legged body —
even ones the bare trot-CPG gait only crouches/slides on (measured: quad CROUCH, hexapod/octopod SLIDE). A customer
building from the exported package was told a walk the body never achieved. Now it VERIFIES the exported gait in
sim and reports verified_walk + the measured distance + verdict, so the claim can never outrun the physics.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "gait verification needs the MuJoCo sim")
class ExportGaitHonestyTests(unittest.TestCase):
    def _program_for(self, prompt):
        from virturoid.services.gene_build import _write_gait_control_program
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot(prompt, ensure_walkable=True)
        out = Path(tempfile.mkdtemp())
        _write_gait_control_program(g, {"id": getattr(g, "id", "robot")}, out)
        p = out / "software" / "control_program.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def test_program_reports_verified_walk_and_never_overclaims(self):
        prog = self._program_for("a quadruped robot dog")
        self.assertIsNotNone(prog, "a legged body must export a gait control program")
        # the honest verification fields are present
        for k in ("verified_walk", "sim_forward_m", "sim_verdict"):
            self.assertIn(k, prog, f"the control program must carry {k}")
        self.assertIsInstance(prog["verified_walk"], bool)
        note = " ".join(prog.get("notes", []))
        # the old UNCONDITIONAL overclaim is gone
        self.assertNotIn("already walks", note)
        # the note is CONSISTENT with the verification verdict (never claims a walk the sim didn't back)
        if prog["verified_walk"]:
            self.assertIn("VERIFIED", note)
            self.assertTrue(str(prog["sim_verdict"]).startswith("CREDIBLE"))
        else:
            self.assertIn("STARTING POINT", note)
            self.assertFalse(str(prog["sim_verdict"]).startswith("CREDIBLE"))

    def test_non_legged_body_exports_no_gait_program(self):
        # a manipulator has no walk to (over)claim -> no gait control program at all
        self.assertIsNone(self._program_for("a robot arm that picks up objects"))

    def test_credible_crawl_is_exported_as_the_runnable_controller(self):
        import importlib.util

        from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
        from virturoid.services.gene_build import _write_gait_control_program

        # A fanned crawl body is the canonical credible open-loop gait. The
        # package must export that wave plan, not fall back to the trot CPG.
        gene = build_from_anatomy(_generic_legged_graph(n_pairs=2, girth=0.22, fan=True))
        out = Path(tempfile.mkdtemp())
        _write_gait_control_program(gene, {"id": gene.id}, out)
        program = json.loads((out / "software" / "control_program.json").read_text(encoding="utf-8"))
        self.assertEqual(program["policy_type"], "crawl_wave_gait")
        self.assertTrue(program["verified_walk"])
        self.assertIn("CREDIBLE", program["sim_verdict"])
        self.assertTrue(program["legs"])

        controller_path = out / "software" / "gait_controller.py"
        spec = importlib.util.spec_from_file_location("exported_crawl", controller_path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        targets = mod.GaitController.from_file(str(out / "software" / "control_program.json")).infer(0.17)
        self.assertEqual(set(targets), set(program["joint_names"]))
        for name, limit in zip(program["joint_names"], program["position_limits"]):
            self.assertGreaterEqual(targets[name], limit[0] - 1e-8)
            self.assertLessEqual(targets[name], limit[1] + 1e-8)


if __name__ == "__main__":
    unittest.main()
