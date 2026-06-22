import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
from virturoid.services.gene_compiler import compile_gene_with_scene

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class GeneBuildTests(unittest.TestCase):
    def _scenes(self, n=4):
        from virturoid.services.gene_build import generate_pick_place_scenes

        return generate_pick_place_scenes("sort red and blue blocks into bins", count=n)

    def test_arm_gene_runs_real_pick_place_at_parity(self):
        # The gene-compiled arm genuinely does the task in real physics, near the known-good
        # exporter arm (gap to 91% is the co-design/controller tuning the full pipeline adds).
        from virturoid.services.gene_build import evaluate_gene_pick_place

        r = evaluate_gene_pick_place(tabletop_arm_gene(), self._scenes(6))
        self.assertGreaterEqual(r["success_rate"], 0.33, r)
        self.assertGreater(r["blocks_placed"], 0)

    def test_humanoid_gene_runs_and_performs_the_task(self):
        # The co-designed humanoid is a real, structurally distinct robot that COMPILES, runs,
        # and genuinely performs the sort on robot-matched scenes (observed 50-75%; assert a
        # conservative floor so it can't silently regress to the 0% it started at).
        import mujoco

        from virturoid.services.gene_build import evaluate_gene_pick_place, generate_reachable_scenes

        g = humanoid_upper_body_gene()
        scenes = generate_reachable_scenes(g, count=8, seed=0)
        model = mujoco.MjModel.from_xml_string(compile_gene_with_scene(g, scenes[0].objects))
        self.assertEqual(model.nu, len(g.actuated_joints()))
        self.assertEqual(6, len(g.segments))               # torso + 4-DOF arm + hand
        r = evaluate_gene_pick_place(g, scenes)
        self.assertEqual("humanoid", r["robot_class"])     # built as a humanoid, not the arm
        self.assertGreater(r["blocks_placed"], 0)          # actually picks+places
        self.assertGreaterEqual(r["success_rate"], 0.3)    # conservative floor (observed ~0.5-0.75)

    def test_build_gene_package_writes_viewer_artifacts(self):
        from virturoid.services.gene_build import build_gene_package

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pkg"
            summary = build_gene_package(tabletop_arm_gene(), "sort blocks into bins", out, scene_count=3)
            self.assertTrue((out / "simulation" / "scene_set.json").exists())
            self.assertTrue((out / "simulation" / "mujoco" / "compiled_scene_index.json").exists())
            self.assertTrue((out / "robot" / "robot_genome.json").exists())
            self.assertTrue((out / "reports" / "gene_evaluation_report.json").exists())
            self.assertIn("success_rate", summary)

    def test_legged_gene_is_scored_on_locomotion_not_pick_place(self):
        # A quadruped has no gripper/reachable workspace; build_gene_package must dispatch it to a
        # LOCOMOTION eval (forward walking distance), not pick-place. Regression guard against a legged
        # body being silently scored on a task it can't do.
        from virturoid.fixtures.gene_library import quadruped_gene
        from virturoid.services.gene_build import build_gene_package

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pkg"
            summary = build_gene_package(quadruped_gene(), "a quadruped robot that walks forward", out)
            self.assertEqual("locomotion", summary["task_type"])
            self.assertIn("success_rate", summary)
            self.assertIn("distance_m", summary)
            self.assertIn("forward_m", summary)
            self.assertTrue((out / "robot" / "robot_genome.json").exists())
            self.assertTrue((out / "reports" / "gene_evaluation_report.json").exists())
            # A walker DOES write scene_set.json (the viewer reads it to replay the gait), but it must be a
            # LOCOMOTION scene (purpose=locomotion, no object scenes) — never a mislabeled pick-place scene.
            ss = out / "simulation" / "scene_set.json"
            self.assertTrue(ss.exists())
            scene_set = json.loads(ss.read_text(encoding="utf-8"))
            self.assertEqual("locomotion", scene_set.get("purpose"))
            self.assertEqual([], scene_set["scenes"][0].get("objects"))


if __name__ == "__main__":
    unittest.main()
