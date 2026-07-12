import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from virturoid.fixtures.gene_library import humanoid_upper_body_gene, tabletop_arm_gene
from virturoid.services.gene_compiler import compile_gene_with_scene

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _lurch_rollout(pitch_deg):
    """A rollout that clears forward+upright+cadence+support but PITCHES over — the exact Goodhart the old
    gate missed. classify() reads the base quaternion (qpos[3:7]) for the roll/pitch orientation gate."""
    import math
    half = math.radians(pitch_deg) / 2.0
    quat = np.array([0.0, 0.0, 0.5, math.cos(half), 0.0, math.sin(half), 0.0])   # xyz + (w,x,y,z) pitch about y
    return {"survived": True, "forward": 0.8, "upright_frac": 0.9, "height_ratio": 0.9, "cadence": 2.5,
            "support_frac": 0.6, "n_feet": 4, "alive": 900, "speed": 0.3, "finite": True,
            "qpos_frames": [quat] * 4}


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

    def test_legged_headline_uses_ungameable_classify_verdict(self):
        """v7-A3: the build HEADLINE shares classify() for legged bodies — a forward PITCH-DIVE lurch (which the
        old forward+upright+cadence gate certified as 'walked') no longer reads 'walked', and a clean level walk
        with the same scalars does. Locks the headline to the same un-gameable verdict verify_robot uses."""
        from virturoid.fixtures.gene_library import quadruped_gene
        from virturoid.services.gene_build import build_gene_package

        for pitch, expect_walked in ((45.0, False), (3.0, True)):     # lurch rejected; clean level walk accepted
            with tempfile.TemporaryDirectory() as tmp, \
                 mock.patch("virturoid.services.morph_policy.recipe_rollout_morph",
                            return_value=_lurch_rollout(pitch)), \
                 mock.patch("virturoid.services.learn_locomotion.banked_policy_for", return_value=None):
                summary = build_gene_package(quadruped_gene(), "a quadruped that walks forward", Path(tmp) / "p")
            self.assertIn("gait_verdict", summary)                     # the classify() string is surfaced
            self.assertEqual(expect_walked, summary["status"] == "walked",
                             f"pitch {pitch}: status={summary['status']} verdict={summary['gait_verdict']}")
            if not expect_walked:
                self.assertEqual(0.0, summary["success_rate"])         # a lurch earns no locomotion credit
                self.assertIn("LURCH", summary["gait_verdict"].upper())


if __name__ == "__main__":
    unittest.main()
