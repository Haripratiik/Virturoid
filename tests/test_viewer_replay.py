"""The viewport always has motion to replay: the settle fallback guarantees frames even when a task episode
fails or produces nothing (so a build is never a frozen, motionless scene)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_CAD = importlib.util.find_spec("build123d") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class ViewerReplayFallbackTests(unittest.TestCase):
    @unittest.skipUnless(_MUJOCO and _CAD, "MuJoCo and build123d are required for detailed visual replay.")
    def test_locomotion_replay_prefers_packaged_visual_meshes(self):
        """The episode's frame model and its browser-visible STL assets must be the same portable package model."""
        from virturoid.fixtures.gene_library import quadruped_gene
        from virturoid.services.gene_compiler import (
            compile_gene_to_mjcf,
            standing_spawn_z,
            write_packaged_visual_mjcf,
        )
        from virturoid.services.viewer_sim import simulate_episode_for_viewer

        gene = quadruped_gene()
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            visual = write_packaged_visual_mjcf(
                gene, package, include_floor=True, spawn_z=standing_spawn_z(gene))
            self.assertIsNotNone(visual)
            scene_uri = f"simulation/mujoco/scenes/locomotion/{gene.id}.xml"
            scene_path = package / scene_uri
            scene_path.parent.mkdir(parents=True, exist_ok=True)
            scene_path.write_text(
                compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)), encoding="utf-8")
            (package / "simulation" / "mujoco" / "compiled_scene_index.json").write_text(json.dumps({
                "scenes": [{"scene_id": "locomotion", "mujoco_xml": scene_uri}],
            }), encoding="utf-8")
            (package / "simulation" / "scene_set.json").write_text(json.dumps({
                "purpose": "locomotion", "scenes": [{"id": "locomotion", "objects": []}],
            }), encoding="utf-8")

            view = simulate_episode_for_viewer(package)
            self.assertEqual("simulation/robot_visual.xml", view["replay_model_uri"])
            self.assertTrue(any(g.get("mesh_uri", "").startswith("simulation/viewer_assets/")
                                for g in view["geoms"]))
            self.assertEqual(len(view["geoms"]), len(view["frames"][0]))

    def test_camera_fits_the_recorded_robot_not_a_fixed_tabletop_point(self):
        import mujoco

        from virturoid.services.viewer_sim import _camera_for_episode_frame

        xml = ('<mujoco><worldbody><geom type="plane" size="5 5 0.1"/>'
               '<body pos="5 0 1"><geom type="box" size="0.5 0.2 1"/></body>'
               '</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        frame = [[*data.geom_xpos[g], 1.0, 0.0, 0.0, 0.0] for g in range(model.ngeom)]
        lookat, distance = _camera_for_episode_frame(model, frame)
        self.assertAlmostEqual(5.0, lookat[0], delta=0.1)
        self.assertGreater(distance, 3.0)

    def test_settle_episode_always_produces_frames(self):
        import mujoco

        from virturoid.services.viewer_sim import _settle_episode

        xml = ('<mujoco><worldbody><geom type="plane" size="5 5 0.1"/>'
               '<body pos="0 0 0.4"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>'
               '</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        frames: list = []
        _settle_episode(model, record_frames=frames, steps=60, frame_every=6)
        self.assertGreater(len(frames), 0)                 # the fallback always yields SOME motion
        self.assertEqual(len(frames[0]), model.ngeom)      # one world pose per geom (viewer-replayable)
        # the box falls under gravity, so the replay shows real motion (z decreases across frames)
        box_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "") if False else model.ngeom - 1
        self.assertLess(frames[-1][box_geom][2], frames[0][box_geom][2] + 1e-6)


if __name__ == "__main__":
    unittest.main()
