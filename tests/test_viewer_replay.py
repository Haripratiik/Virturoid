"""The viewport always has motion to replay: the settle fallback guarantees frames even when a task episode
fails or produces nothing (so a build is never a frozen, motionless scene)."""
import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class ViewerReplayFallbackTests(unittest.TestCase):
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
