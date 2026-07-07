"""I2: dataset/log importer (Input Ingestion plan, Phase 5).

Real parsing where deps allow (virturoid npz demonstrations, LeRobot meta json), honest format detection by
magic bytes for logs. Pure/offline (AGENTS.md).
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.dataset_importer import import_dataset, sniff_format  # noqa: E402


class SniffTests(unittest.TestCase):
    def _write(self, name: str, head: bytes) -> str:
        d = tempfile.mkdtemp(prefix="ds_")
        p = os.path.join(d, name)
        Path(p).write_bytes(head + b"\x00" * 8)
        return p

    def test_magic_byte_detection(self):
        self.assertEqual(sniff_format(self._write("a.bin", b"\x89HDF\r\n\x1a\n")), "robomimic_hdf5")
        self.assertEqual(sniff_format(self._write("b.bin", b"\x89MCAP0\r\n")), "mcap")
        self.assertEqual(sniff_format(self._write("c.bin", b"#ROSBAG V2.0\n")), "rosbag")
        self.assertEqual(sniff_format(self._write("d.bin", b"SQLite format 3\x00")), "ros2_db3")

    def test_extension_detection(self):
        d = tempfile.mkdtemp(prefix="ds_")
        p = os.path.join(d, "log.mcap")
        Path(p).write_bytes(b"whatever")
        self.assertEqual(sniff_format(p), "mcap")

    def test_unknown_format(self):
        p = self._write("x.bin", b"not a known header")
        self.assertEqual(sniff_format(p), "unknown")


class LogImportTests(unittest.TestCase):
    def test_mcap_detected_with_honest_note(self):
        d = tempfile.mkdtemp(prefix="ds_")
        p = os.path.join(d, "run.mcap")
        Path(p).write_bytes(b"\x89MCAP0\r\n" + b"\x00" * 16)
        spec = import_dataset(p)
        self.assertEqual(spec.format, "mcap")
        self.assertTrue(any("mcap" in w.lower() for w in spec.warnings))   # honest: needs the mcap lib


class VirturoidNpzTests(unittest.TestCase):
    def test_real_episode_and_action_shapes(self):
        import numpy as np
        d = tempfile.mkdtemp(prefix="ds_npz_")
        os.makedirs(os.path.join(d, "episodes"))
        obs = np.zeros((10, 6), dtype=np.float32)
        act = np.zeros((10, 3), dtype=np.float32)
        np.savez(os.path.join(d, "episodes", "ep0.npz"), obs=obs, actions=act,
                 rewards=np.zeros(10), dones=np.zeros(10, dtype=np.int8))
        index = [{"episode_id": "ep0", "length": 10, "obs_dim": 6, "action_dim": 3,
                  "uri": "episodes/ep0.npz"}]
        Path(d, "index.json").write_text(json.dumps(index), encoding="utf-8")
        spec = import_dataset(d)
        self.assertEqual(spec.format, "virturoid_npz")
        self.assertEqual(spec.episodes, 1)
        self.assertEqual(spec.total_frames, 10)
        self.assertEqual(spec.candidate_action_dim, 3)         # read from the npz, not just the index
        self.assertIn("obs", spec.candidate_observation_keys)


class LeRobotTests(unittest.TestCase):
    def test_meta_info_json(self):
        d = tempfile.mkdtemp(prefix="ds_lerobot_")
        os.makedirs(os.path.join(d, "meta"))
        info = {
            "fps": 30, "total_episodes": 42, "total_frames": 12600,
            "features": {
                "observation.images.wrist": {"dtype": "video", "shape": [3, 240, 320]},
                "observation.state": {"dtype": "float32", "shape": [7]},
                "action": {"dtype": "float32", "shape": [7]},
            },
        }
        Path(d, "meta", "info.json").write_text(json.dumps(info), encoding="utf-8")
        spec = import_dataset(d)
        self.assertEqual(spec.format, "lerobot")
        self.assertEqual(spec.episodes, 42)
        self.assertEqual(spec.rate_hz, 30.0)
        self.assertEqual(spec.candidate_action_dim, 7)
        self.assertTrue(any("observation" in k for k in spec.candidate_observation_keys))


class ToolTests(unittest.TestCase):
    def test_import_dataset_tool(self):
        from virturoid.services.agent_tools import call_tool
        d = tempfile.mkdtemp(prefix="ds_")
        p = os.path.join(d, "run.mcap")
        Path(p).write_bytes(b"\x89MCAP0\r\n")
        r = call_tool("import_dataset", {"path": p})
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"]["format"], "mcap")


if __name__ == "__main__":
    unittest.main()
