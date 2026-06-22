import importlib.util
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(_FASTAPI, "fastapi not installed.")
class WebAppRouteTests(unittest.TestCase):
    def _client(self, workspace):
        from fastapi.testclient import TestClient

        from virturoid.webapp import create_app

        return TestClient(create_app(Path(workspace)))

    def test_static_and_empty_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            self.assertEqual(200, c.get("/").status_code)
            self.assertIn("<title>Virturoid", c.get("/").text)
            self.assertEqual(200, c.get("/app.js").status_code)
            self.assertEqual(200, c.get("/app.css").status_code)
            self.assertEqual({"built": False}, c.get("/api/project").json())
            # No robot yet -> viewer rejects.
            self.assertEqual(409, c.post("/api/viewer", json={}).status_code)

    def test_chat_starts_a_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            res = c.post("/api/chat", json={"message": "status"}).json()
            self.assertIn("job_id", res)
            self.assertEqual("status", res["intent"])

    def test_memory_endpoint_empty_then_populated(self):
        from virturoid.services.memory_db import MemoryDB

        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            empty = c.get("/api/memory").json()
            self.assertFalse(empty["exists"])
            self.assertEqual(0, empty["stats"]["runs"])

            # Record a run into the same memory dir the app reads from.
            with MemoryDB(Path(tmp) / "memory" / "virturoid_memory.db") as db:
                db.record_run("sort blocks", "manipulator", "pick_place_sort",
                              {"forearm_mm": 340, "wrist_mm": 160, "actuator_torque_nm": 12}, 0.92,
                              species="serial_arm_3dof", succeeded=True, backend="openai")
            populated = c.get("/api/memory").json()
            self.assertTrue(populated["exists"])
            self.assertEqual(1, populated["stats"]["runs"])
            self.assertEqual(1, len(populated["recent"]))

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_preview_composes_robot_from_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._client(tmp)
            res = c.post("/api/preview", json={
                "prompt": "a tabletop arm that sorts red and blue blocks into matching bins",
            })
            self.assertEqual(200, res.status_code, res.text)
            data = res.json()
            self.assertTrue(data["preview"])
            self.assertGreater(len(data["geoms"]), 0)
            self.assertEqual(1, data["frame_count"])
            self.assertEqual(len(data["geoms"]), len(data["frames"][0]))
            self.assertEqual("manipulator", data["robot"]["robot_class"])
            self.assertGreater(data["robot"]["dof"], 0)
            self.assertTrue(data["robot"]["valid"])
            self.assertIn(data["robot"]["design_source"], {"heuristic", "llm"})

    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_project_and_viewer_with_built_robot(self):
        from virturoid.mvp import write_mvp_robot_arm_project

        with tempfile.TemporaryDirectory() as tmp:
            write_mvp_robot_arm_project(Path(tmp) / "project")  # seed the session project
            c = self._client(tmp)
            proj = c.get("/api/project").json()
            self.assertTrue(proj["built"])
            self.assertTrue(proj["artifacts"])

            view = c.post("/api/viewer", json={"scene_index": 0}).json()
            self.assertGreater(len(view["geoms"]), 0)
            self.assertGreater(view["frame_count"], 0)
            self.assertIn(view["outcome"]["status"], {"success", "failure"})
            # Each frame carries a world pose (x,y,z,qw,qx,qy,qz) per geom.
            self.assertEqual(len(view["geoms"]), len(view["frames"][0]))
            self.assertEqual(7, len(view["frames"][0][0]))
            # Transparency payload: the robot spec, the scene spec, and the scene list.
            self.assertEqual("manipulator", view["robot"]["robot_class"])
            self.assertTrue(view["robot"]["links"])
            self.assertTrue(view["robot"]["joints"])
            self.assertTrue(view["robot"]["end_effectors"])
            self.assertTrue(view["scene"]["objects"])
            self.assertGreaterEqual(len(view["scenes"]), 1)
            self.assertEqual(0, view["scene_index"])


if __name__ == "__main__":
    unittest.main()
