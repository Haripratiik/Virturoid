"""The robotics AI GROUNDS the LLM's design: after the model authors a body (submit_design), the robotics
embedding surfaces the nearest PHYSICS-VERIFIED prior bodies + whether a banked gait is likely to warm-start it —
so the model reasons against what has actually worked, not blind and not a deterministic template. This is the
'why we have a robotics AI' loop: the LLM designs, the robotics AI grounds."""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")


class RoboticsGroundingTests(unittest.TestCase):
    def test_grounding_surfaces_nearest_verified_bodies_and_transfer_outlook(self):
        from virturoid.services import agent_design_tools as adt
        from virturoid.services.gait_flywheel import bank_gait
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.robotics_vector_memory import BODY, RoboticsVectorMemory, embed_body
        memdir = Path(tempfile.mkdtemp(prefix="ground_"))
        quad = compose_robot("a small quadruped robot dog")
        with MemoryDB(memdir / "virturoid_memory.db") as db:
            vm = RoboticsVectorMemory(db)
            vm.upsert(BODY, "prior_quad", embed_body(compose_robot("a large quadruped robot")),
                      {"robot_class": "quadruped"})
            r = types.SimpleNamespace(best_survived=True, best_forward=0.9, best_credible=True,
                                      best_height_ratio=0.85, best_params={"freq": 1.5, "hip_amp": 0.9,
                                      "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5})
            bank_gait(db, quad, r)                              # a physics-verified gait for this morphology
        with mock.patch("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: memdir):
            g = adt._robotics_grounding(quad)
        self.assertTrue(g.get("nearest_verified_bodies"), "a verified prior body must ground the design")
        self.assertEqual(g["nearest_verified_bodies"][0]["body"], "prior_quad")
        self.assertTrue(g["warm_start_gait_available"])        # a banked gait exists near this body
        self.assertIn("warm-start", g["transfer_outlook"])     # high similarity + gait -> transfer outlook is positive

    def test_grounding_is_empty_not_a_crash_without_a_corpus(self):
        from virturoid.services import agent_design_tools as adt
        from virturoid.services.morphology_composer import compose_robot
        empty = Path(tempfile.mkdtemp(prefix="empty_"))
        with mock.patch("virturoid.services.agent_tools.safe_build_path", lambda *a, **k: empty):
            self.assertEqual(adt._robotics_grounding(compose_robot("a quadruped robot")), {})


if __name__ == "__main__":
    unittest.main()
