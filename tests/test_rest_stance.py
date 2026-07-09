"""A body with a BAKED REST STANCE spawns standing (not collapsed) in the gait sim.

A spider's legs are bent DOWN in its rest pose (metadata['rest_pose'], emitted by the compiler as a
<keyframe name="rest">). But the gait rollouts reset with mj_resetData -> qpos0 (all joints 0), so the spider
spawned with its legs STRAIGHT OUT (horizontal) and collapsed to the ground (CROUCH, 0 m). Honoring the rest
keyframe on reset makes it spawn in its designed stance -> it stands and walks. This also strengthened every
other rest-posed body's default gait (a quad's default walk rose ~0.65 -> ~0.83 m).
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs the MuJoCo sim")
class RestStanceTests(unittest.TestCase):
    def test_spider_stands_and_walks_instead_of_collapsing(self):
        from virturoid.services.gait_quality import classify
        from virturoid.services.morph_policy import crawl_gait_rollout
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a spider robot", ensure_walkable=True)
        self.assertTrue((g.metadata or {}).get("rest_pose"), "a spider bakes a bent-leg rest pose")
        r = crawl_gait_rollout(g, steps=1200, record_qpos=True)
        self.assertTrue(r["survived"], f"the spider must not collapse: {r}")
        self.assertGreater(r["height_ratio"], 0.6, f"it must stand, not sink to the floor: {r}")
        self.assertGreater(abs(r["forward"]), 0.3, f"a standing spider moves: {r}")

    def test_rest_keyframe_is_honored_on_reset(self):
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        from virturoid.services.morph_policy import _reset_to_rest
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a spider robot", ensure_walkable=True)
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True, spawn_z=standing_spawn_z(g)))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)       # qpos0 -> legs horizontal
        q_flat = np.array(d.qpos[7:]).copy()                     # the actuated-joint pose (skip the 7-DOF free base)
        _reset_to_rest(m, d); mujoco.mj_forward(m, d)            # -> the baked bent-knee rest pose
        q_rest = np.array(d.qpos[7:]).copy()
        self.assertGreater(float(np.abs(q_rest - q_flat).max()), 0.3,
                           "the rest reset must bend the legs (some joint moves well off qpos0)")

    def test_humanoid_reports_real_locomotion_not_a_flat_fall(self):
        # a biped that the multi-leg crawl gait fells must report its REAL best controller: it WALKS FORWARD via
        # the trot (or a banked learned policy), else STANDS -- never a bare 'FELL' implying it can't balance.
        from virturoid.services import session_state as S
        from virturoid.services.ai_native_tools import create_robot, verify_robot
        S.reset()
        rid = create_robot({"prompt": "a humanoid robot"})["robot_id"]
        v = verify_robot({"robot_id": rid, "mode": "quick"})
        self.assertTrue(str(v.get("gait_source", "")).startswith("biped"))
        self.assertNotIn("FELL", v["verdict"])                  # not a flat fall
        self.assertTrue("WALKS FORWARD" in v["verdict"] or "STANDS" in v["verdict"],
                        f"a biped reports forward locomotion or static balance, not a fall: {v['verdict']}")
        self.assertFalse(v.get("credible_walk"))                # honest: a trot/stand is not gated as a credible walk

    def test_a_quad_still_walks(self):
        # the fix must not regress the quad (its rest pose is a valid stance; default walk stays credible)
        from virturoid.services.gait_quality import classify
        from virturoid.services.morph_policy import crawl_gait_rollout
        from virturoid.services.morphology_composer import compose_robot
        r = crawl_gait_rollout(compose_robot("a quadruped robot dog", ensure_walkable=True),
                               steps=1200, record_qpos=True)
        self.assertTrue(classify(r).startswith("CREDIBLE"), f"the quad must still credibly walk: {classify(r)}")


if __name__ == "__main__":
    unittest.main()
