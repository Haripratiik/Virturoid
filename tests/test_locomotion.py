"""Legged locomotion (§11/§32.1): a composed quadruped stands + trots forward, evaluated by distance."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class LocomotionTests(unittest.TestCase):
    def test_composed_quadruped_walks_forward(self):
        import mujoco
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.locomotion_controller import run_locomotion_episode

        g = compose_robot("a quadruped walking robot")
        self.assertEqual(g.robot_class, "quadruped")
        self.assertEqual(g.base_mount, "free")                    # floating base (so it can move)
        # Real-quadruped accuracy (Go1-class): 4 legs x (abduction + thigh + shin + foot) = 16 segments,
        # 3 actuated joints/leg (foot welded) = 12 DOF — matching a real quadruped baseline, not 8.
        self.assertEqual(sum(1 for s in g.segments if "leg" in s.name), 16)
        self.assertEqual(len(g.actuated_joints()), 12)
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g))
        r = run_locomotion_episode(mj)
        self.assertTrue(r["upright"])                             # legs point down -> it stands
        self.assertEqual(r["status"], "walked")                  # the scripted baseline produces locomotion
        self.assertGreater(r["distance_m"], 0.1)                 # (direction is the learned policy's job)

    def test_leg_count_is_parametric(self):
        # "build whatever I want from scratch": a hexapod gets 6 legs (leg count is a PARAMETER), not the
        # 4-leg quadruped template — while the quadruped stays exactly 4 legs / 8 DOF.
        from virturoid.services.morphology_composer import compose_robot
        h = compose_robot("a six-legged hexapod walking robot")
        leg_groups = {s.name.split("_")[0] for s in h.segments if s.name.startswith("leg")}
        self.assertEqual(len(leg_groups), 6)                  # 6 distinct legs
        self.assertEqual(len(h.actuated_joints()), 18)        # 6 legs x 3 actuated joints (abduction+hip+knee)
        q = compose_robot("a quadruped walking robot")
        self.assertEqual(len(q.actuated_joints()), 12)        # quadruped = 4 legs x 3 DOF

    def test_free_bodies_spawn_standing_not_ejected(self):
        # The spawn-height fix: free bodies spawn on their feet, not penetrating the floor. With the old fixed
        # 0.1 m spawn a humanoid launched to z~2.1 under the contact solver; standing_spawn_z keeps it put.
        import mujoco
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        from virturoid.services.morph_graph import encode_robot
        for prompt in ("a quadruped walking robot", "a humanoid robot"):
            g = compose_robot(prompt)
            xml = compile_gene_to_mjcf(g, include_floor=True, spawn_z=standing_spawn_z(g))
            mj = mujoco.MjModel.from_xml_string(xml); bq = encode_robot(mj).base_qadr
            d = mujoco.MjData(mj); mujoco.mj_forward(mj, d)
            # not penetrating the floor — measured from each geom's TRUE oriented AABB. (geom_rbound is a
            # bounding SPHERE: for a long flat foot box it dips ~half the foot length below the sole even when
            # the foot rests flat, so it false-positives once the feet are correctly grounded.)
            import numpy as np
            lowest = 1e9
            for i in range(mj.ngeom):
                if mj.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                c, h = mj.geom_aabb[i, :3], mj.geom_aabb[i, 3:]
                R = d.geom_xmat[i].reshape(3, 3)
                lowest = min(lowest, float(d.geom_xpos[i, 2] + R[2] @ c - np.abs(R[2]) @ h))
            self.assertGreater(lowest, -0.03)
            z0 = float(d.qpos[bq + 2])
            for _ in range(150):
                mujoco.mj_step(mj, d)
            self.assertLess(float(d.qpos[bq + 2]), z0 + 0.5)                              # not ejected upward
            q = d.qpos[bq + 3:bq + 7]
            self.assertGreater(1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2), 0.5)                  # stays upright

    def test_legs_point_down(self):
        # the composition fix: quadruped legs are oriented downward (else the body can't stand)
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        hips = [s for s in g.segments if s.name.endswith("_0") and "leg" in s.name]
        self.assertTrue(hips and all(abs(h.mount_euler[0]) > 1.0 for h in hips))   # rotated to point down


if __name__ == "__main__":
    unittest.main()
