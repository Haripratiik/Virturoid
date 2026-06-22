"""Task-space (operational-space) control: the base controller that makes HIGH-DOF arms reachable.

The species experiment showed joint-PD-to-an-IK-config can't dynamically realize a reachable pose on a
7-DOF redundant arm (settles ~0.22 m off). Jacobian-transpose task-space control servos the end-effector
directly and resolves redundancy, so the same arm reaches. This is the competent base high-DOF arms need
(and a base for a residual policy)."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class TaskSpaceControlTests(unittest.TestCase):
    def _reach(self, prompt, controller):
        import numpy as np
        import mujoco
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.gene_compiler import compile_gene_with_scene
        from virturoid.services.pick_place_controller import (
            task_space_reach_ctrl, plan_joint_targets, _actuated_joint_map,
            _actuator_force_clamps, _ee_site_id, _ee_position)
        from virturoid.schemas.scenes import SceneObject

        g = compose_robot(prompt)
        mj = mujoco.MjModel.from_xml_string(
            compile_gene_with_scene(g, [SceneObject("box", "cube", (0.4, 0, 0.05, 0, 0, 0), mass_kg=0.03, scale=1.0)]))
        ee = _ee_site_id(mj); act = _actuated_joint_map(mj); clamps = _actuator_force_clamps(mj)
        target = np.array([0.4, 0.0, 0.07])
        d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)
        if controller == "task_space":
            for _ in range(800):
                mujoco.mj_forward(mj, d)
                task_space_reach_ctrl(mj, d, ee, target, act, clamps)
                mujoco.mj_step(mj, d)
        else:  # joint-PD to an IK config
            q = plan_joint_targets(mj, tuple(target))[0]
            for _ in range(800):
                for slot, (qadr, vadr, jnt) in enumerate(act):
                    tgt = q[slot] if slot < len(q) else 0.0
                    d.ctrl[slot] = float(np.clip(d.qfrc_bias[vadr] + 60 * (tgt - d.qpos[qadr]) - 6 * d.qvel[vadr],
                                                 -clamps[slot], clamps[slot]))
                mujoco.mj_step(mj, d)
        return float(np.linalg.norm(_ee_position(d, ee) - target))

    def test_task_space_reaches_high_dof_arm(self):
        # a 7-DOF arm reaches the target under task-space control (joint-PD leaves it far away)
        ts = self._reach("a Franka-style 7-dof arm for precise insertion", "task_space")
        pd = self._reach("a Franka-style 7-dof arm for precise insertion", "joint_pd")
        self.assertLess(ts, 0.10)                 # task-space gets the ee to the target
        self.assertLess(ts, pd)                   # and beats joint-PD on the high-DOF arm

    def test_task_space_control_is_stable(self):
        import numpy as np  # noqa: F401
        ts = self._reach("a precise 6-dof industrial arm", "task_space")
        self.assertLess(ts, 0.12)                 # finite + reaches (no blow-up)

    def test_high_dof_arm_performs_pick_place_via_pipeline(self):
        """End-to-end: the pick-place pipeline auto-selects task-space control for a composed 7-DOF arm,
        so it actually grasps + places (joint-PD gave a flat 0). DOF-based base selection working."""
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_runtime import select_task_spec, generate_task_scenes, evaluate_gene_on_task

        g = compose_robot("a Franka-style 7-dof arm that sorts red and blue blocks into bins")
        self.assertGreaterEqual(sum(1 for s in g.segments if s.joint_type == "revolute"
                                    and "finger" not in s.name), 6)   # it really is high-DOF
        spec = select_task_spec("sort red and blue blocks into bins")
        scenes = generate_task_scenes(g, spec, count=3)
        r = evaluate_gene_on_task(g, spec, scenes)
        self.assertGreater(r["blocks_placed"], 0)     # the high-DOF arm places at least one block


if __name__ == "__main__":
    unittest.main()
