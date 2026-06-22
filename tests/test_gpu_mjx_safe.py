"""GPU/MJX safety invariant: the physics_only compile must emit a model MJX can put_model (no CYLINDER geoms,
which MJX cannot collide), while preserving the colliders/joints/actuators that matter. This is THE enabler for
GPU training (MJX PPO crashed on the cosmetic cylinder motor-housings the mechanical-detail work added). If this
regresses, every GPU training run breaks again. See [[task-effectiveness-loop]], [[mjx-rl-debugging-method]]."""

import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _cyl_count(model):
    import mujoco
    return sum(1 for i in range(model.ngeom) if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER))


def _collision_types(model):
    import mujoco
    t = {2: "sphere", 3: "capsule", 5: "cylinder", 6: "box", 0: "plane"}
    out = {}
    for i in range(model.ngeom):
        if int(model.geom_contype[i]) != 0 or int(model.geom_conaffinity[i]) != 0:
            k = t.get(int(model.geom_type[i]), "other")
            out[k] = out.get(k, 0) + 1
    return out


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class MjxSafeCompileTests(unittest.TestCase):
    def test_physics_only_has_no_cylinder_geoms_across_bodies(self):
        # MJX precompiles a collision fn per geom-type-pair PRESENT; a single cylinder + box crashes put_model.
        # physics_only must leave ZERO cylinders for EVERY composed body (the GPU script forces all collidable).
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morphology_composer import compose_robot
        for prompt in ("walk forward across the floor", "a tabletop robot arm", "a wheeled rover", "a humanoid robot"):
            g = compose_robot(prompt, llm=None)
            full = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))
            phys = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True, physics_only=True))
            self.assertGreater(_cyl_count(full), 0, f"{prompt}: expected cosmetic cylinders in the full model")
            self.assertEqual(_cyl_count(phys), 0, f"{prompt}: physics_only must have NO cylinders (MJX-unsafe)")
            # and it still loads + steps (so the GPU all-collidable sweep won't crash)
            for i in range(phys.ngeom):
                phys.geom_contype[i] = 1
                phys.geom_conaffinity[i] = 1
            d = mujoco.MjData(phys)
            mujoco.mj_forward(phys, d)
            mujoco.mj_step(phys, d)

    def test_physics_only_preserves_collision_physics_for_box_capsule_bodies(self):
        # the quad + arm have only box/capsule colliders -> physics_only is byte-identical collision physics
        # (only the visual cylinders are stripped). Bodies WITH cylinder colliders get a capsule conversion.
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morphology_composer import compose_robot
        for prompt in ("walk forward across the floor", "a tabletop robot arm"):
            g = compose_robot(prompt, llm=None)
            full = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))
            phys = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True, physics_only=True))
            self.assertEqual(_collision_types(full), _collision_types(phys), prompt)

    def test_manipulation_scene_compile_is_mjx_safe_and_keeps_gripper(self):
        # the grasp/push MJX trainers compile gene+scene; physics_only must drop cylinders but keep the gripper
        # TCP site + fingers + the free object so the contact-based grasp reward still works.
        import mujoco
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.schemas.scenes import SceneObject
        from virturoid.services.design_critic import add_parallel_gripper
        from virturoid.services.gene_compiler import compile_gene_with_scene
        g = add_parallel_gripper(tabletop_arm_gene())
        box = SceneObject("box", "cube", (0.46, 0.0, 0.05, 0, 0, 0), 0.03, "gray_block", 1.0, 1.0)
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(g, [box], physics_only=True))
        self.assertEqual(_cyl_count(mj), 0)
        sites = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mj.nsite)}
        self.assertIn("grasp_site", sites)            # the grasp TCP the residual policy targets

    def test_manipulation_task_routes_to_a_real_mjx_script(self):
        from pathlib import Path

        from virturoid.services.gpu_trainer import _MANIP_SCRIPT
        for task in ("grasp", "pick_place", "pick_place_sort", "push", "reach"):
            script = _MANIP_SCRIPT[task]
            self.assertTrue((Path("scripts") / script).exists(), f"{task} -> missing {script}")


if __name__ == "__main__":
    unittest.main()
