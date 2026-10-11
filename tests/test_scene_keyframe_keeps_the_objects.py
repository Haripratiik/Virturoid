"""A scene's keyframe keeps every movable object where the scene put it.

MEASURED 2026-10-11 while rendering the README's scene picture: the scene compiler appended the robot's own rest
keyframe, which covers only the robot's joints, and MuJoCo fills the rest of a short key with zeros. Resetting a
sort scene to ``home`` put both blocks at the world origin, inside the arm's base, with a zero quaternion, and every
exported scene file carried that key.
"""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


def _blocks(m):
    return [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("obj_")]


def test_every_key_leaves_the_blocks_where_the_scene_placed_them():
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec

    arm = compose_robot("a 6-axis robot arm that picks up parts", llm=None)
    scene = generate_task_scenes(arm, select_task_spec("sort the red and blue blocks into their bins"), count=1)[0]
    m = mujoco.MjModel.from_xml_string(compile_gene_with_scene(arm, scene.objects))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    blocks = _blocks(m)
    assert len(blocks) == 2 and m.nkey == 2
    placed = {b: d.xpos[b].copy() for b in blocks}
    for k in range(m.nkey):
        mujoco.mj_resetDataKeyframe(m, d, k)
        mujoco.mj_forward(m, d)
        for b in blocks:
            assert np.allclose(d.xpos[b], placed[b], atol=1e-4), (k, d.xpos[b], placed[b])
            assert placed[b][2] > 0.03                     # on the table top, not at the origin


def test_the_robot_part_of_the_key_is_the_robot_rest_pose():
    """The scene key extends the robot's key; it does not replace it. A legged body with a baked rest stance keeps
    that stance at the head of the vector."""
    import re

    from virturoid.services.gene_compiler import _base_z_for, _pose_keyframe, compile_gene_with_scene
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.schemas.scenes import SceneObject

    dog = compose_robot("a four legged robot dog", llm=None)
    cube = SceneObject(name="ball", object_type="block", pose_xyz_rpy=(0.6, 0.0, 0.05, 0.0, 0.0, 0.0),
                       material="matte_red", scale=1.0)
    xml = compile_gene_with_scene(dog, [cube], table=False)
    robot_q = re.search(r'qpos="([^"]*)"', _pose_keyframe(dog, _base_z_for(dog))).group(1).split()
    scene_q = re.search(r'<key name="home" qpos="([^"]*)"', xml).group(1).split()
    assert scene_q[:len(robot_q)] == robot_q
    m = mujoco.MjModel.from_xml_string(xml)
    assert len(scene_q) == m.nq
