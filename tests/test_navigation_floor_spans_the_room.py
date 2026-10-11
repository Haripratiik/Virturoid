"""The navigation course's floor reaches the walls around it.

MEASURED 2026-10-11 while rendering the README's scene picture: the course's walls had been moved out to a
near-square room while its floor kept the old 0.8 m half-width, and a floor scene has no ground plane under it. On
the generated course the walls stood at +-1.53 m, leaving 0.7 m of void inside each long wall, and the navigator
takes its room from the walls, so a detour it planned there drove off the edge of the world.
"""
from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("purpose", ["nominal", "variation", "edge_case"])
def test_there_is_floor_everywhere_inside_the_walls(purpose):
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.scene_generator import generate_scene_set
    from virturoid.services.task_builder import build_task_graph

    rover = compose_robot("a mobile robot base that navigates indoors", llm=None)
    task = build_task_graph(build_requirements_from_prompt("navigate around the obstacles to the goal"))
    assert task.task_type == "navigation"
    for scene in generate_scene_set(task, count=3, purpose=purpose, robot_radius=0.2).scenes:
        m = mujoco.MjModel.from_xml_string(compile_gene_with_scene(rover, scene.objects))
        geoms = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(m.ngeom)}
        if "wall_top" not in geoms:
            continue
        f = geoms["arena_floor"]
        fx, fy = m.geom_pos[f][:2]
        hx, hy = m.geom_size[f][:2]
        inner = {"x": (m.geom_pos[geoms["wall_left"]][0], m.geom_pos[geoms["wall_right"]][0]),
                 "y": (m.geom_pos[geoms["wall_bottom"]][1], m.geom_pos[geoms["wall_top"]][1])}
        assert fx - hx <= inner["x"][0] and inner["x"][1] <= fx + hx, (scene.id, inner, fx, hx)
        assert fy - hy <= inner["y"][0] and inner["y"][1] <= fy + hy, (scene.id, inner, fy, hy)
