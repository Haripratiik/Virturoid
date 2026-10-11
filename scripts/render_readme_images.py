"""Render the README's two pictures from the current generator: ``assets/robot_gallery.png`` (one body per prompt)
and ``assets/scene_generation.png`` (one generated scene per task prompt).

    python scripts/render_readme_images.py [--out assets]

Every body is composed offline from the prompt shown in ``ROBOTS``/``TASKS`` (no LLM, no network) and drawn as the
viewport draws it, standing in its rest pose. The one liberty is the arm's pose: a manipulator rests straight up, so
it is shown bent (gallery) or reaching over its task (scenes), every angle inside its joint's range. Scenes are made
by the same calls the package builder makes (``gene_build``), so a picture of a scene is a picture of that scene.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
# Rendering composes robots and scenes and writes nothing to the memory bank; point the bank at a throwaway
# directory anyway, so a future write on these paths cannot land in the real corpus from a README refresh.
os.environ.setdefault("VIRTUROID_MEMORY_DIR", tempfile.mkdtemp(prefix="virturoid_readme_"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROBOTS = [
    ("manipulator", "a 6-axis robot arm that picks up parts"),
    ("quadruped", "a four legged robot dog"),
    ("hexapod", "a six legged walking robot"),
    ("mobile base", "a mobile robot base that navigates indoors"),
    ("humanoid", "a humanoid robot"),
    ("octopus", "an octopus robot"),
]
ARM, BASE = ROBOTS[0][1], ROBOTS[3][1]
TASKS = [
    ("sort", ARM, "sort the red and blue blocks into their bins"),
    ("stack", ARM, "stack the blocks into a tower"),
    ("push", ARM, "push the block to the target zone"),
    ("lift", ARM, "lift the box onto the shelf"),
    ("navigate", BASE, "navigate around the obstacles to the goal"),
    ("maze", BASE, "navigate through a maze to the goal"),
]
TILE_W, TILE_H = 600, 450
LABEL_H = 0.117                      # share of a tile the label bar takes
BAR, INK = (17, 20, 28), (166, 227, 161)
#: the studio backdrop the gallery gets from the compiler's no-scene path; a scene file has none (render only)
SKYBOX = ('<texture name="readme_sky" type="skybox" builtin="gradient" rgb1="0.52 0.56 0.62" '
          'rgb2="0.15 0.17 0.21" width="512" height="512"/>')


def _font(px: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _geom_corners(m, d, i):
    """The 8 world-frame corners of geom ``i``'s own bounding box (tighter than its bounding sphere)."""
    c, half = m.geom_aabb[i][:3], m.geom_aabb[i][3:]
    rot, x = d.geom_xmat[i].reshape(3, 3), d.geom_xpos[i]
    return [x + rot @ (c + half * np.array((sx, sy, sz))) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _fit_distance(lo, hi, center, azimuth, elevation, fovy_deg, aspect, margin=1.06):
    """The camera distance at which every corner of the box ``lo``..``hi`` fits the frame."""
    az, el = np.radians(azimuth), np.radians(elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    t = np.tan(np.radians(fovy_deg) / 2.0)
    need = 0.0
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                c = np.array([cx, cy, cz]) - center
                need = max(need, abs(c @ up) * margin / t - c @ fwd, abs(c @ right) * margin / (t * aspect) - c @ fwd)
    return need


def _bend(m, d, bends):
    """Pitch the first pitch joints (horizontal axis at the current pose) by ``bends``, each within its range."""
    mujoco.mj_forward(m, d)
    k = 0
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or k >= len(bends):
            continue
        if abs((d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j])[2]) > 0.5:      # a yaw joint: leave it
            continue
        q = bends[k]
        if m.jnt_limited[j]:
            lo, hi = m.jnt_range[j]
            q = q if lo <= q <= hi else (-q if lo <= -q <= hi else float(np.clip(q, lo, hi)))
        d.qpos[m.jnt_qposadr[j]] = q
        mujoco.mj_forward(m, d)
        k += 1


def _reach_over(m, d, target):
    """Turn and bend the arm so its last link hovers 12 cm over ``target``: a search over base yaw and three pitch
    angles, every angle inside its joint's range."""
    mujoco.mj_forward(m, d)
    yaw = next((j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                and abs((d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j])[2]) > 0.5), None)
    depth = [0] * m.nbody
    for b in range(1, m.nbody):
        depth[b] = depth[int(m.body_parentid[b])] + 1
    ee = max((b for b in range(1, m.nbody)
              if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("obj_")),
             key=lambda b: depth[b])
    q0, best = d.qpos.copy(), None
    grid = [(a, b, c) for a in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6) for b in (0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4)
            for c in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)]
    for sign in (1.0, -1.0):
        for bends in grid:
            d.qpos[:] = q0
            _bend(m, d, tuple(sign * v for v in bends))
            bent = d.qpos.copy()
            for yq in np.linspace(-np.pi, np.pi, 37):
                d.qpos[:] = bent
                if yaw is not None:
                    lo, hi = m.jnt_range[yaw] if m.jnt_limited[yaw] else (-np.pi, np.pi)
                    d.qpos[m.jnt_qposadr[yaw]] = float(np.clip(yq, lo, hi))
                mujoco.mj_forward(m, d)
                p = d.xpos[ee]
                cost = float(np.linalg.norm(p[:2] - target[:2])) + abs(float(p[2]) - (float(target[2]) + 0.12))
                if best is None or cost < best[0]:
                    best = (cost, d.qpos.copy())
    d.qpos[:] = best[1]
    mujoco.mj_forward(m, d)


def _render(m, d, pts, w, h, azimuth, elevation, margin):
    pts = np.array(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    lo[2] = max(lo[2], 0.0)
    center = (lo + hi) / 2.0
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 2 * w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 2 * h)
    m.mat_reflectance[:] = np.minimum(m.mat_reflectance, 0.08)          # a quieter floor than the viewport's
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.azimuth, cam.elevation = azimuth, elevation
    cam.distance = _fit_distance(lo, hi, center, azimuth, elevation, float(m.vis.global_.fovy), w / h, margin)
    r = mujoco.Renderer(m, 2 * h, 2 * w)
    r.update_scene(d, cam)
    img = Image.fromarray(r.render()).resize((w, h), Image.LANCZOS)    # rendered at 2x, downsampled
    r.close()
    return img


def render_body(prompt: str, w: int, h: int):
    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot(prompt, llm=None)
    free = getattr(g, "base_mount", "free") == "free"
    m = mujoco.MjModel.from_xml_string(
        gene_to_meshed_mjcf(g, include_floor=True, spawn_z=standing_spawn_z(g) if free else None))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    if g.robot_class == "manipulator":
        _bend(m, d, (0.3, 0.95, 0.75))
    mujoco.mj_forward(m, d)
    pts = [c for i in range(m.ngeom) if m.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE for c in _geom_corners(m, d, i)]
    return _render(m, d, pts, w, h, 140.0, -18.0, 1.06), g


def render_scene(robot_prompt: str, task_prompt: str, w: int, h: int):
    from virturoid.services.gene_build import _robot_footprint_radius, _spec_from_scene
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.requirements_builder import build_requirements_from_prompt
    from virturoid.services.scene_generator import generate_scene_set
    from virturoid.services.task_builder import build_task_graph
    from virturoid.services.task_runtime import generate_task_scenes, select_task_spec
    g = compose_robot(robot_prompt, llm=None)
    task = build_task_graph(build_requirements_from_prompt(task_prompt))
    nav = task.task_type == "navigation"
    if nav:                                # the calls the navigation package makes
        scene = generate_scene_set(task, count=1, purpose="variation", robot_radius=_robot_footprint_radius(g)).scenes[0]
    else:                                  # the calls the manipulation package makes
        spec = select_task_spec(task_prompt)
        if task.task_type == "pick_place_sort":
            scenes = generate_task_scenes(g, spec, count=1)
        else:
            spec = _spec_from_scene(spec, generate_scene_set(task, count=1, purpose="variation").scenes[0])
            scenes = generate_task_scenes(g, spec, count=1) or generate_scene_set(task, count=1, purpose="variation").scenes
        scene = scenes[0]
    xml = compile_gene_with_scene(g, scene.objects).replace("<asset>", "<asset>\n    " + SKYBOX, 1)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    if g.robot_class == "manipulator":
        objs = [d.xpos[b] for b in range(m.nbody)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("obj_")]
        _reach_over(m, d, np.mean(objs, axis=0) if objs else np.array([0.4, 0.0, 0.05]))
    pts = []
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        corners = np.array(_geom_corners(m, d, i))
        span = corners.max(axis=0) - corners.min(axis=0)
        if not nav and m.geom_bodyid[i] == 0 and max(span[0], span[1]) > 0.9:
            continue                       # the table top: frame the task on it, not the whole table
        pts.extend(corners)
    view = (120.0, -52.0) if nav else (140.0, -32.0)
    return _render(m, d, pts, w, h, *view, 1.04), task.task_type


def _tile(label: str, img, w: int, h: int):
    bar = int(round(LABEL_H * h))
    t = Image.new("RGB", (w, h), BAR)
    t.paste(img, (0, bar))
    ImageDraw.Draw(t).text((int(0.03 * w), int(0.14 * bar)), label, font=_font(int(0.62 * bar)), fill=INK)
    return t


def _sheet(tiles, w: int, h: int):
    sheet = Image.new("RGB", (3 * w, 2 * h), BAR)
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 3) * w, (k // 3) * h))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "assets"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = TILE_W, TILE_H
    body_h = h - int(round(LABEL_H * h))
    tiles = []
    for label, prompt in ROBOTS:
        img, g = render_body(prompt, w, body_h)
        print(f"{label:12s} {prompt!r}: {g.robot_class}, {len(g.segments)} segments")
        tiles.append(_tile(label, img, w, h))
    _sheet(tiles, w, h).save(out / "robot_gallery.png", optimize=True)
    tiles = []
    for label, robot, prompt in TASKS:
        img, task_type = render_scene(robot, prompt, w, body_h)
        print(f"{label:12s} {prompt!r}: {task_type}")
        tiles.append(_tile(label, img, w, h))
    _sheet(tiles, w, h).save(out / "scene_generation.png", optimize=True)
    print("wrote", out / "robot_gallery.png", "and", out / "scene_generation.png")


if __name__ == "__main__":
    main()
