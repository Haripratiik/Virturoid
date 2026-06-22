"""METHOD A — REAL-FIRST overhaul prototype (SELF-CONTAINED; does not edit shared source).

Proves the real-model substrate (MuJoCo Menagerie via ``robot_descriptions``) as the DEFAULT body
for every covered robot class, and FIXES the dark-render problem so each model renders clearly.

Root cause of the dark renders (verified by inspecting the loaded MjModel):
  - Unitree H1's dominant material is rgba [0.1, 0.1, 0.1, 1]  (near-black)
  - Unitree Go1's only material is   rgba [0.2, 0.2, 0.2, 1]  (dark grey)
  - many geoms carry NO material at all and fall back to a dark ``geom_rgba``
  brightening only the headlight cannot rescue an albedo that is ~0.1 — almost no light reflects.

The fix (display-only, geometry untouched):
  1. Lift any near-black MATERIAL rgba to a neutral mid grey, preserving hue/relative brightness.
  2. Lift any near-black per-geom rgba (geoms with no material) the same way.
  3. Tame specular/shininess so models do not blow out to white highlights.
  4. Normalize lighting: bright neutral headlight + a soft fill, and a couple of directional lights.
None of this changes mass, inertia, collision geometry, or joints — only how the surfaces shade.

    PYTHONPATH=src python scripts/overhaul/real_first.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from virturoid.services.real_model_library import load_real_model  # noqa: E402

OUT_DIR = ROOT / "build" / "overhaul" / "real_first"

# One representative model per covered class (registry default for each class).
COVERED = [
    "humanoid", "quadruped", "arm", "hand", "gripper",
    "drone", "biped", "bimanual", "mobile_manipulator",
]

DISPLAY_FLOOR = 0.18   # any albedo channel below this gets lifted (near-black -> visible)
DISPLAY_MIN = 0.45     # lifted channels land at/above this neutral grey


def _lift_rgb(rgb: np.ndarray) -> np.ndarray:
    """Lift a near-black colour to a visible neutral grey, preserving relative hue.

    A colour whose brightest channel is below DISPLAY_FLOOR is treated as "display-black": it is
    rescaled so its brightest channel reaches DISPLAY_MIN (keeps any tint, just makes it visible).
    Colours that are already bright enough are returned unchanged.
    """
    rgb = np.asarray(rgb, dtype=float)[:3].copy()
    peak = float(rgb.max())
    if peak < 1e-6:                       # pure black -> flat neutral grey
        return np.array([DISPLAY_MIN, DISPLAY_MIN, DISPLAY_MIN])
    if peak < DISPLAY_FLOOR:
        return np.clip(rgb * (DISPLAY_MIN / peak), 0.0, 1.0)
    return rgb


def normalize_materials(m: mujoco.MjModel) -> dict:
    """Display-only material/colour normalization. Returns a small report of what changed."""
    changed_mats, changed_geoms = 0, 0
    # 1) materials
    for i in range(m.nmat):
        old = m.mat_rgba[i][:3].copy()
        new = _lift_rgb(old)
        if not np.allclose(old, new):
            m.mat_rgba[i][:3] = new
            changed_mats += 1
        # tame highlights so chrome/aluminium parts don't blow out to white
        m.mat_specular[i] = min(float(m.mat_specular[i]), 0.3)
        m.mat_shininess[i] = min(float(m.mat_shininess[i]), 0.3)
        m.mat_reflectance[i] = min(float(m.mat_reflectance[i]), 0.1)
    # 2) per-geom rgba (geoms with no material), skip the floor plane (handled separately)
    floor_gid = -1
    for g in range(m.ngeom):
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            floor_gid = g
            continue
        if int(m.geom_matid[g]) >= 0:
            continue
        old = m.geom_rgba[g][:3].copy()
        new = _lift_rgb(old)
        if not np.allclose(old, new):
            m.geom_rgba[g][:3] = new
            changed_geoms += 1
    return {"materials_lifted": changed_mats, "geoms_lifted": changed_geoms, "floor_gid": floor_gid}


def normalize_lighting(m: mujoco.MjModel) -> None:
    """Bright, neutral, even lighting so every model is legible regardless of its own materials."""
    # headlight follows the camera -> guarantees the front-facing surfaces are always lit
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.vis.headlight.diffuse[:] = [0.75, 0.75, 0.75]
    m.vis.headlight.specular[:] = [0.10, 0.10, 0.10]
    # global scene tint / shadows
    m.vis.rgba.haze[:] = [0.92, 0.94, 0.97, 1.0]


def _studio_background(grid: np.ndarray, bg=(232, 236, 242), bg_floor=(200, 206, 214)) -> np.ndarray:
    """Composite the rendered robots onto a soft vertical-gradient studio backdrop.

    MuJoCo's offscreen renderer fills unlit pixels with black; a flat black sheet makes a production
    model look like a screenshot. We treat near-black pixels as background and replace them with a
    light top-to-bottom gradient, so each robot reads like a catalog render. Robot pixels (anything
    that isn't near-black) are left exactly as rendered — geometry/material are untouched.
    """
    h, w, _ = grid.shape
    top = np.array(bg, dtype=float)
    bot = np.array(bg_floor, dtype=float)
    ramp = np.linspace(0.0, 1.0, h)[:, None]
    backdrop = (top[None, :] * (1 - ramp) + bot[None, :] * ramp)        # (h,3)
    backdrop = np.broadcast_to(backdrop[:, None, :], (h, w, 3))
    lum = grid.astype(float).mean(axis=2)
    is_bg = lum < 8.0                                                    # near-black -> background
    out = grid.astype(float).copy()
    out[is_bg] = backdrop[is_bg]
    return out.astype(np.uint8)


def render_model(prompt_class: str, size: int = 520) -> tuple[str, dict]:
    info = load_real_model(robot_class=prompt_class)
    if not info.get("ok"):
        return "", {"ok": False, "note": info.get("note", "")}
    m = mujoco.MjModel.from_xml_path(info["path"])
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), size)

    report = normalize_materials(m)
    normalize_lighting(m)
    # give the model a clean light ground so it reads against the floor (display-only)
    if report["floor_gid"] >= 0:
        m.geom_rgba[report["floor_gid"]][:] = [0.62, 0.66, 0.72, 1.0]

    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)     # the model's natural "home" pose
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    pts = d.geom_xpos[: m.ngeom]
    center = pts.mean(axis=0)
    extent = float((pts.max(axis=0) - pts.min(axis=0)).max())

    renderer = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = max(0.5, extent * 1.75)        # tighter framing -> robot fills the tile
    shots = []
    for az, el in [(90, -12), (45, -18), (140, -20), (20, -65)]:
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(d, cam)
        shots.append(renderer.render())
    top = np.concatenate(shots[:2], axis=1)
    bot = np.concatenate(shots[2:], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    grid = _studio_background(grid)               # soft studio backdrop instead of flat black

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{prompt_class}__{info['key']}.png"
    Image.fromarray(grid).save(out)

    # brightness probe so we can claim "renders clearly" with a number, not a vibe
    mean_lum = float(grid.mean())
    report.update({
        "ok": True, "out": str(out), "label": info["label"], "kind": info["kind"],
        "bodies": info["bodies"], "actuated": info["actuated"], "meshes": info["meshes"],
        "mean_luminance": round(mean_lum, 1),
    })
    return str(out), report


def main() -> int:
    rows = []
    for cls in COVERED:
        out, rep = render_model(cls)
        if rep.get("ok"):
            print(f"{cls:20} {rep['label']:26} meshes={rep['meshes']:>3} dof={rep['actuated']:>3} "
                  f"lifted(mat/geom)={rep['materials_lifted']}/{rep['geoms_lifted']} "
                  f"lum={rep['mean_luminance']:>5} -> {out}")
        else:
            print(f"{cls:20} FAILED: {rep.get('note')}")
        rows.append((cls, rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
