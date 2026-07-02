"""Terrain difficulty ladder (breakthrough plan v2 §5.3 / gap G3) — procedural floor variants the night shift
and VIRT-Bench climb: flat -> rough -> stairs -> gaps.

Research (stream 4): MJX has supported hfield collision since MuJoCo 3.1.6 (we run 3.9.0), but MuJoCo
Playground's rough-terrain envs keep amplitudes tiny (~5 cm) AND reduce collision to FEET-ONLY, because MJX
throughput falls sharply with contact count; their STAIRS are plain box geoms, not an hfield. So this ladder is
built from PLAIN BOX GEOMS (the cross-version-safe, asset-free path — we control the MJCF), with a FIXED geom
COUNT per level (per-env difficulty comes from randomizing sizes/heights via model DR, never geom count — a
count change would force an MJX recompile). Each function returns the ``<geom>`` XML for the floor region; the
robot's own foot geoms should carry ``condim=3`` and the terrain is the only other colliding surface.
"""

from __future__ import annotations

import random

TERRAIN_LEVELS = ("flat", "rough", "stairs", "gaps")


def flat_floor(*, size: float = 5.0) -> str:
    """L0: the default infinite plane (what every scene uses today)."""
    return f'<geom name="floor" type="plane" size="{size} {size} 0.1" material="grid"/>'


def rough_floor(*, tiles: int = 8, span: float = 4.0, amplitude: float = 0.04, seed: int = 0) -> str:
    """L1: a ``tiles x tiles`` grid of box pads with seeded, bounded height jitter (Playground-style ~cm rough
    terrain, box-tile variant). Deterministic given ``seed`` (reproducible). Includes a base plane underneath so
    a body never falls through the gaps between pads."""
    rng = random.Random(seed)
    step = span / tiles
    half = step / 2.0
    lines = [flat_floor(size=span)]
    for ix in range(tiles):
        for iy in range(tiles):
            x = -span / 2 + step * (ix + 0.5)
            y = -span / 2 + step * (iy + 0.5)
            h = max(0.005, amplitude * rng.random())
            lines.append(f'<geom name="rough_{ix}_{iy}" type="box" pos="{x:.4f} {y:.4f} {h/2:.4f}" '
                         f'size="{half:.4f} {half:.4f} {h/2:.4f}" material="grid"/>')
    return "\n".join(lines)


def stairs(*, n: int = 8, rise: float = 0.08, tread: float = 0.28, width: float = 2.0, x0: float = 0.5) -> str:
    """L2: an ascending box staircase in +x (Playground Go1 uses ~0.25 m tread, 0.05-0.17 m rise). Fixed ``n``
    steps; difficulty knob is ``rise``. A base plane underlies the approach."""
    lines = [flat_floor(size=max(width, x0 + n * tread + 1.0))]
    for i in range(n):
        top = rise * (i + 1)
        cx = x0 + tread * (i + 0.5)
        lines.append(f'<geom name="stair_{i}" type="box" pos="{cx:.4f} 0 {top/2:.4f}" '
                     f'size="{tread/2:.4f} {width/2:.4f} {top/2:.4f}" material="grid"/>')
    return "\n".join(lines)


def gaps(*, n: int = 5, platform: float = 0.6, gap: float = 0.25, width: float = 2.0, height: float = 0.2,
         x0: float = 0.5) -> str:
    """L3: raised platforms separated by gaps in +x — the body must stride across (difficulty knob = ``gap``).
    NO base plane between platforms (the gap is real); a start plane sits under the approach only."""
    lines = [f'<geom name="start_pad" type="box" pos="0 0 {height/2:.4f}" '
             f'size="{x0/2:.4f} {width/2:.4f} {height/2:.4f}" material="grid"/>']
    cx = x0
    for i in range(n):
        cx += gap + platform / 2
        lines.append(f'<geom name="plat_{i}" type="box" pos="{cx:.4f} 0 {height/2:.4f}" '
                     f'size="{platform/2:.4f} {width/2:.4f} {height/2:.4f}" material="grid"/>')
        cx += platform / 2
    return "\n".join(lines)


def terrain_mjcf(level: str = "flat", *, difficulty: float = 0.5, seed: int = 0) -> str:
    """Return the floor-region ``<geom>`` XML for ``level``. ``difficulty`` in [0,1] scales the level's hardness
    knob (amplitude / rise / gap) between gentle and hard bounds -- the curriculum handle the night shift ramps."""
    d = max(0.0, min(1.0, float(difficulty)))
    if level == "flat":
        return flat_floor()
    if level == "rough":
        return rough_floor(amplitude=0.02 + 0.08 * d, seed=seed)          # 2 cm -> 10 cm
    if level == "stairs":
        return stairs(rise=0.05 + 0.12 * d)                               # 5 cm -> 17 cm rise (Playground range)
    if level == "gaps":
        return gaps(gap=0.10 + 0.40 * d)                                  # 10 cm -> 50 cm gap
    raise ValueError(f"unknown terrain level {level!r}; known: {TERRAIN_LEVELS}")


def scene_with_terrain(robot_mjcf_body: str, level: str = "flat", *, difficulty: float = 0.5, seed: int = 0,
                       timestep: float = 0.002) -> str:
    """Wrap a robot ``<body>...</body>`` MJCF fragment in a minimal worldbody over the chosen terrain -> a full
    MJCF string. For quick terrain checks + as a template for the scene compiler's terrain-task path."""
    return (
        f'<mujoco><option timestep="{timestep}" gravity="0 0 -9.81"/>'
        f'<asset><texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" '
        f'width="300" height="300"/><material name="grid" texture="grid" texrepeat="8 8" reflectance=".1"/></asset>'
        f'<worldbody>{terrain_mjcf(level, difficulty=difficulty, seed=seed)}{robot_mjcf_body}</worldbody></mujoco>'
    )
