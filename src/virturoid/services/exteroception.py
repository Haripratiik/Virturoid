"""Exteroception (perception Rung A): give a body a ring of range sensors so it can SENSE its
environment instead of being fed privileged state.

Every robot so far is state-blind — fed its own joints + base pose + a hand-given goal/path. The missing
generality axis is the ENVIRONMENT: a blind robot can't navigate an *unknown* maze or avoid an unscripted
obstacle (see docs/next_phase_research.md). This is the cheap, tractable first rung (no Madrona/camera
rendering): a horizontal fan of MuJoCo ``rangefinder`` sensors on the base, each casting a ray outward
and returning the distance to the nearest surface — exactly the exteroception legged sim-to-real policies
use (raycast/heightmap). It is injected by post-processing the compiled MJCF, so it composes with the
maze walls. CPU-validated here before any GPU training. Relates to [[perception-next-keystone]].
"""

from __future__ import annotations

import math
import re


def inject_rangefinders(xml: str, base_body: str, *, n_rays: int = 8, height: float = 0.12) -> str:
    """Insert a horizontal ring of ``n_rays`` rangefinder sites on ``base_body`` + the matching sensor
    block. Each site's +z axis (the ray direction) points outward at an evenly spaced yaw, so the body
    senses range to obstacles all around it. Ray ``i`` points at body-frame yaw ``2*pi*i/n_rays``."""
    sites = []
    for i in range(n_rays):
        a = 2 * math.pi * i / n_rays
        sites.append(f'      <site name="rf{i}" pos="0 0 {height:.4f}" '
                     f'zaxis="{math.cos(a):.5f} {math.sin(a):.5f} 0" size="0.004"/>')
    sites_xml = "\n" + "\n".join(sites)

    # insert the sites as children right after the base body's opening tag
    m = re.search(rf'<body\b[^>]*\bname="{re.escape(base_body)}"[^>]*>', xml)
    if not m:
        raise ValueError(f"base body {base_body!r} not found in MJCF for rangefinder injection")
    xml = xml[:m.end()] + sites_xml + xml[m.end():]

    sensors = "\n".join(f'    <rangefinder name="rf{i}" site="rf{i}"/>' for i in range(n_rays))
    sensor_block = f'  <sensor>\n{sensors}\n  </sensor>\n'
    return xml.replace("</mujoco>", sensor_block + "</mujoco>")


def with_rangefinders(gene, *, walls=None, n_rays: int = 8, height: float = 0.12) -> str:
    """Compile ``gene`` (optionally with maze ``walls``) and add a rangefinder ring on its base."""
    if walls is not None:
        from virturoid.services.maze import build_maze_mjcf
        xml = build_maze_mjcf(gene, walls)
    else:
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        xml = compile_gene_to_mjcf(gene, include_floor=True)
    return inject_rangefinders(xml, gene.root().name, n_rays=n_rays, height=height)


def read_ranges(model, data, *, n_rays: int = 8, max_range: float = 3.0):
    """Read the rangefinder ring as a fixed-width vector of distances in [0, max_range]. MuJoCo returns
    -1 when a ray hits nothing → mapped to ``max_range`` (open space). Robust to sensor ordering (reads
    each ``rf{i}`` by name)."""
    import mujoco
    import numpy as np

    out = np.full(n_rays, max_range, dtype=float)
    for i in range(n_rays):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"rf{i}")
        if sid < 0:
            continue
        d = float(data.sensordata[int(model.sensor_adr[sid])])
        out[i] = max_range if d < 0 else min(d, max_range)
    return out
