"""Demo: realize several ARBITRARY part shapes (not primitives) and render them in a row — proof the
blueprint vocabulary can model any shape a design needs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from virturoid.services.cad_geometry import realize_shape  # noqa: E402

SPECS = [
    ("L-bracket", {"family": "extrude", "profile": [[0, 0], [60, 0], [60, 18], [18, 18], [18, 60], [0, 60]],
                   "height": 24, "fillet": 3}),
    ("dome shell", {"family": "revolve", "profile": [[40, 0], [38, 18], [30, 32], [16, 42], [2, 46]]}),
    ("tapered limb", {"family": "tapered", "length": 0.14, "r0": 0.04, "r1": 0.016}),
    ("nozzle", {"family": "revolve", "profile": [[14, 0], [14, 30], [22, 40], [8, 56]]}),
    ("star plate", {"family": "extrude",
                    "profile": [[30, 0], [12, 10], [9, 30], [-9, 12], [-30, 9], [-12, -9],
                                [-9, -30], [9, -12]], "height": 12}),
]
out = Path("build/cad_shapes"); out.mkdir(parents=True, exist_ok=True)
meshes, geoms = [], []
for i, (name, spec) in enumerate(SPECS):
    sol = realize_shape(spec)
    import build123d as bd
    f = out / f"{i}.stl"; bd.export_stl(sol, str(f))
    meshes.append(f'<mesh name="m{i}" file="{f.resolve().as_posix()}" scale="0.001 0.001 0.001"/>')
    geoms.append(f'<geom type="mesh" mesh="m{i}" pos="{i*0.18 - 0.36:.3f} 0 0.06" rgba="0.82 0.85 0.9 1"/>')

xml = f"""<mujoco>
  <visual><global offwidth="1100" offheight="320"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.75 0.75 0.75" specular="0.3 0.3 0.3"/></visual>
  <asset>{''.join(meshes)}
    <texture name="g" type="2d" builtin="checker" rgb1="0.2 0.25 0.3" rgb2="0.25 0.3 0.35" width="240" height="240"/>
    <material name="g" texture="g" texrepeat="10 4"/></asset>
  <worldbody><geom name="floor" type="plane" size="2 1 0.1" material="g"/>{''.join(geoms)}</worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 320, 1100); cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.06]; cam.distance = 0.95; cam.azimuth = 70; cam.elevation = -18
r.update_scene(d, cam)
Image.fromarray(r.render()).save("build/inspect/arbitrary_shapes.png")
print("rendered", len(SPECS), "arbitrary shapes ->", ", ".join(n for n, _ in SPECS))
