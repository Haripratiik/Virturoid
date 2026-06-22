"""Probe mesh_synth: which backend is live now, and what does the best no-key path (LLM-CAD) produce?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from virturoid.services.llm_client import get_llm  # noqa: E402
from virturoid.services.mesh_synth import available_backend, synthesize_part  # noqa: E402

print("available_backend (cloud/gpu auto):", available_backend())
llm = get_llm("morphology")
print("LLM for LLM-CAD:", type(llm).__name__ if llm else None)
out = Path("build/inspect/synth_head.stl")
out.parent.mkdir(parents=True, exist_ok=True)
res = synthesize_part("a detailed robot sensor head: rounded cranium, two camera eyes, antennae, "
                      "a neck mount collar", 0.16, 0.08, str(out), llm=llm)
if res:
    import mujoco
    import numpy as np
    from PIL import Image
    m = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <visual><global offwidth="600" offheight="600"/><headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7"/></visual>
      <asset><mesh name="h" file="{out.resolve().as_posix()}" scale="0.001 0.001 0.001"/>
        <texture name="g" type="2d" builtin="checker" rgb1=".2 .25 .3" rgb2=".25 .3 .35" width="200" height="200"/>
        <material name="g" texture="g" texrepeat="6 6"/></asset>
      <worldbody><geom name="f" type="plane" size="1 1 .1" material="g"/>
        <geom type="mesh" mesh="h" pos="0 0 0.1" rgba="0.82 0.85 0.9 1"/></worldbody></mujoco>""")
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 600, 600); cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.12]; cam.distance = 0.45; cam.azimuth = 50; cam.elevation = -10
    shots = []
    for az in (50, 140):
        cam.azimuth = az; r.update_scene(d, cam); shots.append(r.render())
    Image.fromarray(np.concatenate(shots, axis=1)).save("build/inspect/synth_head.png")
    print("synthesized head -> build/inspect/synth_head.png")
else:
    print("synth returned None (no backend / disabled)")
