"""GPU-box generative-3D worker: text -> reference image (SDXL-Turbo) -> 3D mesh (TripoSR), all local on the
RTX 3060. Run over Tailscale SSH by the laptop's mesh_synth 'gpu_box' backend. Prints MESH_OK <path> on success.

    ~/torch/bin/python ~/gen3d.py --prompt "a robot gripper" --out /tmp/part.glb [--res 256]
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True)
ap.add_argument("--out", required=True)                 # .stl (fitted, mm) for the compiler, or .glb (raw)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--length_m", type=float, default=0.1)  # fit: longest axis -> this length along +z
ap.add_argument("--radius_m", type=float, default=0.03)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
TRI = os.path.expanduser("~/TripoSR")
sys.path.insert(0, TRI)            # resolve the torchmcubes shim + the tsr package from here

# 1) text -> image, SDXL-Turbo (local on the GPU)
from diffusers import AutoPipelineForText2Image

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to(dev)
prompt = (args.prompt + ", single isolated object, centered, plain solid white background, "
          "industrial product studio render, sharp focus, high detail, no shadow")
img = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
imgp = args.out + ".input.png"
img.save(imgp)
del pipe
torch.cuda.empty_cache()
print("IMAGE_OK", imgp, flush=True)

# 2) image -> 3D shape, Hunyuan3D-2 (DiT flow-matching; skimage marching cubes via mc_algo='mc' so no nvcc)
sys.path.insert(0, os.path.expanduser("~/Hunyuan3D-2"))
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

img3d = BackgroundRemover()(Image.open(imgp).convert("RGBA"))
pipe3d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
out = pipe3d(image=img3d, num_inference_steps=30, octree_resolution=args.res,
             mc_algo="mc", guidance_scale=5.5, output_type="trimesh")
mesh = out[0]
while isinstance(mesh, list):
    mesh = mesh[0]
print("SHAPE_OK", flush=True)

# 3) fit to the link's [0, length] +z frame (orient longest axis -> z, uniform-scale, center x/y, floor),
#    then export an mm STL the compiler drops straight onto the primitive (0.001 scale). Same convention as
#    part_catalog._fit, done here so the laptop needs no mesh libs — it just pulls the STL.
import trimesh

v = np.asarray(mesh.vertices, dtype=float)
ext = v.max(0) - v.min(0)
axis = int(np.argmax(ext))
if axis == 0:
    v = v @ np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]).T
elif axis == 1:
    v = v @ np.array([[1, 0, 0], [0, 0, -1.0], [0, 1, 0]]).T
ext = v.max(0) - v.min(0)
v *= (max(20.0, args.length_m * 1000.0) / (float(max(ext)) or 1.0))
mn, mx = v.min(0), v.max(0)
v[:, 0] -= (mn[0] + mx[0]) / 2.0
v[:, 1] -= (mn[1] + mx[1]) / 2.0
v[:, 2] -= v.min(0)[2]
trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces)).export(args.out)
print("MESH_OK", args.out, len(v), "verts", flush=True)
