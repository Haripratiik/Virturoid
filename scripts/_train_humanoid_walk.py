"""Train the generated humanoid's locomotion policy on the GPU box (MJX PPO + bipedal-CPG prior + adaptive
per-joint PD gains). Closes the standing headline gap: the 8-DOF generated humanoid SELF-RIGHTS on CPU-ES but
does not WALK -- GPU-PPO at scale is the path to a converged bipedal gait ([[honest-locomotion-audit]] task #33,
[[recipe-gain-generalization]]). Fetches the trained npz to models/humanoid_gpu_v1.npz.

Single GPU job only (thermal). The box host stays internal (gpu_trainer reads VIRTUROID_GPU_SSH from .env).

Usage:  PYTHONPATH=src python scripts/_train_humanoid_walk.py [iters] [envs]
"""
import sys

from virturoid.services.gpu_trainer import train_gene_on_gpu
from virturoid.services.morphology_composer import compose_robot

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ENVS = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

gene = compose_robot("a humanoid robot that walks", llm=None)
ndof = len([s for s in gene.segments if s.joint_type in ("revolute", "prismatic")])
print(f"humanoid gene: {gene.species}, {ndof} DOF; iters={ITERS} envs={ENVS} (cpg + adaptive PD)", flush=True)

out = train_gene_on_gpu(
    gene,
    out_path="models/humanoid_gpu_v1.npz",
    iters=ITERS,
    envs=ENVS,
    cpg=True,          # bipedal-CPG feed-forward prior (l_/r_ leg recognition)
    adaptive=True,     # inertia-scaled per-joint PD gains -- needed for a humanoid (heavy trunk)
    dr=False,          # nominal first; add domain randomization for sim2real once it walks
    timeout=3300.0,
    progress=lambda m: print("[gpu]", m, flush=True),
)
print("RESULT npz:", out, flush=True)
