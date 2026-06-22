"""Drop-in shim for TripoSR's ``from torchmcubes import marching_cubes`` using PyMCubes (prebuilt wheel, no
CUDA build needed — the GPU box has no nvcc, so the real torchmcubes CUDA extension won't compile). Placed
on TripoSR's import path; returns the same (verts, faces) torch tensors TripoSR's MarchingCubeHelper expects."""

import numpy as np
import torch
import mcubes as _mc


def marching_cubes(vol, thresh):
    a = vol.detach().cpu().numpy() if hasattr(vol, "detach") else np.asarray(vol)
    a = np.ascontiguousarray(a, dtype=np.float32)
    v, f = _mc.marching_cubes(a, float(thresh))
    return (torch.from_numpy(np.ascontiguousarray(v)).float(),
            torch.from_numpy(np.ascontiguousarray(f.astype(np.int64))).long())
