"""Generative-3D part synthesis (provider-agnostic) — AI-generated detailed meshes for NOVEL robot parts.

Realistic geometry for a part with no real-robot analog (no kit-bash mesh in ``part_catalog``, beyond the
procedural ``build_anatomy`` roles) comes from AI synthesis. This is the swappable backend behind that, in
priority order — the first one whose credentials are configured wins:

  • CLOUD text/image->3D mesh API — ``replicate`` (hosts TRELLIS / Hunyuan3D on GPUs), ``tripo``, ``meshy``:
    photoreal-grade meshes; needs an API key + ``trimesh`` to process the returned glb/obj.
  • LLM-AUTHORED CAD — the already-configured LLM writes ``build123d`` code for the part, which we sandbox-
    exec into a real solid. Runs on CPU, offline, with ZERO extra setup (reuses the OpenAI key already
    present). Generative *parametric* geometry: richer than the fixed anatomy, not photoreal mesh-diffusion.

The result is fitted to the link's ``[0, length]`` +z frame (mm STL for the compiler's 0.001 scale), cached
by a content hash, and dropped on as a VISUAL-ONLY mesh — physics stays the gene primitive, exactly like
kit-bash. Synthesis is opt-in (it costs LLM/cloud calls), never in the default sim/training path.

To enable photoreal mesh synthesis, set ONE of these in ``.env``: ``REPLICATE_API_TOKEN`` (recommended —
runs the open TRELLIS/Hunyuan3D models), ``TRIPO_API_KEY``, or ``MESHY_API_KEY`` (and ``pip install trimesh``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Sandbox for LLM-authored CAD: only geometry-safe names. Not a hard security boundary, but it blocks
# imports / file / network / process access so a stray model can't do I/O while building a solid.
_SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "abs": abs, "round": round, "sum": sum,
    "float": float, "int": int, "bool": bool, "enumerate": enumerate, "zip": zip, "map": map,
    "list": list, "tuple": tuple, "dict": dict, "set": set, "sorted": sorted, "reversed": reversed,
    "True": True, "False": False, "None": None,
}

_CAD_SYSTEM = (
    "You are a mechanical designer. Build a DETAILED, realistic 3D model of the described robot part by "
    "COMPOSING simple primitives, and assign the final solid to a variable named `part`.\n"
    "USE ONLY these helper functions (do NOT call build123d directly):\n"
    "  box(w, d, h, at=(x,y,z))                 — a box of full sizes w,d,h centered at `at`\n"
    "  cyl(r, h, at=(x,y,z), axis='z')          — a cylinder radius r, height h, centered at `at`, along "
    "axis 'z'|'x'|'y'\n"
    "  sphere(r, at=(x,y,z))\n"
    "  cone(r0, r1, h, at=(x,y,z))              — a frustum: bottom radius r0, top radius r1, height h\n"
    "Combine solids with `+` (union) and `-` (cut). Also available: `L` (part length in mm, along +z), "
    "`R` (radius/half-width in mm), and `math`.\n"
    "Build in the +z frame spanning z=0..L, centered on x=y=0 (it is auto-floored). Make it look like REAL "
    "hardware with MANY features (housings, shells, mounting flanges, bolt bosses, ribs, sensor cutouts) — "
    "typically 8-20 helper calls.\n"
    "EXAMPLE (a sensor head): part = cyl(R, L*0.7, at=(0,0,L*0.35)) + sphere(R, at=(0,0,L*0.7)) + "
    "box(0.6*R, 2*R, 0.4*R, at=(0.6*R,0,0.62*L)) - cyl(0.22*R, 2*R, at=(0.72*R,0,0.62*L), axis='x') + "
    "cyl(0.5*R, 0.2*L, at=(0,0,0.08*L))\n"
    "HARD RULES: use ONLY box/cyl/sphere/cone/`+`/`-`/math/L/R; NO imports, NO build123d, NO I/O. Every "
    "solid must overlap another so the result is one connected body. Output JSON {code, notes}."
)
_CODE_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "notes": {"type": "string"}},
    "required": ["code"],
}


def available_backend() -> str | None:
    """Which synthesis backend is usable right now. Preference: a real GPU box (photoreal, your own
    hardware) > cloud mesh-diffusion > LLM-CAD (parametric, CPU). None if nothing is configured."""
    from virturoid.services.llm_client import _load_local_env
    _load_local_env()
    if os.environ.get("VIRTUROID_GPU_SSH"):          # text->image->3D on your own GPU box over Tailscale
        return "gpu_box"
    if os.environ.get("REPLICATE_API_TOKEN"):
        return "replicate"
    if os.environ.get("TRIPO_API_KEY"):
        return "tripo"
    if os.environ.get("MESHY_API_KEY"):
        return "meshy"
    if os.environ.get("VIRTUROID_LLM_BACKEND", "off").lower() not in ("off", ""):
        return "llm-cad"
    return None


def _find_tailscale() -> str | None:
    import shutil
    for c in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe",
              r"C:\Program Files (x86)\Tailscale\tailscale.exe"):
        if shutil.which(c) or os.path.exists(c):
            return c
    return None


def _gpu_box_part(description: str, length_m: float, radius_m: float, out_path: str) -> str | None:
    """Generate a photoreal mesh on YOUR GPU box (RTX 3060 over Tailscale): run gen3d.py there (SDXL-Turbo
    image -> TripoSR 3D, fitted to the link frame as an mm STL), then pull the STL back. Fully on your own
    hardware — no cloud. Configured via env: VIRTUROID_GPU_SSH (e.g. 'user@host'), optional
    VIRTUROID_GPU_PYTHON (default ~/torch/bin/python), VIRTUROID_GPU_GEN (default ~/gen3d.py),
    VIRTUROID_TAILSCALE (auto-detected)."""
    import base64
    import hashlib
    import subprocess
    host = os.environ.get("VIRTUROID_GPU_SSH")
    if not host:
        return None
    py = os.environ.get("VIRTUROID_GPU_PYTHON", "~/torch/bin/python")
    gen = os.environ.get("VIRTUROID_GPU_GEN", "~/gen3d.py")
    ts = os.environ.get("VIRTUROID_TAILSCALE") or _find_tailscale()
    ssh = [ts, "ssh", host] if ts else ["ssh", host]
    rid = hashlib.md5(f"{description}|{round(length_m,4)}|{round(radius_m,4)}".encode()).hexdigest()[:12]
    remote = f"/tmp/v_{rid}.stl"
    safe = description.replace('"', " ").replace("'", " ").replace("\n", " ")[:220]
    run = (f'{py} {gen} --prompt "{safe}" --out {remote} '
           f'--length_m {float(length_m):.4f} --radius_m {float(radius_m):.4f}')
    try:
        r = subprocess.run(ssh + [run], capture_output=True, text=True, timeout=900)
        if "MESH_OK" not in (r.stdout or ""):
            return None
        pull = subprocess.run(ssh + [f"base64 -w0 {remote}"], capture_output=True, text=True, timeout=180)
        data = base64.b64decode((pull.stdout or "").strip())
        if len(data) < 200:
            return None
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(data)
        return str(Path(out_path).resolve()).replace("\\", "/")
    except Exception:  # noqa: BLE001 - box unreachable / gen failure -> caller falls back
        return None


def _cache_key(description: str, length_m: float, radius_m: float, backend: str) -> str:
    sig = json.dumps([backend, description.strip().lower(), round(length_m, 4), round(radius_m, 4)],
                     sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def synthesize_part(description: str, length_m: float, radius_m: float, out_path: str, *,
                    llm=None, cache: bool = True) -> str | None:
    """Synthesize a detailed VISUAL mesh for ``description``, fitted to the link's [0, length] +z frame, to
    ``out_path`` (mm STL). Uses the best available backend (cloud mesh-diffusion if a key is set, else
    LLM-authored CAD). Returns ``out_path`` on success, else None (caller falls back to procedural anatomy).
    VISUAL ONLY; never affects dynamics."""
    backend = "llm-cad" if llm is not None else available_backend()   # explicit llm forces LLM-CAD (tests)
    if backend is None:
        return None
    try:
        if backend == "gpu_box":
            return _gpu_box_part(description, length_m, radius_m, out_path)
        if backend in ("replicate", "tripo", "meshy"):
            return _cloud_part(backend, description, length_m, radius_m, out_path)
        return _llm_cad_part(description, length_m, radius_m, out_path, llm=llm)
    except Exception:  # noqa: BLE001 - any synthesis failure -> procedural fallback (never breaks a build)
        return None


# ---- LLM-authored CAD (works now: OpenAI key + build123d, CPU) --------------------------------------

class _Mesh:
    """A tiny triangle-soup mesh (N,3,3 array of triangles) with union via ``+``. Subtraction (``-``) is a
    visual no-op (returns the left operand): we don't do CSG — for a cosmetic mesh, an un-cut bolt-boss reads
    fine and we gain instant, hang-free composition (no OpenCascade booleans)."""

    __slots__ = ("tris",)

    def __init__(self, tris):
        import numpy as np
        self.tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)

    def __add__(self, other):
        import numpy as np
        return _Mesh(np.concatenate([self.tris, other.tris])) if isinstance(other, _Mesh) else self

    def __sub__(self, other):           # no CSG — cuts are skipped (cosmetic mesh only)
        return self


def _prim_box(w, d, h):
    import numpy as np
    x, y, z = abs(w) / 2 or 0.5, abs(d) / 2 or 0.5, abs(h) / 2 or 0.5
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])
    f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
         [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]]
    return v[np.array(f)]


def _prim_frustum(r0, r1, h, seg=28):
    import numpy as np
    r0, r1, hz = max(abs(r0), 0.2), max(abs(r1), 0.0), abs(h) / 2 or 0.5
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    bot = np.c_[r0 * np.cos(th), r0 * np.sin(th), np.full(seg, -hz)]
    top = np.c_[r1 * np.cos(th), r1 * np.sin(th), np.full(seg, hz)]
    cb, ct = np.array([0, 0, -hz]), np.array([0, 0, hz])
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        tris += [[bot[i], bot[j], top[j]], [bot[i], top[j], top[i]],
                 [cb, bot[j], bot[i]], [ct, top[i], top[j]]]
    return np.array(tris)


def _prim_sphere(r, lat=14, lon=20):
    import numpy as np
    r = abs(r) or 0.5
    tris = []
    for i in range(lat):
        p0, p1 = np.pi * i / lat, np.pi * (i + 1) / lat
        for k in range(lon):
            t0, t1 = 2 * np.pi * k / lon, 2 * np.pi * (k + 1) / lon
            a = [r * np.sin(p0) * np.cos(t0), r * np.sin(p0) * np.sin(t0), r * np.cos(p0)]
            b = [r * np.sin(p1) * np.cos(t0), r * np.sin(p1) * np.sin(t0), r * np.cos(p1)]
            c = [r * np.sin(p1) * np.cos(t1), r * np.sin(p1) * np.sin(t1), r * np.cos(p1)]
            e = [r * np.sin(p0) * np.cos(t1), r * np.sin(p0) * np.sin(t1), r * np.cos(p0)]
            tris += [[a, b, c], [a, c, e]]
    return np.array(tris)


def _exec_cad(code: str, length_m: float, radius_m: float):
    """Sandbox-exec LLM-authored code (the safe box/cyl/sphere/cone DSL) into a triangle mesh, floored to
    z=0. Pure-numpy analytic primitives + concatenation — instant and hang-free (no OpenCascade booleans).
    The LLM only composes helpers (can't misuse a real CAD API); builtins are restricted to block I/O."""
    import math as _math

    import numpy as np
    L = max(20.0, float(length_m) * 1000.0)
    R = max(6.0, float(radius_m) * 1000.0)

    def _place(tris, at, rot=None):
        if rot is not None:
            tris = tris @ rot.T
        return _Mesh(tris + np.asarray(at, dtype=float))

    _RX = np.array([[1, 0, 0], [0, 0, -1.0], [0, 1, 0]])   # z->y (cyl 'y')
    _RY = np.array([[0, 0, 1.0], [0, 1, 0], [-1, 0, 0]])   # z->x (cyl 'x')

    def box(w, d, h, at=(0, 0, 0)):
        return _place(_prim_box(w, d, h), at)

    def cyl(r, h, at=(0, 0, 0), axis="z"):
        rot = _RY if axis == "x" else (_RX if axis == "y" else None)
        return _place(_prim_frustum(r, r, h), at, rot)

    def cone(r0, r1, h, at=(0, 0, 0), axis="z"):
        rot = _RY if axis == "x" else (_RX if axis == "y" else None)
        return _place(_prim_frustum(r0, r1, h), at, rot)

    def sphere(r, at=(0, 0, 0)):
        return _place(_prim_sphere(r), at)

    ns = {"box": box, "cyl": cyl, "sphere": sphere, "cone": cone,
          "math": _math, "L": L, "R": R, "__builtins__": _SAFE_BUILTINS}
    exec(code, ns)                              # noqa: S102 - safe DSL helpers + restricted builtins only
    part = ns.get("part")
    if not isinstance(part, _Mesh) or part.tris.shape[0] < 4 or not np.isfinite(part.tris).all():
        raise ValueError("code did not assign a valid mesh to `part`")
    ext = part.tris.reshape(-1, 3).max(0) - part.tris.reshape(-1, 3).min(0)
    if float(ext.min()) <= 1e-6:
        raise ValueError("degenerate (flat) part")
    v = part.tris.copy()
    v[:, :, 2] -= v.reshape(-1, 3)[:, 2].min()   # floor to z=0 -> drops onto the [0,length] primitive
    return v


def _llm_cad_part(description: str, length_m: float, radius_m: float, out_path: str, *, llm=None,
                  repair_tries: int = 2) -> str | None:
    from virturoid.services.part_catalog import _write_tris

    if llm is None:
        from virturoid.services.llm_client import get_llm
        llm = get_llm("morphology")             # the strong build model writes the best CAD
    if llm is None:
        return None
    fp = Path(out_path)
    if fp.exists():                             # cache hit (caller keys the path on the description)
        return str(fp.resolve()).replace("\\", "/")
    user = (f"Part: {description}\nL = {max(20.0, length_m * 1000):.0f} mm (length along +z), "
            f"R = {max(6.0, radius_m * 1000):.0f} mm (radius/half-width).\nWrite the code.")
    errors = None
    for _ in range(max(1, repair_tries + 1)):
        u = user if not errors else user + f"\nThe previous code FAILED: {errors}. Return corrected code."
        try:
            out = llm.complete_json(_CAD_SYSTEM, u, _CODE_SCHEMA, max_tokens=2000)
            tris = _exec_cad(out.get("code", ""), length_m, radius_m)
            fp.parent.mkdir(parents=True, exist_ok=True)
            _write_tris(str(fp), tris)
            return str(fp.resolve()).replace("\\", "/")
        except Exception as e:  # noqa: BLE001 - bad code -> feed the error back and retry
            errors = f"{type(e).__name__}: {e}"
    return None


# ---- Cloud text/image->3D (photoreal; activates when a key is configured) ---------------------------

def _cloud_part(backend: str, description: str, length_m: float, radius_m: float, out_path: str) -> str | None:
    """Call a cloud text->3D API, then process the returned mesh into a fitted mm STL. Requires the
    provider key (already checked) and ``trimesh`` (to load glb/obj and re-export). Returns None (graceful
    fallback) if ``trimesh`` is missing or the call fails."""
    try:
        import trimesh  # noqa: F401 - presence gate; processing uses it below
    except Exception:  # noqa: BLE001 - mesh toolchain absent -> caller falls back to LLM-CAD/procedural
        return None
    raw = _cloud_fetch_mesh(backend, description)        # bytes of a glb/obj, or None
    if not raw:
        return None
    import io

    import numpy as np
    import trimesh

    scene = trimesh.load(io.BytesIO(raw), file_type="glb", force="mesh")
    mesh = scene if isinstance(scene, trimesh.Trimesh) else scene.dump(concatenate=True)
    v = np.asarray(mesh.vertices, dtype=float)
    # orient longest axis -> +z, uniform-scale to length (mm), center x/y, floor z=0 (same convention as
    # part_catalog's limb fit) so it drops onto the [0,length] primitive.
    ext = v.max(0) - v.min(0)
    axis = int(np.argmax(ext))
    if axis == 0:
        v = v @ np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]).T
    elif axis == 1:
        v = v @ np.array([[1, 0, 0], [0, 0, -1.0], [0, 1, 0]]).T
    ext = v.max(0) - v.min(0)
    v *= (max(20.0, float(length_m) * 1000.0) / (float(max(ext)) or 1.0))
    mn, mx = v.min(0), v.max(0)
    v[:, 0] -= (mn[0] + mx[0]) / 2.0
    v[:, 1] -= (mn[1] + mx[1]) / 2.0
    v[:, 2] -= v.min(0)[2]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces)).export(out_path)
    return str(Path(out_path).resolve()).replace("\\", "/")


def _cloud_fetch_mesh(backend: str, description: str) -> bytes | None:
    """Provider HTTP adapter: text prompt -> glb bytes. One small, dependency-light implementation per
    provider; all return None on any error so synthesis degrades to LLM-CAD/procedural rather than crashing.
    These run only when the matching key is set (so they are inert until you opt in)."""
    import time

    import requests
    if backend == "replicate":
        token = os.environ["REPLICATE_API_TOKEN"]
        model = os.environ.get("VIRTUROID_REPLICATE_3D_MODEL",
                               "firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c")
        hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
               "Prefer": "wait"}
        # text->3D models vary in input name; TRELLIS-class take an image, so most text->3D is via an
        # image step. Kept generic: pass the prompt; if the model needs an image, configure a text->image
        # model id instead. (Documented in the module header.)
        r = requests.post("https://api.replicate.com/v1/predictions", headers=hdr,
                          json={"version": model.split(":")[-1], "input": {"prompt": description}}, timeout=300)
        r.raise_for_status()
        out = r.json().get("output")
        url = out[0] if isinstance(out, list) else out
        if not url:
            return None
        return requests.get(url, timeout=300).content
    if backend == "tripo":
        key = os.environ["TRIPO_API_KEY"]
        hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        r = requests.post("https://api.tripo3d.ai/v2/openapi/task", headers=hdr,
                          json={"type": "text_to_model", "prompt": description}, timeout=60)
        r.raise_for_status()
        tid = r.json()["data"]["task_id"]
        for _ in range(60):                              # poll up to ~5 min
            time.sleep(5)
            s = requests.get(f"https://api.tripo3d.ai/v2/openapi/task/{tid}", headers=hdr, timeout=60).json()
            d = s.get("data", {})
            if d.get("status") == "success":
                url = (d.get("output") or {}).get("pbr_model") or (d.get("output") or {}).get("model")
                return requests.get(url, timeout=300).content if url else None
            if d.get("status") in ("failed", "cancelled", "banned"):
                return None
        return None
    if backend == "meshy":
        key = os.environ["MESHY_API_KEY"]
        hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        r = requests.post("https://api.meshy.ai/openapi/v2/text-to-3d", headers=hdr,
                          json={"mode": "preview", "prompt": description}, timeout=60)
        r.raise_for_status()
        tid = r.json()["result"]
        for _ in range(60):
            time.sleep(5)
            s = requests.get(f"https://api.meshy.ai/openapi/v2/text-to-3d/{tid}", headers=hdr, timeout=60).json()
            if s.get("status") == "SUCCEEDED":
                url = (s.get("model_urls") or {}).get("glb")
                return requests.get(url, timeout=300).content if url else None
            if s.get("status") in ("FAILED", "CANCELED"):
                return None
        return None
    return None
