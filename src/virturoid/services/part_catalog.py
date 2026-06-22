"""P3 — KIT-BASHING real robot link meshes onto generated bodies for production-grade visual fidelity.

A generated gene is abstract (one primitive per link). For body parts with a real-world analog, we borrow the
ACTUAL detailed link mesh from a license-clean MuJoCo Menagerie model (cached locally by ``robot_descriptions``)
and FIT it to our link's ``[0, length]`` +z body frame — uniform-scaled (no shear), longest axis oriented to
+z, re-origined so it spans the link — then the compiler attaches it as a VISUAL-ONLY mesh over the gene's
primitive collider. True hardware detail, zero modeling.

STRICTLY VISUAL: MuJoCo physics (collision, mass, inertia) stays the gene's primitive — the compiler reads
shape/length/radius/mass and never reads ``geometry``/meshes for dynamics, so kit-bashing cannot change a
single simulated number. LICENSE-GATED: only permissively-licensed source models (BSD-3 / Apache-2.0 / MIT)
are whitelisted for redistribution (see ``LICENSES`` / ``attribution``). Source meshes are chosen as the
ELONGATED structural link for each part (verified by bounding-box aspect) so the longest-axis->+z fit reads as
a real limb/torso rather than a compact joint housing. Roles with no entry (head, hand — the H1 has neither)
fall back to the procedural anatomy in ``cad_geometry.build_anatomy``.
"""

from __future__ import annotations

import struct
from pathlib import Path

# role -> (source model dir, asset STL, fit_mode). fit_mode: "limb" = orient longest axis to the link's +z
# and floor it (arms/legs); "body" = keep the mesh's natural orientation and center it (a torso/trunk that
# lies along its own long axis). Unitree H1 (BSD-3) covers a humanoid's torso/arms/legs/feet; Unitree G1
# (BSD-3) the head + hands; Unitree Go1 (BSD-3) a real quadruped's trunk + legs.
_UNITREE_H1 = "unitree_h1"
_UNITREE_G1 = "unitree_g1"
_UNITREE_GO1 = "unitree_go1"
ROLE_MESHES = {
    # humanoid
    "torso_shell": (_UNITREE_H1, "torso_link.stl", "limb"),       # H1 torso is z-tall -> orient to +z
    "upper_arm":   (_UNITREE_H1, "left_shoulder_yaw_link.stl", "limb"),
    "forearm":     (_UNITREE_H1, "left_elbow_link.stl", "limb"),
    "thigh":       (_UNITREE_H1, "left_hip_pitch_link.stl", "limb"),
    "shin":        (_UNITREE_H1, "left_knee_link.stl", "limb"),
    "foot_pad":    (_UNITREE_H1, "left_ankle_link.stl", "limb"),
    "sensor_head": (_UNITREE_G1, "head_link.STL", "limb"),
    "hand_plate":  (_UNITREE_G1, "left_rubber_hand.STL", "limb"),
    # quadruped (real Go1): a long horizontal trunk + 3-DOF legs (hip, thigh, calf)
    "quad_trunk":  (_UNITREE_GO1, "trunk.stl", "body"),           # long in x -> stays horizontal
    "quad_hip":    (_UNITREE_GO1, "hip.stl", "limb"),
    "quad_thigh":  (_UNITREE_GO1, "thigh.stl", "limb"),
    "quad_calf":   (_UNITREE_GO1, "calf.stl", "limb"),
}
LICENSES = {m: "BSD-3-Clause - Unitree Robotics (via MuJoCo Menagerie)"
            for m in (_UNITREE_H1, _UNITREE_G1, _UNITREE_GO1)}

_STL_HEADER = 80
_STL_TRI_BYTES = 50


def _stl_dtype():
    import numpy as np
    # 50-byte binary-STL triangle: 3-float normal + 3x3-float verts + uint16 attribute (no padding)
    return np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


def _menagerie_root() -> Path | None:
    """The local MuJoCo Menagerie root (cached by robot_descriptions), or None if the package/cache is absent."""
    try:
        from robot_descriptions import h1_mj_description
        return Path(h1_mj_description.MJCF_PATH).resolve().parent.parent
    except Exception:  # noqa: BLE001 - package or cache missing -> kit-bash unavailable (procedural fallback)
        return None


def source_stl(role: str) -> Path | None:
    """Absolute path to the license-clean real link mesh for ``role``, or None if not catalogued / not cached."""
    ent = ROLE_MESHES.get(role)
    root = _menagerie_root()
    if ent is None or root is None:
        return None
    p = root / ent[0] / "assets" / ent[1]
    return p if p.exists() else None


def has_part(role: str | None) -> bool:
    """True iff a license-clean real mesh is available for this role (so kit-bash can apply, else fall back)."""
    return bool(role) and source_stl(role) is not None


def _read_tris(path: str):
    import numpy as np
    with open(path, "rb") as f:
        f.read(_STL_HEADER)
        n = struct.unpack("<I", f.read(4))[0]
        buf = f.read(n * _STL_TRI_BYTES)
    return np.frombuffer(buf, dtype=_stl_dtype(), count=n)["v"].astype(np.float64)   # (n, 3, 3)


def _write_tris(path: str, tris) -> None:
    import numpy as np
    n = len(tris)
    out = np.zeros(n, dtype=_stl_dtype())                 # normals left 0 — MuJoCo recomputes them
    out["v"] = tris.astype("<f4")
    with open(path, "wb") as f:
        f.write(b"\0" * _STL_HEADER)
        f.write(struct.pack("<I", n))
        f.write(out.tobytes())


def _fit(tris, target_mm: float, *, reorient: bool = True, z_floor: bool = True, z_center_mm: float = 0.0):
    """Uniform-scale a mesh so its longest axis equals ``target_mm`` (preserving real proportions — no shear),
    center it in x/y, and seat it in z. ``reorient`` rotates the longest axis to +z (a "limb" along the link
    axis); leave it off for a "body" that keeps its natural orientation. ``z_floor`` drops the base to z=0
    (limb [0,length] frame); else the mesh is centered and shifted to ``z_center_mm`` (a body at the box mid)."""
    import numpy as np
    V = tris.reshape(-1, 3)
    if reorient:
        size = V.max(0) - V.min(0)
        axis = int(np.argmax(size))
        if axis == 0:                                      # x -> +z (proper rotation about y)
            R = np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]])
        elif axis == 1:                                    # y -> +z (proper rotation about x)
            R = np.array([[1, 0, 0], [0, 0, -1.0], [0, 1, 0]])
        else:
            R = np.eye(3)
        V = V @ R.T
    size = V.max(0) - V.min(0)
    V = V * (float(target_mm) / (float(max(size)) or 1.0))     # scale by the LONGEST axis (works for body too)
    mn, mx = V.min(0), V.max(0)
    V[:, 0] -= (mn[0] + mx[0]) / 2.0
    V[:, 1] -= (mn[1] + mx[1]) / 2.0
    if z_floor:
        V[:, 2] -= V.min(0)[2]                             # base at z=0 (limb frame)
    else:
        V[:, 2] -= (mn[2] + mx[2]) / 2.0                   # center, then sit at the box mid-height
        V[:, 2] += float(z_center_mm)
    return V.reshape(-1, 3, 3)


def fitted_part_stl(role: str, length_m: float, radius_m: float, out_path: str) -> str | None:
    """Bake the license-clean real mesh for ``role``, fitted to the link, to ``out_path`` (mm STL for the
    compiler's 0.001 scale). A "limb" role lands in the link's [0, length] +z frame; a "body" role keeps its
    natural orientation, is scaled to the box's largest dimension, and sits at the box mid-height. Returns
    ``out_path`` on success, else None (caller falls back to procedural anatomy). VISUAL ONLY; never affects
    dynamics."""
    ent = ROLE_MESHES.get(role)
    src = source_stl(role)
    if ent is None or src is None:
        return None
    mode = ent[2] if len(ent) > 2 else "limb"
    try:
        tris = _read_tris(str(src))
        if mode == "body":                                 # a torso/trunk: keep orientation, fill the box, center
            target = max(float(length_m), 2.0 * float(radius_m)) * 1000.0
            tris = _fit(tris, target, reorient=False, z_floor=False, z_center_mm=float(length_m) * 1000.0 / 2.0)
        else:                                              # a limb: orient to the link axis, floor to [0,length]
            tris = _fit(tris, max(20.0, float(length_m) * 1000.0), reorient=True, z_floor=True)
        _write_tris(out_path, tris)
        return out_path
    except Exception:  # noqa: BLE001 - any parse/transform/write failure -> procedural fallback
        return None


def attribution(roles=None) -> list[dict]:
    """The (model, license) pairs a body's roles draw real meshes from — for honest provenance/attribution."""
    roles = set(roles) if roles is not None else set(ROLE_MESHES)
    used = {ROLE_MESHES[r][0] for r in roles if r in ROLE_MESHES}
    return [{"model": m, "license": LICENSES.get(m, "unknown")} for m in sorted(used)]
