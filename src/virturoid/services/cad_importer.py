"""CAD / mesh importer (initial) — Input Ingestion plan, Phase 6.

The plan's CAD lane: "STEP/STL/OBJ/DAE/GLB initial support, mesh scale/unit detection, inertial estimate report,
missing-mass warning." STEP needs a CAD kernel (deferred), but STL (ASCII + binary) and OBJ are directly
parseable with the standard library, which already delivers the highest-value facts: triangle/vertex count,
axis-aligned bounding box (the part's real dimensions), a unit-scale guess (parts modeled in mm read ~1000x too
big), a volume estimate, and — with an assumed density — an inertial estimate so a link that ships no mass still
gets a plausible one (with a clear warning). Deterministic, standard-library only, local-only (no network).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

# density priors (kg/m^3) for the inertial estimate; ABS plastic by default (a printed/proto part).
_DENSITY = {"abs": 1040.0, "pla": 1250.0, "aluminum": 2700.0, "steel": 7850.0, "nylon": 1150.0}


@dataclass
class CadImportResult:
    source: str
    format: str = "unknown"          # stl_binary | stl_ascii | obj
    triangles: int = 0
    vertices: int = 0
    bbox_min: tuple = (0.0, 0.0, 0.0)
    bbox_max: tuple = (0.0, 0.0, 0.0)
    size_m: tuple = (0.0, 0.0, 0.0)
    unit_guess: str = "m"            # m | mm | suspicious
    suggested_scale: float = 1.0     # multiply source coords by this to reach metres
    volume_m3: float | None = None
    estimated_mass_kg: float | None = None
    density_kg_m3: float | None = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def _bbox(points):
    xs = [p[0] for p in points]; ys = [p[1] for p in points]; zs = [p[2] for p in points]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _unit_guess(size_max: float) -> tuple[str, float]:
    """Robot parts are ~0.01-2 m. A max dimension >> that is almost certainly millimetres (scale 0.001)."""
    if size_max > 10.0:
        return "mm", 0.001
    if size_max <= 0.0 or size_max < 1e-4:
        return "suspicious", 1.0
    return "m", 1.0


def parse_stl(path: str) -> tuple[str, list, list]:
    """Return (format, triangles-as-vertex-triples, all-vertices). Handles binary and ASCII STL."""
    with open(path, "rb") as handle:
        head = handle.read(84)
    is_ascii = head[:5].lower() == b"solid" and b"facet" in open(path, "rb").read(512).lower()
    verts: list = []
    if is_ascii:
        for line in open(path, encoding="utf-8", errors="ignore"):
            parts = line.split()
            if parts and parts[0] == "vertex" and len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        return "stl_ascii", [verts[i:i + 3] for i in range(0, len(verts) - len(verts) % 3, 3)], verts
    # binary: 80-byte header, uint32 count, then 50 bytes/triangle (normal + 3 verts + attr).
    with open(path, "rb") as handle:
        handle.read(80)
        (count,) = struct.unpack("<I", handle.read(4))
        tris = []
        for _ in range(count):
            data = handle.read(50)
            if len(data) < 50:
                break
            vals = struct.unpack("<12fH", data)
            tri = [(vals[3], vals[4], vals[5]), (vals[6], vals[7], vals[8]), (vals[9], vals[10], vals[11])]
            tris.append(tri)
            verts.extend(tri)
    return "stl_binary", tris, verts


def parse_obj(path: str) -> tuple[str, list, list]:
    verts: list = []
    faces = 0
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f":
            faces += 1
    return "obj", [[] for _ in range(faces)], verts


def _mesh_volume(triangles) -> float:
    """Signed volume via the divergence theorem (sum of tetrahedra); abs value. 0 if no closed triangles."""
    vol = 0.0
    for tri in triangles:
        if len(tri) != 3:
            continue
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = tri
        vol += (x1 * (y2 * z3 - y3 * z2) - x2 * (y1 * z3 - y3 * z1) + x3 * (y1 * z2 - y2 * z1)) / 6.0
    return abs(vol)


def _import_step(path: str, material: str) -> CadImportResult:
    """Import a STEP part via build123d/OCC (a real CAD kernel). OCC normalizes to millimetres, so dimensions and
    volume are scaled to metres. Returns bbox + volume + an inertial mass estimate, or an honest fallback."""
    try:
        import build123d as b3d
    except ImportError:
        return CadImportResult(source=path, format="step",
                               warnings=["build123d not installed; export as STL/OBJ for dimensions + inertia."])
    try:
        part = b3d.import_step(path)
        bb = part.bounding_box()
        lo = (bb.min.X, bb.min.Y, bb.min.Z)
        hi = (bb.max.X, bb.max.Y, bb.max.Z)
        vol_mm3 = float(part.volume)
    except Exception as exc:  # noqa: BLE001
        return CadImportResult(source=path, format="step", warnings=[f"STEP import failed: {exc}"])

    size_m = tuple(round((hi[i] - lo[i]) * 0.001, 6) for i in range(3))
    volume_m3 = round(vol_mm3 * 1e-9, 9) if vol_mm3 > 0 else None
    warnings = ["STEP imported via build123d/OCC (normalized to mm -> scaled to metres)."]
    mass = density = None
    if volume_m3:
        density = _DENSITY.get(material, _DENSITY["abs"])
        mass = round(volume_m3 * density, 5)
        warnings.append(f"mass ESTIMATED from solid volume @ {material} density ({density} kg/m^3); "
                        "provide a BOM/CAD mass to override.")
    return CadImportResult(
        source=path, format="step", triangles=0, vertices=0,
        bbox_min=tuple(round(v * 0.001, 6) for v in lo), bbox_max=tuple(round(v * 0.001, 6) for v in hi),
        size_m=size_m, unit_guess="mm", suggested_scale=0.001, volume_m3=volume_m3,
        estimated_mass_kg=mass, density_kg_m3=density, warnings=warnings)


def import_cad(path: str, *, material: str = "abs") -> CadImportResult:
    """Import an STL/OBJ mesh into a :class:`CadImportResult` (bbox, unit guess, volume + mass estimate)."""
    if not os.path.exists(path):
        return CadImportResult(source=path, warnings=[f"path not found: {path}"])
    ext = os.path.splitext(path.lower())[1]
    warnings: list[str] = []
    try:
        if ext == ".stl":
            fmt, tris, verts = parse_stl(path)
        elif ext == ".obj":
            fmt, tris, verts = parse_obj(path)
        elif ext in (".step", ".stp"):
            return _import_step(path, material)
        elif ext in (".iges", ".igs"):
            return CadImportResult(source=path, format="iges",
                                   warnings=["IGES needs a CAD kernel with an IGES reader; "
                                             "export the part as STEP or STL/OBJ for dimensions + inertia."])
        else:
            return CadImportResult(source=path, warnings=[f"unsupported CAD extension '{ext}'"])
    except Exception as exc:  # noqa: BLE001
        return CadImportResult(source=path, format=ext.lstrip("."), warnings=[f"parse failed: {exc}"])

    if not verts:
        return CadImportResult(source=path, format=fmt, warnings=["mesh had no vertices"])

    lo, hi = _bbox(verts)
    size = tuple(round(hi[i] - lo[i], 6) for i in range(3))
    unit, scale = _unit_guess(max(size))
    size_m = tuple(round(s * scale, 6) for s in size)
    if unit == "mm":
        warnings.append(f"mesh looks like millimetres (max dim {max(size):.1f}); scaling by {scale} to metres.")
    elif unit == "suspicious":
        warnings.append(f"mesh max dimension {max(size):.6f} is implausible; check units/scale.")

    volume = mass = density = None
    if fmt.startswith("stl") and tris:
        vol_src = _mesh_volume(tris)
        volume = round(vol_src * (scale ** 3), 9) if vol_src > 0 else None
        if volume:
            density = _DENSITY.get(material, _DENSITY["abs"])
            mass = round(volume * density, 5)
            warnings.append(f"mass ESTIMATED from mesh volume @ {material} density ({density} kg/m^3); "
                            "provide a BOM/CAD mass to override.")
    elif fmt == "obj":
        warnings.append("OBJ gives geometry but no reliable closed volume; import mass from the BOM.")

    return CadImportResult(
        source=path, format=fmt, triangles=len(tris), vertices=len(verts),
        bbox_min=tuple(round(v, 6) for v in lo), bbox_max=tuple(round(v, 6) for v in hi),
        size_m=size_m, unit_guess=unit, suggested_scale=scale, volume_m3=volume,
        estimated_mass_kg=mass, density_kg_m3=density, warnings=warnings)
