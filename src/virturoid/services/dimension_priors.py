"""Code-owned realistic dimension priors (scene-gen plan S2). The research is unambiguous that an LLM must NOT be
trusted to emit absolute dimensions — probing work shows language models capture only about half the distance to a
scale upper-bound, and GPT-4 answers object-attribute questions at ~50% vs human 84% (NEWTON, EMNLP 2023). The
robust pattern (Holodeck, CVPR 2024) is: the LLM chooses a CATEGORY, and code owns the numbers, snapping/clamping
any proposed size into a cited plausibility band and logging every clamp as an honesty event.

This module is that table + the snap/clamp + the unit-sanity validators (the ones that catch a 10 cm-tall "wall"
or a mm-scaled mesh that MuJoCo would silently give ~1e9x the mass). Dimensions are FULL extents (metres) along
(x, y, z); architectural elements orient so z is the vertical/critical axis. Sources: US Access Board ADA
standards, IRC/IBC, NKBA 4th ed., and YCB caliper measurements (Calli et al. 2015). Pure data + numpy-free logic
so it is trivially testable and importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DimPrior:
    """A category's realistic size prior. ``default`` = canonical full extents (m). ``bounds`` = per-axis
    (min, max) plausibility band a value is clamped into. ``mass_kg`` = (lo, hi) real mass band (None = infer from
    density x volume). ``density`` kg/m^3 for mass inference. ``source`` = the standard/dataset it comes from."""
    category: str
    default: tuple[float, float, float]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    density: float = 700.0
    mass_kg: tuple[float, float] | None = None
    source: str = ""


# --- Architecture / circulation (ADA / IRC / IBC / NKBA). CONVENTION: size_xyz = (x, y, z) full extents with
# Z ALWAYS THE VERTICAL/HEIGHT AXIS (matches the exporter, which rests objects at z=height/2). So a wall is
# (length, thickness, HEIGHT); a table is (width, depth, HEIGHT); a corridor prior is (WIDTH, length, height). ---
_ARCH = [
    DimPrior("corridor", (1.0, 3.0, 2.44), ((0.915, 3.0), (0.6, 30.0), (2.03, 3.5)), source="ADA 403.5.1 / IRC R311.6 (x=width)"),
    DimPrior("wall", (3.0, 0.12, 2.44), ((0.05, 40.0), (0.05, 40.0), (0.9, 4.0)), density=1900.0, source="IRC R305.1 (z=ceiling height; either horizontal axis may be the length)"),
    DimPrior("door", (0.915, 0.045, 2.03), ((0.815, 1.2), (0.035, 0.06), (2.03, 2.44)), density=600.0, source="ADA 404.2"),
    DimPrior("table", (1.2, 0.75, 0.74), ((0.4, 2.4), (0.4, 1.2), (0.71, 0.865)), density=650.0, source="ADA 902.3 / dining-table std"),
    DimPrior("countertop", (1.2, 0.61, 0.914), ((0.4, 3.0), (0.5, 0.7), (0.86, 0.95)), density=700.0, source="kitchen std"),
    DimPrior("desk", (1.2, 0.6, 0.74), ((0.6, 2.0), (0.4, 0.9), (0.71, 0.76)), density=650.0, source="desk std"),
    DimPrior("chair", (0.45, 0.45, 0.45), ((0.35, 0.6), (0.35, 0.6), (0.43, 0.485)), density=500.0, source="ADA 903.5 seat height"),
    DimPrior("shelf", (0.9, 0.305, 0.03), ((0.3, 2.0), (0.25, 0.6), (0.02, 0.05)), density=650.0, source="wall-cabinet depth std (shelf board)"),
    DimPrior("step", (1.0, 0.279, 0.178), ((0.3, 1.6), (0.254, 0.35), (0.10, 0.196)), density=1200.0, source="IRC R311.7.5 (z=riser)"),
    DimPrior("pallet", (1.219, 1.016, 0.144), ((1.0, 1.3), (0.8, 1.1), (0.12, 0.16)), density=500.0, mass_kg=(20.0, 30.0), source="GMA/EPAL pallet"),
    DimPrior("floor", (8.0, 8.0, 0.04), ((0.5, 100.0), (0.5, 100.0), (0.01, 0.1)), density=1000.0, source="arena ground"),
]

# --- Manipulable objects (YCB caliper dims + mass, Calli et al. 2015). (x, y, z) = (footprint_x, footprint_y,
# HEIGHT). ``ycb.`` namespace. ---
_YCB = [
    DimPrior("ycb.mug", (0.080, 0.080, 0.082), ((0.06, 0.11), (0.06, 0.11), (0.06, 0.12)), density=900.0, mass_kg=(0.09, 0.16), source="YCB 025_mug"),
    DimPrior("ycb.soup_can", (0.066, 0.066, 0.101), ((0.05, 0.09), (0.05, 0.09), (0.08, 0.13)), mass_kg=(0.30, 0.40), source="YCB 005_tomato_soup_can"),
    DimPrior("ycb.cracker_box", (0.060, 0.158, 0.210), ((0.05, 0.08), (0.13, 0.18), (0.18, 0.24)), density=350.0, mass_kg=(0.35, 0.45), source="YCB 003_cracker_box"),
    DimPrior("ycb.sugar_box", (0.038, 0.089, 0.175), ((0.03, 0.05), (0.07, 0.11), (0.15, 0.20)), density=900.0, mass_kg=(0.45, 0.55), source="YCB 004_sugar_box"),
    DimPrior("ycb.mustard", (0.058, 0.095, 0.190), ((0.04, 0.08), (0.07, 0.12), (0.16, 0.22)), density=1000.0, mass_kg=(0.55, 0.65), source="YCB 006_mustard_bottle"),
    DimPrior("ycb.foam_brick", (0.050, 0.075, 0.050), ((0.04, 0.06), (0.06, 0.09), (0.04, 0.06)), density=150.0, mass_kg=(0.02, 0.04), source="YCB 061_foam_brick"),
    DimPrior("ycb.wood_block", (0.085, 0.085, 0.200), ((0.07, 0.10), (0.07, 0.10), (0.18, 0.22)), density=700.0, mass_kg=(0.65, 0.80), source="YCB 036_wood_block"),
    DimPrior("ycb.drill", (0.184, 0.184, 0.046), ((0.15, 0.22), (0.15, 0.22), (0.03, 0.06)), density=1200.0, mass_kg=(0.80, 0.95), source="YCB 035_power_drill"),
    DimPrior("ycb.bottle_2l", (0.110, 0.110, 0.315), ((0.09, 0.13), (0.09, 0.13), (0.28, 0.34)), density=1000.0, mass_kg=(1.8, 2.1), source="2L PET bottle"),
    DimPrior("ycb.can_355", (0.066, 0.066, 0.122), ((0.05, 0.08), (0.05, 0.08), (0.10, 0.14)), mass_kg=(0.35, 0.42), source="355 ml can"),
]

# --- Real household furniture + warehouse fixtures (x, y, z=HEIGHT), for house/warehouse environments. Boxes at
# real dimensions is a faithful navigation-obstacle model; detailed meshes are a later fidelity item. ---
_FURNITURE_PRIORS = [
    DimPrior("sofa", (2.0, 0.9, 0.80), ((1.4, 2.6), (0.8, 1.0), (0.7, 0.9)), density=300.0, source="3-seat sofa"),
    DimPrior("armchair", (0.85, 0.85, 0.90), ((0.7, 1.0), (0.7, 1.0), (0.8, 1.0)), density=300.0, source="armchair"),
    DimPrior("coffee_table", (1.1, 0.6, 0.45), ((0.8, 1.4), (0.5, 0.7), (0.40, 0.50)), density=600.0, source="coffee table"),
    DimPrior("tv_stand", (1.5, 0.4, 0.55), ((1.0, 1.9), (0.35, 0.5), (0.4, 0.6)), density=600.0, source="TV stand"),
    DimPrior("dining_table", (1.6, 0.9, 0.75), ((1.0, 2.2), (0.8, 1.1), (0.71, 0.80)), density=650.0, source="dining table"),
    DimPrior("bed", (2.0, 1.5, 0.55), ((1.9, 2.1), (0.9, 1.8), (0.4, 0.65)), density=250.0, source="queen bed"),
    DimPrior("nightstand", (0.5, 0.4, 0.55), ((0.4, 0.6), (0.35, 0.5), (0.45, 0.65)), density=600.0, source="nightstand"),
    DimPrior("wardrobe", (1.2, 0.6, 2.0), ((0.9, 1.8), (0.55, 0.7), (1.8, 2.2)), density=500.0, source="wardrobe"),
    DimPrior("fridge", (0.7, 0.7, 1.80), ((0.6, 0.9), (0.6, 0.8), (1.6, 2.0)), density=350.0, source="refrigerator"),
    DimPrior("counter", (2.0, 0.65, 0.90), ((0.8, 3.0), (0.6, 0.7), (0.86, 0.95)), density=700.0, source="kitchen counter run"),
    DimPrior("toilet", (0.40, 0.66, 0.79), ((0.35, 0.45), (0.6, 0.72), (0.7, 0.85)), density=900.0, source="toilet"),
    DimPrior("sink", (0.60, 0.45, 0.85), ((0.5, 0.8), (0.4, 0.55), (0.8, 0.9)), density=700.0, source="vanity sink"),
    DimPrior("desk", (1.2, 0.6, 0.74), ((0.6, 2.0), (0.4, 0.9), (0.71, 0.76)), density=650.0, source="desk"),
    # warehouse fixtures
    DimPrior("rack", (2.7, 1.0, 2.5), ((1.8, 3.6), (0.8, 1.2), (2.0, 3.5)), density=400.0, source="pallet-rack bay (upright + beams)"),
    DimPrior("crate", (0.5, 0.4, 0.4), ((0.3, 0.7), (0.3, 0.6), (0.25, 0.5)), density=200.0, mass_kg=(2.0, 25.0), source="warehouse crate"),
    DimPrior("carton", (0.4, 0.3, 0.3), ((0.25, 0.6), (0.2, 0.45), (0.2, 0.45)), density=180.0, mass_kg=(0.5, 15.0), source="shipping carton"),
    DimPrior("forklift_zone", (1.2, 1.2, 0.01), ((0.8, 2.0), (0.8, 2.0), (0.005, 0.02)), density=100.0, source="staging pad"),
]

# --- Generic task props (the categories scene_generator uses today), sized realistically. z = HEIGHT. ---
_GENERIC = [
    DimPrior("block", (0.05, 0.05, 0.05), ((0.02, 0.10), (0.02, 0.10), (0.02, 0.10)), density=600.0, mass_kg=(0.02, 0.20), source="graspable block (YCB-scale)"),
    DimPrior("box", (0.20, 0.15, 0.15), ((0.08, 0.6), (0.06, 0.5), (0.06, 0.5)), density=250.0, mass_kg=(0.1, 5.0), source="cardboard shipping box"),
    DimPrior("bin", (0.40, 0.30, 0.20), ((0.15, 0.8), (0.15, 0.8), (0.08, 0.5)), density=600.0, source="tote/bin"),
    DimPrior("obstacle", (0.30, 0.30, 0.60), ((0.05, 1.5), (0.05, 1.5), (0.05, 2.0)), density=400.0, source="generic obstacle/pillar"),
    DimPrior("zone", (0.30, 0.30, 0.01), ((0.05, 2.0), (0.05, 2.0), (0.005, 0.02)), density=100.0, source="target pad (marker)"),
]

PRIORS: dict[str, DimPrior] = {p.category: p for p in (_ARCH + _YCB + _FURNITURE_PRIORS + _GENERIC)}


@dataclass
class SnapResult:
    category: str
    size_xyz: tuple[float, float, float]
    clamped: bool = False
    events: list[str] = field(default_factory=list)     # honesty log: one line per axis that had to be clamped


def default_size(category: str) -> tuple[float, float, float] | None:
    """Canonical realistic full extents (m) for a category, or None if unknown."""
    p = PRIORS.get(category)
    return p.default if p else None


def snap_to_prior(category: str, proposed: tuple[float, float, float] | None = None) -> SnapResult:
    """Resolve a size for a category. With ``proposed=None`` returns the canonical default. With a proposed size,
    clamps each axis into the category's plausibility band and LOGS every clamp (the honesty event) — an LLM
    asking for a 0.1 m-tall wall or a 20 m box is corrected, not obeyed. Unknown category -> proposed unchanged (or
    a neutral default) so the pipeline never hard-fails on a novel label."""
    p = PRIORS.get(category)
    if p is None:
        size = proposed or (0.1, 0.1, 0.1)
        return SnapResult(category, tuple(round(float(v), 4) for v in size), clamped=False,
                          events=[f"unknown category {category!r}: no prior, used {'proposed' if proposed else 'neutral default'}"])
    if proposed is None:
        return SnapResult(category, p.default, clamped=False)
    out, events = [], []
    for i, (v, (lo, hi), ax) in enumerate(zip(proposed, p.bounds, "xyz")):
        cv = min(hi, max(lo, float(v)))
        if abs(cv - float(v)) > 1e-6:
            events.append(f"{category}.{ax}: {float(v):.3f} m out of band [{lo}, {hi}] -> clamped to {cv:.3f} m ({p.source})")
        out.append(round(cv, 4))
    return SnapResult(category, tuple(out), clamped=bool(events), events=events)


def mass_for(category: str, size_xyz: tuple[float, float, float]) -> float:
    """Realistic mass (kg) for an object of ``size_xyz`` in ``category``: density x volume, clamped into the
    category's real mass band when one is known (so a scaled box can't imply an absurd mass)."""
    p = PRIORS.get(category)
    density = (p.density if p and p.density else 700.0)
    vol = float(size_xyz[0]) * float(size_xyz[1]) * float(size_xyz[2])
    m = max(0.005, density * vol)
    if p and p.mass_kg:
        m = min(p.mass_kg[1], max(p.mass_kg[0], m))
    return round(m, 4)


def check_dimensions(category: str, size_xyz: tuple[float, float, float], *, tol: float = 10.0) -> dict:
    """Unit-sanity band check. Flags any axis that is >``tol``x outside its plausibility band (the signature of a
    unit error, not a stylistic choice). Returns ``{"ok", "flags"}``."""
    p = PRIORS.get(category)
    if p is None:
        return {"ok": True, "flags": [f"no prior for {category!r} (unchecked)"]}
    flags = []
    for v, (lo, hi), ax in zip(size_xyz, p.bounds, "xyz"):
        v = float(v)
        if v > hi * tol or v < lo / tol:
            flags.append(f"{category}.{ax}={v:.4f} m is >{tol:g}x outside band [{lo}, {hi}] — likely a unit error")
    return {"ok": not flags, "flags": flags}


_SCALE_SIGNATURES = {"mm->m (x1000 too big)": 0.001, "m->mm (x0.001 too small)": 1000.0,
                     "inch->m (x25.4 too big)": 1.0 / 25.4, "cm->m (x100 too big)": 0.01}


def rescale_signature(category: str, size_xyz: tuple[float, float, float]) -> str | None:
    """If ``size_xyz`` is out of band, test the classic unit-error rescales (the URDF-is-metres / STL-is-unitless
    1000x bug and inch/cm signatures) and report which one would bring it into band. None if already plausible or
    no rescale helps."""
    if check_dimensions(category, size_xyz)["ok"]:
        return None
    for label, factor in _SCALE_SIGNATURES.items():
        scaled = tuple(v * factor for v in size_xyz)
        if check_dimensions(category, scaled)["ok"]:
            return f"{label}: rescaling by {factor:g} brings it into band"
    return None


def robot_scene_ratio_ok(robot_extent_m: float, scene_extent_m: float, *,
                         lo: float = 0.02, hi: float = 0.9) -> dict:
    """Guard the robot:scene size ratio — a robot bigger than its scene, or a speck in a stadium, is almost always
    a unit bug. Default window: the robot spans 2%–90% of the scene's largest extent."""
    r = float(robot_extent_m) / max(1e-6, float(scene_extent_m))
    return {"ok": lo <= r <= hi, "ratio": round(r, 4),
            "flag": None if lo <= r <= hi else f"robot:scene extent ratio {r:.3f} outside [{lo}, {hi}] — check units"}
