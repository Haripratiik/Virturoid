"""STAGE 3 — real CAD geometry via build123d / OpenCascade, replacing the fake ``cad_placeholder``.

Each link becomes a SHAPED solid — a structural body (beam / tube) with a cylindrical motor hub at its
actuated joint — from which we get manufacturable geometry and grounded physics:
  • a real STEP file (B-rep, manufacturable) per link + an assembly STEP,
  • an STL mesh per link (for shaped rendering, and usable as a MuJoCo mesh geom), and
  • REAL mass + a real moment of inertia computed from the solid x material density (not a primitive
    approximation with a guessed scalar).

Units: solids are built in millimetres (CAD convention); STEP is emitted in mm; volumes convert to m^3
for mass. MuJoCo meshes use the STL with a 0.001 scale.
"""

from __future__ import annotations

from pathlib import Path

from virturoid.services.grounded_physics import MATERIALS


def realize_shape(spec: dict):
    """Realize an ARBITRARY part shape from a blueprint spec into a build123d solid (mm). This is what frees
    blocks from box/capsule/cylinder — a block can be ANY shape the design needs. Supported families:

      • ``extrude``  {profile:[[x,y],...], height} — extrude any 2-D polygon (L-brackets, fins, plates…)
      • ``revolve``  {profile:[[r,z],...]}          — revolve a profile about z (domes, cones, nozzles, shells)
      • ``tapered``  {length, r0, r1}               — a frustum (tapered limb segment)
      • ``box``/``cylinder``/``capsule``/``sphere`` {length, radius} — parametric primitives
      • ``compound`` {parts:[sub-spec + ``at``:(x,y,z) + ``subtract``:bool, ...]} — a part is the boolean
        UNION (or SUBTRACT) of several positioned sub-shapes: real multi-feature blocks (a chassis = box +
        side rails - vents), recursively realized. This is the agents' true CAD control over ONE block.
    Optional MODIFIERS on any family: ``fillet`` (mm) rounds edges; ``chamfer`` (mm) bevels them; ``cutouts``:
    [[x,y,z,r,depth],...] (m) bores lightening pockets / vents anywhere; ``holes``:[[x,y,r],...] mounting bores
    (extrude). Falls back to a capsule on any malformed spec, so a bad blueprint never crashes the build."""
    import build123d as bd

    fam = (spec.get("family") or "capsule").lower()
    L = max(8.0, float(spec.get("length", 0.1)) * 1000.0)
    R = max(3.0, float(spec.get("radius", 0.03)) * 1000.0)
    if fam == "compound":                            # a part = boolean of positioned sub-shapes (multi-feature)
        try:
            base = None
            for sub in spec.get("parts", []):
                s = realize_shape(sub)
                ax, ay, az = sub.get("at", (0.0, 0.0, 0.0))
                # per-part ``euler`` (intrinsic xyz, DEGREES) so a sub-shape can be ROTATED, not only translated —
                # needed for mixed-axis assemblies (a wing's leading-edge spar across the panel, a tentacle's
                # suckers following the surface). Default (0,0,0) keeps every existing compound byte-identical.
                rx, ry, rz = sub.get("euler", (0.0, 0.0, 0.0))
                loc = bd.Location((float(ax) * 1000.0, float(ay) * 1000.0, float(az) * 1000.0),
                                  (float(rx), float(ry), float(rz)))
                s = s.moved(loc)
                if base is None:
                    base = s
                elif sub.get("subtract"):
                    base = base - s
                else:
                    base = base + s
            if base is not None and base.is_valid and base.volume > 0:
                return base
        except Exception:  # noqa: BLE001 - malformed compound -> capsule fallback below
            pass
    if fam == "role":                                # role-keyed multi-feature anatomy (build_anatomy)
        try:
            part = build_anatomy(spec.get("role", "actuator_segment"),
                                 float(spec.get("length", 0.1)), float(spec.get("radius", 0.03)))
            if part is not None and part.is_valid and part.volume > 0:
                return part
        except Exception:  # noqa: BLE001 - unknown role / awkward topology -> capsule fallback below
            pass
        with bd.BuildPart() as fb:
            bd.Cylinder(R, L)
        return fb.part
    try:
        with bd.BuildPart() as p:
            MM = 1000.0                               # all spatial inputs are METERS (gene units) -> mm for CAD
            if fam == "extrude":
                pts = [(float(x) * MM, float(y) * MM) for x, y in spec["profile"]]
                with bd.BuildSketch() as sk:
                    with bd.BuildLine():
                        bd.Polyline(*pts, close=True)
                    bd.make_face()
                bd.extrude(amount=float(spec.get("height", spec.get("length", 0.05))) * MM)
                for hx, hy, hr in spec.get("holes", []):
                    with bd.Locations((float(hx) * MM, float(hy) * MM, 0)):
                        bd.Cylinder(float(hr) * MM, 10 * L, mode=bd.Mode.SUBTRACT)
            elif fam == "revolve":
                pts = [(max(0.5, float(r) * MM), float(z) * MM) for r, z in spec["profile"]]
                with bd.BuildSketch(bd.Plane.XZ) as sk:
                    with bd.BuildLine():
                        bd.Polyline(*pts, (0.5, pts[-1][1]), (0.5, pts[0][1]), close=True)
                    bd.make_face()
                bd.revolve(axis=bd.Axis.Z)
            elif fam == "tapered":
                base_r = float(spec.get("r0", spec.get("radius", 0.03)))
                r0 = max(3.0, base_r * MM)
                r1 = max(1.0, float(spec.get("r1", base_r * 0.5)) * MM)
                bd.Cone(bottom_radius=r0, top_radius=r1, height=L)
            elif fam == "loft":
                # ELLIPTICAL loft through sections [[z, half_y, half_x], ...] (meters) — a body whose
                # width (y) and depth (x) vary INDEPENDENTLY along z. This is the one primitive a pure
                # axisymmetric revolve cannot express, and it is what makes a trunk/limb read as HUMAN
                # (broad front, shallow side; broad shoulders -> narrow waist). The credibility primitive.
                faces = []
                for z, hy, hx in spec["sections"]:
                    ry = max(0.5, float(hy) * MM)
                    rx = max(0.5, float(hx) * MM)
                    with bd.BuildSketch() as _sk:
                        bd.Ellipse(x_radius=rx, y_radius=ry)
                    faces.append(_sk.sketch.moved(bd.Location((0, 0, float(z) * MM))))
                bd.loft(faces)
            elif fam == "sole":
                # a forward-pointing foot (rounded heel->toe sole): +x = forward, +z = up.
                Lf = float(spec.get("length", 0.1)) * MM
                Wf = float(spec.get("width", 0.04)) * MM
                Hf = max(2.0, float(spec.get("height", 0.03)) * MM)
                heel = float(spec.get("heel", 0.02)) * MM
                with bd.BuildSketch(bd.Plane.XY) as _sk:
                    with bd.BuildLine():
                        bd.Polyline((-heel, -Wf), (Lf, -Wf * 0.78), (Lf, Wf * 0.78), (-heel, Wf), close=True)
                    bd.make_face()
                bd.extrude(amount=Hf)
                try:
                    bd.fillet(p.edges().filter_by(bd.Axis.Z), min(Wf * 0.5, Hf * 0.8))
                except Exception:  # noqa: BLE001
                    pass
            elif fam == "box":
                bd.Box(2 * R, 2 * R, L)
            elif fam == "sphere":
                bd.Sphere(R)
            elif fam == "cylinder":
                bd.Cylinder(R, L)
            else:                                        # capsule
                bd.Cylinder(R, L)
                for z in (L / 2, -L / 2):
                    with bd.Locations((0, 0, z)):
                        bd.Sphere(R)
            for cut in spec.get("cutouts", []):       # lightening pockets / vents, bored anywhere (any family);
                try:                                  # [x, y, z, r, depth, axis] — axis x|y|z (default z)
                    cx, cy, cz = float(cut[0]) * MM, float(cut[1]) * MM, float(cut[2]) * MM
                    cr = max(0.5, float(cut[3]) * MM)
                    cd = float(cut[4]) * MM if len(cut) > 4 else 4 * R
                    rot = {"x": (0, 90, 0), "y": (90, 0, 0)}.get(str(cut[5]).lower() if len(cut) > 5 else "z", (0, 0, 0))
                    with bd.Locations(bd.Location((cx, cy, cz), rot)):
                        bd.Cylinder(cr, cd, mode=bd.Mode.SUBTRACT)
                except Exception:  # noqa: BLE001 - a bad cutout just doesn't bore; keep the solid
                    pass
            fil = float(spec.get("fillet", 0) or 0) * 1000.0
            if fil > 0:
                try:
                    bd.fillet(p.edges(), min(fil, R * 0.4))
                except Exception:  # noqa: BLE001 - fillet can fail on some topology; skip it
                    pass
            cham = float(spec.get("chamfer", 0) or 0) * 1000.0
            if cham > 0:
                try:
                    bd.chamfer(p.edges(), min(cham, R * 0.35))
                except Exception:  # noqa: BLE001 - chamfer can fail on some topology; skip it
                    pass
        part = p.part
        if part is not None and fam in ("loft", "sole"):     # floor to z=0 in the link's [0,length] frame
            try:
                part = part.moved(bd.Location((0, 0, -float(part.bounding_box().min.Z))))
            except Exception:  # noqa: BLE001
                pass
        if part is not None and part.is_valid and part.volume > 0:
            return part
    except Exception:  # noqa: BLE001 - malformed spec -> safe capsule fallback below
        pass
    with bd.BuildPart() as fb:
        bd.Cylinder(R, L)
    return fb.part


def build_link_solid(shape: str, length_m: float, radius_m: float, actuated: bool):
    """A DETAILED, manufacturable link solid (mm): a slim structural body, mounting collars at both ends,
    and — for an actuated link — a real servo housing with an output horn at the proximal (joint) end.
    This is what closes the mechanical-detail gap vs a bare capsule (real robots show housings/brackets)."""
    import build123d as bd

    L = max(20.0, length_m * 1000.0)
    R = max(6.0, radius_m * 1000.0)
    with bd.BuildPart() as p:
        if shape == "box":
            bd.Box(1.4 * R, 1.4 * R, L)
        else:                                        # slim structural tube body (real limbs are slender)
            bd.Cylinder(R * 0.55, L)
        for z in (L / 2, -L / 2):                    # small mounting collars where it bolts to its neighbours
            with bd.Locations((0, 0, z)):
                bd.Cylinder(R * 0.8, R * 0.35)
        if actuated:                                 # a compact CYLINDRICAL motor can across the joint
            with bd.Locations(bd.Location((0, 0, -L / 2), (0, 90, 0))):
                bd.Cylinder(R * 0.95, R * 2.0)       # slim can spanning the joint width
    return p.part


# Role-keyed ANATOMY library: a small set of MULTI-FEATURE link solids so a generated body reads as real
# hardware instead of one bare box/capsule — the root cause of the "toy" look was that every link compiled
# to a single primitive. A torso becomes a contoured shell + chest plate + neck/pelvis collars; a head a
# domed sensor module with a visor; a limb a tapered shaft with a motor housing at the joint + a coupling
# flange. Each solid is built in the link's [0, length] +z body frame (lowest point at z=0) so it drops
# straight onto the compiler's primitive. VISUAL ONLY — it never changes the gene's MuJoCo collision
# primitive, mass, or inertia (the compiler reads shape/length/radius/mass, never `geometry`).
# `actuator_segment` is the generic tapered limb generator; thigh/shin/upper_arm/forearm are semantic
# aliases for it (same procedural solid) that ALSO key the kit-bash catalog (part_catalog.ROLE_MESHES) to a
# specific real limb mesh. torso/head/hand/foot/base have their own anatomy.
_LIMB_ROLES = {"actuator_segment", "thigh", "shin", "upper_arm", "forearm",
               "quad_hip", "quad_thigh", "quad_calf"}
_ROLES = _LIMB_ROLES | {"torso_shell", "sensor_head", "hand_plate", "foot_pad", "mount_base", "quad_trunk"}


def _loft_solid(bd, sections, *, floor: bool = True):
    """Build an ELLIPTICAL-loft solid (mm) through ``sections`` = [(z, half_y, half_x), ...]; floored to
    z>=0. This is the credibility primitive: width (y) and depth (x) vary independently along z, giving the
    broad-front/shallow-side, broad-shoulders/narrow-waist forms a revolve cannot. Returns a build123d Part."""
    faces = []
    for z, hy, hx in sections:
        with bd.BuildSketch() as _sk:
            bd.Ellipse(x_radius=max(0.5, float(hx)), y_radius=max(0.5, float(hy)))
        faces.append(_sk.sketch.moved(bd.Location((0, 0, float(z)))))
    with bd.BuildPart() as p:
        bd.loft(faces)
    part = p.part
    if floor:
        try:
            part = part.moved(bd.Location((0, 0, -float(part.bounding_box().min.Z))))
        except Exception:  # noqa: BLE001
            pass
    return part


def build_anatomy(role: str, length_m: float, radius_m: float):
    """A detailed, role-specific ORIGINAL link solid (build123d Part, mm), built in the link's [0, length]
    +z frame. Every role is now an ELLIPTICAL LOFT (or a simple analytic part) generated by us — NOT a copied
    real-robot mesh. Unknown roles fall back to a generic limb spindle. VISUAL ONLY (the compiler reads only
    shape/length/radius/mass, never `geometry`), so physics is unaffected."""
    import build123d as bd

    L = max(20.0, length_m * 1000.0)
    R = max(6.0, radius_m * 1000.0)
    role = (role or "actuator_segment").lower()
    if role not in _ROLES:
        role = "actuator_segment"

    def _fil(part, rad):
        try:
            return bd.fillet(part.edges(), max(0.5, rad))
        except Exception:  # noqa: BLE001 - fillet is decorative; awkward edges just stay sharp
            return part

    def _spindle(half):
        """A tapered LIMB shaft: a fat proximal 'shoulder' that bolts onto the joint motor, tapering SMOOTHLY
        and monotonically to a slimmer distal end where the next joint attaches. `half` = the shoulder radius.
        Deliberately ASYMMETRIC + oval (slimmer front-to-back) — reads as an anatomical/mechanical limb, NOT the
        old symmetric bulb-waist-bulb profile (which rendered as a turned furniture baluster)."""
        return _loft_solid(bd, [
            (0.00,     half * 0.82, half * 0.70),     # proximal collar seating onto the motor
            (0.14 * L, half * 1.00, half * 0.86),     # proximal shoulder — widest, the muscle/structure mass
            (0.55 * L, half * 0.74, half * 0.64),     # tapering shaft
            (0.88 * L, half * 0.60, half * 0.52),     # distal narrowing toward the next joint
            (L,        half * 0.50, half * 0.44)])     # distal cap

    if role == "torso_shell":                        # humanoid trunk: hips -> waist -> chest -> shoulder shelf
        return _loft_solid(bd, [
            (0.00,      R * 0.74, R * 0.46),          # pelvis
            (0.12 * L,  R * 0.80, R * 0.50),          # hips
            (0.32 * L,  R * 0.62, R * 0.42),          # waist
            (0.55 * L,  R * 0.74, R * 0.48),          # solar plexus
            (0.74 * L,  R * 0.92, R * 0.52),          # chest
            (0.92 * L,  R * 1.00, R * 0.52),          # shoulder shelf (broadest)
            (L,         R * 0.46, R * 0.40)])         # neck base
    if role == "quad_trunk":                         # an ANIMAL trunk: long front-to-back (x), low + rounded
        return _loft_solid(bd, [
            (0.00,      R * 0.70, R * 1.15),
            (0.30 * L,  R * 0.92, R * 1.40),
            (0.65 * L,  R * 0.88, R * 1.36),
            (L,         R * 0.58, R * 1.05)])
    if role == "sensor_head":                        # a tapered ovoid skull (jaw -> cheeks -> crown)
        head = _loft_solid(bd, [
            (0.00,     R * 0.66, R * 0.60),
            (0.22 * L, R * 0.92, R * 0.82),
            (0.50 * L, R * 1.00, R * 0.94),
            (0.80 * L, R * 0.80, R * 0.74),
            (L,        R * 0.34, R * 0.30)])
        return _fil(head, 0.2 * R)
    if role == "hand_plate":                         # a flat mitten: wide in y, thin in x, tapering to fingers
        return _loft_solid(bd, [
            (0.00,     R * 0.7, R * 0.5),
            (0.35 * L, R * 1.0, R * 0.55),
            (0.75 * L, R * 0.9, R * 0.45),
            (L,        R * 0.5, R * 0.30)])
    if role == "foot_pad":                           # a forward-pointing sole (heel -> toe), +x = forward
        with bd.BuildPart() as p:
            with bd.BuildSketch(bd.Plane.XY) as _sk:
                with bd.BuildLine():
                    bd.Polyline((-0.5 * R, -0.95 * R), (2.4 * R, -0.72 * R),
                                (2.4 * R, 0.72 * R), (-0.5 * R, 0.95 * R), close=True)
                bd.make_face()
            bd.extrude(amount=max(8.0, 0.9 * L))
        return _fil(p.part, 0.28 * R)
    if role == "mount_base":                         # pedestal: wide base flange + a motor drum
        with bd.BuildPart() as p:
            with bd.Locations((0, 0, 0.12 * L)):
                bd.Cylinder(1.5 * R, 0.26 * L)
            with bd.Locations((0, 0, 0.6 * L)):
                bd.Cylinder(1.0 * R, 0.8 * L)
        return _fil(p.part, 0.1 * R)
    return _spindle(R)                               # actuator_segment / thigh / shin / arm / quad limb roles


def _seg_mesh_fp(out, s, *, kitbash: bool, synth: bool):
    """The per-segment visual-mesh cache path + resolved kit-bash role. Shared by ``build_visual_meshes`` and
    ``prebake_synth_meshes`` so a parallel pre-bake writes the exact files the compiler later reads. The
    content hash includes geometry + kit-bash/synth flags, so any change re-bakes."""
    import hashlib
    import json
    actuated = s.joint_type in ("revolute", "prismatic")
    geom = getattr(s, "geometry", None)
    kit_role = (geom.get("role") if isinstance(geom, dict) and geom.get("family") == "role" else None) \
        if kitbash else None
    if kit_role is not None:
        from virturoid.services.part_catalog import has_part
        if not has_part(kit_role):
            kit_role = None
    sig = json.dumps([s.shape, round(s.length_m, 5), round(s.radius_m, 5), actuated, geom,
                      bool(kit_role), bool(synth)], sort_keys=True, default=str)
    return out / f"{s.name}_{hashlib.md5(sig.encode()).hexdigest()[:10]}.stl", kit_role


def prebake_synth_meshes(gene, out_dir: str, *, kitbash: bool = True, llm=None, max_workers: int = 6) -> int:
    """Pre-synthesize, IN PARALLEL, the AI mesh for every ``geometry.family=='synth'`` segment into the
    cache, so a later ``gene_to_meshed_mjcf(gene, synth=True)`` is instant. LLM calls are I/O-bound and the
    mesh build is pure numpy, so threading is safe and a whole body takes ~one part's wall-clock instead of
    the serial sum. Returns the number of parts synthesized this call (cache hits are skipped)."""
    from concurrent.futures import ThreadPoolExecutor

    from virturoid.services.mesh_synth import synthesize_part
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for s in gene.segments:
        if s.joint_type == "prismatic":
            continue
        geom = getattr(s, "geometry", None)
        if not (isinstance(geom, dict) and geom.get("family") == "synth"):
            continue
        fp, _ = _seg_mesh_fp(out, s, kitbash=kitbash, synth=True)
        if fp.exists():
            continue
        jobs.append((geom.get("desc") or f"a robot {s.name} part", s.length_m, s.radius_m, str(fp)))
    if not jobs:
        return 0

    def _run(j):
        try:
            return synthesize_part(j[0], j[1], j[2], j[3], llm=llm)
        except Exception:  # noqa: BLE001 - one part's failure never blocks the others
            return None
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(jobs)))) as ex:
        return sum(1 for r in ex.map(_run, jobs) if r)


def build_visual_meshes(gene, out_dir: str, *, cache: bool = True, kitbash: bool = False,
                        synth: bool = False) -> dict:
    """Generate a per-segment VISUAL STL normalized to the link's ``[0, length]`` body frame so it drops
    straight onto the compiler's primitive (which spans 0..length along local +z). This is the mesh layer
    that makes a compiled body render like real hardware instead of a bare capsule.

    A segment with a custom ``geometry`` spec gets that ARBITRARY/ROLE shape (``realize_shape`` — tapered
    limbs, role anatomy, extruded feet…); a plain segment gets the generic detailed ``build_link_solid``
    (slim body + collars + motor housing). Only PRISMATIC segments (gripper/hand fingers) are skipped — they
    keep their primitive. (Previously geometry segments were ALSO skipped, which silently discarded the
    detailed shapes and was the second root cause of the toy look.)

    Cached by a content hash of the segment's shape params INCLUDING its geometry spec, so re-rendering the
    same body is instant but a changed shape re-bakes. Returns ``{segment_name: absolute_stl_path}``.
    Requires build123d (raises if absent; callers that want a graceful fallback should catch and compile
    without meshes)."""
    import hashlib
    import json

    import build123d as bd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, str] = {}
    for s in gene.segments:
        if s.joint_type == "prismatic":              # gripper/hand fingers keep their primitive
            continue
        actuated = s.joint_type in ("revolute", "prismatic")
        geom = getattr(s, "geometry", None)
        # Kit-bash a real, license-clean link mesh for this part when enabled and one is catalogued for the
        # role; otherwise synth / procedural anatomy / generic detail. Flags are in the cache key (toggle rebakes).
        fp, kit_role = _seg_mesh_fp(out, s, kitbash=kitbash, synth=synth)
        if not (cache and fp.exists()):
            baked = None
            if kit_role is not None:                 # real Menagerie link mesh, fitted to this link's frame
                from virturoid.services.part_catalog import fitted_part_stl
                baked = fitted_part_stl(kit_role, s.length_m, s.radius_m, str(fp))
            if baked is None and synth and isinstance(geom, dict) and geom.get("family") == "synth":
                from virturoid.services.mesh_synth import synthesize_part   # AI-generated novel-part mesh
                baked = synthesize_part(geom.get("desc") or f"a robot {s.name} part",
                                        s.length_m, s.radius_m, str(fp))
            if baked is None and geom:               # arbitrary/role shape -> realize, then drop floor to z=0
                solid = realize_shape(geom)
                try:
                    minz = float(solid.bounding_box().min.Z)
                except Exception:  # noqa: BLE001 - degenerate bbox -> assume already floored
                    minz = 0.0
                solid = solid.moved(bd.Location((0, 0, -minz)))
                bd.export_stl(solid, str(fp))
            elif baked is None:                      # generic detailed link: centered [-L/2,L/2] -> +L/2 to [0,L]
                solid = build_link_solid(s.shape, s.length_m, s.radius_m, actuated)
                solid = solid.moved(bd.Location((0, 0, s.length_m * 1000.0 / 2.0)))
                bd.export_stl(solid, str(fp))
        meshes[s.name] = str(fp.resolve()).replace("\\", "/")
    return meshes


def is_real_cad(step_path) -> bool:
    """True iff ``step_path`` is a REAL B-rep STEP (from ``export_gene_cad``), not a placeholder or a stub.

    The product audit found the blessed path shipped 479-byte comment-only "STEP" files that still carried an
    ISO-10303 header, so a file-EXISTS check accepted fakes. This is the content gate the readiness ledger
    uses for the ``real_cad_exported`` stage: a real STEP is non-trivial in size, carries no placeholder
    marker, and contains actual B-rep entity instances (``#NNN =`` lines, typically with
    ADVANCED_BREP_SHAPE_REPRESENTATION / MANIFOLD_SOLID_BREP). Cheap + dependency-free (text scan)."""
    import re

    p = Path(step_path)
    try:
        if not p.is_file() or p.stat().st_size < 2048:
            return False
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return False
    if "Virturoid MVP placeholder STEP" in text or "placeholder STEP" in text:
        return False
    # Real STEP carries many entity-instance lines like "#123 = CARTESIAN_POINT(...)"; a few isolated comment
    # lines do not. Require a solid B-rep representation OR a healthy population of entity instances.
    if "ADVANCED_BREP_SHAPE_REPRESENTATION" in text or "MANIFOLD_SOLID_BREP" in text:
        return True
    return len(re.findall(r"#\d+\s*=", text)) >= 20


def export_gene_cad(gene, out_dir: str, *, material: str = "aluminum") -> dict:
    """Export real CAD for every link of ``gene``: per-link STEP + STL + assembly STEP, with mass and
    moment of inertia computed from the solids. Returns a manifest dict (the manufacturable artifact)."""
    import build123d as bd

    density = MATERIALS.get(material, 2700.0)
    out = Path(out_dir)
    (out / "step").mkdir(parents=True, exist_ok=True)
    (out / "stl").mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    solids = []
    total_mass = 0.0
    used_grounded = False
    for s in gene.segments:
        if getattr(s, "geometry", None):                 # blueprint specified an ARBITRARY shape for this block
            solid = realize_shape(s.geometry)
        else:                                            # default detailed link (tube + collars + motor can)
            solid = build_link_solid(s.shape, s.length_m, s.radius_m,
                                     s.joint_type in ("revolute", "prismatic"))
        bd.export_step(solid, str(out / "step" / f"{s.name}.step"))
        bd.export_stl(solid, str(out / "stl" / f"{s.name}.stl"))
        vol_m3 = float(solid.volume) * 1e-9          # mm^3 -> m^3
        solid_mass = vol_m3 * density                # mass IF this link were a SOLID `material` billet
        # B3b (2026-07-24 audit): prefer the GROUNDED buildable mass the URDF/BOM/sim all report -- a real link
        # is a hollow CF tube plus a motor, NOT a solid aluminum billet, and treating it as solid invented a
        # ~35 kg dog. When the gene is grounded, ``s.mass_kg`` = structure(@fill) + actuator; use it as the one
        # source of truth and scale the solid's inertia tensor to that mass so ixx stays consistent with what
        # the manifest reports. Ungrounded genes fall back to the solid-volume estimate (flagged in the note).
        grounded = float(getattr(s, "mass_kg", 0.0) or 0.0)
        mass = grounded if grounded > 0 else solid_mass
        used_grounded = used_grounded or grounded > 0
        scale = (mass / solid_mass) if solid_mass > 0 else 1.0
        total_mass += mass
        ixx = float(solid.matrix_of_inertia[0][0]) * density * 1e-15 * scale   # mm^5*kg/m^3 -> kg*m^2 (approx)
        parts.append({"name": s.name, "volume_cm3": round(float(solid.volume) * 1e-3, 2),
                      "mass_kg": round(mass, 3), "ixx_kg_m2": round(ixx, 6),
                      "mass_basis": "grounded" if grounded > 0 else "solid_volume",
                      "step": f"step/{s.name}.step", "stl": f"stl/{s.name}.stl"})
        solids.append(solid)
    try:                                             # one manufacturable assembly STEP
        asm = solids[0]
        for sld in solids[1:]:
            asm = asm + sld
        bd.export_step(asm, str(out / "robot_assembly.step"))
        asm_ok = True
    except Exception:  # noqa: BLE001 - boolean union can be finicky; per-part STEP is the fallback
        asm_ok = False
    return {"material": material, "parts": parts, "part_count": len(parts),
            "total_mass_kg": round(total_mass, 3), "assembly_step": asm_ok, "dir": str(out)}
