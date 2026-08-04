"""Import an EXISTING robot model (MJCF or URDF — i.e. a CAD-exported robot) so the platform can iterate
on it: simulate it, learn control for it (MorphPolicy / MJX-GPU), evaluate it, and bank skills + tips —
the same flywheel as composed robots. This is the "I already have a robot, improve it" half of the goal.

MuJoCo loads both MJCF and URDF natively; we normalize to an MJCF string (ensuring a ground plane) that
flows through ``robot_mjcf`` everywhere. Meshed URDFs need their mesh/asset files alongside the file.
"""

from __future__ import annotations

import re
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------
# URDF REPAIR PASS (P0, agentic-ingestion plan). Real customer URDFs — the AS-PUBLISHED Unitree Go2 among
# them — do NOT load in MuJoCo as-is. The repairs here are deterministic, lossless FOR SIMULATION, and each
# one is REPORTED (never silent) so an ingesting agent can tell the customer exactly what was touched.
# ---------------------------------------------------------------------------------------------------------
_MATERIAL_FULL = re.compile(r"<material\s+name=\"(?P<name>[^\"]+)\"\s*>(?P<body>.*?)</material\s*>", re.S)
# any <material> — a full definition OR a name-only reference (self-closing / empty)
_ANY_MATERIAL = re.compile(r"<material\b[^>]*?/>|<material\b[^>]*?>.*?</material\s*>", re.S)
_VISUAL_BLOCK = re.compile(r"<visual\b[^>]*>.*?</visual\s*>", re.S)
_MESH_REF = re.compile(r'<mesh\b[^>]*\bfilename="([^"]+)"[^>]*?(?:/>|>\s*</mesh\s*>)', re.S)
_MESH_OK = (".stl", ".obj", ".msh")            # formats MuJoCo loads natively


def _within(path: Path, root: Path | None) -> bool:
    """True iff ``path`` canonicalizes INSIDE ``root``. The confinement that stops a HOSTILE URDF from resolving
    a 'mesh' to an absolute path (``file:///etc/passwd``) or a ``../../..`` escape outside the imported project
    -- which would let it use us as a host-filesystem read oracle, and (via _try_convert_dae) write the converted
    .obj outside the project too. mesh_root is the URDF's own directory; a mesh that isn't under it is treated as
    MISSING (the caller then boxes the link), never followed."""
    if root is None:
        return False
    try:
        rp = path.resolve(); rr = root.resolve()
        return rp == rr or rr in rp.parents
    except (OSError, RuntimeError):
        return False


def _resolve_mesh_path(fname: str, mesh_root: Path | None) -> Path | None:
    """Resolve a URDF mesh filename (``package://``, ``file://``, absolute, or relative) to an EXISTING local
    file CONFINED under ``mesh_root``, or None. Tries the ROS-package-relative path, the mesh_root join, and the
    bare basename. A path that resolves OUTSIDE mesh_root (absolute or ``../`` escape) is rejected as missing."""
    f = fname
    if f.startswith("package://"):
        f = f[len("package://"):].split("/", 1)[-1]     # drop the ROS package name, keep the path
    elif f.startswith("file://"):
        f = f[len("file://"):]
    p = Path(f)
    cands = ([p] if p.is_absolute() else []) + (
        [mesh_root / f, mesh_root / p.name] if mesh_root is not None else []) + [p]
    for c in cands:
        try:
            if c.exists() and _within(c, mesh_root):        # only follow meshes that stay inside the project
                return c
        except OSError:
            pass
    # last resort: the URDF's mesh subdir often differs from the on-disk layout (Franka references
    # franka_description/meshes/visual/... but ships them elsewhere) -> find the basename anywhere under mesh_root.
    if mesh_root is not None:
        try:
            hit = next(mesh_root.rglob(p.name), None)
            return hit if (hit is not None and _within(hit, mesh_root)) else None  # a symlink under root may escape
        except OSError:
            pass
    return None


def _try_convert_dae(path: Path) -> Path | None:
    """Convert a Collada ``.dae`` (Franka/KUKA visual meshes) to an ``.obj`` MuJoCo can load, best-effort via
    trimesh. Returns the new path or None when trimesh is absent / the load fails (caller then boxes it)."""
    try:
        import trimesh
        mesh = trimesh.load(str(path), force="mesh")
        if mesh is None or not getattr(mesh, "vertices", None) is not None or len(mesh.vertices) == 0:
            return None
        out = path.with_suffix(".virturoid.obj")
        mesh.export(str(out))
        return out if out.exists() else None
    except Exception:  # noqa: BLE001 - no trimesh / unreadable dae -> primitive fallback
        return None


def _mesh_elem_to_box(text: str, fname: str, size: str = "0.05 0.05 0.05") -> str:
    """Replace every ``<mesh filename="fname" .../>`` with a placeholder ``<box>`` so a link whose mesh is
    missing/unloadable still imports with its FRAME intact (the kinematics are what control iteration needs)."""
    pat = re.compile(r'<mesh\b[^>]*\bfilename="' + re.escape(fname) + r'"[^>]*?(?:/>|>\s*</mesh\s*>)', re.S)
    return pat.sub(f'<box size="{size}"/>', text)


def _resolve_meshes(text: str, mesh_root: Path | None, repairs: list) -> str:
    """Unified mesh pass (I1): resolve every mesh path, convert DAE→OBJ, and fall back a MISSING or unloadable
    mesh to a placeholder box — so an as-published Franka/KUKA (relative + .dae + un-shipped meshes) INGESTS
    instead of hard-failing on the first ``Error opening file``. Every outcome is reported, never silent."""
    names = list(dict.fromkeys(_MESH_REF.findall(text)))
    if not names:
        return text
    resolved = converted = 0
    boxed: list[str] = []
    for f in names:
        real = _resolve_mesh_path(f, mesh_root)
        suffix = real.suffix.lower() if real is not None else ""
        if real is not None and suffix in _MESH_OK:
            if real.as_posix() != f:
                text = text.replace(f'filename="{f}"', f'filename="{real.as_posix()}"')
            resolved += 1
        elif real is not None and suffix == ".dae":
            obj = _try_convert_dae(real)
            if obj is not None:
                text = text.replace(f'filename="{f}"', f'filename="{obj.as_posix()}"')
                converted += 1
            else:
                text = _mesh_elem_to_box(text, f); boxed.append(Path(f).name)
        else:
            text = _mesh_elem_to_box(text, f); boxed.append(Path(f).name)
    parts = []
    if resolved:
        parts.append(f"resolved {resolved} mesh path(s) locally")
    if converted:
        parts.append(f"converted {converted} Collada .dae mesh(es) to .obj")
    if boxed:
        shown = ", ".join(boxed[:4]) + ("..." if len(boxed) > 4 else "")
        parts.append(f"replaced {len(boxed)} missing/unloadable mesh(es) with placeholder boxes ({shown}) — the "
                     f"kinematic structure imports; supply the mesh files for full visual/collision fidelity")
    if parts:
        repairs.append({"kind": "mesh_resolve", "count": len(names), "detail": "; ".join(parts),
                        "boxed": len(boxed)})
    return text


def repair_urdf_text(text: str, *, mesh_root: Path | None = None) -> tuple[str, list[dict]]:
    """Make a real-world URDF compile in MuJoCo, reporting every change.

    Returns ``(repaired_text, repairs)`` where each repair is ``{kind, detail, count}``. Fixes, in order:

    1. **Duplicate material definitions** — the exact failure of the as-published Go2 URDF: a named material
       (e.g. a rubber colour) is FULLY re-defined in many ``<visual>`` blocks, and MuJoCo rejects the second
       full definition ("repeated element: 'material'"). We keep the FIRST definition of each name and rewrite
       every later one to a name-only reference ``<material name="X"/>`` — identical rendering, legal XML.
    2. **``package://`` mesh URIs** — resolved to a real file under ``mesh_root`` when one exists, otherwise
       left for MuJoCo's missing-mesh tolerance. Reported either way.
    3. **``<gazebo>`` / ``<transmission>`` blocks** — simulator-plugin metadata MuJoCo doesn't model; stripped.
    """
    repairs: list[dict] = []

    # MATERIALS. Real exporters (Blender→URDF, the as-published Go2) violate two MuJoCo rules at once: a single
    # <visual> carries MANY <material> children, and the same material name is fully re-defined in block after
    # block. MuJoCo needs exactly ONE material per visual and each name defined ONCE. So:
    #   (a) in every <visual>, keep only the FIRST <material>, capturing any full definition seen;
    #   (b) hoist the captured definitions to one top-level block and leave name-only references inline.
    defs: dict[str, str] = {}
    extra_dropped = {"n": 0}

    def _first_material_only(vis: re.Match) -> str:
        block = vis.group(0)
        mats = list(_ANY_MATERIAL.finditer(block))
        if not mats:
            return block
        for mm in mats:                                     # capture every full def we can, for the hoist
            full = _MATERIAL_FULL.fullmatch(mm.group(0))
            if full:
                defs.setdefault(full.group("name"), full.group(0))
        first = mats[0]
        name = re.search(r'name="([^"]+)"', first.group(0))
        keep = f'<material name="{name.group(1)}"/>' if name else first.group(0)
        extra_dropped["n"] += len(mats) - 1
        # rebuild the visual with just the first material (as a reference), all others removed
        without = _ANY_MATERIAL.sub("", block)
        return without.replace("</visual>", keep + "</visual>", 1) if "</visual>" in without else without + keep

    text = _VISUAL_BLOCK.sub(_first_material_only, text)
    # any material still fully-defined OUTSIDE a visual (top-level) -> capture + reference too
    text = _MATERIAL_FULL.sub(lambda m: (defs.setdefault(m.group("name"), m.group(0)),
                                         f'<material name="{m.group("name")}"/>')[1], text)
    if defs:
        block = "\n  " + "\n  ".join(defs[n] for n in defs) + "\n"
        m_robot = re.search(r"<robot\b[^>]*>", text)
        if m_robot:
            text = text[:m_robot.end()] + block + text[m_robot.end():]
        repairs.append({"kind": "material_normalize", "count": len(defs),
                        "detail": f"hoisted {len(defs)} material definition(s) to one top-level block, "
                                  f"referenced them inline, and dropped {extra_dropped['n']} extra <material> "
                                  f"stuffed into single <visual> elements (MuJoCo allows one per visual); "
                                  f"rendering unchanged"})

    # MESHES (I1): resolve package://+relative paths, convert DAE→OBJ, and box a MISSING/unloadable mesh so an
    # as-published Franka/KUKA/a1 ingests instead of hard-failing on the first "Error opening file".
    text = _resolve_meshes(text, mesh_root, repairs)

    for tag in ("gazebo", "transmission"):
        stripped = re.subn(rf"<{tag}\b.*?</{tag}\s*>", "", text, flags=re.S)
        if stripped[1]:
            text = stripped[0]
            repairs.append({"kind": f"strip_{tag}", "count": stripped[1],
                            "detail": f"removed {stripped[1]} <{tag}> block(s) (simulator-plugin metadata "
                                      f"MuJoCo does not model)"})
    return text, repairs


def _ensure_floor(xml: str) -> str:
    if 'type="plane"' in xml:
        return xml
    floor = '    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.30 0.34 0.40 1"/>\n'
    return xml.replace("<worldbody>", "<worldbody>\n" + floor, 1)


def _add_actuators(mjcf: str, model) -> tuple:
    """Add a motor per movable joint when the imported model has none — URDFs import joints (with effort
    limits) but no MuJoCo actuators, so without this the robot has DOFs but nothing to drive them. Lets an
    imported CAD model be CONTROLLED and improved, not just simulated."""
    import mujoco

    motors = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (2, 3):                    # 2=slide, 3=hinge (skip free/ball)
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name:
                motors.append(f'    <motor joint="{name}" ctrlrange="-1 1" gear="20"/>')
    if not motors:
        return mjcf, model
    block = "  <actuator>\n" + "\n".join(motors) + "\n  </actuator>\n"
    mjcf2 = mjcf.replace("</mujoco>", block + "</mujoco>", 1)
    return mjcf2, mujoco.MjModel.from_xml_string(mjcf2)


def _inject_mujoco_compiler(urdf_text: str) -> str:
    """M3 (2026-07-24 audit): keep a URDF's BASE-LINK inertial alive through MuJoCo's compile.

    MuJoCo's URDF loader defaults to ``fusestatic="true"``, which welds the root link into the world and
    DISCARDS its ``<inertial>`` -- so a customer's ``mass=2 kg`` base silently vanishes, and after we bolt on a
    free joint (``reroot_free_base``) the robot weighs less than the URDF declared. MuJoCo reads compiler
    options from a ``<mujoco>`` extension element inside ``<robot>`` (XMLreference/URDF-extension); adding
    ``<compiler fusestatic="false" inertiafromgeom="auto"/>`` preserves the real base inertial (and lets links
    that genuinely ship no ``<inertial>`` derive one from geometry, rather than defaulting to ~1 kg). Idempotent:
    a URDF that already carries a ``<mujoco>`` block is returned untouched, so we never override a customer's
    explicit MuJoCo directives (maintainer yuvaltassa, mujoco#1176)."""
    if re.search(r"<mujoco\b", urdf_text):
        return urdf_text                                   # respect a customer-provided <mujoco> block
    m = re.search(r"<robot\b[^>]*>", urdf_text)
    if not m:
        return urdf_text
    block = '\n  <mujoco>\n    <compiler fusestatic="false" inertiafromgeom="auto" balanceinertia="true"/>\n  </mujoco>'
    return urdf_text[:m.end()] + block + urdf_text[m.end():]


def import_model(path: str) -> dict:
    """Load an MJCF/URDF robot file. Returns ``{mjcf, name, parts, actuated, free_base, ok, note}`` —
    ``mjcf`` is a normalized MJCF string (ground ensured) usable everywhere a composed gene is."""
    import os
    import tempfile

    import mujoco

    p = Path(path)
    if not p.exists():
        return {"ok": False, "note": f"file not found: {path}"}
    sfx = p.suffix.lower()
    repairs: list[dict] = []
    try:
        if sfx in (".xml", ".mjcf"):
            # COMPILE FROM A PATH IN THE MODEL'S OWN DIRECTORY, never from a bare string. MuJoCo resolves
            # <include file=...>, meshdir and every asset RELATIVE TO THE XML'S LOCATION, so from_xml_string has
            # no directory context and any real-world model that ships its meshes beside it fails to load.
            # Measured: all three canonical MuJoCo Menagerie robots failed this way --
            #   unitree_go2/go2.xml   -> "Error opening file 'assets/base_0.obj'"
            #   unitree_go2/scene.xml -> "Error opening file 'go2.xml'"   (the <include> itself)
            #   unitree_h1/h1.xml     -> "Error opening file 'assets/pelvis.stl'"
            # and each still returned a 0-body import. Menagerie is the most common source a customer brings, so
            # "bring your own robot" was broken for the format it is most often brought in. The URDF branch below
            # already had this right (it writes a SIBLING temp file "so relative meshes still resolve"); the MJCF
            # branch never got the same treatment.
            tmp_xml = p.with_name(p.stem + ".__mjcfprep__.xml")
            try:
                tmp_xml.write_text(_ensure_floor(p.read_text(encoding="utf-8")), encoding="utf-8")
                model = mujoco.MjModel.from_xml_path(str(tmp_xml))
            finally:
                try:
                    tmp_xml.unlink()
                except OSError:
                    pass
            # Round-trip through the compiled model so the MJCF we hand downstream is self-contained (includes
            # expanded, asset paths absolute) rather than only loadable from the source directory.
            tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False); tmp.close()
            mujoco.mj_saveLastXML(tmp.name, model)
            mjcf = _ensure_floor(Path(tmp.name).read_text(encoding="utf-8"))
            os.unlink(tmp.name)
        elif sfx == ".urdf":
            # REPAIR FIRST (P0): a raw customer URDF (Go2!) may not compile. Try as-is, and on failure run the
            # deterministic repair pass and retry — so a clean file is untouched but a real one still loads.
            raw = p.read_text(encoding="utf-8")
            if "<xacro:" in raw or "${" in raw:                    # a xacro TEMPLATE, not a URDF — can't repair it
                return {"ok": False, "repairs": [],
                        "note": "this is a xacro template with unexpanded macros (${...} / <xacro:...>), not a "
                                "loadable URDF. Expand it first: `ros2 run xacro xacro robot.urdf.xacro "
                                "> robot.urdf` (or `rosrun xacro xacro`), then import the generated .urdf."}
            # M3: compile a copy carrying <compiler fusestatic="false"> (sibling file so relative meshes still
            # resolve) so the base-link inertial survives -- from_xml_path on the original would fuse it away.
            tmp_urdf = p.with_name(p.stem + ".__mjcfprep__.urdf")
            tmp_urdf.write_text(_inject_mujoco_compiler(raw), encoding="utf-8")
            try:
                try:
                    model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
                except Exception:  # noqa: BLE001 - the as-published file did not compile; repair + retry
                    fixed, repairs = repair_urdf_text(raw, mesh_root=p.parent)
                    tmp_urdf.write_text(_inject_mujoco_compiler(fixed), encoding="utf-8")
                    model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
            finally:
                try:
                    tmp_urdf.unlink()
                except OSError:
                    pass
            tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False); tmp.close()
            mujoco.mj_saveLastXML(tmp.name, model)
            mjcf = _ensure_floor(Path(tmp.name).read_text(encoding="utf-8"))
            os.unlink(tmp.name)
            model = mujoco.MjModel.from_xml_string(mjcf)            # re-validate the normalized MJCF
        else:
            # A bare "unsupported" costs the customer an afternoon. Name the format and the step that works —
            # USD in particular is the format an Isaac engineer will arrive with, and we have no reader for it
            # (the only Usd.Stage.Open in this repo is usd_exporter re-reading what it just wrote).
            from virturoid.services.input_classifier import (UNSUPPORTED_MODEL_ADVICE,
                                                             UNSUPPORTED_MODEL_FORMATS)
            fam = UNSUPPORTED_MODEL_FORMATS.get(sfx)
            if fam:
                return {"ok": False, "format": fam,
                        "note": f"{sfx} is {fam}. {UNSUPPORTED_MODEL_ADVICE[fam]}"}
            return {"ok": False, "note": f"unsupported format {sfx!r} — use .xml/.mjcf or .urdf"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "note": f"could not load model: {exc}",
                "repairs": repairs}

    if model.nu == 0:                          # URDF joints import without motors -> add them so it's drivable
        try:
            mjcf, model = _add_actuators(mjcf, model)
        except Exception:  # noqa: BLE001 - keep the importable model even if actuator synthesis fails
            pass

    free = any(int(model.jnt_type[j]) == 0 for j in range(model.njnt))   # free joint -> can move/locomote
    note = "ready to iterate on" if model.nu > 0 else "no actuators — can simulate but not learn control"
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
    return {"ok": True, "mjcf": mjcf, "name": p.stem, "parts": int(model.nbody) - 1,
            "actuated": int(model.nu), "free_base": bool(free), "note": note,
            "repairs": repairs, "body_names": body_names}


# ---------------------------------------------------------------------------------------------------------
def reroot_free_base(mjcf: str) -> tuple[str, bool]:
    """Give a FIXED-BASE imported body a free joint so it can actually locomote, reporting whether it did.

    MuJoCo welds a URDF's base link to the world when the file has no floating joint, so a real Go2/quadruped
    imports STANDING-BUT-BOLTED-DOWN — simulable but never walking, and a naive verify reads "WHEELS OFF
    GROUND" / "TIPPED" (measured #214, and the round-trip of our OWN URDF exports). The base geoms are welded
    directly into ``<worldbody>`` and the first ``<body>`` is a child link, so there is nothing to hang a joint
    on — we WRAP the whole robot (every worldbody child except the floor) in a new free-floating base body,
    placed at a standing height derived from the model's own geometry. Idempotent: an already-free model, or a
    non-locomotor we can't lift cleanly, is returned unchanged with ``False``.
    """
    import mujoco
    try:
        m = mujoco.MjModel.from_xml_string(mjcf)
        if any(int(m.jnt_type[j]) == 0 for j in range(m.njnt)):
            return mjcf, False                                      # already free-based
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        lows = [float(d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom)
                if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
        stand_h = max(0.05, -min(lows) + 0.02) if lows else 0.3     # lift so the lowest geom clears the floor
    except Exception:  # noqa: BLE001
        return mjcf, False

    mw = re.search(r"<worldbody\s*>", mjcf)
    mwe = re.search(r"</worldbody\s*>", mjcf)
    if not (mw and mwe):
        return mjcf, False
    inner = mjcf[mw.end():mwe.start()]
    # keep the floor plane OUT of the moving base; everything else (base geoms + link bodies) goes inside it
    floor = re.search(r'<geom\b[^>]*type="plane"[^>]*/>', inner)
    floor_xml = floor.group(0) if floor else ""
    robot_xml = inner.replace(floor_xml, "", 1) if floor_xml else inner
    wrapped = (f"\n    {floor_xml}\n"
               f'    <body name="import_base" pos="0 0 {stand_h:.4f}">\n'
               f"      <freejoint/>\n{robot_xml}\n    </body>\n  ")
    candidate = mjcf[:mw.end()] + wrapped + mjcf[mwe.start():]
    try:
        mujoco.MjModel.from_xml_string(candidate)                  # only adopt if it still compiles
    except Exception:  # noqa: BLE001
        return mjcf, False
    return candidate, True
