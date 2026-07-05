"""Generic gene -> MJCF compiler: realize ANY robot gene as a MuJoCo model.

This is the keystone the platform was missing. The old exporters only knew the five
named links of one fixed arm; this compiler builds a valid MuJoCo model from an arbitrary
``RobotGene`` kinematic tree, so humanoid / quadruped / amended / novel species become
buildable and physics-testable. It is the bridge from "the AI proposed a gene" to "a real
robot runs in MuJoCo".

Each segment becomes a nested ``<body>`` sitting at its parent's distal tip, with a joint
(hinge for revolute, slide for prismatic, none for a fixed/welded attach), a capsule/box/
cylinder geom along local +z, and — for actuated joints — a position actuator. The root is
welded to the world at the mount height (table/floor). The end-effector segment carries an
``ee_site`` so controllers/graspers have a tip reference.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

from virturoid.schemas.gene import RobotGene

TABLE_TOP_Z = 0.025  # shared with the scene exporter
_MOUNT_Z = {"table": TABLE_TOP_Z, "floor": 0.0, "torso": 0.0, "free": 0.1}  # "free" spawns above the floor
_JOINT_KIND = {"revolute": "hinge", "prismatic": "slide"}
# segment-name hints for a WALKING-limb structural link (drives the R2 visual-only fairing/boot pass).
_LIMB_HINT = ("leg", "thigh", "shank", "femur", "tibia", "shin", "calf")

# ---- shared appearance -----------------------------------------------------------------------------
# Generated bodies used to render as flat-grey primitives because this compiler emitted geoms with no
# material and no joint hardware (all the detail lived only in the CAD/STL path, which physics never sees).
# These blocks give every compiled robot a real-hardware read — white structural body, charcoal actuator
# housings, soft studio lighting — entirely through visual-only geoms/materials that never touch dynamics.
_VISUAL_XML = (
    '  <visual>\n'
    # Softer, dimmer headlight so the positioned 3-light rig does the modeling (a strong flat headlight is the
    # #1 "debug viewport" tell). Kept lit enough that a no-rig render is never pitch-black.
    '    <headlight ambient="0.33 0.33 0.36" diffuse="0.42 0.42 0.45" specular="0.12 0.12 0.12"/>\n'
    '    <quality shadowsize="8192" offsamples="8"/>\n'
    '    <map znear="0.01" zfar="60"/>\n'
    '    <global offwidth="1920" offheight="1080"/>\n'
    '  </visual>\n'
)
# Robot surface materials (always emitted so body geoms can reference them even with the floor disabled).
_ROBOT_MATERIALS = (
    '    <material name="mat_body" rgba="0.84 0.86 0.89 1" specular="0.35" shininess="0.45" reflectance="0.05"/>\n'
    # mat_joint = actuator housing. Lifted from near-black (0.15) so a dark-carbon limb still reads its motor
    # bulges as distinct machined housings instead of one black smear (the "black blob crowd" render defect).
    '    <material name="mat_joint" rgba="0.21 0.22 0.25 1" specular="0.55" shininess="0.6" reflectance="0.08"/>\n'
    '    <material name="mat_accent" rgba="0.92 0.47 0.13 1" specular="0.45" shininess="0.55"/>\n'
    '    <material name="mat_ee" rgba="0.22 0.24 0.29 1" specular="0.5" shininess="0.6"/>\n'
    # per-MATERIAL finishes so the viewport shows real material variety (set by GeneSegment.material):
    # mat_alu = the refined limb metal (BOM upgrades 'skeleton'->'aluminum'). Darkened from near-white (0.78) to
    # brushed gunmetal: a stark-white limb stick with near-black motor cans is the Tinkertoy read; anodized
    # gunmetal limbs with charcoal housings read as one cohesive machined limb (and dark anodized aluminum is
    # exactly what premium robot limbs actually use — physically honest, not just prettier).
    '    <material name="mat_alu" rgba="0.50 0.52 0.56 1" specular="0.6" shininess="0.55" reflectance="0.14"/>\n'
    '    <material name="mat_steel" rgba="0.34 0.36 0.40 1" specular="0.7" shininess="0.7" reflectance="0.18"/>\n'
    # mat_cf = the default structural-limb finish (skeleton/frame map here). Lifted from 0.11 so segment/joint
    # contrast reads; a limb in this + mat_joint housings gives the unified dark-limb look real quadrupeds use.
    '    <material name="mat_cf" rgba="0.155 0.165 0.185 1" specular="0.45" shininess="0.4" reflectance="0.06"/>\n'
    '    <material name="mat_ti" rgba="0.60 0.60 0.65 1" specular="0.5" shininess="0.55" reflectance="0.1"/>\n'
    '    <material name="mat_shell" rgba="0.20 0.42 0.72 1" specular="0.4" shininess="0.5" reflectance="0.05"/>\n'
    '    <material name="mat_metal" rgba="0.47 0.49 0.53 1" specular="0.78" shininess="0.78" reflectance="0.22"/>\n'
    '    <material name="mat_rubber" rgba="0.13 0.13 0.15 1" specular="0.15" shininess="0.15"/>\n'
)
# GeneSegment.material key -> the MJCF material above. 'skeleton'/'frame' (the generic structural default) map
# to dark CARBON so a limb reads as one unified dark structure with its motor housings — the design language of
# real quadrupeds/arms (Unitree/Spot black legs) — instead of a stark-white limb alternating with dark cans (the
# "Tinkertoy" render defect the fidelity battery flagged). 'aluminum' stays LIGHT as an explicit opt-in the BOM's
# task refinement can select; steel/carbon_fiber/titanium are honored as-is.
_MATERIAL_KEY_TO_MJCF = {
    "skeleton": "mat_cf", "frame": "mat_cf", "aluminum": "mat_alu", "steel": "mat_steel",
    "carbon_fiber": "mat_cf", "titanium": "mat_ti", "shell": "mat_shell", "metal": "mat_metal",
    "rubber": "mat_rubber",
}
# Scene dressing for the no-scene render/viewport path. A studio-gradient skybox (mid, not a near-black void)
# and a near-uniform CONCRETE floor (fine checker only for scale reference, mild reflectance) — the checkerboard
# tile floor is a classic sim-demo tell; concrete reads as a real workshop/lab. Floor material stays named
# "grid" (referenced by the floor geom + terrain scenes) so only its appearance changes, not any wiring.
_SCENE_ASSETS = (
    '    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.52 0.56 0.62" rgb2="0.15 0.17 0.21" width="512" height="512"/>\n'
    # Near-uniform matte concrete: barely-there checker for scale depth only (no edge grid lines), low
    # reflectance so the robot is not mirrored (a strong floor reflection is a sim tell). texrepeat kept modest.
    '    <texture name="grid" type="2d" builtin="checker" rgb1="0.415 0.420 0.428" rgb2="0.435 0.440 0.448" width="512" height="512"/>\n'
    '    <material name="grid" texture="grid" texrepeat="10 10" specular="0.15" shininess="0.2" reflectance="0.03"/>\n'
)
# 3-point studio rig: a shadow-casting key, a soft fill to open the shadows, and a rim/back light to separate the
# robot from the backdrop. Physics-neutral (lights never touch dynamics). <= 8 lights + headlight (MuJoCo cap).
_THREE_LIGHTS = (
    '    <light name="key" pos="2.2 -1.6 3.0" dir="-0.50 0.36 -1" diffuse="0.62 0.61 0.58" specular="0.30 0.30 0.28" castshadow="true"/>\n'
    '    <light name="fill" pos="-2.4 1.8 2.2" dir="0.50 -0.40 -1" diffuse="0.24 0.25 0.28" castshadow="false"/>\n'
    '    <light name="rim" pos="-1.4 -2.4 1.8" dir="0.35 0.62 -0.7" diffuse="0.20 0.20 0.23" specular="0.35 0.35 0.40" castshadow="false"/>\n'
)


def compile_gene_to_mjcf(gene: RobotGene, *, include_floor: bool = True, spawn_z: float | None = None,
                         meshes: dict | None = None, show_actuators: bool = False,
                         sensor_geoms: dict | None = None, physics_only: bool = False) -> str:
    """Compile a validated gene to a MuJoCo MJCF XML string.

    ``spawn_z`` overrides the free-base spawn height — pass ``standing_spawn_z(gene)`` for locomotion so the
    body spawns standing on its feet instead of the legacy fixed 0.1 m (which makes tall bodies penetrate the
    floor and get ejected). Raises ValueError if the gene is invalid (so a bad proposal never reaches physics).

    ``meshes`` (optional, ``{segment_name: stl_path}`` from ``cad_geometry.build_visual_meshes``) turns on the
    HIGH-FIDELITY render path: those links show a detailed visual mesh (slim body + collars + motor housing)
    while their primitive is kept as an invisible collision geom — same physics, real-hardware look. Omit it
    (the default) for the fast pure-primitive model used by sim/training/scene-gen. See ``gene_to_meshed_mjcf``.

    ``physics_only`` emits an MJX/GPU-SAFE model: it strips ALL visual-only decoration (the cylinder motor cans,
    collars, actuator housings, sensor pucks, palm plates) and converts any cylinder COLLIDER to a capsule —
    because MJX precompiles a collision function per geom-type-pair PRESENT, and cylinder collisions are
    unimplemented in MJX, so a single visual cylinder crashes ``mjx.put_model`` even at contype=0. The colliders
    that matter (box/capsule/sphere/plane) and the joints/actuators/sites are kept, so the trained dynamics are
    the same physics the CPU model uses. Use this for the GPU PPO path (``scripts/mjx_*``)."""
    issues = gene.validate()
    if issues:
        raise ValueError(f"cannot compile invalid gene {gene.id}: {'; '.join(issues)}")

    root = gene.root()
    base_z = _MOUNT_Z.get(gene.base_mount, TABLE_TOP_Z)
    if spawn_z is not None and gene.base_mount == "free":
        base_z = float(spawn_z)
    body_xml = _body_xml(gene, root, pos=(0.0, 0.0, base_z), indent=4, meshes=meshes,
                         show_actuators=show_actuators, sensor_geoms=sensor_geoms, physics_only=physics_only)
    actuators = _actuator_xml(gene)
    keyframe = _pose_keyframe(gene, base_z)   # render the body in its baked rest stance (if any)

    mesh_assets = "".join(
        f'    <mesh name="{escape(n)}_vis" file="{p}" scale="0.001 0.001 0.001"/>\n'
        for n, p in (meshes or {}).items()
    )
    floor = (
        '    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>\n'
        if include_floor else ""
    )
    lights = _THREE_LIGHTS if include_floor else ""  # measurement passes (no floor) don't need scene lights
    return (
        f'<mujoco model="{escape(gene.id)}">\n'
        '  <compiler angle="radian" autolimits="true"/>\n'
        '  <option timestep="0.002" gravity="0 0 -9.81"/>\n'
        f'{_VISUAL_XML}'
        '  <default>\n'
        '    <joint damping="0.8" armature="0.01"/>\n'
        '    <geom friction="1 0.05 0.001"/>\n'
        '  </default>\n'
        '  <asset>\n'
        f'{_SCENE_ASSETS}'
        f'{_ROBOT_MATERIALS}'
        f'{mesh_assets}'
        '  </asset>\n'
        '  <worldbody>\n'
        f'{lights}'
        f'{floor}'
        f'{body_xml}'
        '  </worldbody>\n'
        f'{actuators}'
        f'{keyframe}'
        '</mujoco>\n'
    )


def _pose_keyframe(gene: RobotGene, base_z: float) -> str:
    """Emit a MuJoCo ``<keyframe>`` from ``gene.metadata['rest_pose']`` (joint_name -> angle) so the body
    renders/spawns in a recognizable baked stance instead of a default straight-out star. The qpos vector
    follows MuJoCo's joint order EXACTLY: a free base contributes 7 (xyz + identity quat) then each actuated
    joint contributes 1, in the same pre-order DFS as ``_body_xml``. No-op when there is no rest pose."""
    pose = (getattr(gene, "metadata", None) or {}).get("rest_pose")
    if not pose:
        return ""
    qpos: list[float] = []
    if gene.base_mount == "free":
        qpos += [0.0, 0.0, float(base_z), 1.0, 0.0, 0.0, 0.0]
    base_len = len(qpos)

    def walk(seg) -> None:
        if seg.parent is not None and seg.joint_type in _JOINT_KIND:
            qpos.append(float(pose.get(f"{seg.name}_joint", 0.0)))
        for child in gene.children_of(seg.name):
            walk(child)

    walk(gene.root())
    if len(qpos) == base_len:                 # nothing actuated -> a keyframe adds nothing
        return ""
    qstr = " ".join(f"{v:.5f}" for v in qpos)
    return f'  <keyframe>\n    <key name="rest" qpos="{qstr}"/>\n  </keyframe>\n'


def gene_to_meshed_mjcf(gene: RobotGene, cache_dir: str = "build/_viewmesh", *,
                        include_floor: bool = True, spawn_z: float | None = None,
                        kitbash: bool = False, synth: bool = False, show_actuators: bool = True,
                        task: str = "") -> str:
    """High-fidelity render/viewport MJCF: generate (and cache) detailed per-link visual meshes, then compile
    a model whose visible surface is those meshes over an invisible primitive collider.

    ``kitbash`` defaults to FALSE: the product must produce ORIGINAL designs, so each link's visual is the
    role-keyed PROCEDURAL anatomy generated by ``cad_geometry.build_anatomy`` (build123d) — our own geometry,
    not a copied real-robot part. ``kitbash=True`` is an opt-in path that fits a real, license-clean Menagerie
    link mesh per role; it is a benchmark/reference only, NOT the default, because pasting real parts (a real
    Unitree head/limbs) is copying, not designing. If build123d/mesh generation is unavailable, it falls back
    to the procedural-detail primitive compiler (same physics). Use this for the viewport/hero renders; keep
    ``compile_gene_to_mjcf`` for sim/training.
    """
    meshes: dict | None = None
    try:
        from virturoid.services.cad_geometry import build_visual_meshes
        meshes = build_visual_meshes(gene, cache_dir, kitbash=kitbash, synth=synth) or None
    except Exception:  # noqa: BLE001 - missing CAD kernel / mesh-gen failure -> primitive fallback
        meshes = None
    sensor_geoms = None
    if show_actuators:                  # the render/viewport path also shows materials + sensors in position
        try:
            from virturoid.services.bom_builder import ensure_materials
            from virturoid.services.sensor_geometry import sensor_geoms_for_gene
            ensure_materials(gene)   # colour each part by material (the task-adaptive thickness/material is a
            #                          DESIGN step done at compose-time / explicitly, not in the measure path)
            sensor_geoms = sensor_geoms_for_gene(gene, task=task) or None
        except Exception:  # noqa: BLE001 - materials/sensors are a visual nicety; never block a compile
            sensor_geoms = None
    return compile_gene_to_mjcf(gene, include_floor=include_floor, spawn_z=spawn_z, meshes=meshes,
                                show_actuators=show_actuators, sensor_geoms=sensor_geoms)


def _lowest_world_z(mj, d) -> float:
    """Lowest world-z any geom reaches, from each geom's TIGHT local AABB corners transformed to world. Much
    more accurate than the bounding-sphere ``geom_rbound`` for long capsules and flat pads (a foot, a wheel) —
    the loose sphere both floated horizontal limbs and let flat feet poke through."""
    import numpy as np

    aabb = mj.geom_aabb.reshape(mj.ngeom, 6)                     # [cx,cy,cz, hx,hy,hz] in each geom's frame
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    lo = np.inf
    for g in range(mj.ngeom):
        c, h = aabb[g, :3], aabb[g, 3:]
        corners = c + signs * h                                 # (8,3) local AABB corners
        world = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
        lo = min(lo, float(world[:, 2].min()))
    return lo


def standing_spawn_z(gene: RobotGene, *, clearance: float = 0.03, meshed: bool = True) -> float:
    """Spawn height for a free-base body so its LOWEST point rests ~``clearance`` above the floor — it spawns
    standing on its feet instead of penetrating the floor (legacy fixed 0.1) and getting ejected. ``meshed``
    measures the DISPLAYED visual meshes (what the viewport shows — the mesh can hang below the primitive
    collider, which was the visible foot-penetration bug); pass ``meshed=False`` on hot training paths to
    measure the cheap primitive model instead. Falls back to the legacy height if MuJoCo is unavailable."""
    if gene.base_mount != "free":
        return _MOUNT_Z.get(gene.base_mount, TABLE_TOP_Z)
    ref = _MOUNT_Z["free"]                                       # measure the body's downward reach at 0.1
    for build_xml in ((lambda: gene_to_meshed_mjcf(gene, include_floor=False, spawn_z=ref)) if meshed else None,
                      lambda: compile_gene_to_mjcf(gene, include_floor=False, spawn_z=ref)):
        if build_xml is None:
            continue
        try:
            import mujoco

            mj = mujoco.MjModel.from_xml_string(build_xml())
            d = mujoco.MjData(mj)
            if mj.nkey > 0:                      # measure the body in its baked rest stance, not splayed flat
                mujoco.mj_resetDataKeyframe(mj, d, 0)
            mujoco.mj_forward(mj, d)
            return round((ref - _lowest_world_z(mj, d)) + clearance, 4)
        except Exception:  # noqa: BLE001 - meshing/MuJoCo issue -> try the primitive build, then legacy
            continue
    return _MOUNT_Z["free"]


def compile_gene_with_scene(gene: RobotGene, scene_objects, *, table: bool = True,
                            physics_only: bool = False) -> str:
    """Compile a gene + a pick-and-place scene into one MJCF the controllers/viewer can run.

    Produces the same world structure as the legacy arm exporter (table, materials, blocks
    as free bodies, open-top bins) but with the robot body generated from ANY gene — so the
    morphology-agnostic pick-place controller (which finds actuated joints + the ee_site)
    runs the task on a humanoid or novel body exactly as on the tabletop arm.

    ``physics_only`` emits the MJX/GPU-SAFE robot (no visual cylinder housings/collars; cylinder colliders ->
    capsules) so the manipulation MJX training scripts (mjx_residual_grasp / mjx_push) can run on the GPU —
    the gripper colliders, fingers, ee_site/grasp_site and the scene objects are all kept."""
    from virturoid.services.mujoco_exporter import _scene_objects_xml  # reuse exact scene XML

    issues = gene.validate()
    if issues:
        raise ValueError(f"cannot compile invalid gene {gene.id}: {'; '.join(issues)}")
    root = gene.root()
    base_z = _MOUNT_Z.get(gene.base_mount, TABLE_TOP_Z)
    scene_objects = list(scene_objects)
    # A scene that brings its own ground/walls (navigation, maze) is a FLOOR scene: drop the tabletop so the
    # robot drives on the scene's own floor instead of a 0.7x0.45 m table.
    floor_scene = any(getattr(o, "object_type", None) in ("floor", "wall") for o in scene_objects)
    table_xml = (
        f'    <geom name="table" type="box" size="0.7 0.45 {TABLE_TOP_Z}" pos="0.4 0 0" rgba="0.7 0.7 0.65 1"/>\n'
        if (table and not floor_scene) else ""
    )
    lines = [
        f'<mujoco model="{escape(gene.id)}">',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option timestep="0.002" gravity="0 0 -9.81"/>',
        _VISUAL_XML.rstrip("\n"),
        '  <default>',
        '    <joint damping="0.8" armature="0.01"/>',
        '    <geom friction="1 0.1 0.01"/>',
        '  </default>',
        '  <asset>',
        '    <material name="mat_red" rgba="0.8 0.1 0.1 1"/>',
        '    <material name="mat_blue" rgba="0.1 0.2 0.8 1"/>',
        '    <material name="mat_gray" rgba="0.5 0.5 0.5 1"/>',
        '    <material name="mat_link" rgba="0.35 0.35 0.38 1"/>',
        _ROBOT_MATERIALS.rstrip("\n"),
        '  </asset>',
        '  <worldbody>',
        _THREE_LIGHTS.rstrip("\n"),
        table_xml + _body_xml(gene, root, pos=(0.0, 0.0, base_z), indent=4,
                              physics_only=physics_only).rstrip("\n"),
        *_scene_objects_xml(list(scene_objects)),
        '  </worldbody>',
        _actuator_xml(gene).rstrip("\n") or "  <actuator></actuator>",
        '</mujoco>',
    ]
    return "\n".join(lines) + "\n"


def _body_xml(gene: RobotGene, seg, pos: tuple[float, float, float], indent: int,
              meshes: dict | None = None, show_actuators: bool = False,
              sensor_geoms: dict | None = None, physics_only: bool = False) -> str:
    pad = " " * indent
    px, py, pz = pos
    euler = ""
    me = getattr(seg, "mount_euler", (0.0, 0.0, 0.0))
    if any(abs(v) > 1e-9 for v in me):
        euler = f' euler="{me[0]:.5f} {me[1]:.5f} {me[2]:.5f}"'
    lines = [f'{pad}<body name="{escape(seg.name)}" pos="{px:.5f} {py:.5f} {pz:.5f}"{euler}>']

    # A floating-base robot (mobile_base): the root chassis gets a freejoint so it can drive around,
    # instead of being welded to the world. Only the root (parent is None) and only when base_mount=free.
    if seg.parent is None and gene.base_mount == "free":
        lines.append(f'{pad}  <freejoint name="{escape(seg.name)}_free"/>')

    # Joint attaching this segment to its parent (none -> rigidly welded).
    if seg.joint_type in _JOINT_KIND:
        ax = " ".join(f"{v:.4f}" for v in seg.joint_axis)
        rng = ""
        if seg.joint_lower is not None and seg.joint_upper is not None:
            rng = f' range="{seg.joint_lower:.4f} {seg.joint_upper:.4f}"'
        lines.append(
            f'{pad}  <joint name="{escape(seg.name)}_joint" type="{_JOINT_KIND[seg.joint_type]}" '
            f'axis="{ax}"{rng}/>'
        )

    # Per-part MATERIAL drives the colour/finish: a coloured shell body, metal feet/hands, dark carbon limbs,
    # etc. (GeneSegment.material, set per role + refined per task). A structural link with NO material hint now
    # defaults to dark CARBON (not the legacy light mat_body): every render showed the Tinkertoy defect where
    # some limb segments carried "skeleton" (dark) and their siblings fell through to light mat_body, so a single
    # limb alternated light/dark bead-on-a-string. Shell/body/head parts carry an explicit "shell" material and
    # stay the bright accent; only unhinted structural links are affected, and they should read as one dark limb.
    material = _MATERIAL_KEY_TO_MJCF.get(seg.material or "") or (
        "mat_joint" if seg.joint_type == "prismatic" else "mat_cf")
    meshed = bool(meshes) and seg.name in meshes and not physics_only
    lines.append(_geom_xml(seg, pad + "  ", material=material, meshed=meshed, physics_only=physics_only))
    # physics_only strips ALL visual-only decoration (cylinder motor cans, collars, housings, sensor pucks)
    # so the model is MJX/GPU-safe — those cosmetic cylinders crash mjx.put_model even at contype=0.
    if not meshed and not physics_only:    # the visual mesh already has housings/collars; primitives get them added
        # When real actuators are shown, suppress the generic guessed "motor can" (the real datasheet-sized
        # housing replaces it); the distal collar/flange still helps read the link.
        detail = _detail_geoms_xml(seg, pad + "  ", suppress_motor=show_actuators)
        if detail:
            lines.append(detail)
    # Real off-the-shelf actuator: render the datasheet-sized housing of the part that drives this joint.
    if show_actuators and not physics_only and seg.joint_type == "revolute":
        from virturoid.services.component_geometry import actuator_housing_xml
        act = actuator_housing_xml(seg, pad + "  ")
        if act:
            lines.append(act)

    # Sensors mounted on THIS segment (camera eyes on the head, LiDAR puck on the chassis, an IMU on the
    # torso) — placed in the body's local frame, visual-only so dynamics are untouched.
    if sensor_geoms and not physics_only and seg.name in sensor_geoms:
        for g in sensor_geoms[seg.name].split("\n"):
            if g.strip():
                lines.append(f"{pad}  {g}")

    # End-effector tip reference site at the segment's distal end.
    if seg.is_end_effector:
        lines.append(f'{pad}  <site name="ee_site" pos="0 0 {seg.length_m:.5f}" size="0.01"/>')
        # For a parallel-jaw gripper (prismatic "finger" children), also emit a grasp TCP site at
        # the finger MID-plane between the jaws: the true grasp point. IK targeting ee_site (the
        # finger base) overshoots the object by a finger-length; targeting grasp_site centers the
        # closing jaws on the object.
        fingers = [c for c in gene.children_of(seg.name) if c.joint_type == "prismatic"]
        if fingers:
            tcp_z = seg.length_m + max(c.mount_offset[2] + c.length_m / 2.0 for c in fingers)
            lines.append(f'{pad}  <site name="grasp_site" pos="0 0 {tcp_z:.5f}" size="0.01"/>')
            # VISUAL-ONLY palm plate bridging the slim wrist to the jaws — the fingers mount at ±span/2 in
            # x/y, well beyond the wrist surface, so without it the jaws float in empty space beside the arm
            # tip. Spans the finger mount offsets; mass=0/contype=0 so dynamics, grasp IK and contacts are
            # untouched (the fingers remain the only colliding gripper geoms).
            sx = max(abs(c.mount_offset[0]) for c in fingers)
            sy = max(abs(c.mount_offset[1]) for c in fingers)
            fr = max(c.radius_m for c in fingers)
            hx, hy, hz = max(sx + fr, fr * 1.8), max(sy + fr, fr * 1.8), max(fr * 0.9, 0.01)
            lines.append(
                f'{pad}  <geom name="{escape(seg.name)}_palm" type="box" pos="0 0 {seg.length_m:.5f}" '
                f'size="{hx:.5f} {hy:.5f} {hz:.5f}" material="mat_joint" mass="0" contype="0" conaffinity="0"/>')

    # Children attach at this segment's distal tip (0,0,length), plus any translational mount_offset
    # (lets e.g. two gripper fingers sit side-by-side in y rather than overlapping at the tip).
    for child in gene.children_of(seg.name):
        mo = getattr(child, "mount_offset", (0.0, 0.0, 0.0))
        lines.append(_body_xml(gene, child, pos=(mo[0], mo[1], seg.length_m + mo[2]),
                               indent=indent + 2, meshes=meshes, show_actuators=show_actuators,
                               sensor_geoms=sensor_geoms, physics_only=physics_only))

    lines.append(f"{pad}</body>")
    return "\n".join(lines) + "\n"


def _geom_xml(seg, pad: str, material: str = "mat_body", meshed: bool = False, physics_only: bool = False) -> str:
    name = f'{escape(seg.name)}_geom'
    # When meshed, the primitive becomes collision-only: invisible (alpha 0) + group 3, but its shape/size/
    # mass are untouched, so dynamics & contacts stay byte-identical to the primitive model. The visible
    # surface is the detailed mesh appended below.
    surf = ' rgba="0 0 0 0" group="3"' if meshed else f' material="{material}"'
    if seg.shape == "box":
        h = seg.length_m / 2.0
        coll = (f'{pad}<geom name="{name}" type="box" pos="0 0 {h:.5f}" '
                f'size="{seg.radius_m:.5f} {seg.radius_m:.5f} {h:.5f}" mass="{seg.mass_kg:.5f}"{surf}/>')
    else:
        # MJX has no cylinder collision; in physics_only mode a cylinder COLLIDER becomes a capsule (the closest
        # MJX-safe round primitive) so a cylinder-shaped link can still train on GPU.
        gtype = "cylinder" if (seg.shape == "cylinder" and not physics_only) else "capsule"
        coll = (f'{pad}<geom name="{name}" type="{gtype}" fromto="0 0 0 0 0 {seg.length_m:.5f}" '
                f'size="{seg.radius_m:.5f}" mass="{seg.mass_kg:.5f}"{surf}/>')
    if not meshed:
        return coll
    vis = (f'{pad}<geom name="{escape(seg.name)}_vis" type="mesh" mesh="{escape(seg.name)}_vis" '
           f'material="{material}" mass="0" contype="0" conaffinity="0"/>')
    return coll + "\n" + vis


def _detail_geoms_xml(seg, pad: str, *, suppress_motor: bool = False) -> str:
    """VISUAL-ONLY mechanical detail that makes a generated link read as real hardware instead of a bare
    primitive: a cylindrical actuator housing across each revolute joint, and a flange/collar at the link's
    distal tip where the next segment bolts on. Every detail geom is ``mass=0 contype=0 conaffinity=0`` so it
    contributes NOTHING to dynamics, inertia, or contacts — the load-bearing collision primitive is untouched,
    which is why this changes appearance without any physics regression (grasp/locomotion stay byte-identical).

    ``suppress_motor`` drops the generic guessed can when a real datasheet-sized actuator housing
    (``component_geometry``) is being rendered instead.
    """
    R = float(seg.radius_m)
    L = float(seg.length_m)
    vis = ' mass="0" contype="0" conaffinity="0"'
    parts: list[str] = []
    # Actuator housing: a slightly-fat "motor can" coaxial with the joint axis, centered at the joint
    # (which sits at this body's local origin). This is the single biggest "robot vs toy" cue.
    if seg.joint_type == "revolute" and not suppress_motor:
        ax = seg.joint_axis
        n = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) or 1.0
        hl = min(R * 1.6, R + 0.02)               # half-length of the can along the joint axis
        ex, ey, ez = ax[0] / n * hl, ax[1] / n * hl, ax[2] / n * hl
        cr = min(R * 1.45, R + 0.022)             # can radius (reads as a bulge over the slimmer limb)
        parts.append(
            f'{pad}<geom name="{escape(seg.name)}_motor" type="cylinder" '
            f'fromto="{-ex:.5f} {-ey:.5f} {-ez:.5f} {ex:.5f} {ey:.5f} {ez:.5f}" '
            f'size="{cr:.5f}" material="mat_joint"{vis}/>'
        )
    # Distal collar/flange on a slender structural link (skip boxes, fingers, and short stubs).
    if seg.shape in ("capsule", "cylinder") and L > 0.06 and seg.joint_type != "prismatic":
        t = min(0.012, L * 0.10)
        parts.append(
            f'{pad}<geom name="{escape(seg.name)}_collar" type="cylinder" '
            f'fromto="0 0 {L - t:.5f} 0 0 {L + t:.5f}" size="{R * 1.22:.5f}" material="mat_joint"{vis}/>'
        )
    # R2 FAIRING + BOOT (visual-only): give a walking limb bodywork so it reads as a DESIGNED limb, not a bare
    # capsule chain (the render's remaining toy tell after styling + proportions). The PROXIMAL segments get a
    # shell-accent fairing sleeve over the mid-span (clear of the joint cans); the welded terminal segment (the
    # foot) gets a rubber boot at its contact tip. Distal structural segments stay bare dark, so the limb reads
    # two-tone — accent bodywork over dark structure, the Go2/Spot design language validated in the A/B/C study.
    name_l = (seg.name or "").lower()
    if any(k in name_l for k in _LIMB_HINT):
        welded = (seg.joint_type or "") not in ("revolute", "prismatic")
        tail = name_l.rsplit("_", 1)
        idx = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0
        if welded and 0.02 < L:                        # terminal welded segment == foot -> rubber boot pad
            # Center the boot UP the foot so its distal extent (pos + boot_r) aligns with the foot capsule's own
            # distal cap (L + R) — a boot at the tip with a bigger radius hangs below the collision contact and
            # reads as a spawn penetration (the standing-spawn test measures EVERY geom's oriented AABB). Aligned,
            # the visible pad wraps the foot without ever protruding past the surface the robot actually stands on.
            boot_r = R * 1.35
            boot_z = L + R - boot_r
            parts.append(
                f'{pad}<geom name="{escape(seg.name)}_boot" type="sphere" pos="0 0 {boot_z:.5f}" '
                f'size="{boot_r:.5f}" material="mat_rubber"{vis}/>')
        elif seg.shape in ("capsule", "cylinder") and L > 0.05 and idx == 0:   # top of limb -> shell fairing
            # Only the TOP (hip/thigh) segment gets the accent bodywork, so the segments below stay bare dark and
            # the limb reads two-tone (accent over structure) regardless of how many segments the leg has —
            # fairing every proximal segment turned a 3-segment leg all-accent, losing the structure read.
            p0, p1 = 0.18 * L, 0.78 * L
            parts.append(
                f'{pad}<geom name="{escape(seg.name)}_fairing" type="capsule" '
                f'fromto="0 0 {p0:.5f} 0 0 {p1:.5f}" size="{R * 1.4:.5f}" material="mat_shell"{vis}/>')
    return "\n".join(parts)


def _actuator_xml(gene: RobotGene) -> str:
    """Torque (motor) actuators — the pick-place controller PD-computes torques and writes
    them to data.ctrl, clamped to forcerange. Must match the legacy exporter (gear=1 motors),
    not position servos, or the controller's torques get misread as position setpoints."""
    actuated = gene.actuated_joints()
    if not actuated:
        return ""
    lines = ["  <actuator>"]
    for s in actuated:
        effort = float(s.actuator_torque_nm or 10.0)
        lines.append(
            f'    <motor name="{escape(s.name)}_motor" joint="{escape(s.name)}_joint" '
            f'gear="1" forcerange="{-effort:.2f} {effort:.2f}"/>'
        )
    lines.append("  </actuator>")
    return "\n".join(lines) + "\n"
