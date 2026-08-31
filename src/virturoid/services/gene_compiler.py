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

import json
import math
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape

from virturoid.schemas.gene import RobotGene
from virturoid.services.install_paths import anchored

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
    # mat_shell = the BODYWORK finish (torso/chassis/head/limb fairings). Was a saturated primary blue
    # (0.20 0.42 0.72), the one colour in this palette that no shipping robot wears: against the charcoal
    # limbs and gunmetal housings the chassis read as a painted toy block, and "a blue box with a blue ball
    # for a head" was the first thing a reviewer said about the demo. Real quadruped bodywork is a light
    # composite shell over a dark structural frame (ANYmal, B2, Franka); that also gives the strongest
    # value contrast against mat_cf limbs, so the machined detail on the chassis actually reads.
    '    <material name="mat_shell" rgba="0.80 0.81 0.84 1" specular="0.42" shininess="0.52" reflectance="0.07"/>\n'
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
# THE SOLVER CONTRACT, EMITTED ONCE AND SHARED BY BOTH COMPILE PATHS (plain + scene). It used to be two
# copies of the same literal, which is exactly how a train/deploy divergence starts.
#
# `iterations` was never chosen. The fidelity pass that added `integrator="implicitfast"` and the friction
# cone (bceca7e, "implicitfast, elliptic contacts, structure-aware damping...") set no caps, so MuJoCo's
# defaults applied BY OMISSION -- Newton at 100 iterations / 50 line-search steps -- while the CPU rollouts
# separately call `morph_policy.compiled_model(..., solver_iterations=20)`, which overwrites `opt.iterations`
# AFTER compile. So a generated body was DEPLOYED at 20 iterations on CPU and TRAINED at 100 on MJX: a silent
# train/deploy split. Emitting 20 closes it -- the CPU override becomes a no-op and both paths step the
# identical solver -- and it costs no CPU physics, because MuJoCo's Newton solver EARLY-EXITS on tolerance and
# these bodies converge in a handful of iterations. On MJX/GPU it is NOT free: the jitted kernel cannot branch
# on convergence and runs the FULL nominal count every step, every env, which is why GPU throughput looked
# like a mysterious env-count cliff. 20 (not the 10 that would maximise GPU throughput) is deliberate: 10
# would leave CPU at 20 and GPU at 10, re-creating the very divergence this constant exists to close.
#
# WHAT IS DELIBERATELY *NOT* CAPPED HERE, AND WHY -- both were tried and both were measured to cost real
# product behaviour, so the cheap-looking extra speed is not free:
#
#   * `ls_iterations="8"` SUBSTITUTES A CUSTOMER'S BODY. `ensure_walkable_quad` measures a composed body and
#     swaps in a generic template when it reads as not walking, and the shorter line search pushes one across
#     that threshold: MEASURED, `compose_robot("a large quadruped robot")` returns the authored
#     `anatomy_creature_91b931bf` (20 segments, 2.241 m, 3.569 kg) with the line search at its default and the
#     generic `built_quadruped_18seg` (18 segments, 1.890 m, 14.917 kg) with it at 8. Capping `iterations`
#     alone leaves composer output BYTE-IDENTICAL for that prompt and for "a small quadruped robot dog".
#   * `cone="pyramidal"` costs a verdict. `test_customer_ingest::test_adopt_control_script_utilises_and_
#     improves` tunes the authored quad dog to a credible walk under every other combination but not under
#     pyramidal+caps at its shipped 4x10 search budget (it recovers at 8x20). It is also not needed for
#     parity: CPU deploy and MJX train were already BOTH elliptic for generated bodies.
#
# See [[body-and-gait-are-co-tuned]] -- the bodies are shaped against this contact model, so the contact model
# and the line search are controller-level changes that each need their own measured before/after.
#
# The contact-rich manipulation call sites (grasp_skill, grasp_eval, push_eval, pick_place_controller on CPU;
# every mjx_residual_*/mjx_push_*/mjx_grasp_* script on GPU) pin their own iterations after compile -- 30/12
# and 10/8 respectively -- so this line does not move them at all.
_PHYSICS_OPTION = ('  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"'
                   ' iterations="20"/>')


def _base_z_for(gene: RobotGene) -> float:
    """The fixed base's height: an explicit `base_height_m` when the design gave one, else the named mount.

    `base_mount` says what the robot is bolted TO; it cannot say how high that is. A delta hangs from an overhead
    plate, and table/floor/torso give 0.025/0/0, so its whole mechanism compiled BELOW the floor."""
    h = getattr(gene, "base_height_m", None)
    if h is not None:
        return float(h)
    return _MOUNT_Z.get(gene.base_mount, TABLE_TOP_Z)


def _with_source_meshes(gene: RobotGene, meshes: dict | None, *, physics_only: bool) -> dict | None:
    """Fill in each segment's OWN imported mesh when the caller supplied no mesh map of its own.

    An imported segment carries ``geometry={"family": "source_mesh", "path": ...}`` — that link's real geometry,
    baked into the segment's frame by ``robot_import``. It is part of the gene exactly as ``shape`` and
    ``length_m`` are, so a compiler that ignores it is describing a robot the gene does not declare. Only the
    render path ever read it, via ``build_visual_meshes``; every consumer that compiles the gene directly — the
    exported ``robot.xml`` above all — emitted bare primitives. Measured on a Menagerie Unitree Go2 ingested
    through the product's own front door: the package a customer opens contained 21 capsules and 25 cylinders
    and ZERO meshes, against 16 in the file they handed us, and rendered as an unrecognisable pile of pills.

    Three deliberate limits:

      * VISUAL ONLY. These become ``class="visual"`` geoms (``mass=0 contype=0 conaffinity=0``) over the
        unchanged primitive collider, so the collision set — every geom the physics, the gait and the verdict
        are computed from — is byte-identical with and without them.
      * NEVER under ``physics_only``. That model exists to be MJX-safe, and mesh assets are exactly the kind of
        decoration it strips.
      * ONLY when the caller passed nothing. A caller that supplies ``meshes`` is managing the asset set and its
        portability (``write_packaged_visual_mjcf`` rewrites every path relative to the package); quietly adding
        a machine-local absolute path to that map would put a reference outside the package into a file whose
        whole purpose is to survive being copied.

    A path that is not on disk is dropped rather than emitted: MuJoCo refuses to compile a model with a mesh it
    cannot read, so one missing STL would turn a fidelity gain into a total outage for that robot.
    """
    if physics_only or meshes:
        return meshes
    found: dict[str, str] = {}
    for s in gene.segments:
        g = getattr(s, "geometry", None)
        if not (isinstance(g, dict) and g.get("family") == "source_mesh"):
            continue
        p = str(g.get("path") or "")
        try:
            if p and Path(p).is_file():
                found[s.name] = p.replace("\\", "/")
        except OSError:                              # unreadable path (dead drive, permissions) -> primitive
            continue
    return found or meshes


def compile_gene_to_mjcf(gene: RobotGene, *, include_floor: bool = True, spawn_z: float | None = None,
                         meshes: dict | None = None, show_actuators: bool = False,
                         sensor_geoms: dict | None = None, physics_only: bool = False,
                         source_meshes: bool = True) -> str:
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
    the same physics the CPU model uses. Use this for the GPU PPO path (``scripts/mjx_*``).

    ``source_meshes=False`` builds the PURE PRIMITIVE model: an imported link's own STL is not resolved and not
    referenced, so MuJoCo has no asset to read off disk. Everything that decides physics is unchanged (the
    source meshes are ``class="visual"``, mass=0/contype=0/conaffinity=0), which is exactly why a caller that
    only wants a measurement should not pay for them. MEASURED on the Menagerie models: resolving and loading
    them costs a Unitree G1 0.159 s and a Go2 0.251 s PER COMPILE against 0.032 s / 0.024 s without — a 5-10x
    tax on what ``standing_spawn_z(meshed=False)`` documents as the cheap path. The second reason is not speed:
    ``ai_native_tools.render_sim_parity`` compares the MESHED spawn height against the PRIMITIVE one, and once
    the primitive model carried the same meshes that check compared a number with itself (measured identical to
    the digit on the G1 and the Go2 — 0.8805 and 0.3384 both sides). A tautological gate reads exactly like a
    passing one."""
    issues = gene.validate()
    if issues:
        raise ValueError(f"cannot compile invalid gene {gene.id}: {'; '.join(issues)}")

    if source_meshes:
        meshes = _with_source_meshes(gene, meshes, physics_only=physics_only)
    root = gene.root()
    base_z = _base_z_for(gene)
    if spawn_z is not None and gene.base_mount == "free":
        base_z = float(spawn_z)
    body_xml = _body_xml(gene, root, pos=(0.0, 0.0, base_z), indent=4, meshes=meshes,
                         show_actuators=show_actuators, sensor_geoms=sensor_geoms, physics_only=physics_only)
    self_collision_excludes = _self_collision_excludes_xml(gene)
    actuators = _actuator_xml(gene)
    keyframe = _pose_keyframe(gene, base_z)   # render the body in its baked rest stance (if any)
    sensors = "" if physics_only else _sensor_xml(gene)

    # An imported link is drawn from the CUSTOMER'S baked STL, which has no length field an amend can scale --
    # so ``scale_group``/``set_height`` used to move the child body to the link's new tip while the drawn mesh
    # kept its original size, and the render came apart into floating chunks. ``geometry['scale']`` carries the
    # per-axis factor those edits accumulate; fold it into the asset's own scale (base 0.001 = mm -> m).
    _mesh_scale = {}
    for _s in gene.segments:
        _g = getattr(_s, "geometry", None)
        _sc = (_g or {}).get("scale") if isinstance(_g, dict) else None
        if isinstance(_sc, (list, tuple)) and len(_sc) == 3:
            _mesh_scale[_s.name] = tuple(float(v) for v in _sc)
    mesh_assets = "".join(
        # ``p`` is escaped because a mesh path can now come from the CUSTOMER'S filesystem (an imported link's
        # baked STL), and a directory containing '&' would otherwise emit XML MuJoCo cannot parse.
        '    <mesh name="{n}_vis" file="{p}" scale="{sx:.9g} {sy:.9g} {sz:.9g}"/>\n'.format(
            n=escape(n), p=escape(str(p)), sx=0.001 * _mesh_scale.get(n, (1.0, 1.0, 1.0))[0],
            sy=0.001 * _mesh_scale.get(n, (1.0, 1.0, 1.0))[1],
            sz=0.001 * _mesh_scale.get(n, (1.0, 1.0, 1.0))[2])
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
        f'{_PHYSICS_OPTION}\n'
        f'{_VISUAL_XML}'
        '  <default>\n'
        '    <joint damping="0.8" armature="0.01" frictionloss="0.05"/>\n'
        '    <geom friction="1 0.05 0.001"/>\n'
        '    <default class="visual"><geom mass="0" contype="0" conaffinity="0"/></default>\n'
        '    <default class="collision"><geom friction="1 0.05 0.001"/></default>\n'
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
        f'{self_collision_excludes}'
        f'{_equality_xml(gene)}'
        f'{actuators}'
        f'{sensors}'
        f'{keyframe}'
        '</mujoco>\n'
    )


def _solver_ref_attrs(spec: dict) -> str:
    """`solref`/`solimp` for one constraint, or "" when the design did not state them.

    THE OMISSION IS DELIBERATE AND LOAD-BEARING. An absent attribute means MuJoCo applies its own default,
    which is exactly what "the source did not say" should compile to — writing a number here instead would put
    OUR guess into the customer's model and then let a calibration run "identify" it. Only a value the design
    actually carries is emitted.
    """
    out = []
    for key, n in (("solref", 2), ("solimp", 5)):
        v = (spec or {}).get(key)
        if isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v):
            out.append(f' {key}="{" ".join(f"{float(x):.6g}" for x in v)}"')
    return "".join(out)


def _equality_xml(gene: RobotGene) -> str:
    """Emit the `<equality>` block: `<connect>` per closed kinematic loop, `<joint>` per coupled DOF.

    A gantry's bridge is supported at BOTH columns; a delta's three arms meet at ONE platform. `segments` is a
    strict tree, so those were inexpressible — and the failure was quiet rather than loud: the gantry compiled
    with the right 3 prismatic DOF, rendered convincingly, and its second column carried no load at all. Driving
    the bridge along its rail walked it off that column into mid-air.

    A Panda's two fingers are likewise ONE gripper DOF, and its `<equality><joint>` is what makes them one.
    Both constraint kinds live here for the same reason: neither touches the body tree, they are top-level
    constraints naming things MuJoCo already has, so the gene can model them without `segments` ceasing to be
    a tree.

    NOTE `connect` is a SOFT constraint solved with the rest of the system, not a rigid weld — how hard the
    solver is asked to hold it is `solref`/`solimp`, and those are carried per-constraint from the source
    rather than defaulted here. Measured over the 9 Menagerie packages that declare a `<connect>`, worst anchor
    separation across 2000 stepped frames: Cassie 32.83 -> 14.82 mm, ToddlerBot 2.55 -> 0.01 mm, Robotiq 2F-85
    19.21 -> 0.93 mm, TidyBot 17.79 -> 0.90 mm, xArm7 0.03 -> 0.01 mm.
    """
    names = {s.name for s in gene.segments}
    jointed = {s.name for s in gene.segments if s.joint_type in _JOINT_KIND}
    lines = ["  <equality>"]
    for lc in (getattr(gene, "loop_closures", None) or []):
        a, b = (lc or {}).get("a"), (lc or {}).get("b")
        if a not in names or b not in names or a == b:
            continue                                  # validate() reports these; never emit a broken model
        seg = next((s for s in gene.segments if s.name == a), None)
        anchor = (lc or {}).get("anchor")
        if not (isinstance(anchor, (list, tuple)) and len(anchor) == 3):
            anchor = (0.0, 0.0, float(getattr(seg, "length_m", 0.0) or 0.0))   # a's tip, in a's own frame
        ax, ay, az = (float(v) for v in anchor)
        lines.append(f'    <connect body1="{escape(a)}" body2="{escape(b)}" '
                     f'anchor="{ax:.5f} {ay:.5f} {az:.5f}"{_solver_ref_attrs(lc)}/>')
    for cj in (getattr(gene, "coupled_joints", None) or []):
        a, b = (cj or {}).get("a"), (cj or {}).get("b")
        # A coupling onto a WELDED segment names `<segment>_joint`, which `_body_xml` never emitted — MuJoCo
        # would refuse the whole model. validate() reports it; the compiler must still not produce a broken one.
        if a not in jointed or b not in jointed or a == b:
            continue
        try:
            ratio = float((cj or {}).get("ratio", 1.0))
            offset = float((cj or {}).get("offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if ratio == 0.0:
            continue
        # MuJoCo solves  q_a - q_a0 = c0 + c1*(q_b - q_b0) + c2*(...)^2 + ...  — degree 1 is the whole corpus.
        lines.append(f'    <joint joint1="{escape(a)}_joint" joint2="{escape(b)}_joint" '
                     f'polycoef="{offset:.8g} {ratio:.8g} 0 0 0"{_solver_ref_attrs(cj)}/>')
    if len(lines) == 1:
        return ""
    lines.append("  </equality>")
    return "\n".join(lines) + "\n"


def _self_collision_excludes_xml(gene: RobotGene) -> str:
    """Exclude a link's contact with its structural ancestors, not robot-world contact.

    MuJoCo filters a direct parent/child pair by default, but not a grandchild against its torso ancestor.
    Anatomy appendages intentionally begin within a mounting shell for a continuous silhouette, so those
    missed ancestor contacts caused solver impulses from the robot colliding with its own structure.  Sibling
    limbs and all floor, scene-object, and external-body contacts remain enabled.
    """
    parents = {seg.name: seg.parent for seg in gene.segments}
    pairs: list[tuple[str, str]] = []
    for seg in gene.segments:
        ancestor = parents.get(seg.name)
        while ancestor is not None:
            pairs.append((ancestor, seg.name))
            ancestor = parents.get(ancestor)
    # A LOOP-JOINED pair meets by design, exactly as a parent and child do — but they are not ancestor and
    # descendant, so the walk above never reaches them. Left colliding, the contact solver pushes the two apart
    # while the equality constraint pulls them together, and the model fights itself at the one joint the design
    # cares most about.
    for lc in (getattr(gene, "loop_closures", None) or []):
        a, b = (lc or {}).get("a"), (lc or {}).get("b")
        if a and b and a != b:
            pairs.append((a, b))
    if not pairs:
        return ""
    lines = ["  <contact>"]
    lines.extend(
        f'    <exclude body1="{escape(ancestor)}" body2="{escape(descendant)}"/>'
        for ancestor, descendant in pairs
    )
    lines.append("  </contact>")
    return "\n".join(lines) + "\n"


def _euler_xyz_to_quat(a: float, b: float, c: float) -> tuple[float, float, float, float]:
    """MuJoCo ``euler="a b c"`` (default ``eulerseq="xyz"``, i.e. R = Rx(a) @ Ry(b) @ Rz(c)) as (w, x, y, z).

    Needed because a free base's ``qpos`` quaternion has to AGREE with the orientation baked into the root
    ``<body euler=...>`` tag; MuJoCo derives ``qpos0`` from that tag, so writing a different quaternion in a
    keyframe silently reorients the whole robot."""
    import math

    def mul(p, q):
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

    qx = (math.cos(a / 2.0), math.sin(a / 2.0), 0.0, 0.0)
    qy = (math.cos(b / 2.0), 0.0, math.sin(b / 2.0), 0.0)
    qz = (math.cos(c / 2.0), 0.0, 0.0, math.sin(c / 2.0))
    return mul(mul(qx, qy), qz)


def _pose_keyframe(gene: RobotGene, base_z: float) -> str:
    """Emit a MuJoCo ``<keyframe>`` from ``gene.metadata['rest_pose']`` (joint_name -> angle) so the body
    renders/spawns in a recognizable baked stance instead of a default straight-out star. The qpos vector
    follows MuJoCo's joint order EXACTLY: a free base contributes 7 (xyz + the ROOT'S OWN quat) then each
    actuated joint contributes 1, in the same pre-order DFS as ``_body_xml``. Every actuated/free body gets a
    named ``home`` key; ``rest`` is retained as a compatibility alias for existing rollout/render paths.

    The base quaternion must mirror the root's ``mount_euler``, not be hardcoded to identity. Composed bodies
    have an unrotated root so identity was right by accident; an IMPORTED root carries the rotation that aligns
    its reconstructed link frame (``robot_import._rot_z_to``), and overwriting that with identity threw the whole
    robot into a ~19 deg pitch — an imported Go2 measured 0.803 m tall against a real 0.394 m, legs splayed."""
    pose = (getattr(gene, "metadata", None) or {}).get("rest_pose") or {}
    qpos: list[float] = []
    if gene.base_mount == "free":
        root_euler = getattr(gene.root(), "mount_euler", None) or (0.0, 0.0, 0.0)
        qw, qx, qy, qz = _euler_xyz_to_quat(*(float(v) for v in root_euler))
        qpos += [0.0, 0.0, float(base_z), qw, qx, qy, qz]
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
    return (f'  <keyframe>\n    <key name="home" qpos="{qstr}"/>\n'
            f'    <key name="rest" qpos="{qstr}"/>\n  </keyframe>\n')


def gene_to_meshed_mjcf(gene: RobotGene, cache_dir: str = str(anchored("build/_viewmesh")), *,
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
        # actuator_in_mesh=not show_actuators: whoever is NOT drawing the datasheet _act geom owns the motor.
        # Letting both draw it put two coincident cans in the same place and z-fought into a sawtooth seam.
        meshes = build_visual_meshes(gene, cache_dir, kitbash=kitbash, synth=synth,
                                     actuator_in_mesh=not show_actuators) or None
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


def stage_mesh(src, dst_dir, owner: str, claimed: dict) -> "Path":
    """Copy ONE link's mesh into a package directory under a name no OTHER link can take. Returns the path.

    Every exporter that ships meshes has to answer the same question — what do I call this file next to the
    model? — and every one of them answered it with the source basename, which is not unique. Two links whose
    baked STLs share a basename land on one file and the second link silently renders as the first; the package
    still opens, nothing errors, and the robot is just wrong. So the naming rule lives here, once, and both the
    MJCF and the URDF exporters use it.

    The rule: keep the readable name (``meshes/FL_hip.stl`` reads far better in a shipped package than a hash),
    and only when it is already claimed BY A DIFFERENT LINK append that link's sanitized name, then a digest, so
    the disambiguation itself cannot collide. ``claimed`` maps filename -> owning link and must be shared across
    one export.

    An existing destination is reused only when it is byte-identical to the source (``filecmp`` with
    ``shallow=False``); a same-size-different-content leftover from an earlier export of a different robot would
    otherwise be silently shipped as this robot's geometry.
    """
    import filecmp
    import hashlib
    import shutil

    src, dst_dir = Path(src), Path(dst_dir)
    name = src.name
    if claimed.setdefault(name, owner) != owner:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(owner)).strip("_") or "link"
        name = f"{src.stem}__{safe}{src.suffix}"
        if claimed.setdefault(name, owner) != owner:
            name = f"{src.stem}__{hashlib.md5(str(owner).encode('utf-8', 'replace')).hexdigest()[:8]}{src.suffix}"
            claimed.setdefault(name, owner)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    if not (dst.is_file() and filecmp.cmp(str(src), str(dst), shallow=False)):
        shutil.copyfile(src, dst)
    return dst


#: Where a staged directory records WHAT WE PUT THERE, so a later export can tell its own leftovers from the
#: customer's files. It sits beside the files it describes, so it travels with a copied package.
STAGE_LEDGER_NAME = ".virturoid_staged.json"


def _file_digest(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_stage_ledger(dst_dir) -> dict:
    """``{filename: {"size", "sha256"}}`` -- what OUR last export wrote into ``dst_dir``.

    ``{}`` for a directory we have never staged into AND for one whose ledger has been deleted; the two are
    deliberately indistinguishable, because the safe answer is the same in both cases: we know of nothing here
    that is ours to remove. The redundancy that makes a deleted ledger survivable is the ``prior`` argument of
    ``prune_staged_dir`` -- the index document THIS repo wrote next to the directory (the CAD manifest, the
    URDF's own ``filename=`` refs, ``viewer_mesh_index.json``), which names the same files from the other side.
    """
    p = Path(dst_dir) / STAGE_LEDGER_NAME
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    wrote = rec.get("wrote")
    return wrote if isinstance(wrote, dict) else {}


def prune_staged_dir(dst_dir, keep, suffixes=(".stl",), *, prior=(), dry_run=False) -> dict:
    """Remove the files A PREVIOUS RUN OF OURS left in a staged directory. Never anything else.

    Returns ``{"removed", "kept_foreign", "kept_modified"}`` -- names, not paths, all sorted.

    ``stage_mesh`` above answers "what do I call this file", and this answers the question that has to be asked
    with it: what is still sitting there from LAST time? A build into a REUSED output directory (which is what
    ``autonomous_build`` does after a redesign, and what ``export_held`` does on every re-export of the same
    ``robot_id``) only ever ADDED files. Measured on a second build of a different body into one directory:
    ``cad/step/`` held 27 STEPs of which 18 belonged to a DISCARDED quadruped, and the arm's ROS 2 package
    shipped 25 meshes of which its URDF referenced 7. Every index we write -- the CAD manifest, the URDF's
    ``<mesh filename=>``, the MJCF's ``file=`` -- named only the current robot, so a reader that follows an
    index was safe and a customer running ``glob("cad/step/*.step")`` assembled a chimera out of two robots.
    Documenting that is not a fix: globbing a directory of STEP files is not an unreasonable thing to do.

    REMOVAL IS BY PROVENANCE, NOT BY SHAPE. The first version of this function deleted everything matching a
    suffix that was not in ``keep``, on the reasoning that "these directories have exactly one writer". That is
    true of OUR writers and says nothing about the customer: ``output_dir`` is theirs (``compose.py`` exposes
    ``--build OUTPUT_DIR`` and hands it straight to ``build_gene_package``), and re-exporting THE SAME ROBOT
    into a directory with files planted in it measured five deletions of files we had never written --
    ``cad/reference_fixture.step``, ``cad/step/my_custom_bracket.step``, ``cad/stl/customer_scan.STL`` (matched
    case-insensitively), ``robot/meshes/customer_endeffector.stl`` and the same file inside the ROS 2 package.
    So a file is removable only when we can show it is ours:

      * it is recorded in this directory's ledger (``STAGE_LEDGER_NAME``) AND still byte-identical to what we
        wrote -- a file the customer has since edited is theirs now and is reported in ``kept_modified``; or
      * ``prior`` names it. ``prior`` is what OUR OWN previously-written index for this directory claimed --
        the last ``cad_manifest.json``'s part list, the last URDF's ``filename=`` refs, the last
        ``viewer_mesh_index.json``. That is the redundancy that makes the scheme survive a customer who deletes
        the ledger, and it is also what lets the first post-fix export clean up a directory staged before the
        ledger existed.

    Anything else is left where it is and reported in ``kept_foreign``. An undeletable file (locked, read-only)
    is left alone and reported in ``kept_modified``; an export is never failed over one.

    Call this AFTER the document that references the survivors has been written, never before -- see
    ``write_exported_mjcf`` for why. ``dry_run`` computes the same answer without touching the directory or the
    ledger, so a caller can name the removals inside the artifact it is about to write and then commit them.
    """
    d = Path(dst_dir)
    if not d.is_dir():
        return {"removed": [], "kept_foreign": [], "kept_modified": []}
    keep = {str(k) for k in keep}
    prior = {str(k) for k in prior}
    sfx = tuple(s.lower() for s in suffixes)
    ours = read_stage_ledger(d)
    removed, kept_foreign, kept_modified = [], [], []
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.name == STAGE_LEDGER_NAME:
            continue
        if p.suffix.lower() not in sfx or p.name in keep:
            continue
        rec = ours.get(p.name)
        if isinstance(rec, dict):
            try:
                unchanged = (p.stat().st_size == rec.get("size")
                             and _file_digest(p) == rec.get("sha256"))
            except OSError:
                unchanged = False
            if not unchanged:                        # we wrote it; they changed it -- it is theirs now
                kept_modified.append(p.name)
                continue
        elif p.name not in prior:
            kept_foreign.append(p.name)              # never ours: not in the ledger, not in our last index
            continue
        if dry_run:
            removed.append(p.name)
            continue
        try:
            p.unlink()
        except OSError:                              # locked/read-only -> leave it; never fail an export over it
            kept_modified.append(p.name)
            continue
        removed.append(p.name)
    if not dry_run:
        # DROP the records for what we removed; do NOT re-stamp what we kept. This line was
        # ``_write_stage_ledger(d, keep)``, which re-digested every surviving file with no carry-over --
        # so a mesh the customer had hand-edited, which the cache had served rather than rewritten, was
        # silently re-adopted as ours. MEASURED: edit a file in viewer_assets/, rebuild the SAME robot
        # (the path the package guard explicitly permits), and the ledger swore their bytes were ours;
        # the next rebuild of a different body deleted it and filed it under ``removed_stale``. The
        # digest is the only thing separating "ours, untouched" from "ours once, theirs now", and a
        # prune has no business refreshing it -- only the writer that actually wrote a file may claim it,
        # via ``note_staged``.
        _ledger = read_stage_ledger(d)
        if _ledger:
            _gone = set(removed)
            _write_stage_ledger(d, (), carry_over={k: v for k, v in _ledger.items() if k not in _gone})
    return {"removed": sorted(removed), "kept_foreign": sorted(kept_foreign),
            "kept_modified": sorted(kept_modified)}


def note_staged(dst_dir, names) -> None:
    """Add ``names`` -- files this export JUST WROTE -- to a directory's staging ledger, removing nothing.

    Call it as soon as files are staged, before the document that will reference them is written. If the export
    then fails, the files it left behind are still recorded as ours, so the NEXT export can clear them instead
    of having to leave them forever as unattributable.

    Records already in the ledger for OTHER names are carried over UNCHANGED, digest included. Re-stamping them
    would quietly re-adopt a file the customer has edited since we wrote it -- the digest is the only thing that
    tells "ours, untouched" from "ours once, theirs now", and a helper that refreshes it would hand the next
    prune permission to delete their edit.
    """
    d = Path(dst_dir)
    if not d.is_dir():
        return
    _write_stage_ledger(d, names, carry_over=read_stage_ledger(d))


def _write_stage_ledger(dst_dir, names, *, carry_over: dict | None = None) -> None:
    """Record the files this export staged into ``dst_dir``, with a digest each, so the NEXT export can prove
    which of them are its own to remove. ``names`` are files written by THIS run, so their digests are taken
    now; ``carry_over`` entries survive verbatim for any name not in ``names``. Best-effort: a ledger we cannot
    write costs a future prune, never this export -- and a missing ledger degrades to leaving files alone."""
    d = Path(dst_dir)
    names = {str(n) for n in names}
    wrote: dict = {k: v for k, v in (carry_over or {}).items() if k not in names}
    for name in sorted(names):
        p = d / name
        try:
            if p.is_file():
                wrote[name] = {"size": p.stat().st_size, "sha256": _file_digest(p)}
        except OSError:
            continue
    try:
        (d / STAGE_LEDGER_NAME).write_text(json.dumps(
            {"version": 1,
             "what": "files written into this directory by a Virturoid export; the next export removes only "
                     "these, never anything else it finds here",
             "wrote": dict(sorted(wrote.items()))}, indent=2), encoding="utf-8")
    except OSError:
        pass


def write_exported_mjcf(gene: RobotGene, xml_path: str | Path, *, include_floor: bool = True,
                        spawn_z: float | None = None) -> dict:
    """Write the shipped ``robot.xml`` SELF-CONTAINED: any link mesh it references is copied next to it and
    addressed relatively, so the package opens on a machine that has never seen this one.

    ``compile_gene_to_mjcf`` now resolves an imported link's own mesh (``_with_source_meshes``) to wherever that
    STL happens to live — under ``build/_importmesh`` on the machine that did the import. Writing that string
    straight to a file would ship a model whose every mesh reference is an absolute path into somebody else's
    filesystem: it opens perfectly for whoever exported it and fails for everyone they send it to, which is the
    worst of the three possible outcomes because nothing looks wrong until it is in a customer's hands. There is
    direct precedent — the ROS2 export shipped 22 dangling references by copying a URDF and leaving its meshes
    behind — so the export copies first and references second.

    Returns ``{"path", "meshes", "mesh_dir"}``. Bodies with no meshes at all (anything we generated, which draws
    from primitives) simply write the same XML they always did and report ``meshes: 0``.
    """
    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    # What the PREVIOUS robot.xml in this directory claimed, read before we overwrite it: our own index, and so
    # the second proof of provenance the prune below needs if this package's staging ledger has been deleted.
    prior_meshes: set[str] = set()
    if xml_path.is_file():
        try:
            prior_meshes = {Path(m).name for m in
                            re.findall(r'file="([^"]+)"', xml_path.read_text(encoding="utf-8"))}
        except OSError:
            prior_meshes = set()
    if spawn_z is None:
        spawn_z = standing_spawn_z(gene)
    meshes = _with_source_meshes(gene, None, physics_only=False) or {}
    local: dict[str, str] = {}
    if meshes:
        mesh_dir = xml_path.parent / "meshes"
        claimed: dict[str, str] = {}                 # destination filename -> the segment that owns it
        for name, src in meshes.items():
            try:
                # ``stage_mesh`` keeps the customer's own link filename but never lets two links land on one
                # file: link names come from their file, so two that differ only in a character we sanitize
                # would otherwise silently overwrite each other's geometry.
                dst = stage_mesh(src, mesh_dir, name, claimed)
                # Relative to the XML, which is how MuJoCo resolves a mesh ``file=`` (and how Isaac's and
                # RViz's importers resolve one that is not a package:// URI).
                local[name] = os.path.relpath(dst, start=xml_path.parent).replace("\\", "/")
            except OSError:                          # unreadable source -> that link ships as its primitive
                continue
        note_staged(mesh_dir, {Path(p).name for p in local.values()})   # ours even if the compile below fails
    xml = compile_gene_to_mjcf(gene, include_floor=include_floor, spawn_z=spawn_z,
                               meshes=local or None)
    xml_path.write_text(xml, encoding="utf-8")
    # ONLY NOW. Whatever an EARLIER export into this same directory left in meshes/ is a different robot's
    # geometry -- this XML references none of it -- but the removal has to come AFTER the XML that references
    # the survivors is on disk. Pruning first meant any failure in ``compile_gene_to_mjcf`` (which this function
    # lets propagate) left the previous robot's model beside none of its meshes: a package that USED to load.
    pruned = prune_staged_dir(xml_path.parent / "meshes", {Path(p).name for p in local.values()},
                              prior=prior_meshes)
    return {"path": str(xml_path), "meshes": len(local),
            "mesh_dir": str(xml_path.parent / "meshes") if local else None,
            "removed_stale": pruned["removed"], "left_not_ours": pruned["kept_foreign"],
            "left_modified": pruned["kept_modified"]}


def write_packaged_visual_mjcf(gene: RobotGene, package_dir: str | Path, *,
                               model_uri: str = "simulation/robot_visual.xml",
                               include_floor: bool = True, spawn_z: float | None = None,
                               task: str = "") -> dict | None:
    """Write a portable, detailed visual model beside a gene package's physics artifacts.

    Training and task evaluation deliberately use the primitive collider model.  A viewport, on the other
    hand, needs the same original procedural CAD surfaces that were exported as STEP/STL.  This function bakes
    those link meshes *inside the package*, writes a MJCF model whose mesh paths are relative to that model,
    and records the public package URIs consumed by the browser replay.  The package therefore remains usable
    after being copied or exported; it never points at a machine-local mesh cache.

    It is intentionally fail-open: missing CAD tooling leaves the normal primitive replay intact and returns
    ``None`` rather than writing a misleading visual-model manifest.
    """
    root = Path(package_dir)
    model_path = root / model_uri
    asset_dir = model_path.parent / "viewer_assets"
    index_path = root / "simulation" / "viewer_mesh_index.json"
    # What the PREVIOUS index in this package claimed -- read before anything overwrites it, and used below as
    # the standing proof of which viewer_assets/ files are ours when the directory's own ledger is gone.
    prior_assets: set[str] = set()
    try:
        prior_assets = {Path(m.get("uri", "")).name for m
                        in json.loads(index_path.read_text(encoding="utf-8")).get("meshes", {}).values()}
    except (OSError, ValueError, AttributeError):
        prior_assets = set()
    try:
        from virturoid.services.cad_geometry import build_visual_meshes

        # This package compiles with show_actuators=True below, so the compiler draws the motors; leaving them
        # in the mesh too would double-draw every joint (see build_visual_meshes' actuator_in_mesh).
        # ``wrote`` IS THE CLAIM, NOT ``absolute_meshes``. ``build_visual_meshes`` serves a cached file on
        # EXISTENCE alone (generated links) or on matching SIZE (source links), so the returned map names
        # files this call did not touch. Claiming those re-digested a mesh the customer had hand-edited in
        # ``viewer_assets/`` -- the ledger then swore their bytes were ours, the package shipped their
        # geometry attributed to us, and the next rebuild of a different body DELETED it and filed it under
        # ``removed_stale`` as our own. Measured end to end. A file we did not write keeps whatever the
        # ledger already said about it: ours-and-untouched if it still matches, otherwise left alone.
        wrote_now: set[str] = set()
        absolute_meshes = build_visual_meshes(gene, str(asset_dir), cache=True, actuator_in_mesh=False,
                                              wrote=wrote_now) or {}
        if not absolute_meshes:
            return None
        # Claim them now, so a failure below leaves files that are still provably ours to clear next time.
        note_staged(asset_dir, wrote_now)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        # MuJoCo resolves a mesh filename from the XML's directory.  Store a relative filename in the XML,
        # but a package-relative URI in the sidecar for Three.js/API consumers.
        xml_meshes = {
            name: Path(os.path.relpath(path, start=model_path.parent)).as_posix()
            for name, path in absolute_meshes.items()
        }
        model_path.write_text(
            compile_gene_to_mjcf(gene, include_floor=include_floor, spawn_z=spawn_z,
                                  meshes=xml_meshes, show_actuators=True),
            encoding="utf-8",
        )
        mesh_index = {
            "version": "0.1.0",
            "model_uri": Path(model_uri).as_posix(),
            "mesh_scale": 0.001,  # STL coordinates are millimetres; matches the MJCF mesh asset scale.
            "meshes": {
                f"{name}_vis": {
                    "uri": Path(Path(model_uri).parent, asset_dir.name, Path(path).name).as_posix(),
                    "scale": 0.001,
                }
                for name, path in absolute_meshes.items()
            },
        }
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(mesh_index, indent=2), encoding="utf-8")
        # ONLY NOW, and only files a previous run of ours put here. viewer_assets/ is this package's own asset
        # directory (not the shared bake cache) and it accumulated across rebuilds: a second build shipped the
        # FIRST robot's 18 link STLs beside this one's, unreachable through the index and reachable through the
        # directory. Pruning BEFORE the two writes above turned this function's documented fail-open into a
        # broken package: force ``compile_gene_to_mjcf`` to raise and it still returns None, but on disk
        # viewer_mesh_index.json named 18 meshes of which ZERO existed and MjModel.from_xml_path failed outright
        # -- measured, against a package that loaded before the prune was added.
        pruned = prune_staged_dir(asset_dir, {Path(p).name for p in absolute_meshes.values()},
                                  prior=prior_assets)
        return {
            "model_uri": mesh_index["model_uri"],
            "mesh_index_uri": "simulation/viewer_mesh_index.json",
            "mesh_count": len(mesh_index["meshes"]),
            "removed_stale": pruned["removed"],
            "left_not_ours": pruned["kept_foreign"],
            "left_modified": pruned["kept_modified"],
        }
    except Exception:  # noqa: BLE001 - visual fidelity must never make the core simulator unusable
        return None


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


def standing_spawn_z(gene: RobotGene, *, clearance: float | None = None, meshed: bool = True) -> float:
    """Spawn height for a free-base body so its LOWEST point rests ~``clearance`` above the floor — it spawns
    standing on its feet instead of penetrating the floor (legacy fixed 0.1) and getting ejected. ``meshed``
    measures the DISPLAYED visual meshes (what the viewport shows — the mesh can hang below the primitive
    collider, which was the visible foot-penetration bug); pass ``meshed=False`` on hot training paths to
    measure the cheap primitive model instead. Falls back to the legacy height if MuJoCo is unavailable.

    ``meshed=False`` means PRIMITIVE, and it has to keep meaning that. When ``compile_gene_to_mjcf`` learned to
    resolve an imported link's own STL, this branch silently started loading every one of the customer's meshes
    too: measured, the "cheap" path on a Unitree G1 went 0.032 -> 0.159 s and on a Go2 0.024 -> 0.251 s, and the
    two branches returned the same number to the digit because they had become the same model. So it asks for
    ``source_meshes=False`` explicitly rather than relying on a default that has already changed once.

    ``clearance`` defaults to 2 mm for every free body. Visual/collision disagreement is rejected by the
    visual-physics gate instead of being hidden behind an airborne legged-body safety margin."""
    if clearance is None:
        # The former 30 mm legged margin hid foot visual/collider mismatches by spawning robots visibly airborne.
        # The visual-physics CI gate now rejects those mismatches, so every free body starts at its real contact.
        clearance = 0.002
    if gene.base_mount != "free":
        return _base_z_for(gene)
    ref = _MOUNT_Z["free"]                                       # measure the body's downward reach at 0.1
    for build_xml in ((lambda: gene_to_meshed_mjcf(gene, include_floor=False, spawn_z=ref)) if meshed else None,
                      lambda: compile_gene_to_mjcf(gene, include_floor=False, spawn_z=ref,
                                                   source_meshes=False)):
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
    base_z = _base_z_for(gene)
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
        _PHYSICS_OPTION,
        _VISUAL_XML.rstrip("\n"),
        '  <default>',
        '    <joint damping="0.8" armature="0.01" frictionloss="0.05"/>',
        '    <geom friction="1 0.05 0.001"/>',
        '    <default class="visual"><geom mass="0" contype="0" conaffinity="0"/></default>',
        '    <default class="collision"><geom friction="1 0.05 0.001"/></default>',
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
        # B1: filter the robot's self-contacts (ancestor↔descendant) in scene/manipulation runs too — the plain
        # compile path already does this (line ~162), but the scene path omitted it, so a body in a scene solved
        # spurious impulses against its own structure. Robot↔scene-object + floor contacts stay enabled.
        _self_collision_excludes_xml(gene).rstrip("\n"),
        _equality_xml(gene).rstrip("\n"),          # closed loops, in the scene path too — see _equality_xml
        _actuator_xml(gene).rstrip("\n") or "  <actuator></actuator>",
        *( [] if physics_only else [_sensor_xml(gene).rstrip("\n")] ),
        _pose_keyframe(gene, base_z).rstrip("\n"),
        '</mujoco>',
    ]
    return "\n".join(lines) + "\n"


def _is_source_mesh(seg) -> bool:
    """True when this segment is drawn from the geometry the CUSTOMER imported, not from geometry we generated."""
    g = getattr(seg, "geometry", None)
    return isinstance(g, dict) and g.get("family") == "source_mesh"


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
        damping, armature, frictionloss = _joint_dynamics(gene, seg)
        lines.append(
            f'{pad}  <joint name="{escape(seg.name)}_joint" type="{_JOINT_KIND[seg.joint_type]}" '
            f'axis="{ax}"{rng} damping="{damping:.4f}" armature="{armature:.4f}" '
            f'frictionloss="{frictionloss:.4f}"/>'
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
    # TWO-TONE LIMB on the MESHED path. `_detail_geoms_xml` gives a primitive limb a shell-accent fairing over
    # its proximal segment precisely so a leg reads as bodywork over dark structure (the Go2/Spot language) --
    # but that whole block is skipped when the segment has a visual mesh, which is every render and every
    # viewport. So on the path a customer actually looks at, a leg was one undifferentiated dark mass: a
    # charcoal mesh, a charcoal 82 mm motor can, repeated four times. Paint the HIP/abduction housing in shell
    # instead, which is what carries the body colour on a real quadruped. Appearance only -- `seg.material`
    # (which drives density, the BOM and every mass) is untouched, so this cannot move a gram.
    if meshed and _anatomy_role_of(seg) == "quad_hip":
        material = "mat_shell"
    # Same appearance-only treatment for the FOOT. `_ROLE_MATERIAL` maps every foot/paw/hoof to "metal", so the
    # part that meets the ground rendered as a pale specular metal paddle — the one surface on a real legged
    # robot that is never bare metal, because it is the compliant rubber pad that provides traction. The foot's
    # collider is a capsule with `friction="1 0.05 0.001"`, i.e. the sim is already modelling a high-friction
    # contact; drawing it as polished metal contradicted the physics being run. As above this touches only the
    # MJCF material reference, never `seg.material`, so density, the BOM and every mass are untouched.
    elif meshed and _anatomy_role_of(seg) == "foot_pad":
        material = "mat_rubber"
    lines.append(_geom_xml(seg, pad + "  ", material=material, meshed=meshed, physics_only=physics_only))
    if seg.parent is None and not physics_only:
        lines.append(f'{pad}  <site name="imu_site" pos="0 0 {seg.length_m / 2.0:.5f}" size="0.005" rgba="0 0 0 0"/>')
    # physics_only strips ALL visual-only decoration (cylinder motor cans, collars, housings, sensor pucks)
    # so the model is MJX/GPU-safe — those cosmetic cylinders crash mjx.put_model even at contype=0.
    if not meshed and not physics_only:    # the visual mesh already has housings/collars; primitives get them added
        # When real actuators are shown, suppress the generic guessed "motor can" (the real datasheet-sized
        # housing replaces it); the distal collar/flange still helps read the link.
        detail = _detail_geoms_xml(seg, pad + "  ", suppress_motor=show_actuators)
        if detail:
            lines.append(detail)
    # Real off-the-shelf actuator: render the datasheet-sized housing of the part that drives this joint.
    #
    # NOT on a link the CUSTOMER shipped. Their mesh already contains their own motor, gearbox and bearing
    # housings, so drawing our catalog part on top adds a component that is not on their machine — measured on
    # an imported Go2, 13 real link meshes were rendered under 26 boxes and 49 cylinders of ours, which is why
    # the ingested robot read as "a Go2 wearing grey drums" instead of as a Go2. This is the same reasoning that
    # already suppresses ``_detail_geoms_xml`` for any meshed link; the datasheet housing simply never got the
    # same treatment because for a body WE designed it is the truthful thing to draw. Appearance only: every
    # geom involved is mass=0/contype=0/conaffinity=0, and the BOM still cites the actuator it always did.
    # Keyed on ``meshed and`` source-mesh, not on the geometry spec alone: if the customer's STL could not be
    # read this link falls back to a primitive, and a bare primitive with no housing at all is worse than today.
    if (show_actuators and not physics_only and seg.joint_type == "revolute"
            and not (meshed and _is_source_mesh(seg))):
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
    name_l = (seg.name or "").lower()
    if (not physics_only and seg.joint_type in (None, "fixed")
            and any(token in name_l for token in ("foot", "paw", "leg", "hoof"))):
        lines.append(f'{pad}  <site name="{escape(seg.name)}_touch" pos="0 0 {seg.length_m:.5f}" '
                     f'type="sphere" size="{seg.radius_m:.5f}" rgba="0 0 0 0"/>')

    # Children attach at this segment's distal tip (0,0,length), plus any translational mount_offset
    # (lets e.g. two gripper fingers sit side-by-side in y rather than overlapping at the tip).
    for child in gene.children_of(seg.name):
        mo = getattr(child, "mount_offset", (0.0, 0.0, 0.0))
        lines.append(_body_xml(gene, child, pos=(mo[0], mo[1], seg.length_m + mo[2]),
                               indent=indent + 2, meshes=meshes, show_actuators=show_actuators,
                               sensor_geoms=sensor_geoms, physics_only=physics_only))

    lines.append(f"{pad}</body>")
    return "\n".join(lines) + "\n"


def _joint_dynamics(gene: RobotGene, seg) -> tuple[float, float, float]:
    """Conservative identified-dynamics priors selected from structure, not a closed class taxonomy.

    A table/floor-mounted articulated chain has the reflected inertia and transmission friction of an arm;
    rolling joints and free-base load-bearing limbs use their own lighter priors. New named robot classes still
    receive a useful prior because topology, mounting and the joint's physical role drive the choice.

    A MEASUREMENT OUTRANKS A PRIOR. When a system-identification fit has been applied to this gene
    (``sysid.apply_calibration``), the fitted value for this joint replaces the structural guess below --
    which is the whole point of the calibration wedge: the customer runs one bench experiment and the
    simulator stops being a set of plausible constants. Only parameters that passed the identifiability gates
    are ever in that record, the prior each one replaced is stored beside it, and
    ``sysid.revert_calibration`` puts every joint back on this function. Everything else about the emitted
    joint -- axis, range, name -- is untouched.
    """
    prior = _joint_dynamics_prior(gene, seg)
    fitted = _calibrated_dynamics(gene, seg.name)
    if not fitted:
        return prior
    damping, armature, frictionloss = prior
    return (float(fitted.get("damping", damping)), float(fitted.get("armature", armature)),
            float(fitted.get("frictionloss", frictionloss)))


#: The joint-side drivetrain parameters a compiled source model states about the CUSTOMER'S MACHINE, as
#: opposed to about the solver that was integrating it. ``armature`` is deliberately absent -- see
#: :func:`_declared_joint_dynamics`.
DECLARED_DRIVETRAIN_PARAMS = ("damping", "frictionloss")


def _declared_joint_dynamics(gene: RobotGene, seg) -> tuple[float, float] | None:
    """``(damping, frictionloss)`` as the SOURCE FILE declared them, or ``None``. **Not armature.**

    Only imported genes carry ``metadata['source_joint_dynamics']`` (``robot_import`` is its only writer), so
    a composed body is untouched and keeps the structural prior. Defensive for the same reason
    ``_calibrated_dynamics`` is: this runs inside the compile, ``metadata`` is a free-form dict that survives
    a JSON round trip, and a pasted string or a NaN must fall back rather than reach ``dof_damping``.
    Both or neither -- a half-carried drivetrain is a third value that matches neither model.

    WHY ARMATURE IS RECORDED AND NOT CARRIED, which is the one asymmetry here and was measured the hard way.
    Carrying all three moved simulated dynamics globally and broke eight gates that had held; a
    one-parameter-at-a-time ablation across every one of them put ARMATURE alone on the wrong side of all of
    them, and damping and frictionloss on the right side of all of them::

        gate                                      prior     all 3     damping   armature  frictionloss
        pal_talos coupling residual (rad)         0.00338   0.15944   0.00332   0.13119   0.00285
        toddlerbot coupling residual (rad)        0.02373   1.52841   0.01370   1.88989   0.02371
        cassie loop-closure gap / default          0.495     0.672     0.496     0.720     0.492
        boston_dynamics_spot holds home pose       True      False     True      True      True

    Two independent reasons, and either alone is sufficient:

    * **A compiled ``dof_armature`` of 0 is not a declaration.** MuJoCo's default is 0 and a compiled model
      keeps no record of which attributes the XML wrote, so 0 means "declared zero" OR "never mentioned" and
      nothing can tell them apart. On 14 of the 59 cached Menagerie packages EVERY joint reads 0 --
      anymal_b/c, spot, talos, kinova_gen3, kuka_iiwa_14, tiago, tiago_dual, stretch, leap_hand, tidybot,
      wxai, z1, allegro. Real machines with real harmonic drives and real rotor inertia, all of it un-set.
      Adopting that 0 as the customer's number is a DEFAULT OF MUJOCO'S applied where the source said nothing,
      which is the same class of over-claim as the substitution this whole carry-through exists to remove.
    * **Armature is referenced to a model, not to a machine.** It is added to the diagonal of ``qM``, so what
      it buys is conditioning *relative to the rest of that model* -- and our twin is not that model. We emit
      our own ``<equality>`` couplings and loop closures at the source's own tight ``solref`` (Cassie's
      ``0.005 1`` is 2.5 timesteps), driven by ``<motor>`` actuators under our PD rather than the source's
      ``<position>``, over primitive link inertias. On Cassie the source's armature is 5% of joint inertia
      where our prior is 20%; taking the 5% removes the margin those constraints are solved with. That is a
      number the twin cannot inherit, whatever the source declares.

    Damping and frictionloss carry, because both ARE statements about the hardware: 2.0 N.m.s/rad of joint
    damping is a property of the Go2's drivetrain, and the 0.45 N.m of Coulomb friction we used to invent on a
    Panda joint whose author declared none was a fiction. ``robot_import`` discloses armature as recorded and
    not carried, with our number beside theirs.
    """
    import math

    meta = getattr(gene, "metadata", None)
    if not isinstance(meta, dict):
        return None
    # THE RECORD IS KEPT EVEN WHEN IT IS NOT CARRIED. ``robot_import`` clears this flag for the one case where
    # the customer's declaration makes OUR twin unsteppable (their drivetrain is declared against their link
    # inertias; ours are primitives) -- it falls back to the prior, says so in full, and keeps their numbers
    # verbatim so nothing is lost. Reading the table without reading the flag would re-apply exactly the
    # values that diverge. Absent flag means carried, so a gene serialized before this existed is unaffected.
    if meta.get("source_joint_dynamics_carried") is False:
        return None
    # ``isinstance`` on the CONTAINER too, not only on the row: a metadata dict that round-tripped through
    # JSON as a list (or anything else without ``.get``) would otherwise raise inside the compile, which is
    # the one thing this function exists to prevent.
    table = meta.get("source_joint_dynamics")
    if not isinstance(table, dict):
        return None
    row = table.get(getattr(seg, "name", "") or "")
    if not isinstance(row, dict):
        return None
    out = []
    for param in DECLARED_DRIVETRAIN_PARAMS:
        v = row.get(param)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if not math.isfinite(v) or v < 0.0:
            return None
        out.append(v)
    # A record missing ``armature`` is still MALFORMED and still falls back whole: ``robot_import`` writes all
    # three or none, so a row without it did not come from an import we understand.
    a = row.get("armature")
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(float(a)) or float(a) < 0.0:
        return None
    return (out[0], out[1])


def _calibrated_dynamics(gene: RobotGene, seg_name: str) -> dict:
    """``{param: fitted_value}`` from an applied sysid calibration, or ``{}``.

    Imported lazily and guarded: ``sysid`` needs MuJoCo, and ``gene_compiler`` is imported by paths that run
    without it. A compile must never fail because the calibration package could not load -- it falls back to
    the structural prior, which is the pre-calibration behaviour. The key check comes BEFORE the import so an
    uncalibrated gene -- which is every gene, on every compile, on every path -- does not reach across into
    another package once per joint to be told nothing changed.
    """
    meta = getattr(gene, "metadata", None)
    if not isinstance(meta, dict) or "calibration" not in meta:
        return {}
    try:
        from virturoid.services.sysid.calibration import calibrated_joint_dynamics
        return calibrated_joint_dynamics(gene, seg_name)
    except Exception:  # noqa: BLE001
        return {}


def _joint_dynamics_prior(gene: RobotGene, seg) -> tuple[float, float, float]:
    """The baseline before any BENCH MEASUREMENT. Split out so a calibrated build can still recompute the
    baseline it replaced (``calibration_report`` uses exactly this to flag a stale record).

    A NUMBER THE CUSTOMER DECLARED IS NOT A PRIOR OF OURS, so it wins -- for the two parameters where the
    source is stating something about the MACHINE. ``robot_import`` reads damping/frictionloss off the
    customer's own compiled model and records them on the gene; before this they were recorded and then
    ignored here, and the structural guess below stood in. Measured on a real Menagerie Go2, which declares
    ``damping=2.0 frictionloss=0.2`` on all 12 leg joints, the emitted model carried 0.8 (-60%) and 0.12
    (-40%); on a Panda (``damping=1.0``, no dry friction at all) 2.0 and 0.45 -- i.e. we invented 0.45 N.m of
    Coulomb friction on a joint whose author declared none.

    ARMATURE IS THE EXCEPTION AND STAYS OURS: it is a solver-conditioning term referenced to the model it was
    declared in, and 24% of the corpus leaves it at MuJoCo's 0 so the record cannot even say whether it was
    declared. See :func:`_declared_joint_dynamics` for the ablation that put armature alone on the wrong side
    of every gate that moved. The overlay is written as prior-then-override rather than an early return so
    that this stays true by construction: whatever the record says, ``armature`` comes from below.

    Why it belongs HERE and not one level up in ``_joint_dynamics``: ``sysid.fit`` reads each parameter's
    ``from:`` straight off the compiled model's ``dof_*`` arrays, and ``sysid.calibration.calibration_report``
    re-runs THIS function to decide whether a stored fit has gone stale. Putting the declared value anywhere
    else makes the fit quote its correction against a baseline the staleness check cannot reproduce, and every
    imported joint reads as stale. With it here, all three surfaces agree by construction -- and a fit stops
    "identifying" a substitution of ours and starts measuring the customer's hardware.

    The record carries what the customer's own MuJoCo integrates AT THE JOINT, so an undeclared damping
    arrives as the 0.0 their file leaves there. That zero is carried: an invented 0.45 N.m of stiction is not
    a safer error than none, it is just a less visible one. It is also not the whole story on 18 of 59
    packages, which put their velocity feedback in a ``<position kv=...>`` actuator our ``<motor>`` emitter
    has nowhere to put -- ``robot_import`` names that hole rather than letting "0" imply an undamped machine.
    """
    damping_o, friction_o = _declared_joint_dynamics(gene, seg) or (None, None)

    def _out(damping: float, armature: float, frictionloss: float) -> tuple[float, float, float]:
        return (damping if damping_o is None else damping_o, armature,
                frictionloss if friction_o is None else friction_o)

    name = (seg.name or "").lower()
    # The mount that decides the DRIVETRAIN prior, which is not always the mount the body is compiled with: a
    # sysid bench rig welds a free-base robot to a stand to isolate its actuators, and that weld must not
    # convert every leg joint into an industrial arm axis. `sysid.bench_rig.bench_model` is the only writer.
    mount = (getattr(gene, "metadata", None) or {}).get("joint_dynamics_base_mount") or gene.base_mount
    if seg.joint_type == "prismatic":
        return _out(1.2, 0.02, 0.12)
    if _segment_role(seg) == "wheel" or "wheel" in name or "drive" in name:
        return _out(0.25, 0.02, 0.05)
    if mount in ("table", "floor", "torso"):
        # Scale reflected inertia/friction with the selected actuator instead of
        # assigning a shoulder-sized gearbox to every wrist. The fixed 0.45 Nm
        # Coulomb loss left a grounded 0.52 Nm wrist only 0.07 Nm to move, so the
        # arm could never reach its physically valid grasp pose.
        capacity = max(0.1, abs(float(getattr(seg, "actuator_torque_nm", 0.0) or 0.0)))
        # A fixed-base industrial/manipulator axis carries gearbox and motor
        # inertia even when its output-torque rating is small. Keep the
        # identified manipulator floor for reflected inertia/damping; only
        # Coulomb friction scales down for a light wrist so it can still move.
        damping = max(1.0, min(2.0, 0.2 + 0.08 * capacity))
        armature = max(0.1, min(0.14, 0.003 * capacity))
        friction = max(0.01, min(0.45, 0.02 * capacity))
        return _out(damping, armature, friction)
    if any(token in name for token in ("leg", "hip", "knee", "ankle", "thigh", "shin", "calf")):
        # Preserve the tuned quadruped's measured reflected inertia anchor. A
        # 0.04 armature shifted the closed-loop damping ratio by 37% and broke
        # policy/control parity; the arm/cobot branch retains its identified 0.14.
        return _out(0.8, 0.01, 0.12)
    return _out(0.6, 0.03, 0.08)


def _geom_xml(seg, pad: str, material: str = "mat_body", meshed: bool = False, physics_only: bool = False) -> str:
    name = f'{escape(seg.name)}_geom'
    # When meshed, the primitive becomes collision-only: invisible (alpha 0) + group 3, but its shape/size/
    # mass are untouched, so dynamics & contacts stay byte-identical to the primitive model. The visible
    # surface is the detailed mesh appended below.
    contact = ""
    name_l = (seg.name or "").lower()
    if seg.joint_type in (None, "fixed") and any(token in name_l for token in ("foot", "paw", "leg", "hoof")):
        contact = ' condim="6" priority="1" solimp="0.9 0.95 0.001"'
    surf = (' class="collision" rgba="0 0 0 0" group="3"' if meshed
            else f' class="collision" material="{material}"')
    surf += contact
    if seg.shape == "sphere":
        coll = (f'{pad}<geom name="{name}" type="sphere" pos="0 0 {seg.radius_m:.5f}" '
                f'size="{seg.radius_m:.5f}" mass="{seg.mass_kg:.5f}"{surf}/>')
    elif seg.shape == "box":
        h = seg.length_m / 2.0
        cs = getattr(seg, "cross_section", None)            # laterally-compressed (fish) body if set, else square
        hx, hy = (float(cs[0]), float(cs[1])) if cs else (seg.radius_m, seg.radius_m)
        coll = (f'{pad}<geom name="{name}" type="box" pos="0 0 {h:.5f}" '
                f'size="{hx:.5f} {hy:.5f} {h:.5f}" mass="{seg.mass_kg:.5f}"{surf}/>')
    else:
        # MJX has no cylinder collision; in physics_only mode a cylinder COLLIDER becomes a capsule (the closest
        # MJX-safe round primitive) so a cylinder-shaped link can still train on GPU.
        gtype = "cylinder" if (seg.shape == "cylinder" and not physics_only) else "capsule"
        # A wheel's mount offset denotes its axle CENTER; ordinary links attach at their proximal end. The old
        # shared [0,L] convention shifted both mirrored wheels in the same local direction, leaving one side
        # visibly detached and the other buried in the chassis.
        fromto = (f'0 0 {-seg.length_m / 2.0:.5f} 0 0 {seg.length_m / 2.0:.5f}'
                  if _segment_role(seg) == "wheel" else f'0 0 0 0 0 {seg.length_m:.5f}')
        coll = (f'{pad}<geom name="{name}" type="{gtype}" fromto="{fromto}" '
                f'size="{seg.radius_m:.5f}" mass="{seg.mass_kg:.5f}"{surf}/>')
    if not meshed:
        return coll
    vis = (f'{pad}<geom class="visual" name="{escape(seg.name)}_vis" type="mesh" mesh="{escape(seg.name)}_vis" '
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
    vis = ' class="visual" mass="0" contype="0" conaffinity="0"'
    parts: list[str] = []
    name_l = (seg.name or "").lower()
    is_limb = any(k in name_l for k in _LIMB_HINT)
    welded_limb_tip = is_limb and (seg.joint_type or "") not in ("revolute", "prismatic")
    if seg.shape == "cylinder":
        if _segment_role(seg) == "wheel":
            # A centered axle hub extends past the tire sidewalls and visually mates the wheel to its chassis.
            # It stays above the contact patch, so the tire remains the only contact-visible surface.
            half = L / 2.0 + min(0.025, R * 0.3)
            return (f'{pad}<geom name="{escape(seg.name)}_hub" type="cylinder" '
                    f'fromto="0 0 {-half:.5f} 0 0 {half:.5f}" size="{R * 0.42:.5f}" '
                    f'material="mat_joint"{vis}/>')
        return ""
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
    if (seg.shape in ("capsule", "cylinder") and L > 0.06 and seg.joint_type != "prismatic"
            and not welded_limb_tip):
        t = min(0.012, L * 0.10)
        parts.append(
            f'{pad}<geom name="{escape(seg.name)}_collar" type="cylinder" '
            f'fromto="0 0 {L - 2 * t:.5f} 0 0 {L:.5f}" size="{R:.5f}" material="mat_joint"{vis}/>'
        )
    # R2 FAIRING + BOOT (visual-only): give a walking limb bodywork so it reads as a DESIGNED limb, not a bare
    # capsule chain (the render's remaining toy tell after styling + proportions). The PROXIMAL segments get a
    # shell-accent fairing sleeve over the mid-span (clear of the joint cans); the welded terminal segment (the
    # foot) gets a rubber boot at its contact tip. Distal structural segments stay bare dark, so the limb reads
    # two-tone — accent bodywork over dark structure, the Go2/Spot design language validated in the A/B/C study.
    if is_limb:
        welded = (seg.joint_type or "") not in ("revolute", "prismatic")
        tail = name_l.rsplit("_", 1)
        idx = int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0
        if welded and 0.02 < L:                        # terminal welded segment == foot -> rubber boot pad
            # Center the boot UP the foot so its distal surface lands exactly on the segment's own COLLISION
            # extent — a pad hanging below the contact surface reads as a spawn penetration (the standing-spawn
            # test measures every geom's oriented AABB) and makes the robot look like it is sinking into the floor.
            # That extent is shape-dependent: a capsule's distal cap is at L + R, but a BOX or cylinder foot ends
            # flat at L. This assumed a capsule for every foot, and the composer emits BOX feet — so every walking
            # body carried a pad protruding R (38 mm on a quadruped) past the surface it actually stands on. It
            # went unseen because the foot's VISUAL mesh used to be an oversized brick that reached even further,
            # so standing_spawn_z (which measures the meshed model) lifted the body enough to cover it. Fixing the
            # foot mesh removed that accidental cover and exposed this.
            distal = L + R if seg.shape == "capsule" else L
            boot_r = max(0.004, min(R, 0.45 * L))
            boot_z = distal - boot_r
            parts.append(
                f'{pad}<geom name="{escape(seg.name)}_boot" type="sphere" pos="0 0 {boot_z:.5f}" '
                f'size="{boot_r:.5f}" material="mat_rubber"{vis}/>')
        elif seg.shape in ("capsule", "cylinder", "box") and L > 0.05 and idx == 0:   # top of limb -> shell fairing
            # Only the TOP (hip/thigh) segment gets the accent bodywork, so the segments below stay bare dark and
            # the limb reads two-tone (accent over structure) regardless of how many segments the leg has —
            # fairing every proximal segment turned a 3-segment leg all-accent, losing the structure read.
            p0, p1 = 0.18 * L, 0.78 * L
            parts.append(
                f'{pad}<geom name="{escape(seg.name)}_fairing" type="capsule" '
                f'fromto="0 0 {p0:.5f} 0 0 {p1:.5f}" size="{R * 1.4:.5f}" material="mat_shell"{vis}/>')
    return "\n".join(parts)


def _segment_role(seg) -> str:
    geometry = getattr(seg, "geometry", None)
    return str(geometry.get("semantic_role") or "").lower() if isinstance(geometry, dict) else ""


def _anatomy_role_of(seg) -> str:
    """The detailed-solid role this segment is BUILT from, by either vocabulary: the hard-coded composer
    recipes emit ``family="role"``, the general anatomy compiler stamps ``anatomy_role``."""
    geometry = getattr(seg, "geometry", None)
    if not isinstance(geometry, dict):
        return ""
    if str(geometry.get("family") or "").lower() == "role":
        return str(geometry.get("role") or "").lower()
    return str(geometry.get("anatomy_role") or "").lower()


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
            f'gear="1" ctrllimited="true" ctrlrange="{-effort:.2f} {effort:.2f}" '
            f'forcerange="{-effort:.2f} {effort:.2f}"/>'
        )
    lines.append("  </actuator>")
    return "\n".join(lines) + "\n"


def _sensor_xml(gene: RobotGene) -> str:
    """Shipped proprioception: IMU, joint state and contact sensors over sites emitted by ``_body_xml``."""
    lines = ['  <sensor>',
             '    <accelerometer name="imu_accel" site="imu_site"/>',
             '    <gyro name="imu_gyro" site="imu_site"/>']
    for seg in gene.actuated_joints():
        name = escape(seg.name)
        lines.append(f'    <jointpos name="{name}_position" joint="{name}_joint"/>')
        lines.append(f'    <jointvel name="{name}_velocity" joint="{name}_joint"/>')
    for seg in gene.segments:
        name_l = (seg.name or "").lower()
        if seg.joint_type in (None, "fixed") and any(token in name_l for token in ("foot", "paw", "leg", "hoof")):
            name = escape(seg.name)
            lines.append(f'    <touch name="{name}_contact" site="{name}_touch"/>')
    lines.append('  </sensor>')
    return "\n".join(lines) + "\n"
