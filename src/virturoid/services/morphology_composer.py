"""Agent-driven morphology composition: requirements/prompt -> a block RECIPE -> a validated robot.

This is the seam where an agent BUILDS a robot from blocks rather than picking a hand-coded species.
A "recipe" is a small constrained spec (base + chained links + symmetric limbs + an end-effector) that
either a deterministic heuristic (from the parsed requirements — reach, payload, mobility, grasp) or an
LLM emits over the SAME menu; ``compose_from_spec`` realizes it through ``MorphologyBuilder`` and the
gene's own ``validate()`` rejects anything unbuildable ("proposer proposes, schema/physics disposes").
The result auto-places into the vector-space species tree (``species_discovery``), so composing a novel
robot also discovers a new species — exactly the LEGO + self-organizing-tree premise.
"""

from __future__ import annotations

import math

from virturoid.schemas.gene import RobotGene
from virturoid.services.morphology_builder import MorphologyBuilder

_GRASP_WORDS = ("grasp", "grip", "pick", "sort", "manipulat", "lift", "place", "assemble")
_MOBILE_WORDS = ("mobile", "drive", "navigate", "deliver", "rover", "wheel", "transport", "patrol")
_LEGGED_WORDS = ("quadruped", "legged", "four legs", "four-legged", "walk on legs", "walking robot",
                 "dog robot", "hexapod", "octopod", "six legs", "six-legged", "eight legs", "eight-legged",
                 "spider", "arachnid", "tarantula", "insect", "ant ", "crawler", "crab", "crustacean",
                 "mantis", "gecko", "lizard", "salamander", "newt", "scorpion", "centipede", "beetle",
                 "turtle", "frog", "snake", "serpent", "creature", "animal", "quadrupedal")
_HUMANOID_WORDS = ("humanoid", "biped", "bipedal", "two-legged", "two legged", "human-like", "human like",
                   "android", "person-shaped")
_SPRAY_WORDS = ("spray", "paint", "weld", "dispense", "coat", "coating", "varnish", "lacquer")
_DEXTEROUS_WORDS = ("dexterous", "delicate", "fragile", "multi-finger", "multifinger", "irregular",
                    "in-hand", "envelop", "round object", "ball", "fruit")
_PRECISION_WORDS = ("precise", "precision", "orientation", "6-dof", "7-dof", "six-axis", "seven",
                    "redundant", "industrial arm", "panda", "franka", "ur5", "cobot", "around obstacles")

# Anthropomorphic joint axes per DOF: base yaw, shoulder/elbow pitch, then a wrist (roll/pitch/yaw) for
# orientation control — the standard serial-arm layout (z=yaw/roll about the link, y=pitch).
_ARM_AXES = {
    2: [(0, 0, 1), (0, 1, 0)],
    3: [(0, 0, 1), (0, 1, 0), (0, 1, 0)],
    6: [(0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1)],
    7: [(0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 1, 0)],
}


def _role(role: str, length: float, radius: float) -> dict:
    """A ``geometry`` spec that asks the CAD/mesh layer for detailed, role-specific ANATOMY
    (``cad_geometry.build_anatomy``) instead of a bare primitive — e.g. a tapered limb with a motor
    housing, a contoured torso shell, a domed sensor head. VISUAL/CAD only: the MuJoCo physics model reads
    shape/length/radius/mass and ignores ``geometry``, so attaching this never changes dynamics."""
    return {"family": "role", "role": role, "length": round(length, 4), "radius": round(radius, 4)}


def _mech_beam(length: float, radius: float) -> dict:
    """A chamfered MECHANICAL link beam (a slightly-flattened machined box with bevelled edges, like an
    industrial / Franka-class arm link), built in the link's [0, length] +z frame. Replaces the smooth tapered
    shaft so a manipulator's links match the mechanical aesthetic of the legged/humanoid limbs (the real
    datasheet motor housing is still rendered at each joint by show_actuators). VISUAL-ONLY — the physics model
    reads only shape/length/radius/mass, so the arm's dynamics are byte-identical."""
    depth = max(0.01, radius)
    width = max(0.008, 0.84 * radius)
    return {"family": "extrude", "height": round(length, 4), "fillet": round(0.28 * width, 4),
            "chamfer": round(0.42 * width, 5),
            "profile": [[-depth, -width], [0.9 * depth, -width], [0.9 * depth, width], [-depth, width]]}


def _arm_links(reach: float, dof: int, torque: float) -> list[dict]:
    """Anthropomorphic serial-arm link specs for ``dof`` joints. The PITCH joints (axis~y) extend the
    arm and share the reach; the YAW/ROLL joints (axis~z) are short orientation joints — so a 6-7 DOF
    arm has a real reachable workspace + wrist orientation, like an industrial/Franka-class arm."""
    axes = _ARM_AXES.get(dof, _ARM_AXES[3])
    pitch = [i for i, a in enumerate(axes) if abs(a[1]) > 0.5] or [0]
    seg_len = round(reach / len(pitch), 3)
    links = []
    for i, a in enumerate(axes):
        is_pitch = abs(a[1]) > 0.5
        seg_len_i = seg_len if is_pitch else 0.05        # orientation joints are short
        rad_i = 0.035 if i == 0 else 0.03
        # Torque decays down the chain (less downstream load) but is FLOORED at 45% of the base: an
        # unfloored decay starved the distal joints so a high-DOF arm couldn't hold its folded reaching
        # pose (CPU diagnosis: mean joint-target error stayed ~0.16 rad regardless of gain). Floor it.
        links.append({
            "name": "shoulder" if i == 0 else f"j{i}",
            "length": seg_len_i,
            "joint": "revolute", "axis": a, "radius": rad_i,
            "mass": 0.4 if i == 0 else 0.3,
            "torque": torque if i == 0 else round(torque * max(0.85 ** i, 0.45), 1),
            # chamfered mechanical link beam (matches the legged/humanoid limbs); joint motors via show_actuators
            "geometry": _mech_beam(seg_len_i, rad_i)})
    return links


def _grasp_arm_links(reach: float, torque: float) -> list[dict]:
    """Serial grasp-arm links: a yaw base + two reaching PITCH links + a short wrist PITCH.

    The wrist decouples the gripper's approach ORIENTATION from the arm's reach, so the palm can be
    levelled to point straight DOWN for a top-down grasp. Findings §12: a wrist-less yaw+2-pitch arm
    reaches a tabletop object with the palm tilted ~26 deg and the closing jaws sweep it away — a
    structural (underactuation) failure, not a tuning one. Two pitch links share the reach; the wrist
    is short (orientation only). Use the ``grasp_pose`` orientation IK to drive the vertical approach.
    """
    seg = round(reach / 2.0, 3)
    return [
        {"name": "shoulder", "length": 0.05, "axis": (0, 0, 1), "radius": 0.035, "mass": 0.4,
         "torque": torque, "geometry": _mech_beam(0.05, 0.035)},
        {"name": "j1", "length": seg, "axis": (0, 1, 0), "radius": 0.03, "mass": 0.35,
         "torque": round(torque * 0.9, 1), "geometry": _mech_beam(seg, 0.03)},
        {"name": "j2", "length": seg, "axis": (0, 1, 0), "radius": 0.03, "mass": 0.3,
         "torque": round(torque * 0.75, 1), "geometry": _mech_beam(seg, 0.03)},
        {"name": "wrist", "length": 0.05, "axis": (0, 1, 0), "radius": 0.025, "mass": 0.2,
         "torque": round(max(torque * 0.6, 8.0), 1), "lower": -6.28, "upper": 6.28,
         "geometry": _mech_beam(0.05, 0.025)},
    ]


def compose_from_spec(spec: dict) -> RobotGene:
    """Realize a block recipe into a validated ``RobotGene`` via the builder. Raises if unbuildable."""
    b = MorphologyBuilder(spec.get("robot_class", "manipulator"),
                          base_mount=spec.get("base_mount", "table"),
                          species=spec.get("species"), gene_id=spec.get("gene_id"))
    base = spec.get("base") or {}
    b.base(base.get("name", "base"), shape=base.get("shape", "box"),
           length=float(base.get("length", 0.1)), radius=float(base.get("radius", 0.04)),
           mass=float(base.get("mass", 1.0)), geometry=base.get("geometry"))
    for ln in spec.get("links", []):
        b.link(ln["name"], float(ln["length"]), radius=float(ln.get("radius", 0.03)),
               mass=float(ln.get("mass", 0.3)), joint=ln.get("joint", "revolute"),
               axis=tuple(ln.get("axis", (0.0, 0.0, 1.0))), lower=ln.get("lower", -3.14),
               upper=ln.get("upper", 3.14), torque=float(ln.get("torque", 10.0)),
               geometry=ln.get("geometry"))
    for limb in spec.get("limbs", []):
        b.limb(limb["prefix"], [dict(li) for li in limb["links"]],
               parent=limb["parent"], mount_offset=tuple(limb.get("mount_offset", (0.0, 0.0, 0.0))),
               mount_euler=tuple(limb.get("mount_euler", (0.0, 0.0, 0.0))))
    arm_tip = b._cursor if spec.get("links") else None   # composite (links+wheels): the ee belongs on the ARM tip
    for w in spec.get("wheels", []):
        b.wheel(w["name"], radius=float(w.get("radius", 0.06)), thickness=float(w.get("thickness", 0.04)),
                parent=w["parent"], mount_offset=tuple(w.get("mount_offset", (0.0, 0.0, 0.0))),
                torque=float(w.get("torque", 6.0)))
    if arm_tip is not None and spec.get("wheels"):
        b._cursor = arm_tip                                # wheels moved the cursor; restore so gripper() lands on the arm
    ee = spec.get("end_effector") or {}
    kind = ee.get("kind", "none")
    # Guard: if the design already builds articulated fingers as limbs, an extra hand/gripper end-effector is
    # redundant — the builder welds it to the current chain tip (a fingertip), producing stray jaw stubs. The
    # articulated limbs ARE the hand, so drop the duplicate end-effector.
    if kind in ("hand", "gripper") and any("finger" in (lm.get("prefix") or "") for lm in spec.get("limbs", [])):
        kind = "none"
    if kind == "gripper":
        b.gripper(span=float(ee.get("span", 0.10)), finger_len=float(ee.get("finger_len", 0.06)))
    elif kind == "hand":
        b.hand(n_fingers=int(ee.get("fingers", 3)), span=float(ee.get("span", 0.10)))
    elif kind in ("suction", "spray", "hook"):
        b.end_effector(kind)
    gene = b.build()
    # Plumb a baked render/spawn stance (joint_name -> angle) from the spec into gene.metadata, which
    # gene_compiler._pose_keyframe consumes to emit a <keyframe>. Render/spawn-only: no mass/geom/actuator
    # change (standing_spawn_z resets to the keyframe), so a quadruped renders in a believable bent-leg crouch
    # instead of standing on four straight-down posts.
    if spec.get("rest_pose"):
        gene.metadata = {**(getattr(gene, "metadata", None) or {}), "rest_pose": dict(spec["rest_pose"])}
    return gene


def _leg_count(prompt: str) -> int:
    """How many legs the user asked for — a PARAMETER of the legged body, not a per-species template.
    Defaults to 4 (quadruped). hexapod=6, octopod=8, or '6-legged'/'six legs' etc."""
    import re

    p = (prompt or "").lower()
    if any(w in p for w in ("spider", "arachnid", "octopod", "octopus", "eight-legged", "eight legs")):
        return 8                                          # arachnids/octopods have 8 legs
    if any(w in p for w in ("hexapod", "insect", "ant ", "six-legged", "six legs")):
        return 6                                          # insects/hexapods have 6
    for word, k in (("two", 2), ("four", 4), ("six", 6), ("eight", 8)):
        if f"{word}-leg" in p or f"{word} leg" in p:
            return k
    m = re.search(r"(\d+)\s*-?\s*legs?", p)
    if m and int(m.group(1)) in (2, 4, 6, 8):
        return int(m.group(1))
    return 4


def morphology_from_requirements(reach_m: float, payload_kg: float, *, prompt: str = "",
                                 environment: str | None = None, robot_class: str | None = None,
                                 n_legs: int | None = None, scale_m: float | None = None,
                                 nominal_dims: dict | None = None) -> dict:
    """Deterministic block recipe from task requirements — the no-LLM 'build any robot' path.

    reach -> number + length of arm links; payload+reach -> actuator torque (static moment + margin);
    prompt -> end-effector (gripper/spray) and whether it is a mobile base instead of an arm.

    ``robot_class`` (from the LLM build planner, §31.1) overrides the keyword class decision when given —
    so an off-vocabulary prompt ("a frog-like robot…") still routes to the right body (nearest buildable)
    instead of defaulting to an arm. Unknown/unsupported classes (e.g. humanoid) fall back to manipulator.
    """
    p = (prompt or "").lower()
    import math as _math
    if robot_class is None:                               # keyword fallback (legacy path)
        if any(w in p for w in _HUMANOID_WORDS):
            robot_class = "humanoid"
        elif any(w in p for w in _LEGGED_WORDS):
            robot_class = "quadruped"
        elif any(w in p for w in _MOBILE_WORDS) and any(w in p for w in _GRASP_WORDS):
            robot_class = "mobile_manipulator"            # G4: 'mobile ... picks/carries' = base + arm, not a fixed arm
        elif any(w in p for w in _MOBILE_WORDS):
            robot_class = "mobile_base"
        else:
            robot_class = "manipulator"
    if robot_class in ("humanoid", "biped"):              # torso + neck/head + arms & legs (hands/feet welded)
        # REALISTIC, CONNECTED humanoid proportions (modelled on a real adult-scale biped): long slender
        # limbs, a tall torso, a neck. 2 ACTUATED joints per limb (hip+knee / shoulder+elbow) = 8 DOF
        # (trainable); neck/head/hands/feet are welded so they add anatomy WITHOUT extra DOF. Limbs mount
        # FLUSH to the torso surface (no floating gap). Semantic limb roles (thigh/shin/upper_arm/forearm)
        # drive role anatomy AND key the kit-bash catalog to a matching real Menagerie link mesh.
        T_LEN, T_R = 0.46, 0.13                           # torso length (z) + half-width; arms mount at ~T_R
        # CHUNKY limbs (real humanoid links are thick structural members + fat actuators, not spindly rods):
        # bigger radii + slightly shorter legs so the leg:torso ratio reads human, not stretched-out.
        leg = [{"length": 0.24, "axis": (0, 1, 0), "torque": 28.0, "radius": 0.075, "mass": 1.5,
                "geometry": _role("thigh", 0.24, 0.075)},                                   # thigh
               {"length": 0.24, "axis": (0, 1, 0), "torque": 22.0, "radius": 0.062, "mass": 1.0,
                "geometry": _role("shin", 0.24, 0.062)},                                    # shin
               {"length": 0.16, "axis": (0, 1, 0), "torque": 9.0, "radius": 0.06, "mass": 0.5,
                "joint": "fixed", "shape": "box", "geometry": _role("foot_pad", 0.16, 0.06)}]  # foot
        arm = [{"length": 0.24, "axis": (0, 1, 0), "torque": 14.0, "radius": 0.055, "mass": 0.8,
                "geometry": _role("upper_arm", 0.24, 0.055)},                               # upper arm
               {"length": 0.22, "axis": (0, 1, 0), "torque": 9.0, "radius": 0.046, "mass": 0.6,
                "geometry": _role("forearm", 0.22, 0.046)},                                 # forearm
               {"length": 0.10, "axis": (0, 1, 0), "torque": 5.0, "radius": 0.05, "mass": 0.35,
                "joint": "fixed", "shape": "box", "geometry": _role("hand_plate", 0.10, 0.05)}]  # hand
        return {"robot_class": "humanoid", "base_mount": "free", "species": "humanoid.composed",
                "base": {"name": "torso", "shape": "box", "length": T_LEN, "radius": T_R, "mass": 8.0,
                         "geometry": _role("torso_shell", T_LEN, T_R)},
                # ONE head link mounted directly on the torso top — no separate thin neck segment. The old
                # recipe stacked a thin neck capsule UNDER the head, and the sensor_head mesh ALSO carries its
                # own neck stub, so the two necks read as a desk-lamp stalk. The head mesh's integrated neck
                # stub is the only neck now -> a head-on-shoulders, not a pole.
                "links": [{"name": "head", "length": 0.13, "radius": 0.085, "joint": "fixed", "mass": 0.9,
                           "geometry": _role("sensor_head", 0.13, 0.085)}],
                # mount_offset is relative to the torso TOP tip (z=T_LEN): legs at the bottom (z≈0), arms at
                # the shoulders (z≈T_LEN-0.06). y=±T_R*0.85 keeps each limb root ON the torso surface (the old
                # y=0.12 vs half-width 0.10 floated the arms 2 cm off the body). euler π makes limbs hang DOWN.
                "limbs": [
                    {"prefix": "left_leg", "parent": "torso", "mount_offset": (0.0, T_R * 0.5, -T_LEN),
                     "mount_euler": (_math.pi, 0.0, 0.0), "links": [dict(s) for s in leg]},
                    {"prefix": "right_leg", "parent": "torso", "mount_offset": (0.0, -T_R * 0.5, -T_LEN),
                     "mount_euler": (_math.pi, 0.0, 0.0), "links": [dict(s) for s in leg]},
                    {"prefix": "left_arm", "parent": "torso", "mount_offset": (0.0, T_R * 0.85, -0.06),
                     "mount_euler": (_math.pi, 0.0, 0.0), "links": [dict(s) for s in arm]},
                    {"prefix": "right_arm", "parent": "torso", "mount_offset": (0.0, -T_R * 0.85, -0.06),
                     "mount_euler": (_math.pi, 0.0, 0.0), "links": [dict(s) for s in arm]},
                ],
                "end_effector": {"kind": "none"}}
    if robot_class in ("quadruped", "legged"):            # FREE-floating legged body: N legs (thigh+shin+foot)
        n = int(n_legs) if n_legs else _leg_count(p)      # 4 by default; hexapod=6, octopod=8, … (a parameter)
        per_side = max(2, n // 2); n = per_side * 2
        # Uniform SIZE scale: honor scale_m (animal body length) or nominal_dims['leg_mm'] so a tiny ant and a
        # big pack-mule differ (scale_m used to be a no-op here). s=1.0 by DEFAULT -> every dimension below is
        # byte-identical to the gait-tuned Go1-class baseline that the locomotion tests pin. Clamped so a
        # hallucinated value can't build a degenerate body; masses ~ s^3 (volume), torques ~ s^2 (a rough size
        # law — ground_gene + the validation gate still re-pick/flag real actuators afterward).
        s = 1.0
        if nominal_dims and nominal_dims.get("leg_mm"):
            s = max(0.5, min(2.5, float(nominal_dims["leg_mm"]) / 0.24))
        elif scale_m:
            s = max(0.5, min(2.5, float(scale_m) / 0.45))
        sm, st = s ** 3, s ** 2
        # Real-quadruped leg topology (matches a Go1-class baseline): 3 actuated DOF per leg —
        # hip ABDUCTION (roll about body x) + hip PITCH (thigh) + KNEE (shin) — then a welded foot.
        qleg = [{"length": 0.025 * s, "axis": (1, 0, 0), "torque": round(14.0 * st, 2), "radius": 0.032 * s,
                 "mass": 0.15 * sm, "lower": -0.7, "upper": 0.7,
                 "geometry": _role("quad_hip", 0.025 * s, 0.032 * s)},                       # hip abduction (roll)
                {"length": 0.12 * s, "axis": (0, 1, 0), "torque": round(16.0 * st, 2), "radius": 0.035 * s,
                 "mass": 0.3 * sm, "geometry": _role("quad_thigh", 0.12 * s, 0.035 * s)},     # thigh
                {"length": 0.12 * s, "axis": (0, 1, 0), "torque": round(12.0 * st, 2), "radius": 0.030 * s,
                 "mass": 0.3 * sm, "geometry": _role("quad_calf", 0.12 * s, 0.030 * s)},      # calf (shin)
                {"length": 0.05 * s, "axis": (0, 1, 0), "torque": round(6.0 * st, 2), "radius": 0.038 * s,
                 "mass": 0.3 * sm, "joint": "fixed", "shape": "box",
                 "geometry": {"family": "extrude", "height": 0.05 * s,
                              "profile": [[-0.045 * s, -0.022 * s], [0.045 * s, -0.022 * s],
                                          [0.045 * s, 0.022 * s], [-0.045 * s, 0.022 * s]],
                              "fillet": 0.006 * s}}]                                          # foot pad
        if per_side == 2:
            xs = [0.12 * s, -0.12 * s]                    # exact legacy quadruped layout at s=1 (gait-stable)
        else:                                             # spread legs evenly front→back along the body
            span = 0.14 * s
            xs = [round(span - i * (2 * span / (per_side - 1)), 3) for i in range(per_side)]
        sy_mount = 0.13 * s
        mounts = [(x, sy) for x in xs for sy in (sy_mount, -sy_mount)]   # left + right of each x station
        torso_r = round((0.16 + 0.02 * (per_side - 2)) * s, 3)  # widen the torso for more leg pairs
        cls = "quadruped" if n == 4 else "legged"
        species = "quadruped.composed" if n == 4 else f"legged{n}.composed"
        # Head is a FORWARD-pointing muzzle at the FRONT of the torso (not a thumb on top): mount it as a
        # limb (only limbs carry mount_euler) at +x with a +90° pitch about y so its local +z aims along
        # world +x (forward). x=0.12 < torso_r keeps it flush so the connection audit still passes.
        head_limb = {"prefix": "head", "parent": "torso", "mount_offset": (0.12 * s, 0.0, -0.04 * s),
                     "mount_euler": (0.0, _math.pi / 2, 0.0),
                     "links": [{"length": 0.06 * s, "radius": 0.032 * s, "joint": "fixed", "mass": 0.2 * sm,
                                "geometry": {"family": "revolve",
                                             "profile": [[0.030 * s, 0], [0.028 * s, 0.016 * s],
                                                         [0.020 * s, 0.036 * s], [0.006 * s, 0.052 * s]]}}]}
        leg_limbs = [{"prefix": f"leg{i}", "parent": "torso", "mount_offset": (sx, sy, -0.04 * s),
                      "mount_euler": (_math.pi, 0.0, 0.0), "links": [dict(seg) for seg in qleg]}
                     for i, (sx, sy) in enumerate(mounts)]
        t_len = 0.08 * s
        # Baked standing crouch (render/spawn-only): thigh pitched back + knee folded forward so each leg reads
        # as an articulated limb with a visible knee, not a straight-down post (the dominant "toy" cue on the
        # offline quad). Angles are well within the thigh/calf ranges; physics/training reset to their own init.
        rest_pose = {}
        for limb in leg_limbs:
            pre = limb["prefix"]
            rest_pose[f"{pre}_1_joint"] = -0.55   # thigh: pitch back
            rest_pose[f"{pre}_2_joint"] = 1.10    # calf: fold forward (bent knee)
        return {"robot_class": cls, "base_mount": "free", "species": species, "rest_pose": rest_pose,
                "base": {"name": "torso", "shape": "box", "length": t_len, "radius": torso_r, "mass": 3.0 * sm,
                         # an ORIGINAL rounded ANIMAL trunk (elliptical loft) — longer front-to-back (x) than
                         # wide (y), tapering at the head/tail ends — instead of a square slab. Visual only:
                         # the box collider + gait-tuned legs are unchanged. (sections: [z, half_y, half_x])
                         "geometry": {"family": "loft", "sections": [
                             [0.0,           torso_r * 0.72, torso_r * 1.18],
                             [0.375 * t_len, torso_r * 0.88, torso_r * 1.34],
                             [0.688 * t_len, torso_r * 0.84, torso_r * 1.30],
                             [t_len,         torso_r * 0.60, torso_r * 1.05]]}},
                "limbs": [head_limb] + leg_limbs,
                "end_effector": {"kind": "none"}}
    if robot_class == "mobile_manipulator":              # G4 composite: a wheeled chassis CARRYING a grasp arm.
        # Chassis is wider/heavier than the plain rover (it must counterweight the arm); the 4-DOF grasp arm
        # (yaw + 2 pitch + wrist) mounts at the chassis TOP (serial links chain from the base's distal tip),
        # so the robot can drive to a box, pick it, and carry it — the request the e2e test showed we dropped.
        reach_c = max(0.4, float(reach_m or 0.55))
        t_c = round(max(10.0, (0.5 + 0.35 * 4) * 9.81 * reach_c * 1.6), 1)
        return {"robot_class": "mobile_manipulator", "base_mount": "free",
                "species": "mobile_manipulator.composed",
                "base": {"name": "chassis", "shape": "box", "length": 0.10, "radius": 0.22, "mass": 8.0},
                "links": _grasp_arm_links(reach_c, t_c),
                "wheels": [{"name": f"wheel_{i}", "parent": "chassis", "radius": 0.07, "thickness": 0.05,
                            "mount_offset": (sx, sy, -0.10), "torque": 10.0}
                           for i, (sx, sy) in enumerate([(0.16, 0.19), (0.16, -0.19),
                                                          (-0.16, 0.19), (-0.16, -0.19)])],
                "end_effector": {"kind": "gripper", "span": 0.10}}
    if robot_class == "mobile_base":                      # a driven base: a FREE-floating 4-wheel rover
        # flat wide chassis (box geom = radius,radius,length/2 -> 0.36 sq x 0.08 tall); wheels at the
        # four bottom corners (mount_offset z = -length puts them at the chassis underside) are the
        # ground contact, so applying wheel torque drives it.
        return {"robot_class": "mobile_base", "base_mount": "free", "species": "mobile_base.composed",
                "base": {"name": "chassis", "shape": "box", "length": 0.08, "radius": 0.18, "mass": 3.0},
                "wheels": [{"name": f"wheel_{i}", "parent": "chassis", "radius": 0.06, "thickness": 0.04,
                            "mount_offset": (sx, sy, -0.08), "torque": 8.0}
                           for i, (sx, sy) in enumerate([(0.13, 0.15), (0.13, -0.15),
                                                          (-0.13, 0.15), (-0.13, -0.15)])],
                "end_effector": {"kind": "none"}}

    reach = max(0.2, float(reach_m or 0.65))
    g = 9.81
    # End-effector first (it decides the arm layout): spray nozzle, multi-finger hand for dexterous /
    # irregular objects, else a parallel-jaw gripper (the default manipulator).
    if any(w in p for w in _SPRAY_WORDS):
        ee = {"kind": "spray"}
    elif any(w in p for w in _DEXTEROUS_WORDS):           # dexterous/irregular objects -> multi-finger hand
        ee = {"kind": "hand", "fingers": 3, "span": 0.10}
    else:                                                 # default manipulators to a gripper
        ee = {"kind": "gripper", "span": 0.10}
    # DOF / link layout. Precision tasks (or named industrial arms) get a 6-7 DOF anthropomorphic arm
    # with wrist roll/pitch/yaw (§23 Franka-class). A GRASP arm (gripper/hand) gets a wrist PITCH so the
    # palm can be levelled for a top-down approach — a wrist-less arm reaches the object tilted and the
    # jaws sweep it away (findings §12). A spray/reach arm just needs 2-3 pitch links.
    if any(w in p for w in _PRECISION_WORDS):
        dof = 7 if any(w in p for w in ("7-dof", "seven", "redundant", "panda", "franka")) else 6
    elif ee["kind"] in ("gripper", "hand"):
        dof = 4                                           # yaw + 2 reach pitches + wrist pitch
    else:
        dof = 2 if reach < 0.45 else 3
    arm_mass = 0.35 * dof
    torque = round(max(8.0, (float(payload_kg or 0.25) + arm_mass) * g * reach * 1.6), 1)
    links = _grasp_arm_links(reach, torque) if dof == 4 else _arm_links(reach, dof, torque)
    base_mount = "table" if (environment or "tabletop") == "tabletop" else "table"
    return {"robot_class": "manipulator", "base_mount": base_mount, "species": "manipulator.composed",
            "base": {"name": "base", "shape": "box", "length": 0.08, "radius": 0.04, "mass": 1.0,
                     "geometry": _role("mount_base", 0.08, 0.04)},
            "links": links, "end_effector": ee}


# Over-articulation budgets for the BOUNDED morphologies ONLY. A human / arm body has a small, well-defined
# number of actuated joints; the LLM (gpt-5.2) tends to OVER-articulate one into a 32-DOF "beaded noodle"
# (dozens of tiny stacked segments that satisfy the coarse bounding-box critic but read as a spine of beads).
# These caps make the verify->repair loop reject that. CRITICAL — caps are applied ONLY to humanoid/biped and
# manipulator, NEVER to quadruped/legged: those classes are wildly varied and the coarse intent classifier
# routes crabs (8 legs), spiders (8), octopods, AND snakes (one long serial chain) all into them — a flat cap
# there wrongly rejected an 8-legged crab as if it were a noodle (it became a 4-leg dog). Generality over a
# false cap. Credible references: anthropometric humanoid = 15 segments / 8 DOF; a 7-DOF arm is the ceiling.
_DOF_CAP = {"humanoid": 14, "biped": 14, "manipulator": 10}
_SEG_CAP = {"humanoid": 26, "biped": 26, "manipulator": 16}
_SPINE_CHECK_CLASSES = ("humanoid", "biped")   # NOT quadruped: a snake routes there as one long valid chain


def _articulation_issues(gene) -> list[str]:
    """Catch OVER-articulation in the BOUNDED morphologies (the 'beaded noodle' failure): too many actuated
    DOF / parts, or a humanoid torso split into a long articulated SPINE instead of one coherent part. Applied
    ONLY to humanoid/biped/manipulator — quadruped/legged stay uncapped so crabs/spiders/octopods/snakes keep
    their (legitimately many) segments. Returns repair messages; empty = acceptable."""
    issues: list[str] = []
    cls = (gene.robot_class or "").lower()
    dof = len(gene.actuated_joints())
    n = len(gene.segments)
    if cls in _DOF_CAP and dof > _DOF_CAP[cls]:
        issues.append(
            f"the body has {dof} actuated joints — far too many for a {cls} (max ~{_DOF_CAP[cls]}). It reads "
            f"as an over-articulated 'beaded noodle', not a real machine. CONSOLIDATE: one torso part, one head, "
            f"and 2-3 actuated joints per limb (e.g. a humanoid = shoulder+elbow per arm, hip+knee per leg = 8).")
    if cls in _SEG_CAP and n > _SEG_CAP[cls]:
        issues.append(
            f"the body has {n} parts — too many for a {cls} (target ~{_SEG_CAP[cls]}). Merge stacked segments; "
            f"the torso/chassis must be ONE part, not a stack of small blocks.")
    # An articulated SPINE: a chain of actuated links hanging directly off the root before it branches into
    # limbs. A real humanoid torso is a single rigid part — limbs branch FROM it; it is not a multi-joint column.
    if cls in _SPINE_CHECK_CLASSES:
        spine = 0
        node = gene.root()
        while node is not None:
            kids = gene.children_of(node.name)
            nxt = [c for c in kids if c.joint_type in ("revolute", "prismatic")]
            if len(kids) >= 2 or not nxt:        # branches into limbs, or chain ends -> torso is done
                break
            spine += 1
            node = nxt[0]
        if spine > 1:
            issues.append(
                f"the torso is an articulated spine of {spine} actuated links — a {cls} torso must be ONE rigid "
                f"part that the limbs branch off. Make the root torso a single part (joint-free) and attach the "
                f"head/arms/legs to it directly.")
    return issues


def _morphology_quality_issues(gene) -> list[str]:
    """Quality gate for an LLM-proposed body BEYOND schema validity — catches the failure modes that make a
    generated body look broken: limbs that FLOAT off the parent (the "empty space / parts not attached"
    defect), degenerate single-pole bodies, and OVER-articulation (the 'beaded noodle'). Returns human-readable
    problems (fed back to the LLM to repair); empty means the body is acceptable. Templates all pass this gate."""
    issues: list[str] = _articulation_issues(gene)
    seg = {s.name: s for s in gene.segments}
    for s in gene.segments:
        if s.parent is None or s.joint_type == "prismatic":   # root + gripper jaws (bridged) are exempt
            continue
        p = seg.get(s.parent)
        if p is None:
            continue
        mo = getattr(s, "mount_offset", (0.0, 0.0, 0.0))
        lateral = max(abs(mo[0]), abs(mo[1]))
        if lateral - float(p.radius_m) > 0.02:                # >2 cm beyond the parent surface = a float
            issues.append(
                f"segment '{s.name}' mounts {lateral:.3f} m to the side of parent '{s.parent}', whose "
                f"surface is only {p.radius_m:.3f} m wide — it floats in empty space. Reduce its "
                f"mount_offset x/y to <= the parent's radius so the limb attaches to the body.")
    if len(gene.segments) < 4:
        issues.append("the body has too few parts (<4) — add limb segments and/or a head so it is a real machine.")
    if (gene.robot_class or "").lower() in {"humanoid", "biped", "quadruped", "legged"}:
        branches = sum(1 for s in gene.segments
                       if sum(1 for c in gene.segments
                              if c.parent == s.name and c.joint_type != "prismatic") >= 2)
        if branches == 0:
            issues.append("a legged/humanoid body must BRANCH into multiple limbs from the torso (use "
                          "'limbs'), not collapse into a single chain/pole.")
    return issues


def _pose_quality_issues(gene) -> list[str]:
    """Physics-grounded quality gate: compile the body and inspect its REST pose for DEGENERACY that the
    geometric checks miss — most importantly a legged body whose limbs stack into a tall vertical COLUMN
    (legs not splayed / pointing up instead of down) instead of a wide stable stance. Needs MuJoCo; returns
    [] (lenient) if it is unavailable so the offline path is unaffected."""
    issues: list[str] = []
    try:
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        sz = standing_spawn_z(gene, meshed=False) if gene.base_mount == "free" else None
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False, spawn_z=sz))
        if m.ngeom <= 1:
            return issues
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ext = d.geom_xpos[1:].max(axis=0) - d.geom_xpos[1:].min(axis=0)
        horiz, vert = max(float(ext[0]), float(ext[1])), float(ext[2])
        if (gene.robot_class or "").lower() in {"quadruped", "legged"} and vert > 1.5 * max(horiz, 1e-6):
            issues.append(
                f"the body is a tall vertical column (height {vert:.2f} m vs width {horiz:.2f} m) — a legged "
                f"robot must be WIDER than tall: SPLAY the legs outward to ±x/±y around the torso and set each "
                f"leg's mount_euler to [3.14,0,0] so it points DOWN to the ground, not stacked upward.")
    except Exception:  # noqa: BLE001 - no MuJoCo / transient compile issue -> skip the physics gate
        pass
    return issues


def propose_morphology_spec(prompt: str, *, reach_m: float | None = None, payload_kg: float | None = None,
                            environment: str | None = None, llm=None, robot_class: str | None = None,
                            repair_tries: int = 2) -> dict:
    """Propose a block recipe. With an ``llm`` the LLM DESIGNS the body over the constrained menu (any class,
    P2 — the gate that forced templates for humanoid/quadruped/etc is gone) under a proposer→verify→REPAIR
    loop: each proposal is composed (kinematic/schema validity) and quality-gated (connection + non-
    degeneracy); problems are fed back to the LLM to fix, up to ``repair_tries`` times. Any persistent failure
    (or no backend) falls back to the deterministic template — so a bad/declined LLM proposal never ships a
    broken body. ``robot_class`` (from the build planner) steers the template fallback + hints the LLM."""
    heuristic = morphology_from_requirements(reach_m or 0.65, payload_kg or 0.25, prompt=prompt,
                                             environment=environment, robot_class=robot_class)
    heuristic["design_source"] = "heuristic"
    if llm is None:
        return heuristic
    errors: list[str] | None = None
    for _ in range(max(1, repair_tries + 1)):
        try:
            raw = _llm_spec(prompt, reach_m, payload_kg, llm, robot_class=robot_class, errors=errors)
            gene = compose_from_spec(raw)                 # verifier: realize + validate the kinematic tree
            quality = _morphology_quality_issues(gene) + _pose_quality_issues(gene)   # connection + rest-pose
            try:                                          # + the geometric PROPORTION critic (render->verify)
                from virturoid.services.anatomy_critic import critique_gene
                quality += [i["detail"] for i in critique_gene(gene)["issues"]
                            if i["severity"] in ("high", "fatal")]
            except Exception:  # noqa: BLE001 - critic is best-effort (needs MuJoCo); never block on it
                pass
            if quality:
                errors = quality                          # repair: feed the problems back and retry
                continue
            raw["design_source"] = "llm"
            return raw
        except Exception as e:  # noqa: BLE001 - unbuildable/declined proposal -> repair, then template fallback
            errors = [f"the recipe failed to build ({e}); return a corrected, buildable JSON recipe."]
    return heuristic


def _intent_validation_issues(gene, req) -> list[str]:
    """Fatal/high risk-flag details from the gene-design validation gate (empty = sound). Best-effort: any
    failure of the gate itself (e.g. MuJoCo missing) returns [] so a build is never blocked by the validator."""
    try:
        from virturoid.services.gene_validation import validate_gene_design
        report = validate_gene_design(gene, payload_kg=getattr(req, "payload_kg", 0.0) or 0.0)
        return [f["detail"] for f in report["risk_flags"] if f["severity"] in ("fatal", "high")]
    except Exception:  # noqa: BLE001
        return []


def compose_robot(prompt: str, *, llm="auto", reach_m: float | None = None,
                  payload_kg: float | None = None, plan=None, ensure_walkable: bool = False) -> RobotGene:
    """Compose a robot AND finalize it for the task. Every composition path funnels through here, so the single
    ``finalize_for_task`` step (task-adaptive materials + skeleton thickness) is applied to whatever body was
    built — once, before sim/export — keeping the design, physics, and BOM in agreement.

    ``ensure_walkable=True`` (opt-in — default OFF keeps every build byte-identical + free) runs the
    diagnosis-driven quadruped walkability fallback: if the built quad ROLLS OVER, reach for the fanned
    wide-stance + crawl-gait HINT (per-request, sleek-when-possible, LLM designs preserved). A caller preparing
    a robot for a WALKING task opts in; a generic build pays no eval cost."""
    gene = _compose_robot_impl(prompt, llm=llm, reach_m=reach_m, payload_kg=payload_kg, plan=plan)
    try:
        # An AERIAL prompt (drone/quadcopter/...) is a rotorcraft, not a legged/wheeled body: compose a
        # quadcopter (central hub + 4 rotors + camera pod) that the geometric flight controller actually flies
        # (see aerial.py / _honest_fly). A 'flying fish' stays aquatic — the undulatory swim tier is a better fit.
        from virturoid.services.aerial import build_quadcopter, is_aerial_prompt
        from virturoid.services.aquatic import is_aquatic_prompt as _is_aquatic
        if is_aerial_prompt(prompt) and not _is_aquatic(prompt):
            gene = build_quadcopter(prompt)
    except Exception:  # noqa: BLE001 - aerial routing is value-add; never block a compose
        pass
    try:
        # An aquatic prompt (fish/eel/shark/...) is an UNDULATOR, not a legged/quad body. Compose a serial spine
        # (a snake) if the built body branches into legs, then laterally-compress it into a fish cross-section so
        # it generates REAL undulatory thrust (measured 0.06 m round -> 0.25 m flat: DOES NOT SWIM -> SWIMS).
        from virturoid.services.aquatic import _is_serial_spine, ensure_aquatic_body, is_aquatic_prompt
        if is_aquatic_prompt(prompt):
            if not _is_serial_spine(gene):
                gene = _compose_robot_impl("a snake robot", llm=None)
            ensure_aquatic_body(gene)
    except Exception:  # noqa: BLE001 - aquatic reshape is value-add; never block a compose
        pass
    try:
        from virturoid.services.bom_builder import finalize_for_task
        finalize_for_task(gene, prompt)
    except Exception:  # noqa: BLE001 - finalization is value-add; never block a compose
        pass
    try:
        # G6/G7 FINAL-STAGE slenderness clamp (walking legs only): whatever the source — an LLM graph's own
        # girth/thickness, the heavy-task thickener, a redesign — a leg segment that ships must respect the
        # morphology-prior band (length/diameter >= 2.2). 1:1 sausage legs were the e2e test's core visual +
        # functional defect; realism bands outrank per-part knobs for the load-bearing walking chain.
        # D3a: cap at /4.5 (aspect 2.25), NOT /4.4 (aspect exactly 2.2, which round(,5) nudges UP past the gate's
        # strict `< 2.2` — the T4 seed3 experiment caught a clamped leg failing its own gate by 4e-4). /4.5 lands
        # strictly INSIDE the band so a clamp output always passes the check that motivated it.
        if (gene.robot_class or "").lower() in ("quadruped", "legged"):
            from virturoid.services.bom_builder import _scale_geo
            for s_ in gene.segments:
                n_ = (s_.name or "").lower()
                if "leg" in n_ and (s_.joint_type or "").lower() == "revolute" and s_.length_m > 0:
                    cap = s_.length_m / 4.5
                    if s_.radius_m > cap:
                        f_ = cap / s_.radius_m
                        s_.radius_m = round(cap, 5)
                        s_.mass_kg = round(max(0.01, s_.mass_kg * f_ * f_), 5)
                        _scale_geo(s_.geometry, f_)
    except Exception:  # noqa: BLE001
        pass
    if ensure_walkable and (gene.robot_class or "") not in ("aerial", "aquatic"):
        # opt-in per-request walkability fallback — but a drone (aerial) or a fish (aquatic) is NOT a walker;
        # running the quad-walkability rewrite on one would mangle its purpose-built body.
        try:
            from virturoid.services.anatomy_compiler import ensure_walkable_quad
            gene = ensure_walkable_quad(gene, prompt)
        except Exception:  # noqa: BLE001 - best-effort; never block a compose on the walkability check
            pass
    return gene


def _compose_robot_impl(prompt: str, *, llm="auto", reach_m: float | None = None,
                        payload_kg: float | None = None, plan=None) -> RobotGene:
    """Parse requirements from the prompt, compose a robot from blocks, and return the validated gene.

    The general path: the **LLM designs the body** over the building blocks (``_llm_spec`` — any kinematic
    tree, validated by composition) and the **LLM planner** (``plan_build``) discerns the class/feasibility.
    ``llm="auto"`` resolves the configured backend (real LLM in the app; ``None`` under tests via
    ``VIRTUROID_NO_LOCAL_ENV``, which falls back to the deterministic heuristic). Pass an explicit ``llm``
    object or ``None`` to override. The keyword heuristic is the OFFLINE fallback, not the strategy."""
    from virturoid.services.intent_planner import plan_build
    from virturoid.services.requirements_builder import build_requirements_from_prompt

    if llm == "auto":
        from virturoid.services.llm_client import get_llm
        llm = get_llm("morphology")        # the configured backend (OpenAI gpt-5.2, or the local PC model)
    if plan is None:
        # Classify WITH the LLM when one is available: the class gate below decides whether the general anatomy
        # designer runs, so a keyword-only plan would misroute every un-enumerated creature ("axolotl",
        # "wyvern", "basilisk") to a manipulator and they'd NEVER reach the anatomy path. The LLM knows an
        # axolotl is an animal; the keyword list only knows the species someone typed in. (llm=None offline.)
        plan = plan_build(prompt, llm=llm)
    req = build_requirements_from_prompt(prompt, payload_kg=payload_kg, reach_m=reach_m)
    from virturoid.services.spec_parser import parse_target_height_m
    target_height_m = parse_target_height_m(prompt)        # honor "a 1.2 m tall humanoid" even offline (§4.8E)

    # DESIGN-INTENT COMPILER (general + reliable): with an LLM, it DESIGNS the BODY PLAN — class, scale, how
    # many of which appendages, an end effector — and our deterministic parametric builders synthesize the
    # connected, role-shaped, original geometry (anthropometric humanoid / N-legged template / spider-snake-
    # frog archetype). This is what makes the product general — a crab, a hexapod, a wheeled arm, a humanoid —
    # while staying credible BY CONSTRUCTION. We measured that letting the LLM hand-place every capsule instead
    # produced a different broken body each run (a 'beaded noodle', scattered blocks, a finger welded to a
    # foot); the intent compiler removes that failure mode. (`propose_morphology_spec` keeps the raw-geometry
    # free-design path available for experiments/tests, but it is no longer the product default.)
    # PRIMARY (general + intelligent): for a CREATURE, the LLM describes its ANATOMY as a structured graph
    # (parts/roles/attachment/proportions/joints — what an LLM genuinely knows: a dog has a long body, neck,
    # head, snout, ears, tail, 4 legs) and the GENERAL anatomy compiler builds the geometry around it. NO
    # per-species code, no LLM-placed coordinates. Falls through to the intent compiler/templates if the
    # anatomy doesn't build + validate cleanly. (Manipulators/mobile bases aren't creatures -> skip to intent.)
    if llm is not None and (plan.robot_class or "").lower() in ("humanoid", "biped", "quadruped", "legged"):
        try:
            from virturoid.services.anatomy_compiler import build_from_anatomy
            from virturoid.services.anatomy_designer import propose_anatomy
            gene = build_from_anatomy(propose_anatomy(prompt, plan, req, llm))
            if gene is not None and not gene.validate() and not _intent_validation_issues(gene, req):
                # G6: the LLM's anatomy must ALSO clear the morphology-prior gate (3 actuated joints/leg,
                # slender segments). A non-conformant graph falls through to the archetype/generic builders,
                # which are prior-conformant by construction — so a blob never ships just because the LLM
                # drew one this run.
                from virturoid.services.morphology_priors import validate_morphology
                if validate_morphology(gene).ok:
                    gene.design_source = "anatomy"
                    return gene
        except Exception:  # noqa: BLE001 - any LLM/build/validate failure -> archetype/intent below
            pass
        # The LLM anatomy path can fail (rate-limited / weak model / bad graph). Distinctive exotic SHAPES the
        # general compiler can't yet express (a legless serpent, an 8-radial spider, a leaping frog) have a
        # dedicated builder; EVERYTHING ELSE legged falls to a CLASS-GENERAL generic creature (NOT a per-species
        # lookup) so the body is connected and standing, never a flat slab. The LLM supplies the species.
        try:
            from virturoid.services.novel_anatomy import novel_archetype_gene
            arch = novel_archetype_gene(prompt)
            if arch is not None and not arch.validate():
                arch.design_source = "archetype"
                return arch
        except Exception:  # noqa: BLE001
            pass
        try:
            from virturoid.services.anatomy_compiler import generic_creature_gene
            generic = generic_creature_gene(prompt, plan.robot_class)
            if generic is not None:
                generic.design_source = "anatomy_generic"
                return generic
        except Exception:  # noqa: BLE001
            pass

    if llm is not None and (plan.robot_class or "").lower() != "mobile_manipulator":
        # (the composite goes straight to its deterministic chassis+wheels+arm spec below: the live LLM designer
        # answers the TASK class and drops the mobile platform — G4's exact regression)
        try:
            from virturoid.services.intent_morphology import build_from_intent, propose_intent
            gene = build_from_intent(propose_intent(prompt, plan, req, llm), prompt, req)
            if gene is not None and not gene.validate():
                # Validation as a first-class stage on the path that SHIPS: if the built body trips a
                # fatal/high flag (an over-stressed link, an unstable stance, an under-spec joint), re-prompt
                # the plan ONCE with the issues fed back, then take whichever body is sound (keyword fallback
                # remains the final net). Cheap today (the builders rarely flag); the safety margin that lets
                # the intent gain freer structure (compositional appendages) without shipping a degenerate body.
                fatal = _intent_validation_issues(gene, req)
                if not fatal:
                    return gene
                repaired = build_from_intent(
                    propose_intent(prompt, plan, req, llm, errors=fatal), prompt, req)
                if repaired is not None and not repaired.validate():
                    return repaired
                return gene                                   # repair didn't build cleanly -> keep the first
        except Exception:  # noqa: BLE001 - any LLM/build failure -> deterministic keyword builder below
            pass

    # Offline (no LLM) OR the intent path failed: the deterministic, keyword-routed ORIGINAL builder — the same
    # credible templates, just selected by keywords instead of by the LLM. Quality never regresses.
    spec = morphology_from_requirements(req.reach_m, req.payload_kg, prompt=prompt,
                                        environment=req.environment, robot_class=plan.robot_class)
    spec["design_source"] = "heuristic"
    return _fallback_gene(prompt, spec, target_height_m=target_height_m)


def _fallback_gene(prompt: str, heuristic_spec: dict, *, target_height_m: float | None = None) -> RobotGene:
    """The deterministic ORIGINAL-geometry FALLBACK when the LLM is unavailable or its design fails to
    verify. Never copies a real model: a humanoid/biped -> the anthropometric loft humanoid; a clearly-named
    exotic (spider/snake/frog) -> its archetype; anything else -> the heuristic block recipe.

    ``target_height_m`` (a height parsed from the prompt, §4.8E) scales the humanoid to the requested stature
    via the builder's head-height knob (total stature ~7.7 H), so "a 1.2 m tall humanoid" stands ~1.2 m even
    offline; absent, the head-canon default is used so default builds stay byte-identical."""
    p = (prompt or "").lower()
    if any(w in p for w in _HUMANOID_WORDS):
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        # 8.55 = empirical measured-stature / head-height for this builder (the compiled AABB exceeds the 7.7
        # head-canon because geom bounds extend past the crown); divide by it so MEASURED height ~= requested.
        H = max(0.12, min(0.32, target_height_m / 8.55)) if target_height_m else 0.205
        g = build_anthropometric_humanoid(H)
        g.design_source = "anthropometric"
        return g
    # Distinctive exotic SHAPES the general compiler can't yet express (legless serpent / radial spider /
    # leaping frog) -> their dedicated builder. EVERY OTHER legged creature -> a CLASS-GENERAL generic
    # quadruped through the general compiler (NOT a per-species lookup), so the no-LLM path is a connected
    # standing body, not a flat slab. The LLM path supplies the actual species anatomy when available.
    from virturoid.services.novel_anatomy import novel_archetype_gene
    novel = novel_archetype_gene(prompt)
    if novel is not None:
        novel.design_source = "archetype"
        return novel
    try:
        from virturoid.services.anatomy_compiler import generic_creature_gene
        from virturoid.services.intent_planner import plan_build
        _cls = plan_build(prompt, llm=None).robot_class
        # G4 guard: the generic-creature fallback is for LEGGED bodies only. It used to run for ANY class and
        # silently hijack e.g. a mobile_manipulator ('mobile robot that picks boxes') into a quadruped.
        if (_cls or "").lower() in ("quadruped", "legged", "biped"):
            generic = generic_creature_gene(prompt, _cls)
            if generic is not None:
                generic.design_source = "anatomy_generic"
                return generic
    except Exception:  # noqa: BLE001
        pass
    g = compose_from_spec(heuristic_spec)
    g.design_source = heuristic_spec.get("design_source", "heuristic")
    return g


def compose_working_robot(prompt: str, *, llm=None, reach_m: float | None = None,
                          payload_kg: float | None = None, iterations: int = 4, population: int = 8,
                          scene_count: int = 4, seed: int = 0, best_response: bool | None = None) -> dict:
    """Compose a robot from blocks AND co-design it into a WORKING one (the full autonomous path).

    Composes a gene (``compose_robot``), then co-designs it for ITS task — generalized across
    morphologies (Theme 3): a manipulator runs the controller-aware pick-place co-design
    (``co_optimize_gene``, which also returns tuned controller params), while a legged/mobile/spray body
    runs ``co_design_general`` (CEM over body scale scored by the topology-dispatched task metric). So a
    freshly-composed body of ANY class is tuned to actually perform, not just compile. Returns
    ``{gene, baseline_success, success_rate, controller_params, history}``. Needs MuJoCo.
    """
    from virturoid.services.task_matched_eval import robot_kind

    gene = compose_robot(prompt, llm=llm, reach_m=reach_m, payload_kg=payload_kg)
    kind = robot_kind(gene)
    if kind == "manipulator":
        from virturoid.services.gene_codesign import co_optimize_gene
        from virturoid.services.task_runtime import generate_task_scenes, select_task_spec
        spec = select_task_spec(prompt)
        scenes = generate_task_scenes(gene, spec, count=scene_count, seed=seed)
        # the task's real object->target assignments so co-design scores the actual task, not a vacuous default
        return co_optimize_gene(gene, scenes, assignments=spec.assignments, iterations=iterations,
                                population=population, seed=seed)
    # legged / mobile / spray: body co-design against the general task metric. A LEGGED body is scored by
    # its best-response controller (the Stackelberg / trainability keystone) so the search selects a body
    # that is easy to make WALK, and returns the follower it found; mobile/spray keep the fixed-controller
    # metric (best_response_score has no follower for them yet). Auto-on for legged; override via best_response.
    from virturoid.services.gene_codesign import co_design_general
    use_br = (kind == "legged") if best_response is None else bool(best_response)
    r = co_design_general(gene, prompt, iterations=iterations, population=population, seed=seed,
                          best_response=use_br)
    return {"gene": r["gene"], "controller_params": r.get("best_controller"),
            "success_rate": r["best_value"], "baseline_success": r["baseline_value"],
            "history": r["history"]}


_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "length": {"type": "number"}, "radius": {"type": "number"}, "mass": {"type": "number"},
        "joint": {"type": "string", "enum": ["revolute", "prismatic", "fixed"]},
        "axis": {"type": "array", "items": {"type": "number"}},
        "torque": {"type": "number"}, "lower": {"type": "number"}, "upper": {"type": "number"},
        "geometry": {"type": "object", "properties": {
            "family": {"type": "string",
                       "enum": ["loft", "sole", "revolve", "extrude", "tapered", "role",
                                "box", "cylinder", "capsule", "sphere"]},
            "sections": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "role": {"type": "string", "enum": ["torso_shell", "sensor_head", "upper_arm", "forearm",
                                                "thigh", "shin", "foot_pad", "hand_plate", "mount_base",
                                                "actuator_segment", "quad_trunk", "quad_hip", "quad_thigh",
                                                "quad_calf"]},
            "profile": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "height": {"type": "number"}, "length": {"type": "number"}, "width": {"type": "number"},
            "heel": {"type": "number"}, "r0": {"type": "number"}, "r1": {"type": "number"},
            "radius": {"type": "number"}, "fillet": {"type": "number"}}},
    },
    "required": ["length", "axis"],
}
_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "robot_class": {"type": "string"},
        "base_mount": {"type": "string", "enum": ["table", "floor", "free"]},
        "species": {"type": "string"},
        "base": {"type": "object", "properties": {
            "name": {"type": "string"}, "shape": {"type": "string"},
            "length": {"type": "number"}, "radius": {"type": "number"}, "mass": {"type": "number"},
            "geometry": _LINK_SCHEMA["properties"]["geometry"]}},
        "links": {"type": "array", "items": _LINK_SCHEMA},
        "limbs": {"type": "array", "items": {"type": "object", "properties": {
            "prefix": {"type": "string"}, "parent": {"type": "string"},
            "mount_offset": {"type": "array", "items": {"type": "number"}},
            "mount_euler": {"type": "array", "items": {"type": "number"}},
            "links": {"type": "array", "items": _LINK_SCHEMA}}, "required": ["prefix", "parent", "links"]}},
        "wheels": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "parent": {"type": "string"}, "radius": {"type": "number"},
            "thickness": {"type": "number"}, "torque": {"type": "number"},
            "mount_offset": {"type": "array", "items": {"type": "number"}}}}},
        "end_effector": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["gripper", "hand", "spray", "suction", "hook", "none"]},
            "span": {"type": "number"}, "finger_len": {"type": "number"}, "fingers": {"type": "integer"}}},
    },
    "required": ["robot_class", "base", "end_effector"],
}
_SPEC_SYSTEM = (
    "You are Virturoid's morphology designer. DESIGN a HIGH-FIDELITY robot body from primitive building "
    "blocks to suit the task — do NOT pick from a fixed catalogue, and do NOT under-build. The blocks:\n"
    "• base: the root block (box/capsule/cylinder), welded to the world (base_mount table|floor) or free-"
    "floating (base_mount 'free' for anything that moves: legged, wheeled).\n"
    "• links: a serial chain off the base — each {name,length(m),axis[x,y,z],joint(revolute|prismatic|"
    "fixed),torque(Nm),radius,lower,upper}. Use a 'fixed' link for a welded part (a head, a foot, a hand).\n"
    "• limbs: repeated chains attached at a 'parent', each with mount_offset[x,y,z], mount_euler[r,p,y] and "
    "its own 'links'. THIS is how you make ANY multi-limb body.\n"
    "• wheels / end_effector: driven discs; {kind gripper|hand(fingers:N)|spray|suction|hook|none}.\n"
    "\nGEOMETRY — you DESIGN every part's ORIGINAL shape (we never copy a real robot). Give each block a "
    "'geometry'. TWO ways, and you should use BOTH:\n"
    "1) For STANDARD body parts, tag the block with a ROLE — we render it as detailed, credible, ORIGINAL "
    "anatomy (smooth tapered lofts, correct cross-sections). ALWAYS use the role for these so the body reads "
    "right: {family:'role', role:'torso_shell'} for a torso/chassis; 'sensor_head' for a head; 'upper_arm' "
    "then 'forearm' for an arm; 'thigh' then 'shin' for a leg; 'foot_pad' for a foot; 'hand_plate' for a "
    "hand; 'mount_base' for a pedestal. (A leg of a crab/insect can also use thigh/shin.) Tag EVERY standard "
    "segment with its role — do NOT leave a torso/limb/head as a bare primitive (that is what makes a body "
    "look like a stack of tubes).\n"
    "2) For CUSTOM / exotic parts with no role (a crab claw, a mantis raptorial arm, a wing, a fin, a tail, "
    "an antenna, a shell), author the shape yourself:\n"
    "  - {family:'loft', sections:[[z,half_y,half_x],...]} — a smooth solid through ELLIPTICAL cross-sections "
    "along +z; half_y (side) and half_x (front-back) vary INDEPENDENTLY (broad-and-shallow carapace, a "
    "tapered claw, a streamlined fish body). ~4-7 sections. The most powerful custom shaper.\n"
    "  - {family:'revolve', profile:[[r,z],...]} — axisymmetric rounded parts (dome, nozzle, eyeball).\n"
    "  - {family:'sole', length, width, height, heel} — a forward-pointing FOOT footprint.\n"
    "  - {family:'extrude', profile:[[x,y],...], height, fillet} — flat plates/brackets/fins/claws.\n"
    "NEVER reuse a real product's part; invent every shape. A torso/head/limb WITHOUT a role or custom "
    "geometry is REJECTED as low-fidelity.\n"
    "\nFIDELITY (REQUIRED — match a real machine/animal, not a stick figure):\n"
    "• Give every limb 3+ segments ending in a WELDED end-part: an arm = upper+forearm+hand(fixed); a leg "
    "= thigh+shin+foot(fixed); a finger/antenna can be 2.\n"
    "• Add a HEAD or sensor module (a 'fixed' link on the base) for animals/humanoids; add feet under legs.\n"
    "\nDO NOT OVER-ARTICULATE (this is the #1 failure — a body that becomes a 'beaded noodle' of dozens of "
    "tiny stacked segments). HARD LIMITS:\n"
    "• The TORSO/CHASSIS is exactly ONE part — the base. NEVER build a multi-segment articulated spine/neck "
    "stack. Limbs and the head branch directly OFF the single torso.\n"
    "• Give each limb only 2-3 ACTUATED joints (e.g. shoulder+elbow, hip+knee); the hand/foot/fingertip is a "
    "single FIXED weld, not another joint.\n"
    "• Part / actuated-joint budgets: humanoid ≈ 14 parts, MAX 8-12 actuated joints (2 arms + 2 legs); a "
    "quadruped ≈ torso + 4 legs(3 joints each) ≈ 13 parts, MAX ~16 joints; a manipulator ≤ 7 joints + an "
    "end-effector. (Open-ended animals — a snake, a many-armed octopod — may use more SERIAL segments, but "
    "still keep ONE coherent body and 2-3 joints per appendage.) Prefer the RIGHT detail, never padding.\n"
    "\nGEOMETRY (CRITICAL — get this right or the body collapses into a pole):\n"
    "• A child attaches at its parent's DISTAL TIP (z = parent.length) PLUS its mount_offset. So to put a "
    "limb at the BOTTOM of a torso of length L, use mount_offset z = -L (NOT -L/2). Shoulders sit near the "
    "top (small negative z); hips at the bottom (z ≈ -L).\n"
    "• Links extend along +z; mount_euler [3.14,0,0] flips a limb to hang/point DOWN (legs, hanging arms).\n"
    "• SPLAY limbs in x/y (e.g. left/right at ±y) so they don't overlap into one pipe. Free-floating movers "
    "use base_mount 'free'.\n"
    "• ATTACH every limb to the body: a limb's mount_offset x/y magnitude must be ≤ the parent's radius, so "
    "the limb root sits ON the parent's surface. A limb mounted FARTHER out than the parent is wide will "
    "float in empty space, detached — this is rejected. Keep limbs flush to the torso.\n"
    "\nPROPORTION & END-EFFECTOR (avoid nonsense parts):\n"
    "• Keep parts proportionate: a child link is usually SHORTER than its parent, and a welded end-part "
    "(hand/foot/fingertip) is SHORT (≤ ~0.08 m). NEVER make a fingertip or foot longer than the limb it sits on.\n"
    "• Pick ONE grasping style: EITHER articulated fingers (limbs off the palm) OR a parallel-jaw gripper "
    "(end_effector.kind 'gripper') — never both. Attach fingers/jaws to the PALM or last arm link, never to "
    "another fingertip.\n"
    "Output ONLY the JSON recipe."
)


def _llm_spec(prompt: str, reach_m, payload_kg, llm, *, robot_class: str | None = None,
              errors: list[str] | None = None) -> dict:
    """Ask the LLM to DESIGN a full morphology recipe over the general block menu (not a fixed template),
    via the schema-validated ``complete_json`` interface. ``compose_from_spec`` then realizes + validates
    it (proposer/verifier), so the LLM can build ANY kinematic tree — frog, hexapod, snake, arm, rover.
    ``errors`` (from a prior rejected attempt) drives the REPAIR loop; ``robot_class`` hints the intent."""
    user = f"Task: {prompt}\nreach_m={reach_m}, payload_kg={payload_kg}\n"
    if robot_class:
        user += f"Intended robot_class: {robot_class} (design a detailed body of this kind).\n"
    if errors:
        user += ("Your PREVIOUS design was REJECTED for these problems — return a corrected recipe that "
                 "fixes ALL of them:\n" + "\n".join(f"- {e}" for e in errors) + "\n")
    user += "Design the robot body."
    # generous budget: a multi-limb body (e.g. a hexapod = 6 limbs x several links) is large JSON and
    # truncates at the default 2048 tokens -> unparseable -> silent fallback to the fixed template.
    return llm.complete_json(_SPEC_SYSTEM, user, _SPEC_SCHEMA, max_tokens=6000)
