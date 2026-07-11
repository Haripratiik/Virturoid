"""General ANATOMY COMPILER — the intelligent, non-hard-coded path: the LLM describes a creature's anatomy as
a structured GRAPH (parts + roles + where each attaches + relative sizes + joints), and THIS single compiler
realizes it into connected, credible geometry. There is NO per-species code — a dog, a cat, a bird, a lizard,
a crab are all just different graphs over the SAME reusable role vocabulary.

Why a graph and not raw geometry: letting the LLM place every capsule's length/radius/offset/axis produced
noodles and scattered blocks (LLMs are bad at 3D coordinates). Letting the LLM choose only a class+count gave
generic flat-disc bodies. The reliable middle ground (report-8's "morphology graph"): the LLM owns the
STRUCTURE (which parts, their roles, semantic attachment, relative proportions, joint types); this compiler
owns the GEOMETRY (role->shape from a reusable library, attachment math, orientation solving via aim_euler,
flush connection). The LLM can't weld a finger to a foot because it never gives a coordinate.

Anatomy graph schema (one part per dict):
  {name, role, parent (null=root body), attach (semantic anchor on the parent),
   aim (semantic world direction the part extends), size (m, long axis), girth (m, optional cross-section),
   joint ("fixed"|"revolute"), symmetry ("none"|"left_right")}
Roles (reusable vocabulary): body, neck, head, snout, jaw, ear, eye, horn, tail, leg_upper, leg_lower, paw,
  foot, arm_upper, arm_lower, hand, wing, fin, flipper, claw, beak, antenna, shell, fang. Unknown -> a generic
  tapered limb. attach tokens: front/rear/back, left/right, top/bottom, mid, tip (combine: front_top,
  front_bottom, rear_top, tip). aim tokens: forward/back, up/down, left/right, out (combine: forward_up,
  down, back_up, down_out).
"""

from __future__ import annotations

import math
import re

import numpy as np

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.novel_anatomy import _loft, _revolve, _tapered


def _aim_R(direction):
    """Rotation whose local +z points along world ``direction`` with local +y kept horizontal (so a +y hinge
    bends in the vertical plane). Columns = (x, y, z) of the part's frame expressed in world coords."""
    z = np.array(direction, float)
    z /= (np.linalg.norm(z) or 1.0)
    up = np.array([0.0, 0.0, 1.0])
    y = np.cross(up, z)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= (np.linalg.norm(x) or 1.0)
    return np.column_stack([x, y, z])


def _R_to_euler(R):
    """xyz-intrinsic euler (the compiler's ``mount_euler`` convention) from a rotation matrix."""
    b = math.asin(max(-1.0, min(1.0, R[0, 2])))
    if abs(math.cos(b)) > 1e-6:
        return (math.atan2(-R[1, 2], R[2, 2]), b, math.atan2(-R[0, 1], R[0, 0]))
    return (math.atan2(R[2, 1], R[1, 1]), b, 0.0)

# Roles whose joint is revolute by default (the articulated appendages) — others weld unless the graph says so.
_ARTICULATED = {"leg_upper", "leg_lower", "arm_upper", "arm_lower", "tail", "wing", "fin", "flipper", "claw", "neck", "tentacle", "trunk"}
# Roles that hang/extend DOWN by default when the graph doesn't say (legs).
_DOWN_ROLES = {"leg_upper", "leg_lower", "paw", "foot"}


def _role_geometry(role: str, size: float, girth: float, aspect: str = "", *, authored: bool = False):
    """Map a role + characteristic size/girth to (geometry_dict, length_m, radius_m). Reusable across ALL
    creatures — the LLM never sees this; it only names the role. Unknown roles fall back to a tapered limb.
    ``aspect`` (body only) overrides the default sleek-barrel proportions for non-mammal body plans."""
    L = max(0.02, float(size))
    g = max(0.006, float(girth)) if girth else max(0.01, 0.18 * L)
    r = g  # radius_m carries the collider half-width / connection footprint
    if role == "body":
        # The torso barrel, built along +z=height with the loft's half_x = front-back and half_y = width.
        # DEFAULT ('long'): a sleek mammal barrel — width capped to <= 32% of length so an over-fattened girth
        # can't make a dog a blob. But many creatures are NOT sleek barrels, so an explicit ``aspect`` on the
        # body part unlocks the other plans (the LLM/graph opts in; the safe default is unchanged):
        #   wide  -> a broad carapace (crab/beetle): width can equal/exceed length, modest height.
        #   flat  -> a low disc (ray/turtle/flounder): wide AND shallow.
        #   round -> a bulbous sac (octopus mantle/pufferfish/spider abdomen): near-equal width & height.
        a = (aspect or "").lower()
        hl = L / 2.0

        def loft(hw, h):
            return _loft([(0.0, hw * 0.74, hl * 0.82), (0.40 * h, hw * 1.0, hl * 1.0),
                          (0.82 * h, hw * 0.92, hl * 0.90), (h, hw * 0.58, hl * 0.72)])
        if a in ("deck", "chassis", "flatbed"):
            # a flat machined DECK/CHASSIS (rover/AGV): a low rectangular slab — full length x width x a SHALLOW
            # height, chamfered — NOT a rounded loft. Wheels mount at its bottom corners. A single clean extrude
            # box (verified realize_shape bbox 0.8x0.34x0.05 for size 0.8) so the render is a true flat deck.
            hw = min(max(0.04, g), 0.95 * L) / 2.0
            h = max(0.05, 0.5 * hw)
            deck = {"family": "extrude", "height": round(h, 4), "fillet": round(0.16 * hw, 4),
                    "chamfer": round(0.1 * hw, 5),
                    "profile": [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]]}
            return deck, h, hw
        if a in ("wide", "flat"):
            # broad side-to-side (width can equal/exceed length) and LOW vertically — a carapace/disc, not a
            # tower. A shell stays a lofted organic form, not a box.
            hw = min(max(0.04, g), 1.35 * L) / 2.0
            h = max(0.04, (0.45 if a == "flat" else 0.8) * hw)
            return loft(hw, h), h, hw
        if a in ("round", "bulb", "sac"):
            hw = min(max(0.04, g), 0.95 * L) / 2.0
            h = max(0.04, 1.9 * hw)
            return loft(hw, h), h, hw
        # DEFAULT 'long' robot torso -> a CHASSIS as a multi-feature COMPOUND: a main structural shell PLUS a
        # raised top electronics/sensor deck (a real robot body is a shell with a mounted module bay, not one
        # plain block). The compound is ONE segment (no kinematic cost) but reads as several parts.
        hw = min(max(0.02, g), 0.32 * L) / 2.0
        h = max(0.04, 2.3 * hw)
        # a lighter fillet (was 0.45) so the machined CHAMFER that _apply_detail adds reads as crisp panel
        # edges, not a pillowy slab — matches the humanoid's chamfered chassis modules.
        main = {"family": "extrude", "height": round(h, 4), "fillet": round(0.2 * hw, 4),
                "profile": [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]]}
        deck = {"family": "extrude", "height": round(0.42 * h, 4), "fillet": round(0.16 * hw, 4),
                "chamfer": round(0.12 * hw, 5),
                "profile": [[-0.55 * hl, -0.62 * hw], [0.55 * hl, -0.62 * hw],
                            [0.55 * hl, 0.62 * hw], [-0.55 * hl, 0.62 * hw]],
                "at": (0.0, 0.0, round(0.8 * h, 4))}
        return ({"family": "compound", "parts": [main, deck]}, h, hw)
    if role in ("head", "sensor_head"):
        # a mechanical SENSOR-HEAD module — a rounded box with a flat front face for cameras — NOT an organic
        # skull. This is a ROBOT (think Spot / a humanoid sensor head). Extruded along +z (aimed forward).
        hw = max(0.02, 0.92 * g)
        return ({"family": "extrude", "height": round(L, 4), "fillet": round(0.4 * hw, 4),
                 "chamfer": round(0.3 * hw, 5),   # crisp machined bevel -> a sensor head, not a soft blob
                 "profile": [[-hw, -hw], [hw, -0.85 * hw], [hw, 0.85 * hw], [-hw, hw]]}, L, hw)
    if role == "wheel":
        # T3: a real ROLLING wheel — a flat cylinder. ``size`` = wheel diameter-ish (radius = 0.5*size),
        # ``girth`` = tread WIDTH (thin). Physics is a cylinder (shape='cylinder'); the segment loop lays the
        # axle LATERAL so it rolls. length_m carries the width (the cylinder's axis), radius_m the roll radius.
        rad = max(0.03, 0.5 * L)
        width = max(0.02, min(g if girth else 0.35 * rad, 0.7 * rad))
        return _tapered(width, rad, rad), width, rad          # uniform-radius cylinder of length=width
    if role in ("beak", "horn", "fang", "claw"):
        return _tapered(L, g, max(0.004, 0.28 * g)), L, g     # pointed (beak/horn/fang/claw)
    if role in ("snout", "muzzle"):
        # a robot "snout" reads as a forward SENSOR housing (a short blunt box), not an animal nose
        s = max(0.012, 0.7 * g)
        return ({"family": "extrude", "height": round(L, 4), "fillet": round(0.3 * s, 4),
                 "profile": [[-s, -s], [s, -0.8 * s], [s, 0.8 * s], [-s, s]]}, L, s)
    if role in ("ear", "antenna"):
        return _tapered(L, g, max(0.003, 0.22 * g)), L, g
    if role in ("eye",):
        return _revolve([(g, 0), (g, 0.5 * L), (0.5 * g, L)]), L, g
    if role in ("neck",):
        return _tapered(L, g * 1.05, g * 0.92), L, g
    if role in ("tail",):
        return _tapered(L, g, max(0.005, 0.35 * g)), L, g
    if role == "foot":
        # a flat foot PAD: thin vertically, wider than the leg, length along +z (re-aimed FORWARD at the
        # ankle) — the robot stands on real feet flat on the floor, not on a downward-poking stub.
        # The 1.6x/1.9x widening is calibrated for AUTO-feet that inherit the slender LEG's girth (~0.02 ->
        # a sensible 0.04 pad). An EXPLICITLY-AUTHORED foot part carries the MODEL's own dims — multiplying
        # those double-scales (measured: an LLM foot girth 0.10 realized as a 0.19-radius drum that swallowed
        # the leg and CROUCHed the walk). Respect authored dims (floors only) — the north star: no silent
        # rewrite of the model's numbers; buildability floors are the only clamp.
        if authored:
            Lf = max(0.05, L)
            w = max(0.02, g)
            th = max(0.010, 0.35 * min(g, 0.06))
        else:
            Lf = max(0.05, 1.6 * L)
            w = max(0.03, g * 1.9)
            th = max(0.012, 0.5 * g)
        return ({"family": "extrude", "height": round(Lf, 4), "fillet": round(0.3 * th, 4),
                 "profile": [[-th, -0.5 * w], [th, -0.45 * w], [th, 0.45 * w], [-th, 0.5 * w]]}, Lf, w)
    if role in ("paw", "hand"):
        return _tapered(L, g, g * 0.85), L, g
    if role in ("wing", "fin", "flipper", "shell"):
        # a flat plate: span = length (local +z, aimed along the part's 'out' direction), chord ~= girth
        # (local y), and a THIN vertical thickness (local x) so it reads as a wing/fin, not a fat slab.
        w = g
        th = max(0.004, 0.09 * w)
        return ({"family": "extrude", "height": round(L, 4), "fillet": round(0.12 * w, 4),
                 "profile": [[-th, -0.5 * w], [th, -0.46 * w], [th, 0.46 * w], [-th, 0.5 * w]]}, L, w)
    if role in ("tentacle", "trunk", "pseudopod"):
        # A continuum-like appendage is represented honestly as an articulated
        # tapered rigid chain.  It is not a walking leg: it keeps its requested
        # segment count, can curl, and never receives a welded foot pad.
        return _tapered(L, g, max(0.004, 0.55 * g)), L, g
    if role in ("leg", "arm", "leg_upper", "leg_lower", "arm_upper", "arm_lower", "limb"):
        # a STRAIGHT mechanical link — a flat beam / bracket of constant cross-section (a robot limb segment,
        # think Spot/Go1 leg plates), NOT an organic tapered spindle. Deeper fore-aft (local x), flatter
        # laterally (local y); the real motor housing (show_actuators) sits coaxial at the proximal joint, and
        # the per-segment girth taper down the chain already slims distal links. Extruded along +z (link axis).
        depth = max(0.01, g)
        width = max(0.008, 0.6 * g)
        return ({"family": "extrude", "height": round(L, 4), "fillet": round(0.5 * width, 4),
                 "profile": [[-depth, -width], [0.9 * depth, -width], [0.9 * depth, width], [-depth, width]]},
                L, g)
    # Open-world graph vocabulary: an LLM may name a mechanical module "joint",
    # "battery_pod", or a future concept we have never enumerated. It is still a
    # connected physical part, so realize it as a conservative tapered module
    # instead of rejecting the entire novel body for a missing display role.
    return _tapered(L, g, max(0.004, 0.65 * g)), L, g


def _physics_proxy_from_geometry(geometry: dict | None, fallback_shape: str, fallback_radius: float) -> tuple[str, tuple[float, float] | None]:
    """Select the primitive collider that actually represents a shape program.

    We deliberately keep the fast/MJX-safe primitive vocabulary, but no longer
    discard authored cross-sections: an extrude/loft becomes a box with its real
    x/y bounds, while tapered/revolved forms retain the conservative round
    proxy.  The optional mesh path can add higher visual fidelity; this function
    is the load-bearing physics/export representation used everywhere.
    """
    if not isinstance(geometry, dict):
        return fallback_shape, None
    family = str(geometry.get("family") or "").lower()
    if family == "extrude":
        profile = geometry.get("profile") or []
        points = [(float(p[0]), float(p[1])) for p in profile
                  if isinstance(p, (list, tuple)) and len(p) >= 2]
        if points:
            hx = max(abs(x) for x, _ in points)
            hy = max(abs(y) for _, y in points)
            if hx > 1e-6 and hy > 1e-6:
                return "box", (hx, hy)
    if family == "loft":
        sections = geometry.get("sections") or []
        # Loft sections are [z, half_y, half_x].  A conservative box proxy
        # uses their maximum radial extent rather than a generic square limb.
        valid = [s for s in sections if isinstance(s, (list, tuple)) and len(s) >= 3]
        if valid:
            hx = max(abs(float(s[2])) for s in valid)
            hy = max(abs(float(s[1])) for s in valid)
            if hx > 1e-6 and hy > 1e-6:
                return "box", (hx, hy)
    return fallback_shape, None


def _anchor_on_body(attach: str, half_len: float, half_w: float, height: float, edge: float = 0.6) -> tuple:
    """Semantic attach anchor -> a point in the BODY's local frame (z in [0,height], x=front-back, y=width).

    Anchors sit INSIDE the body's tapered loft so the part OVERLAPS the torso and never floats: x is kept to
    ~0.6*half_len (the loft tapers toward its ends, so 0.9 would stick out past the surface -> a visible gap),
    and limbs root well inside the body (legs into the lower body and extend out the bottom; neck/tail into the
    upper body) — the overlap hides the joint and guarantees a connected silhouette. front=+x, rear=-x."""
    a = (attach or "mid").lower()
    # Five longitudinal anchors front->rear so a many-legged creature (spider/crab/insect = 4 pairs) can
    # spread its limb pairs along the body instead of colliding at one spot. ``front_mid``/``rear_mid`` sit
    # between the ends; check them BEFORE the bare front/rear so the substring match doesn't swallow them.
    # ``edge`` scales how close to the body END an appendage roots: a CHASSIS box has flat surfaces at full
    # half_len (so parts must root near the edge ~0.8 to emerge, not bury inside); a tapered loft shell tapers
    # inward, so its parts root further in (~0.6) to stay overlapping the surface.
    if "front_mid" in a:
        x = 0.5 * edge * half_len
    elif "rear_mid" in a or "back_mid" in a:
        x = -0.5 * edge * half_len
    elif "front" in a:
        x = edge * half_len
    elif "rear" in a or "back" in a:
        x = -edge * half_len
    else:
        x = 0.0
    # legs root INSIDE the lower body (0.40h) and extend down out the belly; top parts root inside the upper
    # body (0.70h); mid otherwise. Rooting inside (not at the surface) is what removes the detached-piece gaps.
    z = (0.70 * height if "top" in a else 0.40 * height if "bottom" in a else 0.5 * height)
    y = (0.82 * half_w if "left" in a else -0.82 * half_w if "right" in a else 0.0)
    return (x, y, z)


_AIM_VEC = {
    "forward": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0), "rear": (-1.0, 0.0, 0.0),
    "up": (0.0, 0.0, 1.0), "down": (0.0, 0.0, -1.0),
    "forward_up": (0.78, 0.0, 0.6), "forward_down": (0.85, 0.0, -0.45),
    "back_up": (-0.78, 0.0, 0.6), "back_down": (-0.72, 0.0, -0.7),
    "down_out": (0.0, 0.5, -0.86), "out": (0.0, 1.0, 0.0),
    # diagonal-lateral aims: fan limbs forward/back AND out at once (radial spider/crab/insect legs). The +y
    # component is mirrored per side by the symmetry logic, so 'forward_out' splays the front pair forward
    # on BOTH sides; 'back_down_out' is the classic sprawled rear arthropod leg.
    "forward_out": (0.66, 0.66, 0.0), "back_out": (-0.66, 0.66, 0.0),
    "forward_down_out": (0.5, 0.5, -0.7), "back_down_out": (-0.5, 0.5, -0.7),
    # RAISED lateral aims — wings/fins that rise up-and-out (a dragon/bird at rest, a sail fin) rather than
    # jutting flat sideways. Mirrored per side like the others.
    "up_out": (0.0, 0.7, 0.72), "back_up_out": (-0.5, 0.62, 0.62), "forward_up_out": (0.5, 0.62, 0.62),
}


def _aim_dir(aim: str, role: str) -> tuple:
    a = (aim or "").lower()
    if a in _AIM_VEC:
        return _AIM_VEC[a]
    if role in _DOWN_ROLES:
        return (0.0, 0.0, -1.0)
    return (1.0, 0.0, 0.0)


_GRAV = 9.81

# Per-ROLE material tier (the render colours each part by this; bom_builder refines it per TASK). The default
# load path is 'skeleton' (a strong structural metal); the outer body/head is a coloured 'shell'; contact parts
# (feet/hands/grippers) are 'metal'; lifting surfaces (wings/fins) are light 'carbon_fiber'.
_ROLE_MATERIAL = {
    "body": "shell", "head": "shell", "sensor_head": "shell", "snout": "shell", "muzzle": "shell",
    "foot": "metal", "paw": "metal", "hand": "metal", "claw": "metal", "fang": "metal", "beak": "metal",
    "horn": "metal", "eye": "metal",
    "wing": "carbon_fiber", "fin": "carbon_fiber", "flipper": "carbon_fiber",
    "leg": "skeleton", "arm": "skeleton", "leg_upper": "skeleton", "leg_lower": "skeleton",
    "arm_upper": "skeleton", "arm_lower": "skeleton", "limb": "skeleton",
    "neck": "frame", "tail": "frame", "antenna": "frame", "ear": "frame", "shell": "shell",
}


def _role_material(role: str) -> str:
    return _ROLE_MATERIAL.get((role or "").lower(), "skeleton")


def _apply_detail(geo, detail: str, length_m: float, girth: float, chamfer: float = 0.0):
    """Post-process a part's geometry spec with MECHANICAL detail (visual-only, physics untouched): bevelled
    (chamfered) edges for a machined look, and vent / lightening cutouts for an industrial read. The agents
    choose the level per robot + task — a sleek drone part is 'smooth'; a rugged loader part is 'vented'."""
    if not isinstance(geo, dict):
        return geo
    if geo.get("family") == "compound":               # detail the MAIN (first) sub-part of a compound body
        parts = list(geo.get("parts", []))
        if parts:
            parts[0] = _apply_detail(parts[0], detail, length_m, girth, chamfer)
            return {**geo, "parts": parts}
        return geo
    d = (detail or "").lower()
    geo = dict(geo)
    cham = chamfer if chamfer else (0.16 * girth if d in ("paneled", "vented", "mechanical", "industrial", "rugged") else 0.0)
    if cham > 0:
        geo["chamfer"] = round(cham, 5)
    if d in ("vented", "industrial", "rugged"):
        cuts = list(geo.get("cutouts", []))
        prof = geo.get("profile") if geo.get("family") == "extrude" else None
        if prof:                                          # a beam/chassis: a ROW of side vents along its LONG
            hx = max((abs(p[0]) for p in prof), default=girth)   # in-plane axis, bored through the SHORT one
            hy = max((abs(p[1]) for p in prof), default=girth)
            n = max(2, min(5, int(2 * max(hx, hy) / (1.7 * max(min(hx, hy), 1e-3)))))
            if hx >= hy:                                  # long along x -> vents stepped in x, bored along y
                r = max(0.004, min(0.3 * hy, 0.5 * (1.4 * hx) / n))
                for i in range(n):
                    cuts.append([round(-0.68 * hx + 1.36 * hx * (i + 0.5) / n, 4), 0.0,
                                 round(0.5 * length_m, 4), round(r, 4), round(2.4 * hy, 4), "y"])
            else:                                         # long along y -> stepped in y, bored along x
                r = max(0.004, min(0.3 * hx, 0.5 * (1.4 * hy) / n))
                for i in range(n):
                    cuts.append([0.0, round(-0.68 * hy + 1.36 * hy * (i + 0.5) / n, 4),
                                 round(0.5 * length_m, 4), round(r, 4), round(2.4 * hx, 4), "x"])
        else:                                             # non-extrude part: vents along its length (z), thru y
            n = max(2, min(6, int(length_m / max(1e-3, 2.6 * girth))))
            r = max(0.004, 0.2 * girth)
            for i in range(n):
                cuts.append([0.0, 0.0, round((i + 0.6) * length_m / (n + 0.2), 4), round(r, 4),
                             round(3.0 * girth, 4), "y"])
        geo["cutouts"] = cuts
    return geo


def _size_actuator_torques(segs: list[GeneSegment]) -> None:
    """Size EACH joint's actuator to the REAL load it bears, in place — not a flat default. A joint must hold
    everything DISTAL to it: the static torque is Σ(distal mass × lever arm to that mass along the chain). A
    weight-bearing LEG joint additionally carries a share of the whole robot's weight at its distal reach
    (ground reaction). A dynamic factor covers acceleration beyond a static hold. Result: a hip/shoulder that
    swings a long heavy limb is sized far higher than a tail-tip or ankle — so the BOM picks differentiated,
    real motors per joint instead of one motor everywhere."""
    by = {s.name: s for s in segs}
    kids: dict = {}
    for s in segs:
        kids.setdefault(s.parent, []).append(s.name)

    def subtree(root):
        out, st = [], [root]
        while st:
            n = st.pop()
            out.append(n)
            st += kids.get(n, [])
        return out

    def static_torque(root):                       # hold the subtree-at-root horizontally off this joint
        t, st = 0.0, [(root, 0.0)]
        while st:
            n, d0 = st.pop()
            s = by[n]
            t += s.mass_kg * (d0 + 0.5 * s.length_m)
            for c in kids.get(n, []):
                st.append((c, d0 + s.length_m))
        return _GRAV * t

    def reaches_foot(root):                         # a weight-bearing leg ends in a welded foot pad (extrude)
        return any((by[n].geometry or {}).get("family") == "extrude" and by[n].joint_type is None
                   for n in subtree(root))

    total = sum(s.mass_kg for s in segs)
    root = next((s.name for s in segs if s.parent is None), None)
    legs = [s.name for s in segs if s.parent == root and s.joint_type and reaches_foot(s.name)]
    n_legs = max(1, len(legs))
    for s in segs:
        if s.joint_type not in ("revolute", "prismatic"):
            continue
        tau = static_torque(s.name)
        leg = reaches_foot(s.name)
        if leg:                                     # + share of body weight, at this joint's distal reach
            reach = sum(by[n].length_m for n in subtree(s.name))
            tau = max(tau, _GRAV * (total / n_legs) * reach)
        s.actuator_torque_nm = round(max(1.0, tau * (2.2 if leg else 1.3)), 2)   # dynamic factor over hold


def build_from_anatomy(graph: dict) -> RobotGene:
    """Realize an anatomy graph into a connected, credible RobotGene. Parent-before-child order assumed;
    a root part (parent null / role 'body') is required. ``symmetry: left_right`` mirrors a part to ±y."""
    parts = list(graph.get("parts") or [])
    robot_class = str(graph.get("robot_class") or "quadruped")
    base_mount = str(graph.get("base_mount") or "free").lower()
    if base_mount not in {"table", "floor", "free"}:
        base_mount = "free"
    by_name = {p.get("name"): p for p in parts if p.get("name")}
    # root = explicit null-parent, else the first 'body' role, else the first part.
    root = next((p for p in parts if not p.get("parent")), None) or \
        next((p for p in parts if p.get("role") == "body"), None) or (parts[0] if parts else None)
    if root is None:
        raise ValueError("anatomy graph has no parts")

    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}
    dims: dict[str, dict] = {}            # name -> {half_len, half_w, height, length_m, role}

    # --- ROOT BODY ---
    b_size = float(root.get("size") or 0.5)
    b_girth = float(root.get("girth") or 0.42 * b_size)
    geo, blen, brad = _role_geometry("body", b_size, b_girth, aspect=str(root.get("aspect") or ""))
    try:
        # a robot body looks MACHINED by default (chamfered panels + vents), not a plain block; the LLM can
        # override per-part with detail 'smooth' for a sleek look.
        geo = _apply_detail(geo, str(root.get("detail") or "vented"), blen, b_girth, float(root.get("chamfer") or 0.0))
    except (TypeError, ValueError):
        pass
    if isinstance(root.get("geometry"), dict):
        geo = root["geometry"]
    # half_w MUST be the ACTUAL realized half-width (brad), not b_girth/2: the width cap / aspect can make the
    # real loft narrower or wider than the authored girth, and mirrored limbs are anchored off half_w. Using the
    # authored girth here (the old bug) floated appendages off a capped body — the detach fragility every exotic
    # creature hit in the stress test.
    half_len, half_w, height = b_size / 2.0, brad, blen
    rname = root.get("name") or "body"
    root_aspect = str(root.get("aspect") or "").lower()
    # A round mantle/abdomen must be round in the model that physics and the
    # URDF exporter consume, not only in the optional CAD description.
    root_shape = "sphere" if root_aspect in ("round", "bulb", "sac") else "box"
    proxy_shape, proxy_cs = _physics_proxy_from_geometry(geo, root_shape, brad)
    # A declared round mantle/abdomen is intentionally a sphere; do not turn
    # its supporting loft description back into a rectangular proxy.
    if root_shape != "sphere":
        root_shape = proxy_shape
    # The body plan's front/back half-length used to disappear here: every
    # torso compiled as a square vertical post.  A box torso now carries the
    # same x/y envelope that anchors and the geometry program use.
    root_cs = proxy_cs or (half_len, brad)
    segs.append(GeneSegment(name=rname, parent=None, shape=root_shape, length_m=blen,
                            radius_m=max(brad, 0.06), mass_kg=max(0.4, 6.0 * (b_size * b_girth) ** 0.5),
                            joint_type=None, cross_section=(root_cs if root_shape == "box" else None),
                            geometry=geo, material="shell"))
    dims[rname] = {"half_len": half_len, "half_w": half_w, "height": height, "length_m": blen, "role": "body"}

    # leaf registry: (part_name, side) -> the segment a CHILD should attach to (the tip of that part's chain
    # on that side). side is "" (unmirrored), "l", or "r". The body is registered unmirrored. This is what lets
    # a symmetric child (foot) attach to the correct side of its symmetric parent (leg) — the bug a naive
    # _l/_r split caused: a foot referencing 'front_leg_pair' must resolve to 'front_leg_pair_l_2' on the left.
    leaf: dict = {(rname, ""): rname}
    # accumulated WORLD rotation per segment, so a child's aim (a world direction) can be solved relative to
    # its parent's frame. The root body sits axis-aligned at rest -> identity.
    world_R: dict = {rname: np.eye(3)}
    # a CHASSIS box has flat end faces at full half_len, so appendages must root NEAR the edge to emerge; a
    # tapered loft shell tapers inward, so they root further in. (Fixes the tail/neck burying in the chassis.)
    body_edge = 0.82 if root_shape == "box" else 0.6

    def _emit_chain(part, base_name, side):
        sign_y = {"l": 1.0, "r": -1.0, "": 0.0}[side]
        pname = part.get("parent") or rname
        # resolve the parent segment to attach to, for THIS side (same-side symmetric parent first, then a
        # shared/unmirrored parent like the body).
        if (pname, side) in leaf:
            parent_seg = leaf[(pname, side)]
        elif (pname, "") in leaf:
            parent_seg = leaf[(pname, "")]
        else:
            return  # parent not built (bad graph order) -> skip this part rather than crash
        body_attached = parent_seg == rname
        role = str(part.get("role") or "limb").lower()
        size = float(part.get("size") or 0.12)
        girth = float(part.get("girth") or 0) or 0.18 * size
        girth *= max(0.3, min(3.0, float(part.get("thickness") or 1.0)))   # per-part THICKNESS knob (load-bearing)
        n = max(1, min(8, int(part.get("segments") or 1)))
        # PROPORTION caps vs the REALIZED torso (blen = trunk length, brad = trunk half-width): the fidelity
        # render — once the styling pass removed the palette noise — showed a head capsule rivaling the whole
        # trunk (a "second body") and connective parts as fat as a limb. Those are the LLM's free sizing running
        # past what reads as a real machine. Code owns these bands (like the leg-slenderness cap); the LLM sizes
        # WITHIN them. Caps only ever shrink (min), so a well-proportioned part is untouched.
        if role in ("head", "sensor_head", "snout", "muzzle", "skull", "face"):
            size = min(size, 0.50 * blen)     # a head must be clearly shorter than the trunk
            girth = min(girth, 0.55 * brad)   # ...and narrower — reads as a sensor head, not a second body
        elif role in ("neck", "tail", "antenna", "ear", "mast", "horn"):
            girth = min(girth, 0.45 * brad)   # slim connective appendages, never torso-fat
        if role == "leg" and (graph.get("robot_class") or "").lower() in ("quadruped", "legged"):
            # G6/G7 morphology-prior NORMALIZATION at the compiler choke point (applies to LLM graphs AND the
            # generic fallback alike): a walking leg needs >=3 ACTUATED joints (Go2 = 3/leg) — with the terminal
            # segment welded as the foot that means >=4 segments — and SLENDER segments (aspect >=~2.5, not the
            # 1:1 sausage stubs the e2e fidelity test shipped). The LLM chooses the leg's overall size; code owns
            # the structural minima, exactly like dimension_priors for scenes.
            n = max(4, n)
            # The slenderness BAND is the invariant (length/diameter >= ~2.2); the per-part thickness knob
            # varies girth WITHIN it. An explicit thickness may not push a walking leg back into 1:1 stubs.
            girth = min(girth, 0.055 * size)
        explicit_joint = str(part.get("joint") or "").lower()
        is_limb = role in ("leg", "arm", "leg_upper", "arm_upper")
        detail = str(part.get("detail") or "")            # mechanical DETAIL: smooth | paneled | vented | rugged
        if not detail and role in ("leg", "arm", "leg_upper", "arm_upper", "leg_lower", "arm_lower"):
            detail = "paneled"                            # limbs read as machined links (chamfered) by default
        try:
            part_chamfer = float(part.get("chamfer") or 0.0)
        except (TypeError, ValueError):
            part_chamfer = 0.0
        try:
            curl = float(part.get("curl") or 0.0)             # total rest-bend (rad) spread across the chain
        except (TypeError, ValueError):
            curl = 0.0
        seg_len = size / n if n > 1 else size
        prev = parent_seg
        prev_world_R = world_R.get(parent_seg, np.eye(3))
        for i in range(n):
            last = i == n - 1
            is_foot = role == "leg" and last and n > 1     # only a walking leg receives a foot
            seg_role = ("foot" if is_foot else role)
            is_wheel = seg_role == "wheel"
            g_i = girth * (0.82 ** i)
            # authored=True when the role "foot" came from the PART itself (the model authored this foot's dims);
            # an auto-foot promoted from a leg's last segment keeps the leg-girth-calibrated widening.
            geo, length_m, radius_m = _role_geometry(seg_role, seg_len * (1.0 if not last or n == 1 else 0.6), g_i,
                                                     authored=(seg_role == role))
            geo = _apply_detail(geo, detail, length_m, g_i, part_chamfer)
            # T4 shape-programs: a part may AUTHOR its own visual geometry (extrude/revolve/tapered/loft +
            # fillet/chamfer/cutouts) — the mesh layer realizes it (cad_geometry.realize_shape, safe capsule
            # fallback). Applies to single-segment parts; the primitive COLLIDER (size/girth) is unchanged, so
            # physics is untouched and only the rendered shape changes (a dome sensor, a bracket, a nozzle).
            _cg = part.get("geometry")
            if isinstance(_cg, dict) and n == 1:
                geo = _cg
            if is_wheel:
                joint_type = "revolute"                            # a wheel is always a (continuous) hinge
            elif explicit_joint == "fixed" or (last and is_limb and n > 1):
                joint_type = None                                  # welded paw / explicitly-fixed part
            elif explicit_joint == "revolute" or role in _ARTICULATED or is_limb:
                joint_type = "revolute"
            else:
                joint_type = None
            seg_name = f"{base_name}_{i}" if n > 1 else base_name
            if i == 0:
                # FIRST segment of the part: AIM it. 'aim' is a WORLD direction; honor it whether the part
                # mounts on the body OR on another part's tip — composing with the parent's accumulated world
                # rotation. This levels a head off an up-angled neck and points ears UP off the head, instead of
                # blindly continuing the parent's direction (the old non-body branch ignored aim entirely).
                d = _aim_dir(part.get("aim"), role)
                if sign_y:                       # mirrored pair: lateral aim follows the side + slight splay
                    aim_world = (d[0], abs(d[1]) * sign_y + sign_y * 0.05, d[2])
                else:
                    aim_world = d
                R_des = _aim_R(aim_world)
                if is_wheel:                     # T3: lay the cylinder axle LATERAL (local z -> world y) so it rolls
                    R_des = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
                mount_euler = _R_to_euler(prev_world_R.T @ R_des)
                seg_world_R = R_des
                if body_attached:
                    pdim = dims[rname]
                    ax, ay, az = _anchor_on_body(part.get("attach", "mid"), pdim["half_len"], pdim["half_w"],
                                                 pdim["height"], edge=body_edge)
                    if is_wheel:
                        # a WHEEL sits OUTSIDE the chassis side (ay past the half-width) at the SAME bottom height
                        # the legs use (az - length_m), so it hangs at the bottom corner like a real wheel.
                        wr = 0.5 * float(part.get("size") or 0.12)
                        ay_w = sign_y * (abs(pdim["half_w"]) + 0.5 * wr) if sign_y else ay
                        mount_offset = (ax, ay_w, az - pdim["length_m"])
                    else:
                        ay = sign_y * abs(pdim["half_w"]) * 0.9 if sign_y else ay
                        mount_offset = (ax, ay, az - pdim["length_m"])
                else:
                    mount_offset = (0.0, 0.0, 0.0)   # attach at the parent segment's tip
            elif is_foot:
                # an ANKLE: turn the foot flat-FORWARD (horizontal) so a real foot PAD lies on the floor and
                # gives ground contact, instead of a downward stub that pokes through the surface.
                R_des = _aim_R((1.0, 0.0, 0.0))
                mount_euler = _R_to_euler(prev_world_R.T @ R_des)
                seg_world_R = R_des
                mount_offset = (0.0, 0.0, 0.0)
            else:                                    # serial continuation -> straight along the chain
                mount_offset, mount_euler = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                seg_world_R = prev_world_R
            # REAL WALKING-LEG JOINT AXES (the anatomy compiler emitted local (0,1,0) for every leg joint, which —
            # given how legs are aimed down-and-out — makes ALL of them swing the foot LATERALLY (measured: dx~0,
            # dy=+0.09); the leg could only paddle sideways, never STRIDE, and had no lateral-balance authority, so
            # the body rolled over during a trot (roll -8deg -> -81deg). Set axes in WORLD terms like the working
            # composed quad: seg 0 = ABDUCTION about world-x (lateral, tight range, balance authority); seg 1.. =
            # HIP + KNEE about world-y (fore-aft STRIDE). Converted into each segment's local frame via seg_world_R
            # so it's correct for any leg aim. Only a real walking leg (role 'leg', >=3 actuated -> n>=4 w/ foot).
            _ja, _jlo, _jhi = (0.0, 1.0, 0.0), -2.6, 2.6
            if is_wheel:                                        # spin about the axle (local z), UNLIMITED (rolls)
                _ja, _jlo, _jhi = (0.0, 0.0, 1.0), None, None
            elif role == "leg" and joint_type == "revolute" and n >= 4:
                _w = np.array([1.0, 0.0, 0.0]) if i == 0 else np.array([0.0, 1.0, 0.0])  # abduction vs stride axis
                _loc = seg_world_R.T @ _w
                _nrm = float(np.linalg.norm(_loc)) or 1.0
                _ja = (float(_loc[0] / _nrm), float(_loc[1] / _nrm), float(_loc[2] / _nrm))
                if i == 0:
                    _jlo, _jhi = -0.6, 0.6                              # abduction range (Go2-class ~+-0.5 rad)
            # The crawl-gait controller is calibrated against the deliberately
            # conservative capsule chain used by the generated walking-leg
            # family.  Preserve that *dynamics* contract for automatic legs;
            # an explicitly authored leg shape still becomes its requested
            # proxy.  Other generated/external extrudes do drive collision.
            if role == "leg" and not isinstance(part.get("geometry"), dict):
                proxy_shape, proxy_cs = ("capsule", None)
            else:
                proxy_shape, proxy_cs = _physics_proxy_from_geometry(
                    geo, "cylinder" if is_wheel else "capsule", radius_m,
                )
            segs.append(GeneSegment(
                name=seg_name, parent=prev, shape=proxy_shape,
                length_m=length_m, radius_m=max(0.006, radius_m),
                mass_kg=max(0.02, 1.2 * length_m * radius_m * 30), joint_type=joint_type, joint_axis=_ja,
                joint_lower=_jlo if joint_type else None, joint_upper=_jhi if joint_type else None,
                mount_offset=mount_offset, mount_euler=mount_euler,
                actuator_torque_nm=14.0 if joint_type else None, geometry=geo,
                cross_section=proxy_cs,
                material=_role_material(seg_role)))
            dims[seg_name] = {"half_len": length_m / 2, "half_w": radius_m, "height": length_m,
                              "length_m": length_m, "role": seg_role}
            world_R[seg_name] = seg_world_R
            if joint_type:
                # rest pose: a slight knee bend on legs (so the body stands TALL, not squatting); a CURL spread
                # across the internal joints of a multi-segment part (scorpion tail arching over, a curved neck/
                # trunk) when the graph asks for it; otherwise straight at rest.
                if is_limb and i == 1:
                    pose[f"{seg_name}_joint"] = 0.12
                elif curl and n > 1 and i >= 1:
                    pose[f"{seg_name}_joint"] = curl / (n - 1)
                else:
                    pose[f"{seg_name}_joint"] = 0.0
            prev = seg_name
            prev_world_R = seg_world_R
        # register this part's chain tip under its ORIGINAL name + side, so a child (e.g. a foot) referencing
        # the part by name resolves to the correct side's leaf segment.
        leaf[(part.get("name") or base_name, side)] = prev

    for part in parts:
        if part is root:
            continue
        nm = part.get("name") or part.get("role") or "part"
        if str(part.get("symmetry") or "none").lower() == "left_right":
            _emit_chain(part, f"{nm}_l", "l")
            _emit_chain(part, f"{nm}_r", "r")
        else:
            _emit_chain(part, nm, "")

    # The LLM selects a TASK TOOL in the graph; this generic compiler attaches
    # its physical mechanism at the nominated distal part. Tool choice remains
    # semantic/open-world, while the collision-safe fingers are grounding.
    ee_spec = graph.get("end_effector") if isinstance(graph.get("end_effector"), dict) else {}
    ee_kind = str(ee_spec.get("kind") or "none").lower()
    ee_parent = str(ee_spec.get("parent") or "")
    ee_target = (leaf.get((ee_parent, "")) if ee_parent else None) or \
        next((s.name for s in reversed(segs) if s.parent is not None), segs[-1].name)
    end_effector_type = "none"
    if ee_kind in {"gripper", "hand"}:
        span = max(0.03, min(0.30, float(ee_spec.get("span") or 0.10)))
        fingers = 2 if ee_kind == "gripper" else max(3, min(5, int(ee_spec.get("fingers") or 3)))
        for s in segs:
            if s.name == ee_target:
                s.is_end_effector = True
        if fingers == 2:
            positions = [(span / 2, (0.0, -1.0, 0.0)), (-span / 2, (0.0, 1.0, 0.0))]
        else:
            positions = [
                (span * math.cos(2 * math.pi * i / fingers) / 2,
                 span * math.sin(2 * math.pi * i / fingers) / 2)
                for i in range(fingers)
            ]
            positions = [((x, y), (-x, -y, 0.0)) for x, y in positions]
        for i, item in enumerate(positions):
            if fingers == 2:
                offset, axis = item
                mount = (0.0, offset, 0.0)
            else:
                mount, axis = item
            ax_norm = math.sqrt(sum(float(v) ** 2 for v in axis)) or 1.0
            axis = tuple(float(v) / ax_norm for v in axis)
            segs.append(GeneSegment(
                name=f"tool_finger_{i}", parent=ee_target, shape="box", length_m=0.06,
                radius_m=0.012, mass_kg=0.03, joint_type="prismatic", joint_axis=axis,
                joint_lower=0.0, joint_upper=0.045, mount_offset=mount,
                actuator_torque_nm=8.0, material="metal"))
        end_effector_type = "gripper"
    elif ee_kind in {"suction", "spray", "hook"}:
        for s in segs:
            s.is_end_effector = s.name == ee_target
        end_effector_type = {"suction": "suction", "spray": "spray_nozzle", "hook": "hook"}[ee_kind]

    # Exactly one end effector is required by the schema: absent a task tool,
    # mark a distal structural part without pretending it can grasp.
    if not any(s.is_end_effector for s in segs):
        ee = next((s for s in reversed(segs) if s.parent is not None), segs[-1])
        ee.is_end_effector = True

    _size_actuator_torques(segs)   # size each joint's actuator to the REAL load it bears (not a flat default)
    mount_bounds = {
        name: {"x": round(float(d.get("half_len", 0.0)), 5),
               "y": round(float(d.get("half_w", 0.0)), 5)}
        for name, d in dims.items()
    }
    metadata = {"anatomy_graph": True, "mount_bounds": mount_bounds}
    if pose:
        metadata["rest_pose"] = pose
    # STRUCTURAL identity: append a short deterministic signature of the actual morphology so bodies that DIFFER
    # (a horse's long legs vs a dog's) get DISTINCT ids — otherwise every anatomy quad collapsed to
    # 'anatomy_creature', banking to ONE skill (keep-best) so the flywheel/transfer-ledger couldn't tell them
    # apart or compound across them. This is deterministic GROUNDING (identity from structure), not a design choice.
    import hashlib
    _n = len(segs)
    _dof = sum(1 for s in segs if s.joint_type in ("revolute", "prismatic"))
    _tl = round(sum(s.length_m for s in segs), 3)          # captures the P4 leg-length diversity
    _sig = hashlib.md5(f"{robot_class}:{_n}:{_dof}:{_tl}".encode("utf-8")).hexdigest()[:8]
    return RobotGene(id="anatomy_" + (graph.get("name") or robot_class) + "_" + _sig,
                     species=f"{robot_class}.anatomy",
                     robot_class=robot_class, segments=segs, base_mount=base_mount,
                     end_effector_type=end_effector_type, design_source="anatomy_graph",
                     composition_notes=["Compiled directly from the supplied anatomy graph."],
                     metadata=metadata)


# ---------------------------------------------------------------------------------------------------------
# CLASS-GENERAL offline fallback — emphatically NOT a per-species catalog. When NO LLM is available, build a
# generic, connected creature of the INFERRED CLASS (a quadruped / a many-legged crawler) through the SAME
# general compiler. There is deliberately no "dog"/"crab"/"lizard" lookup: "dog", "lizard", "axolotl" and
# "wyvern" all yield the same generic legged body offline, and the LLM anatomy designer (the PRIMARY path)
# supplies the species-specific anatomy when present. Hard-coding a graph per animal is the overfitting we
# reject — the software must generalize to creatures nobody enumerated.
def _generic_legged_graph(*, n_pairs: int, body=0.58, girth=0.13, leg=0.42, tail=0.14, fan=False,
                          appendage_role: str = "leg", body_aspect: str = "") -> dict:
    """A generic legged skeleton: torso + neck + head + tail + ``n_pairs`` of 3-segment walking legs, spread
    across the longitudinal anchors and fanned outward. Class-general — 2 pairs = quadruped, 3 = hexapod,
    4 = octopod. No species shape baked in; just a credible standing N-legged body.

    ``fan`` FANS the quadruped legs outward (aim 'down'->'down_out') for a WIDE lateral stance. MEASURED: a
    plain-'down' quad plants its feet NARROW (foot-y spread ~0.15 m, stance ratio 0.22) so a trot rolls over and
    even a statically-stable crawl tips (thin support triangle) — the whole 'quads don't walk' saga. Fanning
    widens the spread to ~0.5 m; paired with a CRAWL gait ([[walking-breakthrough-abduction]]) it gives a real
    open-loop upright walk. Default OFF (byte-identical); the hexapod/octopod already fan by default."""
    n = max(1, min(4, n_pairs))
    is_tentacled = appendage_role == "tentacle"
    parts = [{"name": "mantle" if is_tentacled else "torso", "role": "body", "size": body,
              "girth": girth, "aspect": body_aspect}]
    root_name = parts[0]["name"]
    if not is_tentacled:
        parts += [
            {"name": "neck", "role": "neck", "parent": root_name, "attach": "front_top", "aim": "forward_up",
             "size": 0.22 * body, "girth": 0.42 * girth},
            {"name": "head", "role": "head", "parent": "neck", "attach": "tip", "aim": "forward",
             "size": 0.24 * body, "girth": 0.45 * girth},
            {"name": "tail", "role": "tail", "parent": root_name, "attach": "rear_top", "aim": "back_up",
             "size": tail, "girth": 0.18 * tail, "joint": "revolute"},
        ]
    _dn = "down_out" if fan else "down"                     # fan the quad legs OUT for a wide, roll-stable stance
    layouts = {
        1: [("front_bottom", "down")],
        2: [("front_bottom", _dn), ("rear_bottom", _dn)],
        3: [("front_bottom", "forward_down_out"), ("mid_bottom", "down_out"), ("rear_bottom", "back_down_out")],
        4: [("front_bottom", "forward_down_out"), ("front_mid_bottom", "down_out"),
            ("rear_mid_bottom", "down_out"), ("rear_bottom", "back_down_out")],
    }[n]
    for i, (anchor, aim) in enumerate(layouts):
        # G7 (gap-closure): Go2-band legs — 4 segments = 3 ACTUATED joints + a welded foot (the old 3-segment
        # legs had only 2 actuated joints; every real quadruped has 3/leg), total length 0.42 m (~thigh+shank
        # 0.2+0.2 like the Go2/Mini-Cheetah class), and a SLENDER girth (aspect >=3:1) instead of 1:1 stubs.
        parts.append({"name": f"{appendage_role}{i + 1}", "role": appendage_role, "parent": root_name, "attach": anchor, "aim": aim,
                      "size": leg, "girth": 0.045 * leg, "segments": 4, "symmetry": "left_right",
                      "joint": "revolute"})
    # The public class is intentionally broad for 6/8-legged bodies: downstream
    # capability lookups understand ``legged`` but a hexapod/octopod must never be
    # mislabeled as a quadruped and collapsed into its class-specific memory niche.
    body_class = "legged" if (is_tentacled or n >= 3) else ("biped" if n == 1 else "quadruped")
    return {"robot_class": body_class,
            "name": "creature", "parts": parts}


# A walker prompt = a named legged MORPHOLOGY (quadruped/hexapod/…) OR locomotion INTENT (walk/trot/gait/…).
# Both route to the gait-tuned parametric walker (a proven, well-proportioned walking body) instead of the
# generic anatomy creature — so "walk forward across the floor" gets a body it can actually be trained to walk,
# not a stunted credible-looking dog. (The LLM anatomy path still supplies real species anatomy when present.)
_WALKER_SIGNAL = re.compile(r"\b(walking robot|walker|quadruped|hexapod|octopod|biped|legged|"
                            r"\d+[ -]?legged|\d+ legs|walk|walks|walking|locomot\w*|trot|trots|trotting|"
                            r"gait|crawl|crawls|crawling|march|marches|marching|gallop|gallops|stroll)\b")
# A NAMED CREATURE (animal noun) gets the anatomy body even when it ALSO says it "walks" -- a dog that walks
# is still a dog. Only a PURE functional walker ("a quadruped walking robot", "a hexapod") with no creature
# word routes to the gait-tuned parametric template. Without this, "a dog robot that walks" tripped the walk
# verb and came out a flat generic quadruped instead of a dog.
_CREATURE_WORDS = re.compile(
    r"\b(frog|amphibian|dog|puppy|hound|canine|spider|arachnid|tarantula|insect|lizard|crab|crustacean|"
    r"mantis|gecko|salamander|newt|scorpion|centipede|beetle|ant|turtle|tortoise|snake|serpent|creature|"
    r"animal|octopus|cat|kitten|horse|pony|giraffe|bird|elephant|cow|goat|deer|donkey|sheep|pig|rabbit|fox|"
    r"wolf|bear|tiger|lion|cheetah|leopard|camel|kangaroo|ostrich|penguin|chicken|duck|mammal|reptile|"
    r"dinosaur|raptor|axolotl|rhino|hippo|zebra|antelope|gazelle|panther|puma|lynx|mule|llama|alpaca|moose|"
    r"elk|bison|goose)\b")


def generic_creature_gene(prompt: str, robot_class: str | None = None, n_legs: int | None = None):
    """Offline/no-LLM fallback: a generic, connected creature via the general compiler — a credible standing body
    for an ANIMAL prompt instead of the flat parametric slab. Class-general, with NO per-species lookup: 'dog',
    'lizard', 'axolotl' all get the same generic body, and the LLM anatomy designer (the PRIMARY path) supplies the
    species-specific anatomy when present. ``n_legs`` (parsed from the prompt) sizes the skeleton so a T-rex is a
    2-leg biped and an octopus an 8-leg octopod — the previous fixed 4-leg build was why every off-keyword animal
    collapsed to the same quadruped. Returns None for (a) non-creature classes and (b) FUNCTIONAL/CLASS walker
    prompts ('a quadruped walking robot', 'a hexapod') — those route to the gait-tuned parametric template."""
    p = (prompt or "").lower()
    cls = (robot_class or "").lower()
    if cls in ("manipulator", "mobile_base", "arm", "gripper", "humanoid"):
        return None
    if _WALKER_SIGNAL.search(p) and not _CREATURE_WORDS.search(p):
        return None
    # HONOR the requested appendage plan. An explicit tentacled/radial cue selects a round mantle plus articulated
    # tentacles; it never routes those appendages through the walking-leg/foot machinery.
    n_pairs = max(1, min(4, int(n_legs) // 2)) if n_legs else 2
    tentacled = any(w in p for w in ("octopus", "cephalopod", "tentacle", "tentacled"))
    # P4: animal PROPORTION priors so a horse (long legs), a gecko (short splayed legs) and a turtle (short wide)
    # stop composing to one generic skeleton. 1.0 for generic/dog -> the fixed defaults (byte-identical baseline).
    from virturoid.services.animal_proportions import animal_proportions
    prop = animal_proportions(p)
    try:
        g = build_from_anatomy(_generic_legged_graph(
            n_pairs=n_pairs, appendage_role="tentacle" if tentacled else "leg",
            body_aspect="round" if tentacled else "",
            leg=round(0.42 * prop["leg"], 4), body=round(0.58 * prop["torso"], 4),
            girth=round(0.13 * (prop["mass"] ** (1.0 / 3.0)), 4), fan=(prop["stance"] > 1.2),
        ))
        if tentacled:
            g.robot_class = "legged"
            g.composition_notes = ["Offline structural cue selected a round mantle with articulated tentacles; this is not a walking-leg body."]
        if not g.validate():
            return g
    except Exception:  # noqa: BLE001
        return None
    return None


def ensure_walkable_quad(gene, prompt: str = "", *, force: bool = False):
    """DIAGNOSIS-DRIVEN, per-request walkability fallback for a QUADRUPED — a HINT the build reaches for, never a
    gait/body forced on every dog. If the body already walks under SOME gait (``evaluate_robot`` is gait-aware:
    trot, else the statically-stable crawl), it is returned UNCHANGED (sleek-when-possible). Only if it ROLLS
    OVER (the tippy-narrow-stance failure — distance stays low under any gait) do we reach for the FANNED
    wide-stance hint. To respect design intent we DO NOT silently swap out a bespoke LLM anatomy: a generic/
    heuristic quad is rebuilt fanned (its own narrow form doesn't walk anyway), while an LLM-designed creature is
    kept and merely FLAGGED (metadata.walkability) so the caller/trainer knows to widen its stance. See
    [[walking-breakthrough-abduction]]. Best-effort: any failure returns the gene untouched."""
    try:
        from virturoid.services.task_matched_eval import evaluate_robot, robot_kind
    except Exception:  # noqa: BLE001
        return gene
    try:
        import re
        if robot_kind(gene) != "legged":
            return gene
        legs = {m.group(0) for s in gene.segments
                if (m := re.search(r"(?:(?:front|hind)_)?leg\d*_[lr]", (s.name or "").lower()))}
        # the fan+crawl hint is a QUADRUPED recipe: accept our own leg{n}_{l|r} naming OR any gene already CLASSIFIED
        # a quadruped (an imported robot uses arbitrary leg names but is structurally classed by limb count).
        if len(legs) != 4 and getattr(gene, "robot_class", "") != "quadruped":
            return gene
        base = float(evaluate_robot(gene).get("value", 0.0))
        if base >= 0.5 and not force:                        # already walks under some gait -> leave it sleek
            return gene                                      # (force skips this shortcut but STILL only adopts the
        #                                                      fanned stance below if it beats the original materially)
        src = (getattr(gene, "design_source", None) or "").lower()
        if not force and src not in ("", "anatomy_generic", "archetype", "heuristic", "generic"):
            # a bespoke LLM creature (UNFORCED): preserve the DESIGN, flag that it can't credibly walk as-is (a
            # hint, not a swap). Under force=True (an IMPORT the caller must make simulable — its provenance is
            # 'unknown', not a design to protect) we fall through to the fanned rebuild+compare below, which STILL
            # self-protects a genuinely-walking design (it's adopted only when it beats the original materially).
            md = dict(getattr(gene, "metadata", None) or {})
            md["walkability"] = {"credible_walk": False, "distance_m": round(base, 3),
                                 "hint": "widen the stance (fan the legs out) + use the crawl gait"}
            gene.metadata = md
            return gene
        cand = build_from_anatomy(_generic_legged_graph(n_pairs=2, fan=True, girth=0.18))
        if cand is None:
            return gene
        cval = float(evaluate_robot(cand).get("value", 0.0))
        if cval > max(base, 0.4):                            # the fanned hint walks materially better -> adopt it
            md = dict(getattr(cand, "metadata", None) or {})
            md["walkability_fallback"] = {"applied": True, "from_distance_m": round(base, 3),
                                          "to_distance_m": round(cval, 3), "hint": "fanned_stance+crawl_gait"}
            cand.metadata = md
            cand.design_source = src or "anatomy_generic"
            return cand
        return gene
    except Exception:  # noqa: BLE001 - the fallback is best-effort; a rolling body beats a crash
        return gene
