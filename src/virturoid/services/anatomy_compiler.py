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
Roles (reusable vocabulary): body, segment, neck, head, snout, jaw, ear, eye, horn, tail, leg_upper,
  leg_lower, paw, foot, arm_upper, arm_lower, hand, wing, fin, flipper, claw, beak, antenna, shell, fang.
  Unknown -> a generic tapered limb. attach tokens: front/rear/back, left/right, top/bottom, mid, tip
  (combine: front_top, front_bottom, rear_top, tip). aim tokens: forward/back, up/down, left/right, out
  (combine: forward_up, down, back_up, down_out).

TWO BODY AXES, NOT ONE. A body used to be a single rigid trunk with everything radiating off it, because
``attach`` was read ONLY when the parent was the root: any part whose parent was another PART was welded to
that part's chain TIP at offset (0,0,0), whatever its ``attach`` said. Measured (task #213): a segmented graph
of 6 trunk links carrying 6 leg pairs compiled with all TWELVE leg chains parented to ``trunk_5`` -- the last
link -- at a single point. So a many-legged body had exactly one expressible topology, a radial fan from one
disc, and "centipede" built an urchin. Two additions close that, and neither is a species:
  * role ``segment`` -- a BODY-AXIS link (short, wider than tall, machined) so a trunk can be a serial CHAIN
    of body blocks rather than one rigid barrel or a distally-tapering tail;
  * a STATION on any parent -- ``attach: {"along": 0..1, "lateral": -1..1, "height": 0..1}`` addresses a point
    along a non-root parent's own segment chain, so limbs mount at intervals down a trunk and on its SIDES.
``along`` runs the parent's OWN axis, proximal (0) to distal (1), and picks both which link and where on it.
Everything that omits the dict form keeps the old tip-weld exactly.
"""

from __future__ import annotations

import math
import re

import numpy as np

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.body_kind import FLOATING_BASE_CLASSES
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
_ARTICULATED = {"leg_upper", "leg_lower", "arm_upper", "arm_lower", "tail", "wing", "fin", "flipper", "claw",
                "neck", "tentacle", "trunk", "segment", "body_segment", "spine"}
# Roles that hang/extend DOWN by default when the graph doesn't say (legs).
_DOWN_ROLES = {"leg_upper", "leg_lower", "paw", "foot"}
# BODY-AXIS roles: a link of the trunk itself, not an appendage hanging off it. They differ from limbs in two
# ways that matter — they are WIDER THAN TALL (a body carries organs/electronics side to side; a limb is a
# beam), and they do NOT taper distally (a limb thins toward its tip because it carries less load; a trunk
# does not thin because it is not cantilevered off anything).
_BODY_AXIS_ROLES = {"segment", "body_segment", "spine"}


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
        # THE TRUNK IS THE ONE PART WITH NO ANATOMY ROLE. Every limb/head/foot on this path reaches a
        # multi-feature solid via `anatomy_role` (measured: a dog gets 17/20), but `_anatomy_role_for("body")`
        # returns None on purpose -- build_anatomy's `quad_trunk` is authored for a trunk whose LONG axis is +z,
        # and this compiler builds the body along +z = HEIGHT with the front-back extent in the profile. Routing
        # to it would draw a 0.21 m blob where a 0.58 m trunk belongs. So the trunk's own shape program has to
        # carry the detail, and a single filleted rectangular prism is exactly the "child's blocks" cue: it was
        # the largest object in the render and the least machined thing in it.
        # Now a STEPPED HULL, which is how a real quadruped chassis is actually built and is expressible with
        # the shape-program vocabulary we already have (extrude + `at`, unioned by `compound`):
        #   hull  - the full-width lower structure, plan corners cut so the nose/tail are chamfered, not
        #           brick-flat end-on. This is parts[0], so it is the sub-part _apply_detail vents.
        #   spine - a NARROWER upper hull rising out of it, giving the side view a real shoulder step
        #           instead of one constant-section slab.
        #   deck  - the raised top electronics module (kept, tapered forward so it reads as a sensor bay).
        # Every sub-part starts at z=0 and stays inside the +-half_len x +-hw envelope, so the visual never
        # reaches below or outside the collider it stands in (the mesh is floored to z=0 after realize, and a
        # sub-part poking below would silently shift the whole trunk up off its own primitive).
        # ALL of it is visual: `family` stays "compound", which _physics_proxy_from_geometry maps to
        # (fallback_shape, None) exactly as before, so the collider stays the box built from (half_len, brad),
        # and h/hw -- the numbers that become length_m/radius_m -- are untouched. Verified byte-identical
        # collider hash + mass + spawn height on dog / cat / hexapod / humanoid.
        # a lighter fillet (was 0.45) so the machined CHAMFER that _apply_detail adds reads as crisp panel
        # edges, not a pillowy slab — matches the humanoid's chamfered chassis modules.
        hull = {"family": "extrude", "height": round(0.66 * h, 4), "fillet": round(0.2 * hw, 4),
                "profile": [[-hl, -0.70 * hw], [-0.88 * hl, -hw], [0.88 * hl, -hw], [hl, -0.70 * hw],
                            [hl, 0.70 * hw], [0.88 * hl, hw], [-0.88 * hl, hw], [-hl, 0.70 * hw]]}
        spine = {"family": "extrude", "height": round(h, 4), "fillet": round(0.18 * hw, 4),
                 "chamfer": round(0.13 * hw, 5),
                 "profile": [[-0.92 * hl, -0.76 * hw], [0.92 * hl, -0.76 * hw],
                             [0.92 * hl, 0.76 * hw], [-0.92 * hl, 0.76 * hw]]}
        deck = {"family": "extrude", "height": round(0.42 * h, 4), "fillet": round(0.16 * hw, 4),
                "chamfer": round(0.12 * hw, 5),
                "profile": [[-0.52 * hl, -0.56 * hw], [0.46 * hl, -0.68 * hw],
                            [0.46 * hl, 0.68 * hw], [-0.52 * hl, 0.56 * hw]],
                "at": (0.0, 0.0, round(0.8 * h, 4))}
        return ({"family": "compound", "parts": [hull, spine, deck]}, h, hw)
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
    if role in _BODY_AXIS_ROLES:
        # A BODY-AXIS LINK: one block of a segmented trunk. ``size`` is its length ALONG the body axis and
        # ``girth`` its lateral half-width, and it is extruded along local +z like every other link — which is
        # what lets a chain of them run head-to-tail through the same aim/mount machinery limbs use.
        #
        # The cross-section is the whole point. `_aim_R` guarantees local +y is HORIZONTAL, so for a trunk
        # aimed along the ground local y is LATERAL and local x is VERTICAL. A body block is therefore
        # half_y = g (wide) and half_x = 0.62*g (low) — wider than tall, with a flat underside, which is how
        # every segmented animal and every modular robot train is actually built. A limb's `leg` profile is
        # the opposite (deeper fore-aft than wide) and a `tail`'s is a distally-shrinking spindle; neither
        # reads as a body, which is why a segmented trunk built out of them looked like a string of sausages.
        hy = max(0.008, g)
        hx = max(0.006, 0.62 * hy)
        return ({"family": "extrude", "height": round(L, 4), "fillet": round(0.34 * hx, 4),
                 "chamfer": round(0.18 * hx, 5),
                 "profile": [[-hx, -0.88 * hy], [0.86 * hx, -hy], [0.86 * hx, hy], [-hx, 0.88 * hy]]}, L, hy)
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


def _limb_length_shares(role: str, n: int, has_foot: bool) -> list[float]:
    """How one limb's total length divides among its ``n`` segments (fractions, summing to 1).

    A leg was cut into n EQUAL pieces, which is the single biggest reason our legged bodies read as toys. Real
    legged robots are nothing like that -- they put a SHORT proximal abduction block, then two LONG load-bearing
    links, then a small contact pad:

        Unitree Go2      hip/abad ~0.08, thigh 0.213, calf 0.213, foot = a 22 mm sphere   (leg 0.426 m)
        MIT Mini Cheetah abad 0.062, thigh 0.209, calf 0.195, foot radius 0.02
        ANYmal C         hip 0.09, thigh 0.285, shank 0.33

    Measured on our own dog, 4 equal 105 mm stubs put an 82 mm motor can on a 105 mm segment -- can/length 0.78,
    where a Go2 hangs the same size motor on a 213 mm thigh (0.38). At 0.78 the ACTUATOR IS THE SILHOUETTE, which
    is exactly the bead-chain look; the shares below bring us to ~0.4, in the real range.

    Arms are deliberately left UNIFORM: a manipulator's links really are near-equal (a UR5 is 0.425/0.392), and
    the arm is the one archetype that already renders convincingly -- there is nothing here to fix."""
    n = max(1, int(n))
    if n == 1:
        return [1.0]
    if not role.startswith("leg"):
        return [1.0 / n] * n                          # arms/tails/necks: unchanged
    foot = 0.07 if has_foot else 0.0                  # a foot is a CONTACT PAD, not a limb link
    n_links = n - (1 if has_foot else 0)
    if n_links <= 1:
        return [1.0 - foot] + ([foot] if has_foot else [])
    hip = 0.16 * (1.0 - foot)                         # short proximal abduction/rotation block
    each = (1.0 - foot - hip) / (n_links - 1)
    return [hip] + [each] * (n_links - 1) + ([foot] if has_foot else [])


def _anatomy_role_for(seg_role: str, index: int, n_segments: int) -> str | None:
    """Map this compiler's SEMANTIC role onto ``cad_geometry.build_anatomy``'s detailed-solid vocabulary.

    The two halves of the pipeline grew separate role words: we describe a part by what it IS ("leg", "foot",
    "head"), while the mesh layer indexes multi-feature solids by what to BUILD ("quad_thigh", "foot_pad",
    "sensor_head"). Nothing joined them, so every body from this compiler rendered as extruded rectangles.

    A limb's segments are distinguished BY POSITION, which is how a real leg is built: the first segment carries
    the abduction/hip block, the second is the thigh, and everything below it is calf. Returns None for roles
    with no better solid than the generic one (the caller then leaves the authored geometry alone)."""
    r = (seg_role or "").lower()
    if r in ("foot", "paw", "hoof"):
        return "foot_pad"
    if r in ("hand", "gripper", "claw"):
        return "hand_plate"
    if r in ("head", "skull", "sensor_head", "snout", "muzzle"):
        return "sensor_head"
    if r in ("leg", "leg_upper", "leg_lower", "thigh", "shin", "limb"):
        if r in ("thigh", "leg_upper"):
            return "quad_thigh"
        if r in ("shin", "leg_lower"):
            return "quad_calf"
        if n_segments <= 1:
            return "quad_thigh"                      # a single-segment leg is a thigh, not a hip block
        return "quad_hip" if index == 0 else ("quad_thigh" if index == 1 else "quad_calf")
    if r in ("arm", "arm_upper", "arm_lower", "forearm", "upper_arm"):
        if r in ("arm_lower", "forearm"):
            return "forearm"
        if r in ("arm_upper", "upper_arm"):
            return "upper_arm"
        return "upper_arm" if (n_segments <= 1 or index == 0) else "forearm"
    return None


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
    # PARAMETRIC ANCHOR. The eight named sites are an animal's vocabulary -- shoulder, hip, withers, tail root --
    # and they cannot say "60% along the deck": a turret mid-chassis, a carriage partway down its rail, a sensor
    # mast at 70% of the body. A dict form is a strict GENERALISATION, not a second code path: along=1.0/0.5/0.0
    # reproduces front/mid/rear exactly, lateral=+-1 reproduces left/right, and height=0.40/0.50/0.70 reproduces
    # bottom/mid/top, so every named site is a point in this space and the equivalence is asserted in the tests.
    if isinstance(attach, dict):
        along = min(1.0, max(0.0, float(attach.get("along", 0.5))))
        lateral = min(1.0, max(-1.0, float(attach.get("lateral", 0.0))))
        frac_h = min(1.0, max(0.0, float(attach.get("height", 0.5))))
        return ((along - 0.5) * 2.0 * edge * half_len, lateral * 0.82 * half_w, frac_h * height)
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


def _station_on_chain(links, attach, sign_y, world_R_of):
    """Where on a NON-ROOT parent a child mounts. ``links`` is that parent part's OWN segment chain in order,
    as ``(seg_name, length_m, radius_m)``; ``world_R_of`` maps a segment name to its world rotation. Returns
    ``(seg_name, (dx, dy, dz))``, or ``None`` to keep the historical tip-weld.

    THIS IS THE PRIMITIVE A SEGMENTED BODY WAS MISSING. ``_anchor_on_body`` gives the ROOT five longitudinal
    stations and two sides, which is how a quadruped gets shoulders and hips; every other parent got
    ``mount_offset = (0, 0, 0)`` — its chain tip — no matter what its ``attach`` said. So a trunk built as a
    chain could carry limbs at exactly one place: the end of it. Measured on a 6-link trunk with 6 leg pairs,
    all twelve legs came out parented to the last link at one point.

    The address mixes two frames on purpose, because the two questions live in different ones:
      ``along``   0 = the chain's proximal base, 1 = its distal tip — the parent's OWN axis, the only frame in
                  which "how far down the parent" means anything. Picks BOTH which link and where on it.
      ``lateral`` -1..+1 out to the side, and ``height`` 0 (underside) .. 1 (top), which are WORLD ±y and ±z.
    Those two are world-framed because ``symmetry: left_right`` mirrors a part's AIM in world y, and the mount
    has to end up on the same flank the limb points at. Measured while building this: expressing ``lateral`` in
    the parent's local +y instead put every left leg's mount on the RIGHT flank of a trunk aimed backwards
    (``_aim_R`` sends local +y to world -y there), so the pair crossed underneath the body and the foot spread
    collapsed from a sprawled 0.22 m to 0.058 m — narrower than the body itself. The world offset is projected
    off the parent's own axis first, so ``along`` stays the sole author of the longitudinal position however
    the parent is oriented.

    Offsets are 0.9x the half-extent so the child ROOTS INSIDE the parent's surface and the joint is covered,
    the same overlap rule ``_anchor_on_body`` uses. The returned dz is measured from the chosen link's TIP,
    matching this compiler's convention that ``(0,0,0)`` means "at the parent's tip".
    """
    if not links:
        return None
    try:
        along = min(1.0, max(0.0, float(attach.get("along", 0.5))))
        lateral = min(1.0, max(-1.0, float(attach.get("lateral", 0.0))))
        frac_h = min(1.0, max(0.0, float(attach.get("height", 0.5))))
    except (TypeError, ValueError):
        return None
    if sign_y:                                   # a mirrored pair defaults to the SIDES, not the centreline
        lateral = abs(lateral) if lateral else 1.0
        lateral *= sign_y
    total = sum(max(0.0, l) for _, l, _ in links) or 1e-9
    want = along * total
    acc = 0.0
    for idx, (name, L, r) in enumerate(links):
        last = idx == len(links) - 1
        if want <= acc + L or last:
            f = 0.0 if L <= 0 else min(1.0, max(0.0, (want - acc) / L))
            R = world_R_of(name)
            off_w = np.array([0.0, lateral * 0.9 * r, (frac_h - 0.5) * 2.0 * 0.9 * r])
            zw = np.asarray(R)[:, 2]
            off_w = off_w - float(off_w @ zw) * zw          # `along` owns the longitudinal position, not this
            off_l = np.asarray(R).T @ off_w
            return name, (round(float(off_l[0]), 6), round(float(off_l[1]), 6),
                          round(float(off_l[2]) + f * L - L, 6))
        acc += L
    return None


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
    """Which way this part points. An OMITTED aim takes the role's default; an UNRECOGNISED one is an error.

    Those two cases used to be the same line, and any token we did not know silently became `forward`. Measured:
    a gantry authored with `aim: "right"` -- not a token, though it reads like one -- built its bridge pointing
    down the machine instead of across it, and nothing said so. The design compiled, passed its DOF check, and
    was wrong in the only place a customer looks.

    This is the same rule the ROLE vocabulary already follows: an unknown role is never guessed at, it has to
    declare what it is `like`. Silence about a word we do not know is the one thing a design tool must not do."""
    # An explicit DIRECTION VECTOR, for anything the 18 tokens cannot say. Every token has y >= 0, because they
    # were written for animals whose limbs come in mirrored pairs and are placed by `symmetry: left_right` -- so
    # there is no way to point a single part at -y at all, and a radial layout (three delta arms at 120 degrees,
    # a radial urchin) is unauthorable however the tokens are combined. Same move as the parametric `attach`:
    # the vocabulary stays as convenient shorthand and stops being a ceiling.
    if isinstance(aim, (list, tuple)) and len(aim) == 3:
        try:
            v = [float(c) for c in aim]
        except (TypeError, ValueError):
            raise ValueError(f"aim {aim!r} must be three numbers [x, y, z], or one of: "
                             f"{', '.join(sorted(_AIM_VEC))}") from None
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        if n < 1e-9:
            raise ValueError(f"aim {aim!r} has no direction (zero length); a part must point somewhere")
        return (v[0] / n, v[1] / n, v[2] / n)
    a = (aim or "").strip().lower()
    if a in _AIM_VEC:
        return _AIM_VEC[a]
    if a:
        raise ValueError(
            f"aim {aim!r} is not a direction this compiler knows (part role {role!r}). It would have been "
            f"silently treated as 'forward'. Use one of: {', '.join(sorted(_AIM_VEC))} — or give an explicit "
            "[x, y, z] vector, which is the only way to reach a direction the tokens cannot name (every token "
            "has y >= 0, so a single part pointing at -y, or a radial fan at 120 degrees, needs the vector). "
            "Note 'left'/'right' are not tokens: lateral placement is a job for `attach` (its `lateral` runs "
            "-1..+1) or for `symmetry: 'left_right'`, which mirrors a part into a +y/-y pair.")
    if role in _DOWN_ROLES:                               # nothing declared: the role's own default is intended
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
    "segment": "shell", "body_segment": "shell", "spine": "shell",   # trunk links are body, not load path
    "wheel": "rubber",
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
            # A MACHINED VENT ROW, not portholes. The vent radius used to be capped at 0.3 x the SHORT half-axis
            # with a count capped at 5, which is fine on a beam but wrong on a chassis: a 580 x 180 x 207 mm
            # quadruped trunk got THREE 54 mm holes bored clean through its flank, and that -- more than any
            # other single feature -- is what made the render read as a wooden block rather than a robot. Size a
            # vent against the panel it sits in (the part's own extruded HEIGHT as well as its short half-axis)
            # and let the count rise, so a long chassis gets a row of small louvres and a short beam is
            # unchanged in character. Visual only: cutouts never reach _physics_proxy_from_geometry.
            lo, hi = min(hx, hy), max(hx, hy)
            n = max(3, min(7, int(1.7 * hi / max(lo, 1e-3))))
            r = max(0.003, min(0.14 * lo, 0.085 * max(length_m, 1e-3), 0.5 * (1.1 * hi) / n))
            cz = round(0.40 * length_m, 4)                 # in the lower hull panel, clear of the shoulder step
            if hx >= hy:                                  # long along x -> vents stepped in x, bored along y
                for i in range(n):
                    cuts.append([round(-0.66 * hx + 1.32 * hx * (i + 0.5) / n, 4), 0.0,
                                 cz, round(r, 4), round(2.4 * hy, 4), "y"])
            else:                                         # long along y -> stepped in y, bored along x
                for i in range(n):
                    cuts.append([0.0, round(-0.66 * hy + 1.32 * hy * (i + 0.5) / n, 4),
                                 cz, round(r, 4), round(2.4 * hx, 4), "x"])
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
    # A FIXED-BASE MACHINE MUST NOT FLOAT. The default was "free" for every class, so an arm -- and now a gantry,
    # SCARA, rail carriage or turret, which are all bolted down -- got a 6-DOF floating base and simply fell over,
    # making any verdict on it meaningless. Default by class instead, the same rule robot_import already applies
    # when it decides an imported body's base (free for things that locomote, table for things that are mounted).
    # An explicit graph value still wins, so a caller can bolt a quadruped to a rig or float an arm on purpose.
    # The set is body_kind.FLOATING_BASE_CLASSES -- shared with robot_import, whose private copy had drifted.
    _declared = str(graph.get("base_mount") or "").lower()
    base_mount = _declared or ("free" if str(robot_class or "").lower() in FLOATING_BASE_CLASSES else "table")
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
    # chain registry: (part_name, side) -> that part's FULL ordered segment list as (name, length_m, radius_m).
    # ``leaf`` records only the TIP, which is why a child naming a multi-segment parent could reach nothing but
    # its last link. A station address (`_station_on_chain`) needs the whole chain to find which link it lands on.
    chain_of: dict = {}
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
            parent_seg, parent_key = leaf[(pname, side)], (pname, side)
        elif (pname, "") in leaf:
            parent_seg, parent_key = leaf[(pname, "")], (pname, "")
        else:
            return  # parent not built (bad graph order) -> skip this part rather than crash
        body_attached = parent_seg == rname
        # A STATION ON A NON-ROOT PARENT (see `_station_on_chain`). Resolved BEFORE the emit loop because it can
        # re-point the parent onto an INTERMEDIATE link of the parent's chain -- the whole difference between a
        # trunk that carries limbs along its length and one that can only carry them off its tail.
        station_offset = None
        if not body_attached and isinstance(part.get("attach"), dict):
            _st = _station_on_chain(chain_of.get(parent_key) or [], part["attach"], sign_y,
                                    lambda nm: world_R.get(nm, np.eye(3)))
            if _st is not None:
                parent_seg, station_offset = _st
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
        # REVERTED 2026-07-29: an anatomical split (short hip + two long links + a contact pad) is measurably
        # closer to a real leg -- it took stride-link aspect from 2.27 to 4.33/5.28, the band our own priors
        # file asks for -- but it moved every legged body off the operating point the SCRIPTED crawl gait was
        # co-tuned with, and 12 locomotion tests went red. The geometry was right and the control was not, and
        # shipping a prettier robot that stopped walking is the wrong trade. See _limb_length_shares: the work
        # to land it is re-fitting the gait layer to non-uniform limbs, not reshaping the body again.
        seg_len = size / n if n > 1 else size
        prev = parent_seg
        prev_world_R = world_R.get(parent_seg, np.eye(3))
        built_links: list = []
        # DISTAL TAPER IS A LIMB FACT, NOT A BODY FACT. 0.82 per link thins a 4-link leg to 0.55x by the foot,
        # which is right: a limb carries less load the further out it goes. A TRUNK is not cantilevered off
        # anything, so the same rule turns an 8-link segmented body into a cone (0.82**7 = 0.25x) — a tail, not
        # a torso. Body-axis links taper only gently, head to tail, the way a real segmented animal does.
        _taper = 0.97 if role in _BODY_AXIS_ROLES else 0.82
        for i in range(n):
            last = i == n - 1
            is_foot = role == "leg" and last and n > 1     # only a walking leg receives a foot
            seg_role = ("foot" if is_foot else role)
            is_wheel = seg_role == "wheel"
            g_i = girth * (_taper ** i)
            geo, length_m, radius_m = _role_geometry(seg_role, seg_len * (1.0 if not last or n == 1 else 0.6), g_i,
                                                     authored=(seg_role == role))
            geo = _apply_detail(geo, detail, length_m, g_i, part_chamfer)
            # T4 shape-programs: a part may AUTHOR its own visual geometry (extrude/revolve/tapered/loft +
            # fillet/chamfer/cutouts) — the mesh layer realizes it (cad_geometry.realize_shape, safe capsule
            # fallback). Applies to single-segment parts; the primitive COLLIDER (size/girth) is unchanged, so
            # physics is untouched and only the rendered shape changes (a dome sensor, a bracket, a nozzle).
            _cg = part.get("geometry")
            _authored = isinstance(_cg, dict) and n == 1
            if _authored:
                geo = _cg
            if isinstance(geo, dict):
                geo = {**geo, "semantic_role": seg_role}
                # ANATOMY VOCABULARY FOR THE GENERAL PATH. cad_geometry.build_anatomy already ships multi-feature
                # ORIGINAL solids -- a 5-section tapered limb spindle with a proximal shoulder that seats on the
                # motor, a heel->toe sole, a jaw->cheeks->crown skull -- but they are only reachable through
                # `family="role"`, which ONLY the hard-coded morphology_composer recipes emitted. Every body from
                # THIS compiler (the general LLM anatomy-graph path: dogs, horses, geckos, novel creatures) fell
                # through to a 4-point rectangle extruded into a bar, which is why the hexapod's legs read as real
                # limbs while the dog's read as sticks. Stamp the mapped role so the mesh layer can use the rich
                # solid. ADDITIVE ONLY: `family` is untouched, so _physics_proxy_from_geometry picks exactly the
                # same collider and no mass, inertia or verdict can move. Never stamped over an AUTHORED shape
                # program (a T4 dome/bracket the model designed on purpose still wins).
                if not _authored:
                    _ar = _anatomy_role_for(seg_role, i, n)
                    if _ar:
                        geo = {**geo, "anatomy_role": _ar}
            # An explicit `joint` beats the role's inference — including for a wheel. The wheel branch used to run
            # FIRST, so `joint: "fixed"` on a wheel-role part was silently overridden into a revolute and the
            # design quietly gained an actuator it never asked for. That matters because the compiler's own
            # teaching error tells an agent that a TRACK is "like a wheel": a tracked loader would declare two
            # fixed track units and be handed two motors. This module already states the rule further down —
            # "SPECIFIED joints beat inferred ones" — and the wheel case simply predated it.
            if explicit_joint == "fixed" or (last and is_limb and n > 1):
                joint_type = None                                  # welded paw / explicitly-fixed part
            elif is_wheel and not explicit_joint:
                joint_type = "revolute"                            # a wheel with nothing declared is a free hinge
            elif explicit_joint == "prismatic":
                # A SLIDING joint. `GeneSegment` and `gene_compiler` have always supported prismatic; the anatomy
                # path just never offered it, so a whole family of real machines was inexpressible -- a gantry
                # axis, a telescoping mast, a linear rail carriage, an excavator's stick. Nothing about animals
                # needs it, which is exactly why it was missing.
                joint_type = "prismatic"
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
                        # Mount offsets describe the axle center. Seat the inner tire sidewall on the chassis
                        # using the wheel WIDTH (length_m), not a fraction of its rolling radius.
                        ay_w = sign_y * (abs(pdim["half_w"]) + 0.5 * length_m) if sign_y else ay
                        mount_offset = (ax, ay_w, az - pdim["length_m"])
                    else:
                        ay = sign_y * abs(pdim["half_w"]) * 0.9 if sign_y else ay
                        mount_offset = (ax, ay, az - pdim["length_m"])
                elif station_offset is not None:
                    mount_offset = station_offset    # an addressed station along a non-root parent's chain
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
                # ANATOMICAL per-role ranges. Only abduction was limited; hip-pitch and knee inherited the +-2.6
                # catch-all, i.e. a knee free to hyperextend ~149 deg -- physically impossible and the first thing
                # a reviewer spots in the XML. Bounds are MEASURED from this body's own walking excursion
                # (aggregated over ALL legs so left/right mirrors are covered: abduction +-0.08, hip pitch
                # -0.51..+0.90, knee -0.06..+0.49) and set to contain it with margin, so they are honest anatomy
                # that cannot clamp the gait. Knee keeps ~11 deg of slack below zero rather than a hard 0 floor,
                # because the measured motion dips slightly negative. Cf. Unitree Go2 (Menagerie): calf is
                # likewise flexion-only (-2.72..-0.84).
                if i == 0:
                    _jlo, _jhi = -0.6, 0.6                              # abduction range (Go2-class ~+-0.5 rad)
                elif i == 1:
                    _jlo, _jhi = -1.4, 1.6                              # hip pitch: fore/aft stride
                else:
                    _jlo, _jhi = -0.2, 2.4                              # knee/ankle: flexion, no hyperextension
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
            # SPECIFIED joints beat inferred ones. Everything above derives axis and limits from the ANIMAL role
            # (abduction-vs-stride for a walking leg, unlimited spin for a wheel, else a +-2.6 catch-all), which is
            # right for creatures and useless for machines: a gantry axis, a SCARA elbow, a turret yaw and an
            # excavator boom are each defined BY their axis and travel. GeneSegment and gene_compiler have always
            # carried joint_axis/lower/upper -- only this path never let the caller say them. A declared value wins;
            # anything omitted keeps the role-derived default, so every existing graph compiles unchanged.
            if joint_type:
                _uax = part.get("axis")
                if isinstance(_uax, (list, tuple)) and len(_uax) == 3:
                    _n = (sum(float(v) ** 2 for v in _uax)) ** 0.5 or 1.0
                    _ja = tuple(float(v) / _n for v in _uax)       # normalised: a direction, not a magnitude
                if part.get("lower") is not None:
                    _jlo = float(part["lower"])
                if part.get("upper") is not None:
                    _jhi = float(part["upper"])
                if _jlo is not None and _jhi is not None and _jlo > _jhi:
                    _jlo, _jhi = _jhi, _jlo                        # tolerate a swapped range rather than emit one
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
                #
                # A DECLARED rest angle wins over all of it. The derived cases above are animal defaults, and a
                # machine has none: every joint sat at 0, so a chain of forward-aimed links came out as a straight
                # horizontal line. Measured by rendering the eight non-animal benchmark families -- the excavator
                # passed its DOF check 4/4 and lay flat on the ground, and declaring a stance on the same body
                # stood it up 0.025 -> 1.248 m. Which stance is right belongs to the DESIGNER, not to a per-class
                # table here: a parked excavator and one loading a truck are the same machine.
                if part.get("rest") is not None:
                    _r = float(part["rest"])
                    if _jlo is not None:                  # a stance MuJoCo cannot hold is worse than none -- the
                        _r = max(_r, float(_jlo))         # solver would snap it on step 1 and the body a customer
                    if _jhi is not None:                  # was shown would not be the body that simulates
                        _r = min(_r, float(_jhi))
                    pose[f"{seg_name}_joint"] = _r
                elif is_limb and i == 1:
                    pose[f"{seg_name}_joint"] = 0.12
                elif curl and n > 1 and i >= 1:
                    pose[f"{seg_name}_joint"] = curl / (n - 1)
                else:
                    pose[f"{seg_name}_joint"] = 0.0
            built_links.append((seg_name, float(length_m), float(max(0.006, radius_m))))
            prev = seg_name
            prev_world_R = seg_world_R
        # register this part's chain tip under its ORIGINAL name + side, so a child (e.g. a foot) referencing
        # the part by name resolves to the correct side's leaf segment.
        leaf[(part.get("name") or base_name, side)] = prev
        chain_of[(part.get("name") or base_name, side)] = built_links

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
                     # A parallel mechanism is declared at the GRAPH level, not per part: a loop joins two parts
                     # and belongs to neither, and how high the whole thing is mounted is a fact about the body.
                     # Without these two the graph could describe a delta and never build one.
                     loop_closures=[dict(lc) for lc in (graph.get("loop_closures") or [])
                                    if isinstance(lc, dict)],
                     base_height_m=(None if graph.get("base_height_m") is None
                                    else float(graph["base_height_m"])),
                     metadata=metadata)


# ---------------------------------------------------------------------------------------------------------
# CLASS-GENERAL offline fallback — emphatically NOT a per-species catalog. When NO LLM is available, build a
# generic, connected creature of the INFERRED CLASS (a quadruped / a many-legged crawler) through the SAME
# general compiler. There is deliberately no "dog"/"crab"/"lizard" lookup: "dog", "lizard", "axolotl" and
# "wyvern" all yield the same generic legged body offline, and the LLM anatomy designer (the PRIMARY path)
# supplies the species-specific anatomy when present. Hard-coding a graph per animal is the overfitting we
# reject — the software must generalize to creatures nobody enumerated.
def _generic_legged_graph(*, n_pairs: int, body=0.58, girth=0.13, leg=0.42, tail=0.14, fan=False,
                          appendage_role: str = "leg", body_aspect: str = "",
                          leg_girth_mult: float = 1.0, center_leg: bool = False) -> dict:
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
                      "size": leg, "girth": 0.045 * leg * leg_girth_mult, "segments": 4,
                      "symmetry": "left_right", "joint": "revolute"})
    if center_leg:
        # An ODD leg count is a real request, not a rounding error. The pairs above carry the body (a bilateral
        # animal is built that way); the odd one goes on the CENTRELINE at the rear -- a genuine tripedal /
        # five-legged layout -- instead of being rounded up into another pair the customer never asked for.
        # ``symmetry`` is omitted, which the compiler reads as a single un-mirrored chain.
        parts.append({"name": f"{appendage_role}{n + 1}", "role": appendage_role, "parent": root_name,
                      "attach": "rear_bottom", "aim": "back_down", "size": leg,
                      "girth": 0.045 * leg * leg_girth_mult, "segments": 4, "joint": "revolute"})
    # The public class is intentionally broad for 6/8-legged bodies: downstream
    # capability lookups understand ``legged`` but a hexapod/octopod must never be
    # mislabeled as a quadruped and collapsed into its class-specific memory niche.
    n_legs_total = 2 * n + (1 if center_leg else 0)
    body_class = ("legged" if (is_tentacled or n_legs_total >= 5)
                  else ("biped" if n_legs_total == 2 else
                        "quadruped" if n_legs_total == 4 else "legged"))
    return {"robot_class": body_class,
            "name": "creature", "parts": parts}


# A RIGID trunk offers five longitudinal stations (front / front_mid / mid / rear_mid / rear), so it can carry
# at most four leg pairs before they start colliding at one anchor -- which is exactly where
# `_generic_legged_graph`'s layout table stops. Past that the trunk itself has to articulate. This is a
# STRUCTURAL capacity rule, not a species: a hexapod and an octopod are unaffected, and a 14-legged request is
# segmented whether the prompt said "centipede", "millipede" or "fourteen-legged".
RIGID_TRUNK_MAX_PAIRS = 4
# ... and a chain is bounded above too, by the appendage discovery that has to read the result:
# ``appendage_map.main_paths`` recurses past a branching one-token stub with ``depth < 8``, so a trunk deeper
# than 8 links stops being descended and the leg pairs hanging off its tail silently stop being found as legs.
SEGMENTED_MAX_PAIRS = 8


def _segmented_crawler_graph(*, n_pairs: int, seg_len: float = 0.092, seg_width: float = 0.055,
                             leg: float = 0.280, leg_girth_mult: float = 1.0) -> dict:
    """A SEGMENTED CHAIN-WITH-LEGS: a head block, an articulated trunk of ``n_pairs`` body-axis links, and one
    leg pair mounted at the middle of EACH link. The second body plan this compiler can express, and the one a
    many-legged body actually needs.

    NOT A CENTIPEDE FUNCTION. It takes a pair count and nothing else about any animal; the caller decides when
    a request needs more leg stations than a rigid trunk has (``RIGID_TRUNK_MAX_PAIRS``). The same plan is what
    a modular snake-robot-with-feet, a caterpillar, an articulated pipe crawler or a bogied train is: repeated
    identical modules along a flexing spine. What makes it possible at all is the station attach
    (``_station_on_chain``) -- without it every pair would weld to the trunk's last link at one point, which is
    precisely how "centipede" used to compile.

    The trunk hinges in YAW (local +x is vertical for a level link, so a hinge about it swings side to side),
    which is how a long body steers and how a segmented animal actually flexes as it walks; the range is kept
    narrow so the chain cannot fold back on itself and knot.
    """
    n = max(2, min(SEGMENTED_MAX_PAIRS, int(n_pairs)))
    head = 1.15 * seg_len
    parts = [
        # named "forebody", not "head": ``morphology_priors`` reads any segment whose NAME contains "head" as a
        # sensor head and compares its extent to the trunk, so a root module called "head" was measured against
        # itself and failed its own gate at 1.33x. This part is the front BODY module, which is what it is.
        {"name": "forebody", "role": "body", "size": round(head, 4), "girth": round(2.0 * seg_width, 4),
         "aspect": "wide"},
        {"name": "antenna", "role": "antenna", "parent": "forebody", "attach": "front_top", "aim": "forward_up",
         "size": round(1.1 * seg_len, 4), "girth": round(0.16 * seg_width, 4), "symmetry": "left_right"},
        {"name": "trunk", "role": "segment", "parent": "forebody", "attach": "rear_mid", "aim": "back",
         "size": round(n * seg_len, 4), "girth": round(seg_width, 4), "segments": n,
         "joint": "revolute", "axis": [1.0, 0.0, 0.0], "lower": -0.35, "upper": 0.35, "detail": "paneled"},
    ]
    for i in range(n):
        parts.append({
            "name": f"leg{i + 1}", "role": "leg", "parent": "trunk",
            # the MIDDLE of link i, on both flanks, just below the body's mid-height: one pair per body segment.
            "attach": {"along": (i + 0.5) / n, "lateral": 1.0, "height": 0.30},
            "aim": "down_out", "size": leg, "girth": round(0.045 * leg * leg_girth_mult, 5),
            "segments": 4, "symmetry": "left_right", "joint": "revolute"})
    return {"robot_class": "legged", "name": "segmented_crawler", "parts": parts}


def segmented_crawler_gene(n_legs: int, prompt: str = ""):
    """A many-legged body as a SEGMENTED CHAIN. Returns a ``RobotGene``, or ``None`` if it does not build.

    This is the offline answer for any request whose leg count exceeds what a rigid trunk can station
    (``RIGID_TRUNK_MAX_PAIRS``). It used to be answered by the radial spider archetype at the requested leg
    count -- 14 legs fanned around one 75 mm disc -- which is a body plan no many-legged animal or machine has
    and which gave the metachronal wave gait no fore-aft leg ordering to work with.

    Odd counts round DOWN to whole pairs here, deliberately: a segmented body is built out of modules and a
    module carries a pair. The count actually built is reported by the caller's coercion note.
    """
    from virturoid.services.animal_proportions import size_scale
    from virturoid.services.morphology_priors import body_proportions
    ask = body_proportions(prompt or "")
    s, _ = size_scale(prompt or "")
    pairs = max(2, min(SEGMENTED_MAX_PAIRS, int(n_legs) // 2))
    # THE LEG LENGTH IS A CLEARANCE BUDGET, NOT A LOOK. A real centipede's legs are ~0.12x its body length,
    # and at that ratio the shipped many-leg wave gait cannot walk this body: `crawl_gait_rollout` caps
    # ``knee_amp`` at 0.35 for a >=10-leg body, so foot lift is about 0.35 x the shank, and on a 0.165 m leg
    # (shank 0.041) that is 14 mm — less than the auto foot pad's own thickness, so the feet never clear.
    # Swept on the grounded body at 1500 steps: leg 0.165 -> 0.118 m travelled, 0.28 -> 0.179 m, 0.34 -> 0.124 m
    # (past 0.28 the actuator mass grounding adds outruns the extra reach: 57 kg -> 83 kg). 0.28 it is, which
    # puts the foot span at 2.9x the body width — house-centipede (Scutigera) proportions rather than
    # Scolopendra's, and the leggier of the two is the one that can actually be driven.
    try:
        g = build_from_anatomy(_segmented_crawler_graph(
            n_pairs=pairs,
            seg_len=round(0.092 * ask["torso"] * s, 4),
            seg_width=round(0.055 * ask["girth"] * s, 4),
            leg=round(0.280 * ask["leg"] * s, 4),
            leg_girth_mult=ask["thick"]))
    except Exception:  # noqa: BLE001
        return None
    if g.validate():
        return None
    g.composition_notes = [
        f"Segmented body plan: {pairs} articulated trunk links, one leg pair on each ({2 * pairs} legs). "
        "A rigid trunk carries at most 4 pairs, so the trunk itself articulates."]
    return g


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
    # The pairs the body is built from, plus ONE centreline appendage when the requested count is odd -- a
    # three- or five-legged request used to be rounded to the nearest pair (3 -> 2, 5 -> 4) and composed a body
    # byte-identical to a plain quadruped (#285). Radial/tentacled bodies keep the pair-only layout.
    _req = int(n_legs) if n_legs else 4
    tentacled = any(w in p for w in ("octopus", "cephalopod", "tentacle", "tentacled"))
    n_pairs = max(1, min(4, _req // 2))
    center_leg = bool(_req % 2) and not tentacled and _req >= 3
    # P4: animal PROPORTION priors so a horse (long legs), a gecko (short splayed legs) and a turtle (short wide)
    # stop composing to one generic skeleton. 1.0 for generic/dog -> the fixed defaults (byte-identical baseline).
    # ...and SIZE words scale the whole body on top of those ratios, so "a small quadruped" is not the same body
    # as "a large quadruped" (measured 2026-07-21: they were byte-identical). Neutral prompt -> (1.0, 1.0).
    from virturoid.services.animal_proportions import animal_proportions, size_scale

    from virturoid.services.morphology_priors import body_proportions
    prop = animal_proportions(p)
    ask = body_proportions(p)          # #285: what the prompt LITERALLY asked for, on top of the species prior
    s, size_mass = size_scale(p)
    try:
        g = build_from_anatomy(_generic_legged_graph(
            n_pairs=n_pairs, center_leg=center_leg, appendage_role="tentacle" if tentacled else "leg",
            body_aspect="round" if tentacled else "",
            leg=round(0.42 * prop["leg"] * ask["leg"] * s, 4),
            body=round(0.58 * prop["torso"] * ask["torso"] * s, 4),
            girth=round(0.13 * ((prop["mass"] * size_mass) ** (1.0 / 3.0)) * ask["girth"] * s, 4),
            leg_girth_mult=ask["thick"],
            # A wide/splayed stance is asked for either by the species prior (gecko, crocodile) or in so many
            # words ("a wide stance"); both reach the same fanned layout.
            fan=(prop["stance"] * ask["stance"] > 1.2),
        ))
        if tentacled:
            g.robot_class = "legged"
            g.composition_notes = ["Offline structural cue selected a round mantle with articulated tentacles; this is not a walking-leg body."]
        if not g.validate():
            return g
    except Exception:  # noqa: BLE001
        return None
    return None


def _scale_geometry(gene, s: float):
    """A PURE isometric geometric scale (NO re-grounding), so the body's PROPORTIONS -- and thus the anatomy
    critic's volume-FRACTION checks and its proven gait -- are preserved exactly; only the size changes. Lengths/
    radii/mount-offsets × s, mass × s^3 (realistic monotone mass: a small dog is lighter, a large one heavier).
    Used to fit the walkable template to a requested body size without the mass/proportion drift re-grounding adds."""
    import copy
    g = copy.deepcopy(gene)
    for seg in g.segments:
        seg.length_m = float(seg.length_m or 0.0) * s
        seg.radius_m = float(seg.radius_m or 0.0) * s
        seg.mass_kg = float(seg.mass_kg or 0.0) * s ** 3
        seg.mount_offset = tuple(float(v) * s for v in (seg.mount_offset or (0.0, 0.0, 0.0)))
        if getattr(seg, "cross_section", None) is not None:
            seg.cross_section = tuple(float(v) * s for v in seg.cross_section)
        if getattr(seg, "torque_req_nm", None) is not None:      # torque ~ mass*length ~ s^4 (re-grounded later anyway)
            seg.torque_req_nm = float(seg.torque_req_nm) * s ** 4
        # Keep a real actuator clamp (never null it): nulling dropped the walkable body's actuator-fidelity from
        # L1 to L0 before grounding ran (cert_v2 regression). Size it to the scaled load (demand ~ s^4) WITH real
        # motor margin, and never below the template's proven-walking value: a plain s^4 scale STARVES a small
        # body (s<1) so the leg can't drive the gait, while a demand-tight clamp clips the scripted gait's
        # transient peaks on a large body -- either way the template stops walking. `max(1, s^4) x margin` keeps
        # a heavier body's clamp above its (s^4) demand and a smaller one's at the template value. Grounding
        # re-sizes the real motor honestly at build time (ground_gene is idempotent).
        if getattr(seg, "actuator_torque_nm", None) is not None:
            seg.actuator_torque_nm = round(float(seg.actuator_torque_nm) * max(1.0, s ** 4) * 3.0, 3)
    return g


def _leg_len(gene) -> float:
    """The characteristic leg length of a legged body = summed length of its leg segments (the gait-relevant
    linear scale for Froude similarity). Falls back to the body's summed segment length if no legs are named."""
    import re as _re
    legs = [float(getattr(s, "length_m", 0.0) or 0.0) for s in gene.segments
            if _re.search(r"leg", (s.name or "").lower())]
    if legs:
        return sum(legs)
    return sum(float(getattr(s, "length_m", 0.0) or 0.0) for s in gene.segments) or 1.0


_STANCE_GATE_GAIT = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}


def _splay_before_substituting(gene, base: float):
    """Last resort BEFORE replacing a rolling quadruped: give it its own STANCE back. Returns ``(gene, base, fit)``.

    ORDER IS THE ETHIC ([[make-it-learn-dont-teach-it]]). Control gets the first and fairest chance -- the body has
    already had an operating point searched for it by the time this runs -- and only when that was not enough does
    any geometry move. When it does move, the body moves AS ITSELF: ``widen_stance`` rotates each leg's proximal
    MOUNT and preserves every authored segment, its geometry, its proportions and its part count. Nothing is
    substituted, nothing is rebuilt from a template.

    The angle is MEASURED per body, never assumed. ``stance_repair`` was shelved partly because its gate ran the
    default crawl while deploy ran a different gait, so a "win" could degrade on delivery; here the screen uses
    THIS BODY'S OWN cached op-point -- the controller that will actually deploy -- and the winner is then re-fitted
    for its new stance and re-judged through ``evaluate_robot``, the same gate the substitution decision uses.
    Adopted only on a strictly better verdict. 35 deg beat 25 on the horse and the cheetah, but the dog's crawl
    verdict was measured DEGRADING at 46, so the angle is chosen from evidence and never hard-coded.
    """
    from virturoid.services.gait_flywheel import fit_gait_for_body
    from virturoid.services.gait_search import evaluate_gait
    from virturoid.services.stance_repair import SPLAY_ANGLES_DEG, _leg_sides, widen_stance
    from virturoid.services.task_matched_eval import evaluate_robot

    md0 = dict(getattr(gene, "metadata", None) or {})
    fit0 = md0.get("gait_fit") or {}
    sides = _leg_sides(gene)
    if not sides:
        return gene, base, fit0
    params = {**_STANCE_GATE_GAIT, **{k: float(v) for k, v in (md0.get("gait_params") or {}).items()}}
    best_deg, best_score = 0.0, None
    for deg in SPLAY_ANGLES_DEG:
        if not deg:
            continue                                     # 0 deg is the incumbent, already measured as `base`
        r = evaluate_gait(widen_stance(gene, deg, sides), params, steps=900)
        score = (1 if r.get("credible") else 0, round(float(r.get("forward", 0.0)), 3))
        if best_score is None or score > best_score:
            best_score, best_deg = score, deg
    if not best_deg:
        return gene, base, fit0
    cand = widen_stance(gene, best_deg, sides)
    md = dict(getattr(cand, "metadata", None) or {})
    md.pop("gait_params", None)                          # the old op-point was fitted to the OLD stance
    md.pop("gait_fit", None)
    cand.metadata = md
    fit = fit_gait_for_body(cand, cache=True) or {}
    cval = float(evaluate_robot(cand).get("value", 0.0))
    if cval <= base:                                     # the splay did not earn its place -> keep the body as authored
        return gene, base, fit0
    md = dict(getattr(cand, "metadata", None) or {})
    md["stance_repair"] = {"applied": True, "roll_deg": best_deg, "from_value": round(base, 3),
                           "to_value": round(cval, 3), "screened_with": "this body's own operating point",
                           "note": (f"widened the stance {best_deg:.0f} deg per side after a per-body operating "
                                    "point was not enough on its own; every authored part is kept")}
    cand.metadata = md
    return cand, cval, fit


def ensure_walkable_quad(gene, prompt: str = "", *, force: bool = False, decline: dict | None = None):
    """DIAGNOSIS-DRIVEN, per-request walkability fallback for a QUADRUPED — a HINT the build reaches for, never a
    gait/body forced on every dog. If the body already walks under SOME gait (``evaluate_robot`` is gait-aware:
    trot, else the statically-stable crawl), it is returned UNCHANGED (sleek-when-possible). Only if it ROLLS
    OVER (the tippy-narrow-stance failure — distance stays low under any gait) do we reach for the FANNED
    wide-stance hint. To respect design intent we DO NOT silently swap out a bespoke LLM anatomy: a generic/
    heuristic quad is rebuilt fanned (its own narrow form doesn't walk anyway), while an LLM-designed creature is
    kept and merely FLAGGED (metadata.walkability) so the caller/trainer knows to widen its stance. See
    [[walking-breakthrough-abduction]]. Best-effort: any failure returns the gene untouched.

    ``decline`` is an OPTIONAL out-dict a caller passes to learn WHY nothing was adopted. It is written only on
    the unchanged-return paths, never read, and defaults to ``None`` so every existing call site behaves
    byte-identically. It exists because this function has EIGHT distinct ways of handing the gene back and the
    customer-facing operator that wraps it (``edit_operators.adopt_walkable_template``) could not tell them
    apart, so it printed one sentence for all of them. MEASURED 2026-08-12 through that operator: a real
    Menagerie ``unitree_g1`` (30 links, 33.341 kg, 0.000 m, CROUCH) and a composed 6-axis ARM both came back
    with "the original body already walks or no better template was found" — false on its first disjunct for a
    humanoid that measurably does not walk, and meaningless for a manipulator that never entered the gate. The
    ingest surface already gives the right reason for the same humanoid ("the walkable reference template is a
    QUADRUPED fan/crawl recipe; this import is a humanoid body"); two surfaces answering one question in two
    framings is the #215/#218 shape. Each key here names ONE return path so the decision cannot drift from the
    explanation."""
    try:
        from virturoid.services.task_matched_eval import evaluate_robot, robot_kind
    except Exception:  # noqa: BLE001
        if decline is not None:
            decline.update(reason="evaluator_unavailable",
                           detail="the walk evaluator could not be loaded, so no body was measured and nothing "
                                  "was changed")
        return gene
    try:
        import re
        if robot_kind(gene) != "legged":
            if decline is not None:
                decline.update(reason="not_a_legged_body", robot_kind=str(robot_kind(gene)),
                               detail=f"the walkable reference template is a LEGGED (quadruped) fan/crawl recipe "
                                      f"and this robot is a {robot_kind(gene)}; adopting it would hand back a "
                                      f"different machine, not a walking version of yours")
            return gene
        legs = {m.group(0) for s in gene.segments
                if (m := re.search(r"(?:(?:front|hind)_)?leg\d*_[lr]", (s.name or "").lower()))}
        # the fan+crawl hint is a QUADRUPED recipe: accept our own leg{n}_{l|r} naming OR any gene already CLASSIFIED
        # a quadruped (an imported robot uses arbitrary leg names but is structurally classed by limb count).
        if len(legs) != 4 and getattr(gene, "robot_class", "") != "quadruped":
            if decline is not None:
                _cls = str(getattr(gene, "robot_class", "") or "legged")
                decline.update(reason="not_a_quadruped", robot_class=_cls,
                               detail=f"the walkable reference template is a QUADRUPED fan/crawl recipe and this "
                                      f"body is a {_cls}; adopting it would hand back a different animal rather "
                                      f"than a walking version of yours. Train the real body with train_reward "
                                      f"instead -- for a non-quadruped, walking is a learned-control problem")
            return gene
        base = float(evaluate_robot(gene).get("value", 0.0))
        # WHAT CONTROLLER WAS `base` MEASURED WITH? That is the whole question, and for years the answer was "one
        # other robot's". `evaluate_robot` -> `crawl_gait_rollout` reads metadata['gait_params'] when the body has
        # one, so a body that arrives here already fitted (ai_native_tools.create_robot runs
        # gait_flywheel.fit_gait_for_body after grounding) is judged with a controller of its own, and a body that
        # does not is judged at the shipped freq 1.5 / kp 32 -- which is the canonical fanned template's hand-tuned
        # op-point and nothing else's. Measured on this checkout: the authored dog / cat / horse / cheetah all
        # score 0.000 here at the shipped op-point and all reach a CREDIBLE WALK with one of their own (0.644 /
        # 1.321 / 4.151 / 3.940 -- three of the four beating the template that is substituted for them).
        #
        # HISTORY, because the shape of the fix was found the hard way. TRIED AND REVERTED 2026-07-29: tune the
        # body HERE, before deciding. The instinct was right ([[make-it-learn-dont-teach-it]]) and the evidence
        # real, but it cannot be done at COMPOSE time -- grounding has not embodied actuator and structure mass
        # yet, so a horse is 4.2 kg here and 21 kg once grounded and the op-point is fitted to a 5x lighter robot.
        # Caching it shipped that wrong gait; not caching it made the gate incoherent (it decided "this walks WITH
        # tuning" and shipped the body WITHOUT it -- design-bench verdict@1 0.45 -> 0.40). The sound version fits
        # AFTER grounding and keeps the result, which meant moving the DECISION into the build path. It has now
        # moved; this function is the decision, and it is called from there. See docs/breaking_the_cotuning_wall.md.
        _fit = (getattr(gene, "metadata", None) or {}).get("gait_fit") or {}
        if base < 0.5 and not _fit.get("searched") and (getattr(gene, "metadata", None) or {}).get("grounding"):
            # NEVER DISCARD A BODY THAT WAS NEVER GIVEN A CONTROLLER OF ITS OWN. The caller normally fits one
            # before calling (create_robot does), but the two use different questions -- the fit asks "is the
            # default a credible crawl", this gate asks "does evaluate_robot clear 0.5" -- and a body can pass the
            # first and fail the second, in which case it would be substituted having never been searched.
            # Gated on metadata['grounding'] because that is the ONLY safe place to do it: at compose time the
            # mass is a placeholder and the op-point would be fitted to a robot a fifth of the shipped weight,
            # which is the version of this that was tried and reverted on 2026-07-29.
            try:
                from virturoid.services.gait_flywheel import fit_gait_for_body
                _fit = fit_gait_for_body(gene, cache=True) or {}
                if _fit.get("adopted"):
                    base = float(evaluate_robot(gene).get("value", 0.0))
            except Exception:  # noqa: BLE001 - fitting is an accelerant; the honest verdict stands without it
                _fit = (getattr(gene, "metadata", None) or {}).get("gait_fit") or {}
        # WHATEVER CHANCE THIS BUILD ALLOWED CONTROL, IT HAS NOW HAD IT. ``searched`` alone was the guard until
        # 2026-08-11, and it went silently unreachable the day ``tune_gait=false`` started being honoured inside
        # ``fit_gait_for_body`` (which now returns ``skipped`` without searching, so the net just above measures
        # nothing and ``searched`` can never become True). The splay then never ran and a build that declined to
        # pay for a SEARCH went straight to throwing the customer's design away.
        #
        # MEASURED on ``create_robot("a quadruped robot dog that walks", tune_gait=false)``, same prompt, same
        # commit, guard on vs off: with ``searched`` alone the authored dog (12.791 kg) is replaced by the fanned
        # template (15.000 kg, metadata.walkability_fallback) and that template TURNS OFF COURSE -- 1.228 m
        # forward for 4.66 m walked, heading swung 148 deg, travel rate decaying 0.75 -> 0.20 m/1000. The splayed
        # AUTHORED body is a CREDIBLE WALK: 5.428 m, straightness 0.807, rate holding 1.31 -> 0.90.
        #
        # DECLINING TO PAY FOR A SEARCH IS NOT CONSENT TO SHIP A DIFFERENT ROBOT. That is the same rule the net
        # above states ("NEVER DISCARD A BODY THAT WAS NEVER GIVEN A CONTROLLER OF ITS OWN"), and the splay is
        # the cheap half of it: it keeps every authored part, and under ``tune_gait=false`` it costs three short
        # screening rollouts because the re-fit inside it is a no-op too.
        # ``not_applicable`` is excluded on purpose: that is the fitter saying the crawl gait never DRIVES this
        # body, and the splay is screened with the crawl, so widening a stance on its word would be measuring a
        # controller the body does not use -- the artefact this whole gate was rebuilt to remove.
        _had_its_chance = bool(_fit.get("searched")
                               or (_fit.get("skipped") and not _fit.get("not_applicable")))
        if base < 0.5 and _had_its_chance and (getattr(gene, "metadata", None) or {}).get("grounding"):
            # Control had its fair chance and it was not enough. Before throwing the design away, try the ONE
            # change that keeps every authored part -- widening the body's own stance. See _splay_before_substituting.
            try:
                gene, base, _fit = _splay_before_substituting(gene, base)
            except Exception:  # noqa: BLE001 - stance repair is an accelerant; the honest verdict stands without it
                pass
        _own = (getattr(gene, "metadata", None) or {}).get("gait_params") or {}
        if _fit.get("searched"):
            _tried = (f"with an operating point searched for this body: {int(_fit.get('n_evals') or 0)} gaits "
                      f"evaluated, best {_fit.get('forward_m')} m, plus the trot and the crawl")
        elif _own:
            _tried = "under this body's own cached operating point, plus the trot and the crawl"
        else:
            _tried = "under the shipped default operating point (freq 1.5 / kp 32), trot and crawl"
        if base >= 0.5 and not force:                        # already walks under some gait -> leave it sleek
            if decline is not None:
                decline.update(reason="already_walks", distance_m=round(base, 3), measured=_tried,
                               detail=f"your body already walks -- {round(base, 3)} m {_tried} -- so there is "
                                      f"nothing for a template to improve and your geometry was kept")
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
                                 "measured": _tried,
                                 "hint": "widen the stance (fan the legs out) + use the crawl gait"}
            gene.metadata = md
            if decline is not None:
                decline.update(reason="bespoke_design_preserved", design_source=src, distance_m=round(base, 3),
                               detail=f"this is a bespoke design (design_source {src!r}), so its geometry was "
                                      f"preserved and only FLAGGED (metadata.walkability): {round(base, 3)} m "
                                      f"{_tried}. The hint is to widen the stance and use the crawl gait")
            return gene
        # The fanned wide-stance rebuild uses the gait-tuned CANONICAL dimensions.
        #
        # Rebuilding at the input body's own size was built and REVERTED 2026-07-21. It is the right instinct
        # (this rebuild is why differently-sized quadruped prompts could collapse onto one body) but it does not
        # work yet, for a physics reason rather than a code one: the crawl gait is tuned for THIS scale, so
        # off-scale bodies measurably walk worse and the adopt-only-if-better comparison collapsed back to the
        # canonical body anyway -- while an imported URDF, whose link names the measurement could not parse,
        # mis-read as legs six times its torso and built a 4.76 m body that CROUCHed. Per-size gait tuning is
        # the real prerequisite. Prompt-level differentiation (size words + animal proportions) is unaffected
        # and still applies; only bodies that CANNOT walk are normalised here, and that is recorded in
        # metadata.walkability_fallback rather than hidden. See docs/mvp_readiness_report.md.
        # ...but the rebuild now CARRIES THE PROPORTIONS THE PROMPT ASKED FOR (#285). Substituting a fixed
        # template was the second half of the diversity collapse: measured 2026-08-08, every offline
        # "walking robot" prompt shipped this one shape at one of three clamped scales, so whatever the
        # composer expressed was discarded here. Only the EXPLICIT ask is carried (``body_proportions``,
        # all-1.0 for a prompt that says nothing) — the species priors are deliberately not, because this
        # rebuild exists precisely for bodies whose own shape did not walk. A prompt with no proportion
        # words rebuilds the byte-identical canonical template, so nothing that ships today moves.
        from virturoid.services.morphology_priors import body_proportions
        _ask = body_proportions(prompt or "")
        cand = build_from_anatomy(_generic_legged_graph(
            # fan stays ON unless a NARROW stance was explicitly asked for: the wide fanned stance is what
            # makes this rebuild walk at all, so a neutral prompt must keep it (and stay byte-identical).
            n_pairs=2, fan=(_ask["stance"] > 0.8), girth=round(0.18 * _ask["girth"], 4),
            leg=round(0.42 * _ask["leg"], 4), body=round(0.58 * _ask["torso"], 4),
            leg_girth_mult=_ask["thick"]))
        if cand is None:
            if decline is not None:
                decline.update(reason="template_compile_failed", distance_m=round(base, 3),
                               detail="the walkable reference template could not be compiled for this request, "
                                      "so nothing was changed")
            return gene
        # B1 (2026-07-24 audit): SCALE the fanned template to the authored body's size instead of substituting a
        # fixed one, so "small dog" and "large horse" no longer collapse to one byte-identical robot (the 5-prompts
        # -> 1 render blocker). MEASURED (2026-07-24): the fanned template walks at s in ~[0.7, 1.4] under its own
        # default crawl gait via dynamic similarity (Alexander/Froude) -- and the DEFAULT gait beats a Froude-tuned
        # override (which blew kp up with the s^3 mass), so we scale GEOMETRY only and keep the robust gait. Bodies
        # bigger than ~1.4x roll (heavy + tall), so we clamp; the authored body's honest verdict is kept there.
        try:
            s = _leg_len(gene) / max(1e-6, _leg_len(cand))
            s = min(1.4, max(0.6, s))                        # the measured walkable band for the fanned crawl
            if abs(s - 1.0) > 0.03:
                cand = _scale_geometry(cand, s)                  # pure isometric scale -> proportions + gait preserved
                m1 = sum(float(getattr(x, "mass_kg", 0.0) or 0.0) for x in cand.segments)
                md0 = dict(getattr(cand, "metadata", None) or {})
                md0["scaled_to_body"] = {"scale": round(s, 3), "mass_kg": round(m1, 3),
                                         "note": "fanned walkable template scaled to the requested body size"}
                cand.metadata = md0
        except Exception:  # noqa: BLE001 - scaling is an enhancement; fall back to the unscaled template
            pass
        # A geometry fallback can survive and cover distance while still
        # rearing/rocking enough to fail the orientation credibility gate.
        # Tune the general structural crawl on the candidate before judging
        # it. The tuner is morphology-driven, confirms a candidate on a
        # longer rollout, and only caches a CREDIBLE result; it contains no
        # species branch or closed robot-class table.
        gait_tuning = None
        # COMPARE LIKE WITH LIKE. `base` is now often measured on a GROUNDED body (this gate moved onto the build
        # path, after ground_and_repair), while `cand` comes fresh out of the composer weighing a third of what it
        # will ship at. Judging a 4 kg substitute against a 13.5 kg original flatters the substitute -- exactly the
        # asymmetry that made an ungrounded gate throw good bodies away. If the incoming body was grounded, ground
        # the candidate the same way before either tuning or scoring it.
        if (getattr(gene, "metadata", None) or {}).get("grounding"):
            try:
                from virturoid.services.gene_build import ground_and_repair
                ground_and_repair(cand)
            except Exception:  # noqa: BLE001 - an ungrounded comparison is worse but must not crash the build
                pass
        try:
            from virturoid.services.morph_policy import tune_crawl_gait
            gait_tuning = tune_crawl_gait(cand, steps=800, cache=True)
        except Exception:  # noqa: BLE001 - fallback selection remains fail-safe if tuning is unavailable
            gait_tuning = None
        cval = float(evaluate_robot(cand).get("value", 0.0))
        if cval > max(base, 0.4):                            # the fanned hint walks materially better -> adopt it
            md = dict(getattr(cand, "metadata", None) or {})
            md["walkability_fallback"] = {"applied": True, "from_distance_m": round(base, 3),
                                          "to_distance_m": round(cval, 3), "hint": "fanned_stance+crawl_gait",
                                          "authored_gait": _tried, "authored_gait_fit": (_fit or None)}
            if gait_tuning is not None:
                md["walkability_fallback"]["gait_tuning"] = gait_tuning
            cand.metadata = md
            cand.design_source = src or "anatomy_generic"
            # HONESTY AT THE TOOL SURFACE (final-drive finding 2026-07-22): the substituted body inherited the
            # generic builder's note "Compiled directly from the supplied anatomy graph." — which, for a body we
            # just REPLACED, is actively false, and create_robot surfaces composition_notes while the true trace
            # sat only in gene.metadata. Say what happened where the caller will actually see it.
            #
            # AND SAY IT ACCURATELY. This note used to read "could not walk (0.0 m under ANY gait)". That was
            # false: it meant "under ONE gait" -- the shipped freq 1.5 / kp 32, which is one other robot's
            # hand-tuned op-point. Three of the four authored animals that string was written about walk credibly
            # with an op-point of their own. The claim the customer reads must be the claim we actually measured,
            # so it now names WHICH controller the body failed under and how hard we looked.
            cand.composition_notes = [
                f"Walkability fallback applied: the authored body could not walk ({round(base, 3)} m {_tried}), "
                f"so a fanned wide-stance quadruped was substituted ({round(cval, 3)} m). "
                "Details in metadata.walkability_fallback.",
            ]
            return cand
        if decline is not None:
            decline.update(reason="template_no_better", distance_m=round(base, 3),
                           template_distance_m=round(cval, 3), measured=_tried,
                           detail=f"the walkable reference template was BUILT AND MEASURED on this body's scale "
                                  f"and did not beat it: template {round(cval, 3)} m vs your {round(base, 3)} m "
                                  f"{_tried}, and it must clear both that and 0.4 m to be worth your geometry. "
                                  f"Your body was kept")
        return gene
    except Exception as _exc:  # noqa: BLE001 - the fallback is best-effort; a rolling body beats a crash
        if decline is not None:
            decline.update(reason="error", error=f"{type(_exc).__name__}: {_exc}",
                           detail="the walkability check failed part-way through, so your body was returned "
                                  "untouched rather than half-substituted")
        return gene
