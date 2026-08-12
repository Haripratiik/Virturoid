"""BILL OF MATERIALS — turn a generated gene into a real, buildable parts list.

Every robot gets its own BOM: each actuated joint is matched to the smallest real actuator that can drive it
(by torque, with a safety margin), each structural link is assigned a material, and the robot's CLASS decides
its sensor suite — a humanoid gets stereo camera "eyes" + an IMU + LiDAR; a rover gets drive wheels + motors +
a scanning LiDAR; an arm gets a wrist camera + a 6-axis force/torque sensor + a gripper. Compute and a power
pack are sized to the build. The result is a structured BOM with per-line mass/price and rolled-up totals, so
the robot is not just a shape — it is a thing you could actually source and assemble.

This reads the gene's ``actuator_torque_nm`` per joint (the compiler already sizes these from link
length/mass), so the actuator selection tracks the real mechanical demand of the body we generated.

A PART'S RATING IS NOT THE ROBOT'S CAPABILITY, AND A PART THE CUSTOMER ALREADY OWNS IS NOT A PURCHASE. Both
halves of that sentence were false in the same document, measured on a real Menagerie Unitree Go2 imported
through ``agent_tools.call_tool``. Its 12 joints declare 23.7 / 23.7 / 45.43 N.m in the customer's own file,
and those numbers survive import, grounding, and the MJCF this package ships. But the BOM's only torque was
the one inside an actuator line's detail string — the CATALOG part's rating — so ``spec_sheet`` printed
``peak_joint_torque_nm: 360.0``, 8x the truth. And the same four "Unitree M107" lines, already correctly
labelled "EQUIVALENT for a motor already fitted to your robot" and already excluded from the mass total, put
$3,600 into ``totals.price_usd``. So:

* ``joint_limits`` states the ROBOT's own per-joint capability with its provenance, so nobody downstream has
  to reconstruct it from a part number. The catalog rating still ships, on the part's own line, labelled.
* ``BomLine.in_price_total`` is the money twin of ``in_mass_total``. A catalog equivalent for hardware already
  bolted to the machine is a REPLACEMENT OPTION THE CUSTOMER HAS NOT ASKED FOR: it keeps its real price, in
  ``totals.already_fitted_usd``, and stays out of ``totals.price_usd``, which is the bill.
* ``spec_provenance`` is the sweep — every field this list reports, and whether it was READ from the customer's
  model, DERIVED by our physics, or ASSUMED from a catalog part.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass

from virturoid.schemas.gene import RobotGene
from virturoid.services.component_catalog import (
    by_category,
    component,
    material,
    select_actuator,
)

_DEFAULT_JOINT_TORQUE_NM = 6.0       # if the gene didn't size a joint, assume a modest mid-range demand
#: must match ``grounded_physics.ground_gene``'s ``margin`` default, so the BOM re-selects the SAME motor
#: grounding already fitted instead of a second, larger opinion of the same joint.
_GROUNDING_MARGIN = 1.3


@dataclass
class BomLine:
    part: str
    category: str                    # actuator | material | camera | lidar | imu | force_torque | compute |
    #                                  power | wheel | drive_motor | gripper | balance_of_system
    qty: int
    unit_mass_kg: float
    unit_price_usd: float
    detail: str
    #: Does this line's mass ADD to the robot, or is this part already inside another line's mass?
    #:
    #: It is not always a new part. On an IMPORTED robot the link masses are the manufacturer's and already
    #: contain that machine's motors; our actuator lines are catalog EQUIVALENTS for hardware the customer
    #: already owns, so adding them again claimed a real Unitree G1 -- 33.341 kg, preserved verbatim -- weighs
    #: 94.176 kg. The line still ships (you need the part number, the torque and the price); it just does not
    #: get counted a second time, and ``totals.mass_note`` says so.
    in_mass_total: bool = True
    #: Does this line's price belong in the BILL — the money this list asks the customer to SPEND?
    #:
    #: The money twin of ``in_mass_total``, and it was missing for exactly as long. Measured on a real
    #: Menagerie Go2 through ``agent_tools.call_tool``: the actuator lines were already marked "EQUIVALENT for
    #: a motor already fitted to your robot" and excluded from the mass, and then $6,480 of them (4x Unitree
    #: M107 at $900 + 8x AK80-64 at $360) went straight into ``totals.price_usd``, which the spec sheet prints
    #: as "Est. parts cost". Billing a customer for motors already bolted to their machine — motors they told
    #: us to keep — is the same claim the mass double-count made, in dollars.
    #:
    #: The line still ships at its real catalog price, because "what does a spare cost / what would replacing
    #: this run to" is a question an engineer legitimately asks. It is a REPLACEMENT OPTION THEY HAVE NOT ASKED
    #: FOR, so it is quoted in ``totals.already_fitted_usd`` and left out of the bill. Always True on a robot
    #: we designed: there is no machine yet, so every line is a purchase.
    in_price_total: bool = True
    #: Is this part FITTED to the machine, or are we PROPOSING it?
    #:
    #: On a robot we designed the whole list is a proposal and this stays False (there is no machine yet to
    #: contrast it with). On an IMPORTED robot the distinction is money: their model is the record of what they
    #: own, and everything not in it is a quote for hardware to buy. A Menagerie Go2 compiles to ``ncam 0,
    #: nsensor 0`` and we billed it $334 for an Intel RealSense D435i inside its headline price -- see
    #: ``sensor_provenance``. A proposed line still ships (the recommendation is the value); its cost is broken
    #: out in ``totals.proposed_additions_usd`` and its detail says it is not fitted.
    proposed: bool = False

    @property
    def mass_kg(self) -> float:
        return round(self.qty * self.unit_mass_kg, 4)

    @property
    def price_usd(self) -> float:
        return round(self.qty * self.unit_price_usd, 2)


# render material KEY (GeneSegment.material) -> the catalog Material it cites in the BOM.
_MATERIAL_KEY_TO_CATALOG = {
    "shell": "Polycarbonate shell", "aluminum": "Aluminum 6061-T6", "skeleton": "Aluminum 7075-T6",
    "frame": "Aluminum 6061-T6", "steel": "Steel 4140", "carbon_fiber": "Carbon-fiber composite",
    "titanium": "Titanium Ti-6Al-4V", "metal": "Hardened steel (gripper)", "rubber": "TPU (rubber)",
}
# TASK -> the load-bearing SKELETON material. Heavy work wants steel; flight/agility wants carbon-fibre;
# precision/medical wants titanium; otherwise aircraft-grade aluminium. (The outer shell, contact metal, etc.
# are set per role; only the skeleton load path is re-chosen by the task.)
_SKELETON_BY_TASK = (
    (r"heavy|payload|haul|\btow|forge|construction|load[- ]?bearing|reinforc|heavy[- ]?duty", "steel"),
    (r"aeria|drone|\bfly|flight|\bjump|leap|hop|agile|\bfast\b|sprint|lightweight|\blight\b|climb", "carbon_fiber"),
    (r"surg|medical|\blab\b|precision|micro|semiconductor|cleanroom|aerospace", "titanium"),
)

# THE CUSTOMER'S STATED MATERIAL WINS OVER OUR INFERENCE. Measured defect: "…carries a 5 kg payload, aluminium
# frame" shipped 12 Steel 4140 links, because `_SKELETON_BY_TASK` matched "payload" and nothing ever read the word
# "aluminium". The stated material is a REQUIREMENT (the customer has a machine shop, a supplier, a weight budget);
# our task heuristic is only a default for when they did not say. So: parse the request first, honour it when the
# load path allows, and when it does not, SUBSTITUTE WITH THE REASON RECORDED — never silently.
_REQUESTED_SKELETON = (
    (r"\b(carbon[- ]?fib(?:er|re)|cfrp|carbon composite)\b", "carbon_fiber"),
    (r"\b(titanium|ti[- ]?6al[- ]?4v|grade ?5 ti)\b", "titanium"),
    (r"\b(alumini?um|6061|7075|alu(?:-|\s)?alloy)\b", "aluminum"),
    (r"\b(stainless|mild steel|steel|4140|chromoly)\b", "steel"),
    (r"\b(polycarbonate|\bpc\b plastic)\b", "shell"),
    (r"\b(abs plastic|\babs\b)\b", "abs"),
    (r"\b(pla|3d[- ]?printed|fdm|printed plastic|plastic frame|plastic chassis)\b", "pla"),
)
#: material keys that are NOT a legitimate load path for a robot's skeleton, and why. A stated material from this
#: set is substituted for aluminium with the reason spelled out in the BOM.
_UNFIT_SKELETON = {
    "shell": "polycarbonate is a shell-tier polymer (creeps under sustained load)",
    "abs": "ABS is a shell-tier print material (low stiffness, layer-delaminates under joint torque)",
    "pla": "PLA is a rapid-prototype print material (low stiffness, creeps and softens above ~55 C)",
    "rubber": "TPU is a soft grip material, not a load path",
}
_SKELETON_SUBSTITUTE = "aluminum"


def _requested_skeleton(task: str) -> str | None:
    """The load-bearing material the CUSTOMER asked for in their prompt, or None if they did not say."""
    t = (task or "").lower()
    for pat, mat in _REQUESTED_SKELETON:
        if re.search(pat, t):
            return mat
    return None


def _worst_joint_torque_nm(gene: RobotGene) -> float:
    return max((abs(_joint_requirement_nm(s)) for s in gene.actuated_joints()), default=0.0)


def refine_materials_for_task(gene: RobotGene, task: str = "") -> RobotGene:
    """Resolve each part's material for the task (mutates GeneSegment.material in place, which the render colours
    by). THE CUSTOMER'S STATED MATERIAL COMES FIRST: if the prompt names one ("aluminium frame", "carbon fibre
    limbs", "titanium"), the load-bearing skeleton is built from it — unless it is not a load path at all (a print
    polymer), in which case it is substituted for aluminium and the reason is recorded on the gene
    (``metadata['material_policy']``) and reported in the BOM. Only when the customer said nothing does the task
    heuristic choose: steel (heavy), carbon-fibre (flight/agility), titanium (precision/medical), else aluminium.
    Contact parts pick up soft TPU pads when the task is delicate handling. Idempotent — once 'skeleton' is
    resolved to a concrete metal, re-running is a no-op."""
    t = (task or "").lower()
    asked = _requested_skeleton(t)
    if asked and asked in _UNFIT_SKELETON:
        tau = _worst_joint_torque_nm(gene)
        policy = {"requested": asked, "skeleton": _SKELETON_SUBSTITUTE, "honoured": False,
                  "reason": (f"{_UNFIT_SKELETON[asked]}; this robot's worst joint carries {tau:.1f} Nm, so the "
                             f"skeleton is built from {_SKELETON_SUBSTITUTE} instead. Shell/cover parts can still "
                             f"be printed in the requested material.")}
        skel = _SKELETON_SUBSTITUTE
    elif asked:
        policy = {"requested": asked, "skeleton": asked, "honoured": True,
                  "reason": "customer-specified material; the task heuristic was not applied to the load path"}
        skel = asked
    else:
        skel = "aluminum"
        why = "default structural aluminium (no material specified, no task driver)"
        for pat, mat in _SKELETON_BY_TASK:
            if re.search(pat, t):
                skel, why = mat, f"task-inferred ({mat}) — no material was specified"
                break
        policy = {"requested": None, "skeleton": skel, "honoured": True, "reason": why}
    if any(s.material == "skeleton" for s in gene.segments) or asked:
        try:
            gene.metadata["material_policy"] = policy
        except (AttributeError, TypeError):   # a gene without usable metadata is still buildable
            pass
    soft = bool(re.search(r"delicate|fragile|\bsoft\b|gentle|handle with care|egg|fruit|tissue", t))
    for s in gene.segments:
        if s.material == "skeleton":
            s.material = skel
        elif s.material == "metal" and soft:
            s.material = "rubber"
    return gene


def _material_for_key(key: str | None):
    return material(_MATERIAL_KEY_TO_CATALOG.get(key or "skeleton", "Aluminum 6061-T6"))


# TASK -> a thickness multiplier on the load-bearing SKELETON. A heavier payload demands a thicker, stronger
# frame; a flight/agility robot wants a slimmer, lighter one.
_SKELETON_THICKNESS = (
    (r"heavy|payload|haul|forge|\btow|load[- ]?bear|reinforc|rugged|heavy[- ]?duty", 1.5),
    (r"lightweight|\blight\b|aeria|drone|\bfly|flight|agile|\bfast\b|sprint|nimble|racing|climb", 0.78),
)


def _scale_geo(geo, f: float) -> None:
    """Scale a part's visual CROSS-SECTION (girth) by ``f`` in place — extrude profile, tapered/role radii, loft
    ring widths, compound sub-parts. Does NOT touch the length axis (see ``_scale_geo_length``)."""
    if not isinstance(geo, dict):
        return
    fam = geo.get("family")
    if fam == "source_mesh":                               # the customer's own STL: scale its cross-section axes
        sc = geo.get("scale")
        sc = list(sc) if isinstance(sc, (list, tuple)) and len(sc) == 3 else [1.0, 1.0, 1.0]
        sc[0] = round(float(sc[0]) * f, 6); sc[1] = round(float(sc[1]) * f, 6)
        geo["scale"] = sc
        return
    if fam == "extrude" and isinstance(geo.get("profile"), list):
        geo["profile"] = [[round(p[0] * f, 5), round(p[1] * f, 5)] for p in geo["profile"]]
    elif fam in ("tapered", "role"):
        for k in ("r0", "r1", "radius"):
            if k in geo:
                geo[k] = round(geo[k] * f, 5)
    elif fam == "loft" and isinstance(geo.get("sections"), list):
        geo["sections"] = [[z, round(hy * f, 5), round(hx * f, 5)] for z, hy, hx in geo["sections"]]
    if fam == "compound":
        for sub in geo.get("parts", []):
            _scale_geo(sub, f)


def _scale_geo_length(geo, f: float) -> None:
    """Scale a part's visual LENGTH axis by ``f`` in place, so the RENDERED link stays as long as its collider.

    #216: ``scale_group(dims='length')`` lengthened ``length_m`` (which moves the child body to the parent's new
    distal tip) but never touched the visual geometry, so a lengthened arm link RENDERED short while the gripper
    attached at the new, longer tip — the gripper appeared to float in a gap. Scaling the length-axis field of
    each geometry family (extrude ``height``, tapered/role ``length``, loft ring ``z``, compound sub-part ``at``
    offsets) keeps the visual and the kinematics in lockstep."""
    if not isinstance(geo, dict):
        return
    fam = geo.get("family")
    if fam == "source_mesh":
        # AN IMPORTED LINK IS DRAWN FROM THE CUSTOMER'S OWN BAKED STL, which has no length field to scale — so a
        # lengthened link kept its ORIGINAL drawn size while its child moved to the new, longer tip. Measured on
        # a Menagerie Unitree G1 asked to stand at 1.5 m: the collider chain stayed mated (physics fine) while
        # the RENDER came apart into a torso, a floating thigh, a floating shin and floating feet. Carry an
        # explicit per-axis mesh scale instead; gene_compiler folds it into the <mesh> asset's scale.
        sc = geo.get("scale")
        sc = list(sc) if isinstance(sc, (list, tuple)) and len(sc) == 3 else [1.0, 1.0, 1.0]
        sc[2] = round(float(sc[2]) * f, 6)                 # +z is the link's length axis
        geo["scale"] = sc
        return
    if fam == "extrude" and "height" in geo:
        geo["height"] = round(geo["height"] * f, 5)
    elif fam in ("tapered", "role") and "length" in geo:
        geo["length"] = round(geo["length"] * f, 5)
    elif fam == "loft" and isinstance(geo.get("sections"), list):
        geo["sections"] = [[round(z * f, 5), hy, hx] for z, hy, hx in geo["sections"]]
    if fam == "compound":
        for sub in geo.get("parts", []):
            _scale_geo_length(sub, f)
            at = sub.get("at")
            if isinstance(at, (list, tuple)) and len(at) == 3:   # push a stacked sub-part out along the length axis
                sub["at"] = [at[0], at[1], round(at[2] * f, 5)]


def refine_skeleton_for_task(gene: RobotGene, task: str = "") -> RobotGene:
    """Thicken (heavy work) or slim (flight/agility) the load-bearing SKELETON for the task — collider radius +
    mass + the visual beam, in place. Must run BEFORE ``refine_materials_for_task`` resolves the 'skeleton'
    tag to a concrete metal. So the design, the physics, and the BOM all reflect one coherently-sized robot."""
    t = (task or "").lower()
    f = 1.0
    for pat, fac in _SKELETON_THICKNESS:
        if re.search(pat, t):
            f = fac
            break
    if abs(f - 1.0) < 1e-3:
        return gene
    for s in gene.segments:
        if s.material != "skeleton":
            continue
        r_new = max(0.006, s.radius_m * f)
        if f > 1.0 and "leg" in (s.name or "").lower() and s.length_m > 0:
            # G6/G7: a 'heavy-duty' thicken must NOT turn walking legs back into 1:1 sausages — cap the leg
            # radius at the slenderness band (length/diameter >= 2.2, the morphology-prior minimum). D3a: /4.5
            # (aspect 2.25) not /4.4 (2.2 exactly, which rounds up past the strict gate).
            r_new = min(r_new, s.length_m / 4.5)
            f_eff = r_new / max(1e-9, s.radius_m)
        else:
            f_eff = f
        s.radius_m = round(r_new, 5)
        s.mass_kg = round(max(0.01, s.mass_kg * f_eff * f_eff), 5)
        _scale_geo(s.geometry, f_eff)
    return gene


_ALU_DENSITY = 2700.0
# the task-chosen SKELETON metals (the load path) — their mass scales vs the aluminium baseline the tuned
# masses sit at. (Shell/contact/frame parts keep their tuned masses, so a default robot's dynamics are
# unchanged; only when the task SELECTS a denser/lighter frame does the robot get heavier/lighter.)
_SKELETON_DENSITY = {"steel": 7850.0, "carbon_fiber": 1600.0, "titanium": 4430.0}


def refine_mass_for_materials(gene: RobotGene) -> RobotGene:
    """Couple the task-chosen SKELETON MATERIAL -> PHYSICS: a steel frame is ~2.9x heavier in sim than the same
    aluminium one, carbon-fibre ~0.6x lighter, titanium ~1.6x — so choosing a heavy-duty steel skeleton really
    makes the robot heavier to move, not just darker. Default (aluminium) frames are untouched, preserving the
    tuned baseline. (Size + thickness already scale mass via the collider.) One-shot, so finalize is idempotent."""
    if gene.metadata.get("_mass_materialized"):
        return gene
    for s in gene.segments:
        d = _SKELETON_DENSITY.get(s.material)
        if d:
            s.mass_kg = round(max(0.01, s.mass_kg * (d / _ALU_DENSITY)), 5)
    gene.metadata["_mass_materialized"] = True
    return gene


# TASK -> a load factor on every joint's required torque, so the BOM picks BIGGER motors (and the sim gets
# stronger ones) — a humanoid hauling a super-heavy payload needs significantly larger hip/knee actuators.
_ACTUATOR_LOAD = (
    (r"super[- ]?heavy|extreme load|massive load|very heavy|heavy[- ]?duty", 3.0),
    (r"heavy|payload|haul|load[- ]?bearing|reinforc|carry heavy|lift heavy|forge|tow", 2.0),
    (r"lightweight|\blight\b|agile|nimble|delicate|racing|drone", 0.85),
)


def refine_actuators_for_task(gene: RobotGene, task: str = "") -> RobotGene:
    """Scale every joint's required torque for the TASK load, so the BOM selects bigger/stronger actuators (and
    the sim's motors get the matching effort) for heavy-duty work — and smaller ones for a light/agile robot.
    One-shot via a flag so finalize stays idempotent."""
    if gene.metadata.get("_actuators_loaded"):
        return gene
    t = (task or "").lower()
    f = 1.0
    for pat, fac in _ACTUATOR_LOAD:
        if re.search(pat, t):
            f = fac
            break
    if abs(f - 1.0) >= 1e-3:
        for s in gene.segments:
            if s.actuator_torque_nm:
                s.actuator_torque_nm = round(s.actuator_torque_nm * f, 2)
    gene.metadata["_actuators_loaded"] = True
    return gene


def finalize_for_task(gene: RobotGene, task: str = "") -> RobotGene:
    """Apply ALL task-adaptive DESIGN choices to a composed gene before sim/export/render (mutates in place):
    tag any untagged parts, size the SKELETON thickness for the load, choose each skeleton's MATERIAL, then make
    the MASS reflect that material — so the design, the physics sim, and the BOM agree on one robot. Idempotent
    (re-running is a no-op)."""
    ensure_materials(gene)
    refine_skeleton_for_task(gene, task)
    refine_actuators_for_task(gene, task)   # bigger motors for heavy loads (BOM + sim)
    refine_materials_for_task(gene, task)
    refine_mass_for_materials(gene)
    return gene


# Infer a material tier from a segment NAME, for genes whose builder didn't tag parts (the anthropometric
# humanoid, the heuristic arm). Checked in order, most-specific first.
_NAME_MATERIAL = (
    (("hand", "foot", "paw", "gripper", "finger", "claw", "jaw", "palm", "wheel", "tool"), "metal"),
    (("head", "sensor", "skull", "face"), "shell"),
    (("torso", "chassis", "body", "pelvis", "abdomen", "trunk"), "shell"),
    (("wing", "fin", "rotor", "blade", "prop"), "carbon_fiber"),
    (("neck", "tail", "antenna", "ear", "spine", "mast"), "frame"),
    (("thigh", "shin", "calf", "uparm", "forearm", "upper", "lower", "leg", "arm", "limb", "femur", "tibia"),
     "skeleton"),
)


def ensure_materials(gene: RobotGene) -> RobotGene:
    """Give every UNTAGGED part a sensible material tier (inferred from its name/role), so a gene from ANY
    builder — not just the anatomy compiler — renders with material colours + lands in the BOM. Idempotent:
    parts already tagged keep their material."""
    for s in gene.segments:
        if s.material:
            continue
        if s.parent is None:
            s.material = "shell"
            continue
        if s.joint_type == "prismatic":
            s.material = "metal"
            continue
        nm = s.name.lower()
        s.material = next((mat for keys, mat in _NAME_MATERIAL if any(k in nm for k in keys)), "skeleton")
    return gene


# TASK -> extra sensors. The TASK the robot must do (not just its body class) decides perception: a navigator
# needs a scanning LiDAR, an inspector a thermal core, a sorter a detection camera, a fragile-contact task a
# force/torque sensor. Keyword-matched against the prompt + capabilities so any build adapts to its job.
_NAV_TASK = re.compile(r"navigat|\bmap\b|mapping|explore|patrol|slam|autonom|deliver|\brove|warehouse|outdoor")
_TASK_SENSORS: tuple[tuple[str, str, int, str], ...] = (
    (r"inspect|thermal|\bheat\b|temperatur|\bleak|hotspot|electrical fault",
     "FLIR Boson 640", 1, "task: thermal inspection camera"),
    (r"outdoor|in the field|\bgps\b|gnss|survey|agricultur|orchard|construction site",
     "u-blox ZED-F9P RTK GPS", 1, "task: RTK GNSS positioning"),
    (r"sort|pick|grasp|assembl|\bbin\b|by color|by colour|label|defect|inventory",
     "Luxonis OAK-D Pro", 1, "task: object-detection camera"),
    (r"fragile|delicate|\bforce\b|contact|polish|insert|peg|deburr|wipe",
     "Robotiq FT 300-S", 1, "task: contact force/torque sensing"),
    (r"voice|speech|listen|convers|social|\bhear|command",
     "ReSpeaker Mic Array v2", 1, "task: far-field microphone array"),
    (r"follow|track|\bperson|human|crowd|guide",
     "Intel RealSense D435i", 1, "task: person-tracking depth camera"),
)


def _dedupe_components(items):
    """Collapse duplicate component picks by name (class baseline + task additions can overlap) — keep the
    highest qty and the first mounting note."""
    merged: dict = {}
    for name, qty, mount in items:
        if name in merged:
            merged[name] = (max(merged[name][0], qty), merged[name][1])
        else:
            merged[name] = (qty, mount)
    return [(n, q, m) for n, (q, m) in merged.items()]


#: A LiDAR's price is set by its RANGE, and range is a task requirement — not a proxy for how heavy the robot is.
#: Sizing by mass alone put an $8,000 Ouster OS1-32 on a 60 kg humanoid and an $18,000 OS2-128 on a 108 kg one,
#: while every real machine of that class (Go2, B2, Unitree H1, Spot's nav payload) ships a ~$749 Livox Mid-360.
#: So: buy the long-range unit when the JOB needs long range (outdoors, survey, autonomous driving, mapping a
#: yard/site), and buy the class-standard indoor 3D unit otherwise.
_LONG_RANGE_TASK = re.compile(
    r"outdoor|in the field|field robot|orchard|agricultur|forest|construction site|survey|highway|road|street|"
    r"\bdriv\w*|autonomous vehicle|long[- ]?range|\bacre|\bfarm|mine|quarry|perimeter|\byard\b", re.I)


def _pick_lidar(scale_kg: float, task_blob: str = "") -> str:
    """The LiDAR the JOB needs. A long-range outdoor/survey job buys the long-range unit (and a big machine on
    that job buys the 128-beam); an indoor navigator buys the class-standard 3D dome; a small indoor robot with
    no 3D need gets a 2D puck. Cost follows the requirement, not the robot's mass."""
    long_range = bool(_LONG_RANGE_TASK.search(task_blob or ""))
    if long_range:
        return "Ouster OS2-128" if scale_kg > 55 else "Ouster OS1-32"
    if scale_kg > 7:
        return "Livox Mid-360"
    return "Slamtec RPLIDAR A2M12"


def _sensor_suite(robot_class: str, capabilities, task_text: str = "", scale_kg: float = 0.0) -> list[tuple[str, int, str]]:
    """(component_name, qty, mounting) the robot needs — a CLASS baseline (a humanoid's camera eyes, an arm's
    F/T) PLUS task-adaptive perception (what it must DO) PLUS size-adaptive perception (a bigger robot gets a
    longer-range LiDAR + extra camera coverage). So the parts scale with the robot type, task AND size."""
    cls = (robot_class or "").lower()
    blob = ((task_text or "") + " " + " ".join(capabilities or [])).lower()
    suite: list[tuple[str, int, str]] = []
    wants_lidar = bool(_NAV_TASK.search(blob))
    if cls in ("humanoid", "biped"):
        suite += [("Intel RealSense D435i", 2, "head: stereo camera eyes"),
                  ("VectorNav VN-100", 1, "torso: balance/AHRS IMU")]
        wants_lidar = True                            # a humanoid navigates -> always a 3D LiDAR (size-picked)
    elif cls in ("quadruped", "legged"):
        suite += [("Intel RealSense D435i", 1, "head: forward depth camera"),
                  ("Bosch BNO055", 1, "trunk: body IMU")]
    elif cls in ("manipulator", "arm"):
        suite += [("Luxonis OAK-D Pro", 1, "wrist: eye-in-hand camera"),
                  ("Bosch BNO055", 1, "base: IMU")]
    elif cls in ("mobile_base", "mobile", "rover"):
        suite += [("Intel RealSense D435i", 1, "front: depth camera"),
                  ("Bosch BNO055", 1, "chassis: IMU")]
        wants_lidar = True
    else:
        suite += [("Intel RealSense D435i", 1, "forward depth camera"),
                  ("Bosch BNO055", 1, "body IMU")]

    for pat, name, qty, mount in _TASK_SENSORS:
        if re.search(pat, blob):
            suite.append((name, qty, mount))
    if wants_lidar:                                   # navigation perception -> the LiDAR the JOB's range needs
        suite.append((_pick_lidar(scale_kg, blob), 1, "mast: scanning LiDAR for navigation"))
    if scale_kg > 25 and cls not in ("manipulator", "arm"):   # a BIG robot needs 360 coverage + redundancy
        suite.append(("Intel RealSense D435i", 1, "rear: extra depth camera (360 coverage)"))
        suite.append(("VectorNav VN-100", 1, "redundant high-grade IMU"))
    return _dedupe_components(suite)


# Prompt hints that flip power AWAY from the class default. A robot that carries its own body around is
# battery-powered by construction (see _CARRIES_ITS_OWN_POWER); a bolted-down machine runs off the wall. These
# only matter when the customer OVERRIDES that: "a tethered treadmill test rig" (socket) or "a battery-powered
# benchtop arm" (battery).
_BATTERY_HINTS = re.compile(
    r"\b(batter|portable|untether|cordless|wireless|roam|off[- ]?grid|on[- ]the[- ]go|handheld|backpack|"
    r"drone|free[- ]?roam|wander|mobile robot|in the field|field robot)\w*", re.I)
_SOCKET_HINTS = re.compile(
    r"\b(bench(?!mark)|bench[- ]top|tethered|wall[- ]?power|mains[- ]?power|stationary|fixed[- ]base|corded|"
    r"plugged[- ]?in|desktop|tabletop|treadmill|test rig)\w*", re.I)

# THE ROBOT KIND DECIDES WHERE ITS POWER COMES FROM. A legged/mobile/flying robot moves its whole body away from
# the socket, so a 4.2 kg enclosed wall PSU is not a candidate part for it at any wattage — it is a lab bench
# supply. A bolted-down arm/gantry/SCARA legitimately runs off the wall. Measured defect: every walking robot we
# generated shipped a "Mean Well RSP-1500-48 (socketed/wall)" because the default was "wall unless the prompt
# says battery", and no prompt has to say "battery" to mean a walking robot.
_CARRIES_ITS_OWN_POWER = {"humanoid", "biped", "quadruped", "legged", "hexapod", "multiped", "crawler",
                          "mobile_base", "mobile", "rover", "wheeled", "tracked", "aerial", "aquatic",
                          "swimmer", "snake", "serpentine"}
_FIXED_BASE = {"manipulator", "arm", "gantry", "scara", "delta", "turret", "rail", "cell", "fixture"}
# Real parts for sizing: wall PSUs (name, supply watts, rail V) and battery packs (name, usable Wh, rail V).
_WALL_PSUS: tuple[tuple[str, float, float], ...] = (
    ("Mean Well RSP-150-24", 150.0, 24.0), ("Mean Well RSP-320-48", 320.0, 48.0),
    ("Mean Well RSP-750-48", 750.0, 48.0), ("Mean Well RSP-1500-48", 1500.0, 48.0))
_BATTERY_PACKS: tuple[tuple[str, float, float], ...] = (
    ("LiPo 4S 5200mAh (14.8V)", 77.0, 14.8), ("LiPo 6S 8000mAh (22.2V)", 178.0, 22.2),
    ("Li-ion 48V 12Ah pack", 576.0, 48.0))
#: The runtime a battery is sized FOR — stated in the BOM next to the pack it bought, so "how long does it run"
#: is answered by the parts list instead of implied. A flying robot is sized for a realistic flight, not an hour.
_TARGET_RUNTIME_H = {"aerial": 0.5}
_DEFAULT_RUNTIME_H = 1.0
_USABLE_SOC = 0.80          # a Li-ion/LiPo pack is sized on usable energy, not nameplate (20% reserve)
_PSU_HEADROOM = 1.4         # a wall supply is bought with headroom over the continuous draw
_VISION_RE = re.compile(r"vision|camera|visual|perceiv|detect|recogn|inspect|\bsee\b|slam|navigat|\bmap\b", re.I)
_LIDAR_RE = re.compile(r"lidar|point[- ]?cloud|3d map|slam|navigat", re.I)


def _select_compute(robot_class: str, n_actuators: int, task: str, capabilities) -> tuple[str, str]:
    """Size the 'brain' from the robot's actual COMPUTE LOAD — DOF count + vision/LiDAR/SLAM + whole-body
    class — not the class alone. A simple low-DOF arm gets a Pi; a vision-guided or whole-body robot gets an
    AGX Orin. (Real boards from the catalog's compute tier.)"""
    blob = f"{task or ''} {' '.join(capabilities or [])}"
    has_vision = bool(_VISION_RE.search(blob))
    has_lidar = bool(_LIDAR_RE.search(blob))
    whole_body = (robot_class or "").lower() in ("humanoid", "biped")
    load = n_actuators + (8 if has_vision else 0) + (6 if has_lidar else 0) + (10 if whole_body else 0)
    # whole-body bipeds need the top board (real humanoids run 25-75 DOF + perception; our gene under-counts it);
    # everything else scales by DOF + sensing load.
    board = ("NVIDIA Jetson AGX Orin 64GB" if whole_body or load >= 24
             else "NVIDIA Jetson Orin Nano 8GB" if load >= 9
             else "Raspberry Pi 5 (8GB)")
    drivers = [d for d, on in (("whole-body", whole_body), ("vision", has_vision), ("LiDAR/SLAM", has_lidar)) if on]
    why = f"compute load {load} ({n_actuators} DOF{', ' + ', '.join(drivers) if drivers else ''})"
    return board, why


# --- THE POWER BUDGET: computed ONCE, quoted everywhere -----------------------------------------------------
#
# The old code computed the robot's draw TWICE by two different routes, so one package carried three different
# power numbers: totals said 800.6 W (electronics datasheet draw + an actuator bus), the PSU line said
# "1500 W >= 1133 W draw" (1133 = a DIFFERENT draw, 809 W, multiplied by the 1.4 PSU headroom and then LABELLED
# as the draw), and the part itself was a 1500 W supply. Nothing in the package told a buyer which number to
# believe. There is now exactly one `PowerBudget`; the totals, the power-source sizing and the detail string all
# read the same fields off it, and the headroom is printed AS headroom.
#
# The actuator term also had to change. It was `rated_torque x max_speed x 0.3` summed over the SELECTED motors —
# the motor's corner power, at an operating point (full rated torque AND full no-load speed at once) that no motor
# ever reaches. It also made the draw a function of WHICH MOTOR WE BOUGHT: a motor two rungs oversized reported
# twice the power, though a robot draws what its load demands, not what its motor could deliver. On the generated
# quadruped it produced 1409 W for a 14 kg machine whose real-world equivalent draws ~150-250 W walking.
# The load-driven model below: each joint's own torque REQUIREMENT x a class-typical joint speed x a phase factor
# (torque peaks in stance, speed peaks in swing — their product never averages to the corner value) / drivetrain
# and drive efficiency, plus a per-axis driver/holding allowance. Conservative by design (a sizing figure), and
# independent of how much motor was bought to cover the joint.
_JOINT_SPEED_RADPS = {"humanoid": 3.0, "biped": 3.0, "quadruped": 3.0, "legged": 3.0, "hexapod": 3.0,
                      "manipulator": 2.0, "arm": 2.0, "gantry": 1.5, "scara": 2.5,
                      "mobile_base": 2.5, "mobile": 2.5, "rover": 2.5}
_DEFAULT_JOINT_SPEED_RADPS = 2.5
_TORQUE_SPEED_PHASE = 0.20      # peak torque and peak speed happen at different times in a stride/stroke
_DRIVETRAIN_EFF = 0.65          # motor + gearbox + drive electronics, continuous
_AXIS_QUIESCENT_W = 2.0         # per driven axis: servo driver + encoder + holding current


@dataclass
class PowerBudget:
    """One robot, one power budget. ``actuators_w`` is the load-driven continuous mechanical draw of the driven
    axes; ``electronics_w`` is the datasheet draw of every non-actuator line in the BOM."""
    actuators_w: float
    electronics_w: float
    n_axes: int
    joint_speed_radps: float

    @property
    def total_w(self) -> float:
        return round(self.actuators_w + self.electronics_w, 1)

    def to_dict(self) -> dict:
        return {"total_draw_w": self.total_w, "actuators_w": round(self.actuators_w, 1),
                "electronics_w": round(self.electronics_w, 1), "driven_axes": self.n_axes,
                "basis": (f"per axis: joint torque requirement x {self.joint_speed_radps:g} rad/s x "
                          f"{_TORQUE_SPEED_PHASE:g} phase factor / {_DRIVETRAIN_EFF:g} drivetrain efficiency, "
                          f"+ {_AXIS_QUIESCENT_W:g} W driver/holding per axis; electronics from datasheets. "
                          f"Continuous sizing figure, not a stall peak.")}


def _actuator_bus_w(joint_reqs, robot_class: str) -> tuple[float, float]:
    """Continuous electrical draw of the driven axes, from the JOINTS' torque requirements. Returns
    (watts, the class-typical joint speed used) so the figure is reproducible from the BOM itself."""
    w = _JOINT_SPEED_RADPS.get((robot_class or "").lower(), _DEFAULT_JOINT_SPEED_RADPS)
    mech = sum(abs(float(r)) * w * _TORQUE_SPEED_PHASE for r in joint_reqs)
    return mech / _DRIVETRAIN_EFF + _AXIS_QUIESCENT_W * len(list(joint_reqs)), w


def _wants_battery(robot_class: str, task: str) -> tuple[bool, str]:
    """Does this robot carry its own power? The ROBOT KIND decides; the prompt can override it either way."""
    cls = (robot_class or "").lower()
    blob = task or ""
    if cls == "aerial":
        return True, "a flying robot cannot be tethered to a wall supply"
    said_battery = bool(_BATTERY_HINTS.search(blob))
    said_socket = bool(_SOCKET_HINTS.search(blob))
    if cls in _CARRIES_ITS_OWN_POWER:
        if said_socket and not said_battery:
            return False, f"a {cls} is normally battery-powered; the prompt asked for a tethered/benchtop build"
        return True, f"a {cls} moves its whole body, so it carries its own power - a wall cord is not an option"
    if cls in _FIXED_BASE:
        if said_battery and not said_socket:
            return True, f"a {cls} is normally wall-powered; the prompt asked for a portable/battery build"
        return False, f"a {cls} is a fixed-base machine, so it runs off a socketed supply"
    return (True, "the prompt asked for untethered operation") if (said_battery and not said_socket) \
        else (False, "unknown class, no untethered requirement stated — socketed supply")


def _select_power(robot_class: str, task: str, budget: PowerBudget,
                  bus_v: float = 48.0) -> tuple[str, int, str]:
    """Choose the power SOURCE from the robot KIND and the ONE power budget. A robot that carries itself gets a
    battery sized for a STATED runtime at the budgeted draw (and more than one pack when one is not enough — the
    old code silently returned 'the largest pack' and let the runtime be whatever it was); a fixed-base machine
    gets a wall PSU with stated headroom. Every number in the detail string is the budget's, or is labelled as
    the margin applied to it."""
    draw = max(1.0, budget.total_w)
    battery, why = _wants_battery(robot_class, task)
    if battery:
        target_h = _TARGET_RUNTIME_H.get((robot_class or "").lower(), _DEFAULT_RUNTIME_H)
        need_wh = draw * target_h / _USABLE_SOC
        fits = [(n, wh) for n, wh, v in _BATTERY_PACKS if v >= bus_v * 0.9] or \
               [(n, wh) for n, wh, _v in _BATTERY_PACKS]
        for name, wh in fits:
            if wh >= need_wh:
                hrs = wh * _USABLE_SOC / draw
                return name, 1, (f"battery ({why}): {wh:g} Wh pack >= {need_wh:.0f} Wh needed for "
                                 f"{target_h:g} h at the {draw:.0f} W budgeted draw "
                                 f"({_USABLE_SOC:.0%} usable) -> ~{hrs:.1f} h runtime")
        name, wh = fits[-1]
        qty = max(1, int(-(-need_wh // wh)))                 # parallel packs until the runtime is really met
        hrs = qty * wh * _USABLE_SOC / draw
        return name, qty, (f"battery ({why}): {qty}x {wh:g} Wh = {qty * wh:g} Wh >= {need_wh:.0f} Wh needed for "
                           f"{target_h:g} h at the {draw:.0f} W budgeted draw ({_USABLE_SOC:.0%} usable) "
                           f"-> ~{hrs:.1f} h runtime")
    need_w = max(60.0, draw * _PSU_HEADROOM)
    fits = [(n, w) for n, w, v in _WALL_PSUS if v >= bus_v * 0.9] or [(n, w) for n, w, _v in _WALL_PSUS]
    for name, w in fits:
        if w >= need_w:
            # say WHY when the supply is much bigger than the requirement: it is the smallest one that serves the
            # actuator rail, not a bigger-is-better pick. Without this it reads like the over-sizing bug it isn't.
            floor = " (the smallest supply on this actuator rail)" if w > need_w * 1.6 else ""
            return name, 1, (f"socketed PSU ({why}): {w:g} W / {bus_v:g} V supply{floor} for a {draw:.0f} W "
                             f"budgeted draw ({_PSU_HEADROOM:g}x headroom = {need_w:.0f} W required)")
    name, w = fits[-1]
    qty = max(1, int(-(-need_w // w)))
    return name, qty, (f"socketed PSU ({why}): {qty}x {w:g} W = {qty * w:g} W for a {draw:.0f} W budgeted draw "
                       f"({_PSU_HEADROOM:g}x headroom = {need_w:.0f} W required)")


def _select_compute_line(robot_class: str, n_actuators: int, task: str, capabilities) -> tuple[str, int, str]:
    board, why = _select_compute(robot_class, n_actuators, task, capabilities)
    return board, 1, why


def _mobile_drive(gene: RobotGene) -> list[tuple[str, int, str]]:
    """Wheels + drive motors for a mobile base (count from the gene's wheel-ish segments, else a 2+caster default)."""
    if (gene.robot_class or "").lower() not in ("mobile_base", "mobile", "rover"):
        return []
    wheels = sum(1 for s in gene.segments if "wheel" in s.name.lower()) or 2
    return [("Pololu 37Dx70L 12V gearmotor + encoder", wheels, "differential drive"),
            ("100mm rubber drive wheel + hub", wheels, "driven wheels"),
            ("Caster wheel 2in", 1, "passive balance caster")]


# --- ACTUATOR SKU POLICY ------------------------------------------------------------------------------------
#
# Sizing every joint independently is correct engineering and a terrible parts list. Measured on a generated
# 16-joint humanoid: 6 different motor SKUs (Harmonic Drive FHA-40C-160, Unitree M107, T-Motor AK80-64, AK10-9,
# Dynamixel XC430, XL330) from four vendors. Nobody builds a robot that way — it means six sets of drivers,
# firmware, cabling, spares and calibration procedures for one machine, and the cost/spares story is nonsense.
# Real robots standardise: a Go2 runs one motor family; an H1 runs two or three sizes of the same family.
#
# THE RULE (reported in the BOM as ``actuator_policy``): size every joint from its own torque requirement, then
# STANDARDISE onto at most ``_MAX_ACTUATOR_SKUS`` part numbers by rolling the smallest-quantity group up into the
# next-larger part already on the list. A roll-up is only legal while the standard part stays within
# ``_SKU_UPSPEC_LIMIT`` x the joint's own peak requirement — past that the extra motor mass hanging off the limb
# costs more than the standardisation saves, so the SKU is kept and the BOM says why. Roll-ups only ever go UP:
# no joint is ever handed a motor that cannot drive it.
_MAX_ACTUATOR_SKUS = 3
_SKU_UPSPEC_LIMIT = 8.0
#: …and standardising is a PROCUREMENT convenience, so it must not cost more than it saves. Measured: rolling a
#: humanoid's single 53 Nm torso joint up into its 520 Nm leg part removed one SKU and added $3,900 — 11% of the
#: whole robot to avoid stocking one spare. A roll-up may add at most 10% of the actuator budget, and small
#: absolute swaps (a $24 servo to a $60 one) are always allowed regardless of percentage.
_SKU_ROLLUP_COST_FRAC = 0.10
_SKU_ROLLUP_COST_FLOOR_USD = 250.0


def _joint_requirement_nm(seg) -> float:
    """The joint's TORQUE REQUIREMENT — the pinned ``torque_req_nm`` on a grounded body (``actuator_torque_nm``
    there is the SELECTED motor's peak, i.e. a capacity, not a demand), else the authored demand."""
    req = getattr(seg, "torque_req_nm", None)
    if req:
        return float(req)
    return float(getattr(seg, "actuator_torque_nm", None) or _DEFAULT_JOINT_TORQUE_NM)


# --- WHAT THE ROBOT CAN DO vs WHAT THE PART IS RATED FOR ----------------------------------------------------
#
# MEASURED DEFECT (2026-08-12, real MuJoCo Menagerie Go2 through ``agent_tools.call_tool`` -> ``export_held``).
# The shipped ``robot.xml`` in the package declares ``forcerange="-45.43 45.43"`` on the calf joints and
# ``-23.70 23.70`` on hip/thigh — the customer's own numbers, preserved by ``grounded_physics.ground_gene``.
# ``reports/spec_sheet.md`` in the SAME package said:
#
#     | Peak joint torque | 360.0 Nm |
#
# 8x the truth, because the only torque anywhere in the BOM was the one embedded in an actuator line's detail
# string ("peak 360 Nm @ 48 V, gear 1:1"), and that is the CATALOG PART's rating. The part is by construction
# the smallest one we stock that COVERS the joint, so its rating is always >= the robot's — reading it as the
# robot's capability can only ever overstate, and on this machine it overstated by 8x.
#
# So the BOM now states the robot's own capability itself, per joint, with its provenance, instead of leaving
# every downstream reader to reconstruct it from a part number. Two different quantities, two different fields,
# and the one named after the ROBOT is read from the ROBOT.
def _robot_joint_limits(gene, joints) -> dict:
    """THE ROBOT's own actuation capability + where each number came from — never a catalog part's rating.

    Per actuated joint: the torque (or, on a slider, force) limit the SHIPPED MODEL declares, which is what
    ``gene_compiler`` writes into the MJCF ``forcerange``/``ctrlrange`` and what the physics was run at. On an
    IMPORTED machine that is the manufacturer's number, read out of the customer's file at import and protected
    through grounding by ``source_declared_torques``; on a robot we designed it is the peak of the actuator we
    fitted, which genuinely IS that robot's capability because we chose it.

    Torque and force are kept apart. ``actuator_torque_nm`` on a prismatic joint carries a LINEAR FORCE in
    newtons (see ``grounded_physics.ground_gene``), so a max taken across both would be comparing N.m with N.
    """
    try:
        from virturoid.services.grounded_physics import source_declared_torques
        declared = source_declared_torques(gene) or {}
    except Exception:  # noqa: BLE001 - provenance is value-add; a limit with no provenance still beats a part rating
        declared = {}
    where = ((getattr(gene, "metadata", None) or {}).get("source_actuator_torque_where") or {})
    per: dict[str, dict] = {}
    for s in joints:
        nm = float(declared.get(s.name) or 0.0) or abs(float(getattr(s, "actuator_torque_nm", 0.0) or 0.0))
        if not nm:
            continue
        row = {"limit": round(nm, 3),
               "unit": "N" if s.joint_type == "prismatic" else "N.m",
               "source": "your model" if s.name in declared else "the actuator we fitted"}
        if s.name in declared:
            row["declared_in"] = str(where.get(s.name) or "the source model")
        per[str(s.name)] = row
    return _aggregate_joint_limits(per)


def _aggregate_joint_limits(per: dict[str, dict]) -> dict:
    """Roll per-joint limits up into the headline capability + the sentence that keeps it distinguishable from
    a part rating. Torque (N.m) and linear force (N) are aggregated separately — they are not the same unit."""
    rot = {k: v for k, v in per.items() if v["unit"] == "N.m"}
    lin = {k: v for k, v in per.items() if v["unit"] == "N"}
    n_declared = sum(1 for v in per.values() if v["source"] == "your model")
    out: dict = {
        "per_joint": per,
        "n_joints": len(per),
        "n_declared_by_your_model": n_declared,
        "n_sized_by_us": len(per) - n_declared,
        "declared_by_your_model": n_declared > 0,
    }
    if rot:
        worst = max(rot, key=lambda k: rot[k]["limit"])
        out["peak_joint_torque_nm"] = round(rot[worst]["limit"], 3)
        out["peak_joint"] = worst
    if lin:
        worst = max(lin, key=lambda k: lin[k]["limit"])
        out["peak_joint_force_n"] = round(lin[worst]["limit"], 3)
        out["peak_prismatic_joint"] = worst
    out["note"] = (
        "THIS IS THE ROBOT'S CAPABILITY, not the rating of the catalog part on the actuator lines. "
        + (f"{n_declared} of {len(per)} joint limit(s) were READ from your own model" if n_declared else
           "every limit here is the peak of the actuator this design fitted, which is this robot's capability "
           "because we chose the motor")
        + ". A catalog part is selected as the smallest one that COVERS its joint, so its rating is always at "
          "or above the joint's own limit; quoting the part number as the robot's capability can only overstate "
          "it, and on a real Unitree Go2 it overstated 45.43 N.m as 360 N.m.")
    return out


#: WHICH NUMBERS ARE THE CUSTOMER'S AND WHICH ARE OURS — one table, shipped with the parts list.
#:
#: The peak-torque defect was not a lone bug; it was the visible end of a category. A BOM mixes three kinds of
#: number and prints them in one font: values READ from the customer's own model, values DERIVED by our physics,
#: and values ASSUMED from whichever catalog part we selected. Only the first is a fact about their machine.
#: This is the sweep of everything ``bom_builder`` and ``spec_sheet`` report, per field, so a reader never has
#: to guess which kind they are looking at. ``sensor_provenance`` does the same job for perception and is
#: referenced rather than duplicated here.
_ASSUMED = "assumed from the catalog part"
_MODELLED = "computed by our model"


def _spec_provenance(gene, preserved: bool, joint_limits: dict, has_sensor_table: bool) -> dict:
    """READ vs DERIVED vs ASSUMED, field by field, for every number this BOM (and the spec sheet built from it)
    reports about the robot."""
    imported = bool(str(((getattr(gene, "metadata", None) or {}).get("imported_from")) or ""))
    n_read = int(joint_limits.get("n_declared_by_your_model") or 0)
    rows: list[dict] = [
        {"field": "joint_limits.peak_joint_torque_nm / per_joint",
         "source": "READ from your model" if n_read else "derived: the peak of the actuator this design fitted",
         "evidence": (f"{n_read}/{joint_limits.get('n_joints')} joint limits taken verbatim from your file "
                      f"(actuator forcerange / a force-mode motor's ctrlrange / a joint's actuatorfrcrange) and "
                      f"protected through grounding" if n_read else
                      "this robot does not exist yet; we chose the motor, so its peak IS the joint's limit")},
        {"field": "lines[actuator].detail — peak/rated torque, voltage, gear ratio, no-load speed",
         "source": _ASSUMED,
         "evidence": ("the selected part's datasheet. NONE of it is read from your model: MJCF/URDF declare a "
                      "torque limit, not a bus voltage or a gearbox ratio. On an imported machine the part is a "
                      "catalog EQUIVALENT chosen to cover your joint, so its ratings sit at or above yours.")},
        {"field": "power_budget.* / totals.est_power_w", "source": _MODELLED,
         "evidence": ("joint torque requirement x a class-typical joint speed x a phase factor / drivetrain "
                      "efficiency + a per-axis driver allowance — see power_budget.basis. The joint SPEED is a "
                      "class table, not a number from your model; no electrical current is measured or claimed "
                      "anywhere in this list.")},
        {"field": "lines[*].price_usd / totals.*_usd", "source": _ASSUMED,
         "evidence": "single-unit list prices for representative real parts (~2024) — see cost_drivers.basis"},
        {"field": "lines[material].mass_kg / totals.mass_kg",
         "source": "READ from your model" if preserved else "derived from geometry x material density",
         "evidence": ("your own per-link masses, preserved verbatim at import and de-duplicated against the "
                      "actuator lines" if preserved else
                      "link volume x the task-chosen material's density x fill, plus each link's actuator")},
        {"field": "lines[material].part / density / tier", "source": _ASSUMED,
         "evidence": ("the material our task/prompt heuristic chose — see material_policy. An imported model "
                      "declares masses and inertias, never what the part is made of.")},
        {"field": "lines[compute] / lines[power]", "source": _ASSUMED,
         "evidence": ("sized from DOF + perception load and from the power budget above. A robot model cannot "
                      "declare its battery or its compute board, so we can neither confirm nor deny the one you "
                      "already have — these are our recommendation, not an inventory of your machine.")},
        # no pipes in a field name -- this table is also rendered as markdown, where a pipe ends the cell.
        {"field": "lines[camera / imu / lidar / force_torque / ...]",
         "source": "see sensor_provenance (per line)" if has_sensor_table else _ASSUMED,
         "evidence": ("each perception line says whether your model declares that instrument; the ones it does "
                      "not are marked PROPOSED and priced separately" if has_sensor_table else
                      "chosen from robot_class + task keywords + robot mass")},
    ]
    return {
        "robot_is": "your imported machine" if imported else "a design of ours",
        "rule": ("A number READ from your model is a fact about your robot; a number from a catalog part is a "
                 "fact about the part we would buy. They are never printed as the same thing." if imported else
                 "This robot does not exist yet, so every part is a proposal — but the JOINT LIMITS are still "
                 "this design's own, not a restatement of the part rating that happens to cover them."),
        "fields": rows,
    }


def _standardize_actuators(picks: list[tuple[str, object, float]]) -> tuple[dict[str, str], dict]:
    """Roll the per-joint picks onto a small number of part numbers. ``picks`` is (joint_name, Actuator, req_nm).
    Returns (joint -> ORDERED part name, policy dict explaining the rule and every roll-up)."""
    act = {a.name: a for _n, a, _r in picks}
    by_sku: dict[str, list[tuple[str, float]]] = {}
    for jn, a, r in picks:
        by_sku.setdefault(a.name, []).append((jn, r))
    budget_usd = sum(act[s].price_usd * len(js) for s, js in by_sku.items())
    cost_cap = max(_SKU_ROLLUP_COST_FLOOR_USD, _SKU_ROLLUP_COST_FRAC * budget_usd)
    rolled: list[dict] = []
    blocked: list[dict] = []
    frozen: set[str] = set()                       # SKUs that provably cannot be rolled up
    while len(by_sku) > _MAX_ACTUATOR_SKUS:
        cands = [s for s in by_sku if s not in frozen]
        if not cands:
            break
        smallest = min(cands, key=lambda s: (act[s].peak_torque_nm, len(by_sku[s])))
        bigger = sorted((s for s in by_sku if act[s].peak_torque_nm > act[smallest].peak_torque_nm),
                        key=lambda s: act[s].peak_torque_nm)
        target = bigger[0] if bigger else None
        n = len(by_sku[smallest])
        worst = max((r for _j, r in by_sku[smallest]), default=0.0)
        need = max(worst * _GROUNDING_MARGIN, 1e-6)
        added = (act[target].price_usd - act[smallest].price_usd) * n if target else 0.0
        if target is None:
            reason = "already the largest part on the list"
        elif act[target].peak_torque_nm > _SKU_UPSPEC_LIMIT * need:
            reason = (f"the next standard part up ({target}, {act[target].peak_torque_nm:g} Nm) would over-spec "
                      f"this group's {worst:.2f} Nm joint by {act[target].peak_torque_nm / need:.0f}x "
                      f"(limit {_SKU_UPSPEC_LIMIT:g}x)")
        elif added > cost_cap:
            reason = (f"standardising {n} joint(s) onto {target} would add ${added:,.0f} to the actuator budget "
                      f"(cap ${cost_cap:,.0f}); one fewer spare is not worth that")
        else:
            reason = ""
        if reason:
            blocked.append({"sku": smallest, "joints": n, "worst_requirement_nm": round(worst, 2),
                            "reason": reason})
            frozen.add(smallest)
            continue
        rolled.append({"from": smallest, "to": target, "joints": n, "worst_requirement_nm": round(worst, 2),
                       "upspec_x": round(act[target].peak_torque_nm / need, 1),
                       "added_cost_usd": round(added, 2),
                       "added_mass_kg": round((act[target].mass_kg - act[smallest].mass_kg) * n, 3)})
        by_sku[target].extend(by_sku.pop(smallest))
    ordered_map = {jn: sku for sku, js in by_sku.items() for jn, _r in js}
    policy = {
        "rule": (f"size each joint from its own torque requirement (peak >= {_GROUNDING_MARGIN:g}x), then "
                 f"standardise onto at most {_MAX_ACTUATOR_SKUS} part numbers by rolling small groups up into a "
                 f"larger part already on the list. A roll-up is refused when it would over-spec a joint by more "
                 f"than {_SKU_UPSPEC_LIMIT:g}x, or add more than ${cost_cap:,.0f} "
                 f"({_SKU_ROLLUP_COST_FRAC:.0%} of the actuator budget) to the parts cost. Roll-ups only ever go "
                 f"up, so no joint is handed a motor it outgrows; a SKU that survives both tests is a joint that "
                 f"genuinely needs a different actuator, and the reason is listed."),
        "max_skus": _MAX_ACTUATOR_SKUS, "skus": len({v for v in ordered_map.values()}),
        "sized_skus": len({a.name for _n, a, _r in picks}),
        "actuator_cost_usd": round(sum(act[s].price_usd * len(js) for s, js in by_sku.items()), 2),
    }
    if rolled:
        policy["standardised"] = rolled
    if blocked:
        policy["kept_separate"] = blocked
    return ordered_map, policy


def _pick_gripper(gene: RobotGene, scale_kg: float, task: str) -> str:
    """An end effector sized to the robot, not to its class label. The $5,500 Robotiq 2F-85 was handed to every
    ``manipulator`` — including a 5 kg benchtop sorting arm, where it was 77% of the whole parts list and heavier
    than two of the arm's links. An industrial/high-force job on a substantial arm still gets it."""
    heavy = bool(re.search(r"industrial|heavy|payload|\d+\s?kg|production line|palletis|palletiz|machine tend",
                           task or "", re.I))
    return "Robotiq 2F-85 gripper" if (scale_kg >= 10.0 or heavy) else "Dynamixel-driven 2-finger gripper"


def _cost_drivers(lines: list[BomLine], total: float) -> dict:
    """WHAT MAKES THIS ROBOT COST WHAT IT COSTS — per-category subtotals + the biggest single lines, so a buyer
    can see at a glance whether the number is motors, perception or the end effector, and argue with it.

    Explains THE BILL, so it reads only the lines the bill charges for (``in_price_total``). A category share
    computed over hardware the customer already owns answers a question nobody asked: on the imported Go2 it
    said "actuator 85%" of a total the customer is not being charged.
    """
    billed = [ln for ln in lines if ln.in_price_total]
    excluded = round(sum(ln.price_usd for ln in lines if not ln.in_price_total), 2)
    by_cat: dict[str, float] = {}
    for ln in billed:
        by_cat[ln.category] = round(by_cat.get(ln.category, 0.0) + ln.price_usd, 2)
    top_cat = sorted(by_cat.items(), key=lambda kv: -kv[1])
    top_lines = sorted(billed, key=lambda ln: -ln.price_usd)[:3]
    out = {
        "by_category": {k: v for k, v in top_cat},
        "share_pct": {k: round(100.0 * v / max(total, 1e-6), 1) for k, v in top_cat[:4]},
        "top_lines": [{"part": ln.part, "qty": ln.qty, "price_usd": ln.price_usd} for ln in top_lines],
        "basis": ("single-unit list prices for representative real parts (~2024). A comparable commercial robot's "
                  "RETAIL price is not this number: it is a volume-manufactured product with in-house actuators, "
                  "and can sell below its own one-off parts cost. Compare like for like before concluding we are "
                  "expensive — start with the category shares above."),
    }
    if excluded > 0:
        out["excluded_already_fitted_usd"] = excluded
        out["excluded_note"] = (f"${excluded:,.2f} of catalog equivalents for hardware already on your machine is "
                                f"NOT in these shares — see totals.already_fitted_usd")
    return out


def _aerial_propulsion(gene: RobotGene) -> list[tuple[str, int, str]]:
    """A quadcopter's propulsion: one brushless motor+prop and one ESC PER rotor, plus a flight controller. The
    rotors are welded thrust points (not actuated joints), so the joint-driven actuator loop misses them entirely
    — without this a drone shipped a BOM with NO propulsion at all."""
    md = getattr(gene, "metadata", None) or {}
    if (gene.robot_class or "").lower() != "aerial" and not md.get("rotor_offsets"):
        return []
    n = len(md.get("rotor_offsets") or []) or sum(1 for s in gene.segments if "rotor" in (s.name or "").lower()) or 4
    return [("T-Motor MN3110 700KV + 9x4.5 prop", n, "rotor propulsion (thrust)"),
            ("BLHeli-32 40A ESC", n, "one ESC per rotor"),
            ("Holybro Pixhawk 6C flight controller", 1, "autopilot: stabilization + IMU/baro")]


def build_bom(gene: RobotGene, *, capabilities=None, task: str = "", pins: dict | None = None) -> dict:
    """Spec the whole robot as a real parts list. ``task`` (the build prompt / task description) makes the
    sensor suite TASK-ADAPTIVE — a navigator gets a LiDAR, an inspector a thermal camera, a sorter a detection
    camera. ``pins`` (or ``gene.metadata['pinned_parts']``) lets a user SPECIFY an exact part for a category
    (e.g. {'lidar': 'Ouster OS1-32', 'actuator': 'T-Motor AK80-9'}) — the auto-selected part is swapped for it.
    Returns per-line items, the joint->actuator map, and rolled-up totals (count/mass/price/power/DOF)."""
    lines: list[BomLine] = []

    # 1) ACTUATORS — one per actuated joint, smallest real motor that meets the joint's torque (with margin).
    #
    # SIZE FROM THE REQUIREMENT, NOT FROM THE MOTOR ALREADY CHOSEN. ``actuator_torque_nm`` is not a demand: on a
    # grounded body ``ground_gene`` overwrote it with the SELECTED motor's PEAK. Feeding that back through
    # ``select_actuator``'s default 1.3x margin re-read a capacity as a demand and bought one rung bigger every
    # time, so ONE package shipped two contradictory motor lists -- bom.json said 8x AK80-64 + 4x XM540-W270-T
    # while the certificate's margins.bom (which grades the SHIPPED clamp, margin=1.0) said 8x AK10-9 +
    # 4x XM430-W350-T. A buyer procuring from that gets different part numbers depending on which file they open.
    # ``torque_req_nm`` is the requirement grounding pinned for exactly this reason; reproducing its call here
    # makes the two agree by construction. An UNGROUNDED body has no pin, and there ``actuator_torque_nm`` really
    # is the authored requirement, so the original call stands.
    joints = gene.actuated_joints()
    picks: list[tuple[str, object, float]] = []
    for s in joints:
        req = getattr(s, "torque_req_nm", None)
        if req:
            a = select_actuator(float(req), margin=_GROUNDING_MARGIN,
                                continuous_torque_nm=float(req) * _GROUNDING_MARGIN)
        else:
            a = select_actuator(s.actuator_torque_nm or _DEFAULT_JOINT_TORQUE_NM)
        picks.append((s.name, a, _joint_requirement_nm(s)))
    sized_map = {jn: a.name for jn, a, _r in picks}
    # STANDARDISE the parts list onto a few part numbers (see _standardize_actuators) — `actuator_map` is what you
    # ORDER; `actuator_policy.sized_per_joint` keeps the per-joint sizing that produced it, so nothing is hidden.
    actuator_map, actuator_policy = _standardize_actuators(picks)
    _by_name = {a.name: a for _n, a, _r in picks}
    ordered = [_by_name[actuator_map[jn]] for jn, _a, _r in picks]
    for a, qty in Counter(ordered).items():
        lines.append(BomLine(a.name, "actuator", qty, a.mass_kg, a.price_usd,
                             f"{a.kind}, peak {a.peak_torque_nm:g} Nm @ {a.voltage_v:g} V, gear {a.gear_ratio:g}:1"))
    if actuator_map != sized_map:
        actuator_policy["sized_per_joint"] = {jn: sized_map[jn] for jn in sized_map if sized_map[jn] != actuator_map[jn]}
    bus_v = max((a.voltage_v for a in ordered), default=12.0)      # the pack/PSU must serve the actuator rail

    # 2) STRUCTURE — resolve each part's material for the TASK, then one line PER material actually used
    # (a coloured shell, an aluminium/steel/carbon skeleton, metal hands/feet) with its links + mass + cost.
    # A MATERIAL LINE IS THE RAW STOCK A FABRICATOR BUYS, not the finished link. This summed ``s.mass_kg``,
    # which on a grounded body is structure + the motor grounding folded into it (and, after
    # ``embody_component_masses``, the battery and compute too) -- while every one of those parts ALSO has its
    # own line above. So the same hardware was billed twice and the totals were fiction: an authored horse
    # simulated at 15.951 kg carried a 29.212 kg parts list, 10.410 kg of it its own motors counted again.
    # ``structural_link_masses`` subtracts exactly what was folded in, per link, and floors at zero.
    ensure_materials(gene)
    refine_materials_for_task(gene, task)
    from virturoid.services.grounded_physics import structural_link_masses
    struct_kg = structural_link_masses(gene)
    groups: dict = {}
    for s in gene.segments:
        m = _material_for_key(s.material)
        if m is None:
            continue
        g = groups.setdefault(m.name, [m, 0.0, 0])
        g[1] += float(struct_kg.get(s.name, s.mass_kg))
        g[2] += 1
    for mname, (m, mass, cnt) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        lines.append(BomLine(mname, "material", cnt, round(mass / max(1, cnt), 4),
                             round(mass * m.cost_per_kg_usd / max(1, cnt), 2),
                             f"{cnt} links ({m.tier}), {m.density_kgm3:g} kg/m^3 - {m.note}"))

    # 3) SENSORS / 4) COMPUTE / 5) MOBILE DRIVE / 6) END EFFECTOR. The POWER SOURCE is deliberately NOT chosen
    # here: it is chosen last, from the finished parts list, so the pack/PSU it buys is sized by the same numbers
    # the totals report. (Rotor draw needs no special case either — a rotor is a catalog part with a datasheet
    # power_w, so it lands in the electronics term once, instead of being added to one budget and not the other.)
    scale_kg = sum(s.mass_kg for s in gene.segments)        # the robot's size drives sensor selection
    spec_items = (_sensor_suite(gene.robot_class, capabilities, task, scale_kg)
                  + _aerial_propulsion(gene)
                  + [_select_compute_line(gene.robot_class, len(joints), task, capabilities)]
                  + _mobile_drive(gene))
    if (gene.end_effector_type or "none") in ("gripper", "suction") and \
            (gene.robot_class or "").lower() in ("manipulator", "arm", "humanoid"):
        spec_items.append((_pick_gripper(gene, scale_kg, task), 1, "end effector"))
    for name, qty, mounting in spec_items:
        c = component(name)
        if c is None:
            continue
        lines.append(BomLine(c.name, c.category, qty, c.mass_kg, c.price_usd, f"{c.spec} - {mounting}"))

    # PART PINS — honor any user-SPECIFIED exact parts (gene.metadata['pinned_parts'] merged with the `pins` arg):
    # swap the auto-selected part of a category for the pinned one. A pin the catalog lacks, or a category mismatch,
    # is REJECTED (reported), never silently dropped. This is how "use the Ouster OS1-32 lidar" or "swap to the
    # AK80-9 motor" gets specified. Applied BEFORE the power budget is struck, so a pinned sensor's draw is in it.
    from virturoid.services.component_catalog import resolve_part
    all_pins = {**(gene.metadata.get("pinned_parts") or {}), **(pins or {})}
    power_pin = None
    pins_applied, pins_rejected = [], []
    for cat, pname in all_pins.items():
        part = resolve_part(pname)
        if part is None:
            pins_rejected.append({"category": cat, "part": pname, "reason": "no such part in the catalog"}); continue
        pcat = "actuator" if hasattr(part, "peak_torque_nm") else part.category
        if cat != pcat:
            pins_rejected.append({"category": cat, "part": part.name, "reason": f"that part is a {pcat}, not a {cat}"}); continue
        if cat == "actuator":
            n = len(joints)
            lines = [ln for ln in lines if ln.category != "actuator"]
            lines.insert(0, BomLine(part.name, "actuator", n, part.mass_kg, part.price_usd,
                                    f"{part.kind}, peak {part.peak_torque_nm:g} Nm @ {part.voltage_v:g} V, "
                                    f"{part.no_load_rpm:g} rpm, gear {part.gear_ratio:g}:1 (pinned)"))
            for k in actuator_map:
                actuator_map[k] = part.name
            actuator_policy = {**actuator_policy, "skus": 1, "pinned": part.name,
                               "rule": "every joint pinned to a user-specified part number"}
            bus_v = part.voltage_v
        elif cat == "power":
            power_pin = part                     # applied after the budget, so its detail can quote the real draw
        else:
            qty = sum(ln.qty for ln in lines if ln.category == cat) or 1
            lines = [ln for ln in lines if ln.category != cat]
            lines.append(BomLine(part.name, cat, qty, part.mass_kg, part.price_usd, f"{part.spec} (pinned)"))
        pins_applied.append({"category": cat, "part": part.name})

    # 7) THE POWER BUDGET — computed ONCE from the finished parts list, then the source that must carry it.
    act_w, joint_w = _actuator_bus_w([r for _n, _a, r in picks], gene.robot_class)
    elec_w = 0.0
    for ln in lines:                                # datasheet draw, and the rail every powered part needs
        c = component(ln.part)
        if c is None:
            continue
        elec_w += ln.qty * c.power_w
        if c.power_w > 0 and isinstance(c.specs.get("voltage_v"), (int, float)):
            bus_v = max(bus_v, float(c.specs["voltage_v"]))
    budget = PowerBudget(act_w, elec_w, len(joints), joint_w)
    if power_pin is not None:
        lines.append(BomLine(power_pin.name, "power", 1, power_pin.mass_kg, power_pin.price_usd,
                             f"{power_pin.spec} (pinned) - carries the {budget.total_w:g} W budgeted draw"))
    else:
        pname, pqty, pdetail = _select_power(gene.robot_class, task, budget, bus_v)
        pc = component(pname)
        if pc is not None:
            lines.append(BomLine(pc.name, pc.category, pqty, pc.mass_kg, pc.price_usd, f"{pc.spec} - {pdetail}"))

    # 8) THE BALANCE OF SYSTEM THAT HAS NO PART NUMBER — wiring harness, fasteners, covers. It is on the body
    # (``grounded_physics._apply_balance_of_system`` puts it on the trunk so the class band is met), so a parts
    # list that omits it cannot add up to the robot the customer will lift. Priced at zero and labelled an
    # ESTIMATE, because pretending to a price would be the same invention one level along.
    bos_kg = sum(float(v or 0.0) for v in
                 (((gene.metadata or {}).get("embodied_mass") or {}).get("balance_of_system_kg") or {}).values())
    if bos_kg > 1e-6:
        lines.append(BomLine("Wiring, fasteners and covers (estimate)", "balance_of_system", 1,
                             round(bos_kg, 4), 0.0,
                             "NOT a part number: the residue of a finished machine that this parts list does "
                             "not itemise, carried on the trunk so the simulated mass matches the built mass"))

    # THE ROBOT'S OWN ACTUATION CAPABILITY, stated by the BOM rather than reconstructed from a part number by
    # whoever reads it next. See _robot_joint_limits: the spec sheet used to regex the catalog rating out of the
    # detail string below and print it as the robot's peak joint torque (Go2: 360 N.m for a 45.43 N.m machine).
    joint_limits = _robot_joint_limits(gene, joints)

    # ONE PART, COUNTED ONCE — IN MASS AND IN MONEY. On an imported robot the link masses are the manufacturer's
    # and already contain that machine's motors, so our actuator lines are equivalents for hardware the customer
    # already owns. They are excluded from the mass (they are already inside it) AND from the bill (the customer
    # is not buying them). The line keeps its real catalog price as a REPLACEMENT figure, quoted separately.
    preserved = str((gene.metadata or {}).get("mass_source") or "") == "source_model"
    if preserved:
        _joints_for = {}
        for jn, part in (actuator_map or {}).items():
            _joints_for.setdefault(str(part), []).append(str(jn))
        for ln in lines:
            if ln.category != "actuator":
                continue
            ln.in_mass_total = False
            ln.in_price_total = False
            # WHOSE NUMBER IS WHOSE. Everything before the dash is the CATALOG PART's datasheet — peak torque,
            # voltage, gear ratio — and none of it describes the customer's motor. State their joints' own
            # declared limit here, on the line, so the two can never be read as one number again.
            mine = [joint_limits["per_joint"][j] for j in _joints_for.get(ln.part, [])
                    if j in joint_limits.get("per_joint", {})]
            theirs = [r for r in mine if r["source"] == "your model"]
            if theirs:
                worst = max(theirs, key=lambda r: r["limit"])
                ln.detail = (f"catalog part rating: {ln.detail} -- the PART's datasheet, NOT your robot's. "
                             f"YOUR ROBOT's own limit on {'this' if len(mine) == 1 else 'these'} "
                             f"{len(mine)} joint(s) is {worst['limit']:g} {worst['unit']} "
                             f"(read from {worst['declared_in']})")
            ln.detail += (" - EQUIVALENT for a motor already fitted to your robot; its mass is already "
                          "inside your model's link masses and is not added again, and it is NOT billed in "
                          "totals.price_usd — its catalog price is quoted in totals.already_fitted_usd as a "
                          "replacement/spares option you have not asked for")

    # A SENSOR THE CUSTOMER'S MODEL DOES NOT DECLARE IS A QUOTE, NOT AN INVENTORY LINE. ``_sensor_suite`` picks
    # perception from ``robot_class`` alone: every quadruped gets a RealSense + a BNO055 whether or not the file
    # we were handed contains one. On a body we composed that is the design; on the customer's machine it was a
    # $334 camera billed inside the price of a robot that has none (Menagerie Go2: ncam 0, nsensor 0). The line
    # stays -- "you will need a depth camera, here is the one we would fit" is the useful part -- but it is
    # marked, its detail says so, and its cost is broken out of the as-imported total below.
    from virturoid.services.sensor_provenance import (SENSOR_CATEGORIES, carries_our_proposed_sensors,
                                                      category_declared, declared, sensor_reality_table)
    sensor_prov = None
    if carries_our_proposed_sensors(gene):
        inv = declared(gene) or {}
        _where = (f"your model declares ncam {inv.get('ncam')} / nsensor {inv.get('nsensor')}"
                  if inv else "your model's sensing was not read at import")
        _pinned = {str(v) for k, v in ((gene.metadata or {}).get("pinned_parts") or {}).items()
                   if k in SENSOR_CATEGORIES}
        for ln in lines:
            if ln.category not in SENSOR_CATEGORIES or ln.part in _pinned:
                continue
            # ...AND THE TABLE HAS TO BE RIGHT IN BOTH DIRECTIONS. A model that declares ``<gyro>`` and
            # ``<accelerometer>`` HAS an IMU; quoting one as an addition would be the same failure to read their
            # file, only cheaper. Where the model evidences the instrument, the line becomes what the actuator
            # lines already are -- an EQUIVALENT part number for hardware already fitted, not a purchase.
            _has, _ev = category_declared(gene, ln.category)
            if _has:
                ln.detail += (f" - EQUIVALENT for a {ln.category} your model already declares ({_ev}); a part "
                              f"number for something you have, not an addition to buy")
                continue
            ln.proposed = True
            ln.detail += (f" - PROPOSED ADDITION, not fitted to your robot: {_where}, so this is a part we "
                          f"recommend buying, quoted separately in totals.proposed_additions_usd")
        sensor_prov = sensor_reality_table(gene, _sensor_suite(gene.robot_class, capabilities, task, scale_kg))

    # TOTALS. ``est_power_w`` IS ``power_budget.total_draw_w`` and IS the number the power line was sized against —
    # one figure, quoted in three places, instead of three figures quoted once each.
    total_mass = round(sum(ln.mass_kg for ln in lines if ln.in_mass_total), 3)
    # price_usd IS THE BILL: what this list asks the customer to spend. On a robot we designed that is every
    # line (nothing exists yet). On an imported machine it excludes the catalog equivalents for hardware already
    # bolted on, whose catalog value is quoted beside it as a replacement option instead.
    catalog_list_price = round(sum(ln.price_usd for ln in lines), 2)
    total_price = round(sum(ln.price_usd for ln in lines if ln.in_price_total), 2)
    already_fitted_price = round(catalog_list_price - total_price, 2)
    proposed_price = round(sum(ln.price_usd for ln in lines if ln.proposed), 2)
    _embodied = ((gene.metadata or {}).get("embodied_mass") or {}).get("component_kg") or {}
    mass_note = ("every part counted once: material lines are STRUCTURE only (the motors, electronics and "
                 "balance of system folded into the link masses are billed on their own lines)")
    if preserved:
        mass_note = ("your robot's own per-link masses, plus the parts this list proposes ADDING. The actuator "
                     "lines are catalog equivalents for motors already fitted and are not counted again")
    elif not _embodied:
        mass_note += ("; the simulated body does NOT yet carry the sensors, compute or power below — call "
                      "grounded_physics.embody_component_masses to bolt them on")
    return {
        "robot_class": gene.robot_class,
        "dof": len(joints),
        "actuator_map": actuator_map,
        "actuator_policy": actuator_policy,
        "material_policy": (gene.metadata or {}).get("material_policy"),
        "power_budget": budget.to_dict(),
        # THE ROBOT's capability, stated once, with provenance — never re-derived from a part number.
        "joint_limits": joint_limits,
        "lines": [asdict(ln) | {"mass_kg": ln.mass_kg, "price_usd": ln.price_usd} for ln in lines],
        "totals": {"line_items": len(lines), "actuators": len(joints), "mass_kg": total_mass,
                   "price_usd": total_price, "est_power_w": budget.total_w, "mass_note": mass_note,
                   # THE MONEY SPLIT. Without it "$6,414" reads as the price of the machine in front of them,
                   # and $334 of that was a camera their robot does not have.
                   **({"already_fitted_usd": already_fitted_price,
                       "catalog_list_price_usd": catalog_list_price,
                       "already_fitted_note": (
                           f"${already_fitted_price:,.2f} of catalog EQUIVALENTS for hardware already on your "
                           f"machine is NOT in price_usd. You own these parts and did not ask to replace them, "
                           f"so this is a replacement/spares figure, not a bill. catalog_list_price_usd "
                           f"(${catalog_list_price:,.2f}) is what the whole list would cost if you were buying "
                           f"the machine from scratch.")}
                      if already_fitted_price > 0 else {}),
                   **({"proposed_additions_usd": proposed_price,
                       "as_imported_price_usd": round(catalog_list_price - proposed_price, 2),
                       "price_note": ("price_usd is THE BILL — what this list asks you to spend. "
                                      "proposed_additions_usd is the part of that bill which is hardware your "
                                      "model does not declare (sensors we recommend adding). "
                                      "as_imported_price_usd values the machine you already have at our catalog "
                                      "prices; it is not money you are being asked for.")}
                      if proposed_price > 0 else {})},
        **({"sensor_provenance": sensor_prov} if sensor_prov else {}),
        "spec_provenance": _spec_provenance(gene, preserved, joint_limits, bool(sensor_prov)),
        "cost_drivers": _cost_drivers(lines, total_price),
        **({"pins": {"applied": pins_applied, "rejected": pins_rejected}} if (pins_applied or pins_rejected) else {}),
        "note": ("Representative real-world components (manufacturer datasheets, ~2024); verify exact specs "
                 "against the live datasheet before procurement."),
    }


def attach_bom(gene: RobotGene, *, capabilities=None, task: str = "") -> RobotGene:
    """Compute the BOM and stash it on the gene (``metadata['bom']``) so it travels with the robot/export."""
    try:
        gene.metadata["bom"] = build_bom(gene, capabilities=capabilities, task=task)
    except Exception:  # noqa: BLE001 - a BOM is value-add; never let it block a build
        pass
    return gene


def format_bom_markdown(bom: dict) -> str:
    """Human-readable BOM table (for the export package / evidence ledger)."""
    t = bom.get("totals", {})
    rows = ["# Bill of Materials",
            f"\n**Class:** {bom.get('robot_class')}  |  **DOF:** {bom.get('dof')}  |  "
            f"**Mass:** {t.get('mass_kg')} kg  |  **To buy:** ${t.get('price_usd')}  |  "
            f"**Est. power:** {t.get('est_power_w')} W\n"]
    if t.get("already_fitted_usd"):
        rows.append(f"> **You are not being billed for hardware you already own.** ${t['already_fitted_usd']:,.2f} "
                    f"of this list is catalog EQUIVALENTS for parts already fitted to your machine (priced as a "
                    f"replacement option you have not asked for). The whole list bought from scratch would be "
                    f"${t.get('catalog_list_price_usd', 0.0):,.2f}.\n")
    jl = bom.get("joint_limits") or {}
    if jl.get("peak_joint_torque_nm") is not None or jl.get("peak_joint_force_n") is not None:
        # THE ROBOT's number, above the parts table, so it can never be confused with a part rating below it.
        bits = []
        if jl.get("peak_joint_torque_nm") is not None:
            bits.append(f"**{jl['peak_joint_torque_nm']:g} N.m** at `{jl.get('peak_joint')}`")
        if jl.get("peak_joint_force_n") is not None:
            bits.append(f"**{jl['peak_joint_force_n']:g} N** at `{jl.get('peak_prismatic_joint')}` (linear)")
        rows.append(f"\n## This robot's actuation capability\n\nPeak joint limit: {' and '.join(bits)}. "
                    f"{jl.get('note', '')}\n")
    rows += ["| Part | Category | Qty | Mass (kg) | Price (USD) | Detail |",
             "|------|----------|----:|----------:|------------:|--------|"]
    for ln in bom.get("lines", []):
        price = (f"{ln['price_usd']} (already owned)" if ln.get("in_price_total") is False
                 else str(ln["price_usd"]))
        rows.append(f"| {ln['part']} | {ln['category']} | {ln['qty']} | {ln['mass_kg']} | "
                    f"{price} | {ln['detail']} |")
    # The SELECTION RULES, in the same document as the parts they chose — an engineer ordering from this list
    # should not have to reverse-engineer why it says what it says.
    pb = bom.get("power_budget") or {}
    if pb:
        rows.append(f"\n## Power budget\n\n**{pb.get('total_draw_w')} W** continuous = "
                    f"{pb.get('actuators_w')} W actuators ({pb.get('driven_axes')} driven axes) + "
                    f"{pb.get('electronics_w')} W electronics. This is the number the power source above was "
                    f"sized against, and the `est_power_w` in the totals.\n\n_{pb.get('basis', '')}_")
    ap = bom.get("actuator_policy") or {}
    if ap:
        rows.append(f"\n## Actuator selection\n\n{ap.get('rule', '')}\n\n"
                    f"Sized {ap.get('sized_skus', '?')} distinct parts, ordering **{ap.get('skus', '?')}**.")
        for r in ap.get("standardised", []):
            rows.append(f"- standardised {r['joints']} joint(s): `{r['from']}` -> `{r['to']}` "
                        f"({r['upspec_x']}x the group's {r['worst_requirement_nm']} Nm requirement)")
        for b in ap.get("kept_separate", []):
            rows.append(f"- kept `{b['sku']}` separate ({b['joints']} joint(s)): {b['reason']}")
    mp = bom.get("material_policy") or {}
    if mp:
        verdict = "honoured" if mp.get("honoured") else "SUBSTITUTED"
        rows.append(f"\n## Structural material\n\nRequested: **{mp.get('requested') or 'not specified'}** -> "
                    f"skeleton built from **{mp.get('skeleton')}** ({verdict}). {mp.get('reason', '')}")
    cd = bom.get("cost_drivers") or {}
    if cd:
        shares = ", ".join(f"{k} {v}%" for k, v in (cd.get("share_pct") or {}).items())
        rows.append(f"\n## What drives the cost\n\n{shares}\n")
        for t3 in cd.get("top_lines", []):
            rows.append(f"- {t3['part']} x{t3['qty']}: ${t3['price_usd']}")
        if cd.get("excluded_note"):
            rows.append(f"\n{cd['excluded_note']}.")
        rows.append(f"\n_{cd.get('basis', '')}_")
    sp = bom.get("spec_provenance") or {}
    if sp.get("fields"):
        # WHICH NUMBERS ARE YOURS AND WHICH ARE OURS, in the same document as the numbers.
        rows.append(f"\n## Where each number comes from\n\nThis robot is **{sp.get('robot_is')}**. "
                    f"{sp.get('rule', '')}\n")
        rows.append("| Field | Source | Evidence |")
        rows.append("|-------|--------|----------|")
        for f in sp["fields"]:
            rows.append(f"| `{f['field']}` | {f['source']} | {f['evidence']} |")
    rows.append(f"\n_{bom.get('note', '')}_")
    return "\n".join(rows)


def build_bom_from_genome(genome: dict, *, task: str = "", capabilities=None) -> dict:
    """G1 (fidelity gap-closure): a REAL parts list for a template-path RobotGenome package (the path that shipped
    with NO BOM at all — the e2e fidelity test's worst correctness finding). Uses the genome's own joint effort
    limits to size real actuators, one aluminium structure line for the links, and the SAME class/task-adaptive
    sensor/compute/power selection as the gene path, so both paths emit the identical BOM schema."""
    joints = [j for j in genome.get("joints", []) if (j.get("joint_type") or "").lower() in ("revolute", "prismatic")]
    links = genome.get("links", [])
    species = (genome.get("species") or "").lower()
    robot_class = (genome.get("robot_class")
                   or ("mobile_base" if "mobile" in species else "manipulator"))
    lines: list[BomLine] = []
    picks: list[tuple[str, object, float]] = []
    # THE GENOME'S OWN EFFORT LIMITS ARE THE ROBOT'S CAPABILITY on this path too — same rule as the gene path,
    # so both emit the same field and neither leaves the reader to regex a part rating out of a detail string.
    per_limit: dict[str, dict] = {}
    for j in joints:
        declared = (j.get("limit") or {}).get("effort")
        eff = float(declared or _DEFAULT_JOINT_TORQUE_NM)
        name = j.get("name", f"joint{len(picks)}")
        picks.append((name, select_actuator(eff), eff))
        per_limit[str(name)] = {
            "limit": round(abs(eff), 3),
            "unit": "N" if (j.get("joint_type") or "").lower() == "prismatic" else "N.m",
            "source": "your model" if declared else "the actuator we fitted",
            **({"declared_in": "the genome's joint effort limit"} if declared else {})}
    actuator_map, actuator_policy = _standardize_actuators(picks)
    _by_name = {a.name: a for _n, a, _r in picks}
    ordered = [_by_name[actuator_map[jn]] for jn, _a, _r in picks]
    for a, qty in Counter(ordered).items():
        lines.append(BomLine(a.name, "actuator", qty, a.mass_kg, a.price_usd,
                             f"{a.kind}, peak {a.peak_torque_nm:g} Nm @ {a.voltage_v:g} V, gear {a.gear_ratio:g}:1"))
    bus_v = max((a.voltage_v for a in ordered), default=12.0)
    if links:
        m = _material_for_key("skeleton")
        if m is not None:
            est_mass = 0.35 * len(links)                    # structural estimate: template links carry no mass
            lines.append(BomLine(m.name, "material", len(links), round(est_mass / len(links), 4),
                                 round(est_mass * m.cost_per_kg_usd / len(links), 2),
                                 f"{len(links)} links ({m.tier}) - template-path structural estimate"))
    # BomLine.mass_kg / .price_usd are PROPERTIES that already fold in qty (qty * unit_*), so summing them
    # directly IS the line-total roll-up -- multiplying by qty again double-counted (qty^2) every multi-unit
    # line (4 wheels, 12 leg motors). Roll up FROM the mass column exactly like build_bom does (2026-07-24 audit).
    scale_kg = sum(ln.mass_kg for ln in lines)
    for name, qty, mounting in (_sensor_suite(robot_class, capabilities, task, scale_kg)
                                + [_select_compute_line(robot_class, len(joints), task, capabilities)]):
        c = component(name)
        if c is None:
            continue
        lines.append(BomLine(c.name, c.category, qty, c.mass_kg, c.price_usd, f"{c.spec} - {mounting}"))
    # the SAME single power budget + kind-driven source as the gene path, so both paths' numbers mean one thing
    act_w, joint_w = _actuator_bus_w([r for _n, _a, r in picks], robot_class)
    elec_w = sum(ln.qty * (component(ln.part).power_w if component(ln.part) else 0.0) for ln in lines)
    budget = PowerBudget(act_w, elec_w, len(joints), joint_w)
    pname, pqty, pdetail = _select_power(robot_class, task, budget, bus_v)
    pc = component(pname)
    if pc is not None:
        lines.append(BomLine(pc.name, pc.category, pqty, pc.mass_kg, pc.price_usd, f"{pc.spec} - {pdetail}"))
    total_mass = round(sum(ln.mass_kg for ln in lines), 3)
    total_price = round(sum(ln.price_usd for ln in lines), 2)
    return {"robot_class": robot_class, "dof": len(joints), "actuator_map": actuator_map,
            "actuator_policy": actuator_policy, "power_budget": budget.to_dict(),
            "joint_limits": _aggregate_joint_limits(per_limit),
            "lines": [asdict(ln) | {"mass_kg": ln.mass_kg, "price_usd": ln.price_usd} for ln in lines],
            "totals": {"line_items": len(lines), "actuators": len(joints), "mass_kg": total_mass,
                       "price_usd": total_price, "est_power_w": budget.total_w,
                       # Same key as ``build_bom``, so both paths' totals answer the same question. No double
                       # count to correct for here: the material line is a 0.35 kg/link ESTIMATE, not a sum of
                       # finished link masses, because a template genome's links carry no mass at all.
                       "mass_note": ("every part counted once; the structure line is a 0.35 kg/link estimate, "
                                     "not a measured body — a template genome carries no link masses")},
            "cost_drivers": _cost_drivers(lines, total_price),
            "note": ("Template-path BOM derived from the genome's joint effort limits + class suite. "
                     "Structural masses are estimates; gene-path packages carry measured link masses.")}


def emit_genome_bom(package_dir, *, task: str = "") -> dict | None:
    """Write robot/bill_of_materials.json for a genome-based package (idempotent; returns the BOM or None if the
    package has no genome). The fail-closed rule: a package without a genome gets NO fabricated BOM."""
    import json as _json
    from pathlib import Path as _P
    pkg = _P(package_dir)
    gp = pkg / "robot" / "robot_genome.json"
    if not gp.exists():
        return None
    bom = build_bom_from_genome(_json.loads(gp.read_text(encoding="utf-8")), task=task)
    (pkg / "robot" / "bill_of_materials.json").write_text(_json.dumps(bom, indent=2), encoding="utf-8")
    return bom
