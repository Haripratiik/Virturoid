"""BILL OF MATERIALS — turn a generated gene into a real, buildable parts list.

Every robot gets its own BOM: each actuated joint is matched to the smallest real actuator that can drive it
(by torque, with a safety margin), each structural link is assigned a material, and the robot's CLASS decides
its sensor suite — a humanoid gets stereo camera "eyes" + an IMU + LiDAR; a rover gets drive wheels + motors +
a scanning LiDAR; an arm gets a wrist camera + a 6-axis force/torque sensor + a gripper. Compute and a power
pack are sized to the build. The result is a structured BOM with per-line mass/price and rolled-up totals, so
the robot is not just a shape — it is a thing you could actually source and assemble.

This reads the gene's ``actuator_torque_nm`` per joint (the compiler already sizes these from link
length/mass), so the actuator selection tracks the real mechanical demand of the body we generated.
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


@dataclass
class BomLine:
    part: str
    category: str                    # actuator | material | camera | lidar | imu | force_torque | compute |
    #                                  power | wheel | drive_motor | gripper
    qty: int
    unit_mass_kg: float
    unit_price_usd: float
    detail: str

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


def refine_materials_for_task(gene: RobotGene, task: str = "") -> RobotGene:
    """Resolve each part's material INTELLIGENTLY for the task (mutates GeneSegment.material in place, which the
    render colours by): the load-bearing SKELETON becomes steel (heavy), carbon-fibre (flight/agility) or
    titanium (precision/medical), else aircraft aluminium; contact parts pick up soft TPU pads when the task is
    delicate handling. Idempotent — once 'skeleton' is resolved to a concrete metal, re-running is a no-op."""
    t = (task or "").lower()
    skel = "aluminum"
    for pat, mat in _SKELETON_BY_TASK:
        if re.search(pat, t):
            skel = mat
            break
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
    """Scale a part's visual cross-section by ``f`` in place (extrude profile / compound sub-parts)."""
    if not isinstance(geo, dict):
        return
    if geo.get("family") == "extrude" and isinstance(geo.get("profile"), list):
        geo["profile"] = [[round(p[0] * f, 5), round(p[1] * f, 5)] for p in geo["profile"]]
    if geo.get("family") == "compound":
        for sub in geo.get("parts", []):
            _scale_geo(sub, f)


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
        s.radius_m = round(max(0.006, s.radius_m * f), 5)
        s.mass_kg = round(max(0.01, s.mass_kg * f * f), 5)
        _scale_geo(s.geometry, f)
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


def _pick_lidar(scale_kg: float) -> str:
    """A BIGGER robot needs a longer-range LiDAR: a tiny rover gets a 2D puck, a humanoid a 3D dome, a large
    machine a long-range 128-beam unit. Scales the part selection with the robot's actual size."""
    if scale_kg > 55:
        return "Ouster OS2-128"
    if scale_kg > 22:
        return "Ouster OS1-32"
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
    if wants_lidar:                                   # navigation perception -> a LiDAR sized to the robot
        suite.append((_pick_lidar(scale_kg), 1, "mast: 3D LiDAR for navigation"))
    if scale_kg > 25 and cls not in ("manipulator", "arm"):   # a BIG robot needs 360 coverage + redundancy
        suite.append(("Intel RealSense D435i", 1, "rear: extra depth camera (360 coverage)"))
        suite.append(("VectorNav VN-100", 1, "redundant high-grade IMU"))
    return _dedupe_components(suite)


def _compute_and_power(robot_class: str, n_actuators: int) -> list[tuple[str, int, str]]:
    cls = (robot_class or "").lower()
    items: list[tuple[str, int, str]] = []
    if cls in ("humanoid", "biped"):
        items.append(("NVIDIA Jetson AGX Orin 64GB", 1, "whole-body control + perception"))
        items.append(("Li-ion 48V 12Ah pack", 1, "high-torque joint bus"))
    elif cls in ("quadruped", "legged"):
        items.append(("NVIDIA Jetson Orin Nano 8GB", 1, "locomotion + perception"))
        items.append(("Li-ion 48V 12Ah pack" if n_actuators >= 10 else "LiPo 6S 8000mAh (22.2V)", 1, "joint bus"))
    elif cls in ("mobile_base", "mobile", "rover"):
        items.append(("NVIDIA Jetson Orin Nano 8GB", 1, "SLAM + navigation"))
        items.append(("LiPo 4S 5200mAh (14.8V)", 1, "drive + logic"))
    else:                                          # manipulator / default
        items.append(("NVIDIA Jetson Orin Nano 8GB", 1, "planning + vision"))
        items.append(("LiPo 4S 5200mAh (14.8V)", 1, "logic + servo bus"))
    return items


def _mobile_drive(gene: RobotGene) -> list[tuple[str, int, str]]:
    """Wheels + drive motors for a mobile base (count from the gene's wheel-ish segments, else a 2+caster default)."""
    if (gene.robot_class or "").lower() not in ("mobile_base", "mobile", "rover"):
        return []
    wheels = sum(1 for s in gene.segments if "wheel" in s.name.lower()) or 2
    return [("Pololu 37Dx70L 12V gearmotor + encoder", wheels, "differential drive"),
            ("100mm rubber drive wheel + hub", wheels, "driven wheels"),
            ("Caster wheel 2in", 1, "passive balance caster")]


def build_bom(gene: RobotGene, *, capabilities=None, task: str = "") -> dict:
    """Spec the whole robot as a real parts list. ``task`` (the build prompt / task description) makes the
    sensor suite TASK-ADAPTIVE — a navigator gets a LiDAR, an inspector a thermal camera, a sorter a detection
    camera. Returns per-line items, the joint->actuator map, and rolled-up totals (count/mass/price/power/DOF)."""
    lines: list[BomLine] = []

    # 1) ACTUATORS — one per actuated joint, smallest real motor that meets the joint's torque (with margin).
    joints = gene.actuated_joints()
    actuator_map: dict[str, str] = {}
    chosen = []
    for s in joints:
        a = select_actuator(s.actuator_torque_nm or _DEFAULT_JOINT_TORQUE_NM)
        actuator_map[s.name] = a.name
        chosen.append(a)
    for a, qty in Counter(chosen).items():
        lines.append(BomLine(a.name, "actuator", qty, a.mass_kg, a.price_usd,
                             f"{a.kind}, peak {a.peak_torque_nm:g} Nm @ {a.voltage_v:g} V, gear {a.gear_ratio:g}:1"))

    # 2) STRUCTURE — resolve each part's material for the TASK, then one line PER material actually used
    # (a coloured shell, an aluminium/steel/carbon skeleton, metal hands/feet) with its links + mass + cost.
    ensure_materials(gene)
    refine_materials_for_task(gene, task)
    groups: dict = {}
    for s in gene.segments:
        m = _material_for_key(s.material)
        if m is None:
            continue
        g = groups.setdefault(m.name, [m, 0.0, 0])
        g[1] += s.mass_kg
        g[2] += 1
    for mname, (m, mass, cnt) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        lines.append(BomLine(mname, "material", cnt, round(mass / max(1, cnt), 4),
                             round(mass * m.cost_per_kg_usd / max(1, cnt), 2),
                             f"{cnt} links ({m.tier}), {m.density_kgm3:g} kg/m^3 - {m.note}"))

    # 3) SENSORS / 4) COMPUTE+POWER / 5) MOBILE DRIVE / 6) END EFFECTOR
    scale_kg = sum(s.mass_kg for s in gene.segments)        # the robot's size drives sensor selection
    spec_items = (_sensor_suite(gene.robot_class, capabilities, task, scale_kg)
                  + _compute_and_power(gene.robot_class, len(joints))
                  + _mobile_drive(gene))
    if (gene.end_effector_type or "none") in ("gripper", "suction") and \
            (gene.robot_class or "").lower() in ("manipulator", "arm", "humanoid"):
        spec_items.append(("Robotiq 2F-85 gripper" if gene.robot_class == "manipulator"
                           else "Dynamixel-driven 2-finger gripper", 1, "end effector"))
    for name, qty, mounting in spec_items:
        c = component(name)
        if c is None:
            continue
        lines.append(BomLine(c.name, c.category, qty, c.mass_kg, c.price_usd, f"{c.spec} - {mounting}"))

    # TOTALS
    total_mass = round(sum(ln.mass_kg for ln in lines), 3)
    total_price = round(sum(ln.price_usd for ln in lines), 2)
    # electronics draw is exact (datasheet); the actuator bus is a duty-weighted continuous estimate
    # (rated torque x rated speed x ~30% duty / drivetrain efficiency) — a battery-sizing figure, not a stall peak.
    elec_w = sum(ln.qty * (component(ln.part).power_w if component(ln.part) else 0.0) for ln in lines)
    bus_w = sum(a.rated_torque_nm * a.max_speed_radps * 0.3 for a in chosen)
    est_power = round(elec_w + bus_w, 1)
    return {
        "robot_class": gene.robot_class,
        "dof": len(joints),
        "actuator_map": actuator_map,
        "lines": [asdict(ln) | {"mass_kg": ln.mass_kg, "price_usd": ln.price_usd} for ln in lines],
        "totals": {"line_items": len(lines), "actuators": len(joints), "mass_kg": total_mass,
                   "price_usd": total_price, "est_power_w": est_power},
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
            f"**Mass:** {t.get('mass_kg')} kg  |  **Est. cost:** ${t.get('price_usd')}  |  "
            f"**Est. power:** {t.get('est_power_w')} W\n",
            "| Part | Category | Qty | Mass (kg) | Price (USD) | Detail |",
            "|------|----------|----:|----------:|------------:|--------|"]
    for ln in bom.get("lines", []):
        rows.append(f"| {ln['part']} | {ln['category']} | {ln['qty']} | {ln['mass_kg']} | "
                    f"{ln['price_usd']} | {ln['detail']} |")
    rows.append(f"\n_{bom.get('note', '')}_")
    return "\n".join(rows)
