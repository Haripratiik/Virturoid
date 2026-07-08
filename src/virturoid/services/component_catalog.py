"""REAL-WORLD COMPONENT CATALOG — the off-the-shelf hardware a generated robot is actually built from.

The original-generation mandate ([[original-generation-mandate]]) is about GEOMETRY: we author our own link
shapes, we never ship a real robot's shell. But the ACTUATORS, SENSORS, COMPUTE, POWER and FASTENERS that
animate that geometry ARE real purchasable parts — a robot is only credible (and buildable) if every joint
maps to a motor that can actually drive it and every capability maps to a real sensor. This module is a
curated catalog of representative real products with their headline specs, so the BOM builder
(``bom_builder.build_bom``) can spec any robot: pick the smallest actuator that meets each joint's torque,
attach the sensors a class needs (a humanoid's camera "eyes", a rover's LiDAR), and total the mass/price/power.

Specs are representative real-world figures (manufacturer datasheets, ~2024); treat them as engineering
estimates and re-verify against the live datasheet before procurement — they are accurate to ~the right part,
not a purchase order. Torque/voltage/mass drive the selection logic; price is a planning figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Actuator:
    name: str
    kind: str                 # "servo" | "qdd" (quasi-direct-drive) | "harmonic" | "gearmotor"
    peak_torque_nm: float     # short-term peak (selection ceiling)
    rated_torque_nm: float    # continuous
    mass_kg: float
    voltage_v: float
    gear_ratio: float
    max_speed_radps: float
    price_usd: float
    vendor: str
    # --- physical housing (drives the rendered motor geom, so the viewport shows the SAME part the BOM cites)
    shape: str = "cylinder"   # "box" (Dynamixel-style servo) | "cylinder" (qdd/harmonic pancake)
    envelope_m: tuple = (0.08, 0.08, 0.04)   # datasheet bounding box (x, y, z) in metres
    axis_dim: int = 2         # which envelope dim the output/rotation axis runs along (0=x servos, 2=z pancakes)


@dataclass(frozen=True)
class Component:
    name: str
    category: str             # camera | lidar | imu | force_torque | compute | power | wheel | drive_motor | gripper
    mass_kg: float
    power_w: float            # nominal electrical draw (0 for passive parts like wheels)
    price_usd: float
    vendor: str
    spec: str                 # one-line headline spec


@dataclass(frozen=True)
class Material:
    name: str
    density_kgm3: float
    cost_per_kg_usd: float
    note: str
    tier: str = "structural"   # skeleton-strong | structural | shell | contact-metal | grip-soft
    stiffness: str = "medium"  # high | medium | low — informs the load-path choice
    render_rgba: tuple = (0.82, 0.84, 0.88, 1.0)   # how this material looks in the viewport
    finish: float = 0.4        # 0 matte -> 1 polished (specular/shininess hint)


# --- ACTUATORS: a torque ladder from fingertip servos to humanoid hip motors. Selection picks the smallest
# whose PEAK torque clears the joint's requirement (with margin). Specs ~ manufacturer datasheets. The trailing
# (shape, envelope_m, axis_dim) is the physical housing so the rendered motor geom and the BOM cite ONE part:
# Dynamixel servos are boxes with the shaft on x (axis_dim=0); qdd/harmonic motors are pancake cylinders on z.
ACTUATORS: tuple[Actuator, ...] = (
    Actuator("Dynamixel XL330-M288-T", "servo", 0.52, 0.42, 0.018, 5.0, 288.0, 12.6, 24.0, "ROBOTIS",
             "box", (0.0203, 0.0340, 0.0220), 0),
    Actuator("Dynamixel XC430-W150-T", "servo", 1.5, 1.0, 0.065, 12.0, 150.0, 12.0, 60.0, "ROBOTIS",
             "box", (0.0287, 0.0466, 0.0288), 0),
    Actuator("Dynamixel XM430-W350-T", "servo", 4.1, 3.0, 0.082, 12.0, 353.5, 4.8, 290.0, "ROBOTIS",
             "box", (0.0285, 0.0465, 0.0340), 0),
    Actuator("Dynamixel XM540-W270-T", "servo", 10.6, 7.0, 0.165, 12.0, 272.5, 6.7, 430.0, "ROBOTIS",
             "box", (0.0335, 0.0585, 0.0440), 0),
    Actuator("T-Motor AK70-10", "qdd", 24.8, 8.3, 0.521, 24.0, 10.0, 50.3, 280.0, "CubeMars/T-Motor",
             "cylinder", (0.0830, 0.0830, 0.0385), 2),
    Actuator("Unitree GO-M8010-6", "qdd", 23.7, 9.1, 0.340, 24.0, 6.33, 39.3, 300.0, "Unitree",
             "cylinder", (0.0830, 0.0830, 0.0450), 2),
    Actuator("T-Motor AK80-9", "qdd", 18.0, 9.0, 0.485, 48.0, 9.0, 41.9, 320.0, "CubeMars/T-Motor",
             "cylinder", (0.0800, 0.0800, 0.0394), 2),
    Actuator("T-Motor AK10-9", "qdd", 48.0, 18.0, 0.790, 48.0, 9.0, 31.4, 480.0, "CubeMars/T-Motor",
             "cylinder", (0.0985, 0.0985, 0.0460), 2),
    Actuator("T-Motor AK80-64", "qdd", 120.0, 48.0, 0.485, 48.0, 64.0, 6.3, 360.0, "CubeMars/T-Motor",
             "cylinder", (0.0820, 0.0820, 0.0540), 2),
    Actuator("Unitree M107 (B2/H1-class)", "qdd", 360.0, 90.0, 1.6, 48.0, 1.0, 33.5, 900.0, "Unitree",
             "cylinder", (0.1100, 0.1100, 0.0600), 2),
    Actuator("Harmonic Drive FHA-25C-100", "harmonic", 137.0, 28.0, 2.1, 48.0, 100.0, 12.0, 2500.0, "Harmonic Drive",
             "cylinder", (0.0720, 0.0720, 0.0500), 2),
    Actuator("Harmonic Drive FHA-40C-160", "harmonic", 520.0, 137.0, 3.8, 48.0, 160.0, 6.3, 4800.0, "Harmonic Drive",
             "cylinder", (0.1040, 0.1040, 0.0660), 2),
)

# --- SENSORS / COMPUTE / POWER / DRIVE / END-EFFECTORS ------------------------------------------------------
COMPONENTS: tuple[Component, ...] = (
    # cameras (a robot's "eyes")
    Component("Intel RealSense D435i", "camera", 0.072, 3.5, 334.0, "Intel", "RGB-D stereo + IMU, 0.2-10m"),
    Component("Luxonis OAK-D Pro", "camera", 0.091, 5.0, 299.0, "Luxonis", "stereo depth + 12MP RGB + on-board AI"),
    Component("Logitech C920", "camera", 0.162, 1.5, 70.0, "Logitech", "1080p RGB webcam"),
    Component("Arducam OV9281 global-shutter", "camera", 0.010, 1.0, 30.0, "Arducam", "1MP global-shutter mono (VIO/tracking)"),
    Component("Basler ace 5MP", "camera", 0.090, 2.5, 420.0, "Basler", "5MP industrial machine-vision"),
    # lidar
    Component("Slamtec RPLIDAR A2M12", "lidar", 0.190, 1.5, 320.0, "Slamtec", "2D 360deg, 12m, 8kHz"),
    Component("Livox Mid-360", "lidar", 0.265, 6.5, 749.0, "Livox", "3D 360deg x 59deg, 40m"),
    Component("Ouster OS1-32", "lidar", 0.447, 14.0, 8000.0, "Ouster", "3D 32-beam, 120m"),
    Component("Ouster OS2-128", "lidar", 0.92, 22.0, 18000.0, "Ouster", "3D 128-beam, 240m (large/outdoor)"),
    # imu / force-torque / gps / thermal (task-specific perception)
    Component("Bosch BNO055", "imu", 0.003, 0.04, 35.0, "Bosch", "9-DOF IMU + sensor fusion"),
    Component("VectorNav VN-100", "imu", 0.015, 0.2, 800.0, "VectorNav", "industrial AHRS/IMU"),
    Component("Robotiq FT 300-S", "force_torque", 0.300, 2.0, 4500.0, "Robotiq", "6-axis wrist F/T sensor"),
    Component("FLIR Boson 640", "thermal", 0.0075, 0.5, 1200.0, "Teledyne FLIR", "640x512 LWIR thermal core"),
    Component("u-blox ZED-F9P RTK GPS", "gps", 0.030, 0.5, 220.0, "u-blox", "cm-level RTK GNSS (outdoor)"),
    Component("ReSpeaker Mic Array v2", "microphone", 0.030, 1.0, 70.0, "Seeed", "4-mic far-field array"),
    # compute
    Component("Raspberry Pi 5 (8GB)", "compute", 0.046, 8.0, 80.0, "Raspberry Pi", "quad A76, no GPU AI"),
    Component("NVIDIA Jetson Orin Nano 8GB", "compute", 0.176, 15.0, 249.0, "NVIDIA", "40 TOPS edge AI"),
    Component("NVIDIA Jetson AGX Orin 64GB", "compute", 0.700, 60.0, 1999.0, "NVIDIA", "275 TOPS edge AI"),
    # power
    Component("LiPo 4S 5200mAh (14.8V)", "power", 0.440, 0.0, 45.0, "Turnigy", "77Wh, ~12V rail"),
    Component("LiPo 6S 8000mAh (22.2V)", "power", 1.05, 0.0, 110.0, "Turnigy", "178Wh, 24V rail"),
    Component("Li-ion 48V 12Ah pack", "power", 2.6, 0.0, 320.0, "custom", "576Wh, 48V rail (legged/humanoid)"),
    # power — socketed (wall) PSUs: the DEFAULT supply for tethered/benchtop robots, sized above the draw
    Component("Mean Well RSP-150-24", "power", 0.65, 0.0, 55.0, "Mean Well", "150W 24V enclosed PSU (socketed/wall)"),
    Component("Mean Well RSP-320-48", "power", 1.30, 0.0, 78.0, "Mean Well", "320W 48V enclosed PSU (socketed/wall)"),
    Component("Mean Well RSP-750-48", "power", 2.40, 0.0, 140.0, "Mean Well", "750W 48V enclosed PSU (socketed/wall)"),
    Component("Mean Well RSP-1500-48", "power", 4.20, 0.0, 320.0, "Mean Well", "1500W 48V enclosed PSU (socketed/wall)"),
    # mobile drive
    Component("Pololu 37Dx70L 12V gearmotor + encoder", "drive_motor", 0.215, 11.0, 40.0, "Pololu", "12V, 0.8Nm, 64CPR"),
    Component("100mm rubber drive wheel + hub", "wheel", 0.150, 0.0, 15.0, "Pololu", "100mm dia, 6mm hex"),
    Component("100mm Mecanum wheel", "wheel", 0.350, 0.0, 42.0, "Nexus", "100mm omni/strafe"),
    Component("Caster wheel 2in", "wheel", 0.080, 0.0, 8.0, "generic", "passive support caster"),
    # end effectors
    Component("Robotiq 2F-85 gripper", "gripper", 0.900, 6.0, 5500.0, "Robotiq", "85mm 2-finger adaptive"),
    Component("Dynamixel-driven 2-finger gripper", "gripper", 0.250, 4.0, 380.0, "ROBOTIS", "XM430 parallel jaw"),
)

# --- STRUCTURAL MATERIALS for the links. ``tier`` lets the per-part selector pick a STRONG load-path material
# for the skeleton, a lighter polymer for the outer SHELL, and METAL for contact parts (hands/feet/grippers) —
# intelligently, per task/robot. ``render_rgba`` gives each material its real look so the viewport shows
# material variety (dark steel, near-black carbon weave, a coloured polymer shell, bright machined metal).
MATERIALS: tuple[Material, ...] = (
    Material("Steel 4140", 7850.0, 6.0, "high-load skeleton / heavy joints", "skeleton-strong", "high",
             (0.34, 0.36, 0.40, 1.0), 0.7),
    Material("Titanium Ti-6Al-4V", 4430.0, 90.0, "strong + light skeleton (premium/aerospace)",
             "skeleton-strong", "high", (0.60, 0.60, 0.65, 1.0), 0.6),
    Material("Aluminum 7075-T6", 2810.0, 18.0, "aircraft-grade structural skeleton", "skeleton-strong",
             "high", (0.80, 0.82, 0.86, 1.0), 0.55),
    Material("Aluminum 6061-T6", 2700.0, 12.0, "CNC structural links (default)", "structural", "medium",
             (0.78, 0.80, 0.84, 1.0), 0.5),
    Material("Carbon-fiber composite", 1600.0, 90.0, "stiff, very light limbs (drones/legged)",
             "skeleton-strong", "high", (0.11, 0.12, 0.14, 1.0), 0.35),
    Material("Polycarbonate shell", 1200.0, 14.0, "tough coloured outer body panels", "shell", "medium",
             (0.20, 0.42, 0.72, 1.0), 0.45),
    Material("ABS (FDM print)", 1040.0, 30.0, "tough printed shell / covers", "shell", "low",
             (0.86, 0.55, 0.18, 1.0), 0.3),
    Material("PLA (FDM print)", 1240.0, 25.0, "rapid-prototype light shell", "shell", "low",
             (0.85, 0.85, 0.88, 1.0), 0.25),
    Material("TPU (rubber)", 1210.0, 35.0, "soft grippy pads (feet / fingertips)", "grip-soft", "low",
             (0.13, 0.13, 0.15, 1.0), 0.2),
    Material("Hardened steel (gripper)", 7850.0, 9.0, "wear-resistant metal hands/jaws", "contact-metal",
             "high", (0.46, 0.48, 0.52, 1.0), 0.8),
)


def materials_by_tier(tier: str) -> list[Material]:
    return [m for m in MATERIALS if m.tier == tier]

_BY_NAME = {c.name: c for c in COMPONENTS}


def component(name: str) -> Component | None:
    return _BY_NAME.get(name)


def by_category(category: str) -> list[Component]:
    return [c for c in COMPONENTS if c.category == category]


def material(name: str) -> Material | None:
    return next((m for m in MATERIALS if m.name == name), None)


def select_actuator(required_torque_nm: float, *, margin: float = 1.3,
                    required_speed_radps: float = 0.0, continuous_torque_nm: float = 0.0) -> Actuator:
    """Smallest actuator that meets the joint's real operating point. Always clears PEAK torque
    ``required_torque_nm * margin`` (a real engineering safety factor). When a required joint SPEED and/or a
    sustained (continuous) torque are given, it also requires ``max_speed_radps`` and ``rated_torque_nm`` to
    cover them -- because a real servo cannot be sized on peak torque alone: a high-torque QDD motor is often too
    SLOW for a fast gait, and running near stall continuously overheats. Falls back to the strongest actuator (by
    a torque*speed capability score) if nothing in the catalog meets every constraint."""
    need_pk = max(0.05, float(required_torque_nm)) * margin
    need_spd = max(0.0, float(required_speed_radps))
    need_cont = max(0.0, float(continuous_torque_nm))
    ladder = sorted(ACTUATORS, key=lambda a: (a.peak_torque_nm, a.max_speed_radps))
    for a in ladder:
        if a.peak_torque_nm >= need_pk and a.max_speed_radps >= need_spd and a.rated_torque_nm >= need_cont:
            return a
    # Nothing clears every constraint -> offer the CLOSEST: best worst-case coverage across the ACTIVE
    # constraints. This is monotone in the requirement (a bigger load never selects a weaker motor) and
    # matches the docstring's "strongest": when torque is the shortfall it will NOT trade peak torque away
    # for raw speed (the old ``max(peak*speed)`` did -- handing back a fast-but-weaker motor under overload,
    # so asking a robot to carry MORE could silently downsize its actuators).
    def _coverage(a):
        ratios = [a.peak_torque_nm / need_pk]
        if need_spd > 0:
            ratios.append(a.max_speed_radps / need_spd)
        if need_cont > 0:
            ratios.append(a.rated_torque_nm / need_cont)
        return min(ratios)
    return max(ladder, key=lambda a: (_coverage(a), a.peak_torque_nm * a.max_speed_radps))
