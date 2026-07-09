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

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Actuator:
    name: str
    kind: str                 # "servo" | "qdd" (quasi-direct-drive) | "harmonic" | "gearmotor"
    peak_torque_nm: float     # short-term peak / stall (selection ceiling)
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

    @property
    def no_load_rpm(self) -> float:
        """No-load output speed in RPM (the servo/motor's rated shaft speed), from ``max_speed_radps``."""
        return round(self.max_speed_radps * 60.0 / (2.0 * math.pi), 1)

    def to_spec(self) -> dict:
        """The part's real, STRUCTURED operating specs (queryable numbers, not a headline string)."""
        return {
            "part": self.name, "category": "actuator", "kind": self.kind, "vendor": self.vendor,
            "peak_torque_nm": self.peak_torque_nm, "rated_torque_nm": self.rated_torque_nm,
            "no_load_rpm": self.no_load_rpm, "max_speed_radps": round(self.max_speed_radps, 2),
            "voltage_v": self.voltage_v, "gear_ratio": self.gear_ratio,
            "mass_kg": self.mass_kg, "price_usd": self.price_usd,
        }


@dataclass(frozen=True)
class Component:
    name: str
    category: str             # camera | lidar | imu | force_torque | compute | power | wheel | drive_motor | gripper
    mass_kg: float
    power_w: float            # nominal electrical draw (0 for passive parts like wheels)
    price_usd: float
    vendor: str
    spec: str                 # one-line headline spec (human-readable)
    specs: dict = field(default_factory=dict)   # STRUCTURED real datasheet specs (resolution_mp, range_m, channels, fps, fov_deg, rate_hz, ...)

    def to_spec(self) -> dict:
        """The part's real, STRUCTURED specs merged with its commercial fields — queryable, not a headline string."""
        return {"part": self.name, "category": self.category, "vendor": self.vendor, "headline": self.spec,
                "mass_kg": self.mass_kg, "power_w": self.power_w, "price_usd": self.price_usd, **self.specs}


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
    # cameras (a robot's "eyes") — resolution (MP), frame rate (fps), depth range + FOV are real datasheet figures
    Component("Intel RealSense D435i", "camera", 0.072, 3.5, 334.0, "Intel", "RGB-D stereo + IMU, 0.2-10m",
              {"modality": "rgbd", "rgb_mp": 2.1, "depth_resolution": "1280x720", "fps": 90,
               "range_m": 10.0, "min_range_m": 0.2, "fov_deg": [87, 58], "onboard_imu": True}),
    Component("Luxonis OAK-D Pro", "camera", 0.091, 5.0, 299.0, "Luxonis", "stereo depth + 12MP RGB + on-board AI",
              {"modality": "rgbd", "rgb_mp": 12.0, "fps": 60, "range_m": 19.0, "fov_deg": [80, 55], "onboard_ai_tops": 4}),
    Component("Logitech C920", "camera", 0.162, 1.5, 70.0, "Logitech", "1080p RGB webcam",
              {"modality": "rgb", "rgb_mp": 2.1, "resolution": "1920x1080", "fps": 30, "fov_deg": [78]}),
    Component("Arducam OV9281 global-shutter", "camera", 0.010, 1.0, 30.0, "Arducam", "1MP global-shutter mono (VIO/tracking)",
              {"modality": "mono", "rgb_mp": 1.0, "resolution": "1280x800", "fps": 120, "shutter": "global"}),
    Component("Basler ace 5MP", "camera", 0.090, 2.5, 420.0, "Basler", "5MP industrial machine-vision",
              {"modality": "rgb", "rgb_mp": 5.0, "resolution": "2592x2048", "fps": 35}),
    # lidar — channels (beam count), range (m), FOV (deg), point rate are real datasheet figures
    Component("Slamtec RPLIDAR A2M12", "lidar", 0.190, 1.5, 320.0, "Slamtec", "2D 360deg, 12m, 8kHz",
              {"dimensions": "2D", "channels": 1, "range_m": 12.0, "fov_deg": [360, 0], "scan_hz": 15, "points_per_s": 8000}),
    Component("Livox Mid-360", "lidar", 0.265, 6.5, 749.0, "Livox", "3D 360deg x 59deg, 40m",
              {"dimensions": "3D", "channels": "non-repetitive", "range_m": 40.0, "fov_deg": [360, 59], "points_per_s": 200000}),
    Component("Ouster OS1-32", "lidar", 0.447, 14.0, 8000.0, "Ouster", "3D 32-beam, 120m",
              {"dimensions": "3D", "channels": 32, "range_m": 120.0, "fov_deg": [360, 45], "points_per_s": 655360}),
    Component("Ouster OS2-128", "lidar", 0.92, 22.0, 18000.0, "Ouster", "3D 128-beam, 240m (large/outdoor)",
              {"dimensions": "3D", "channels": 128, "range_m": 240.0, "fov_deg": [360, 22.5], "points_per_s": 2621440}),
    # imu / force-torque / gps / thermal (task-specific perception)
    Component("Bosch BNO055", "imu", 0.003, 0.04, 35.0, "Bosch", "9-DOF IMU + sensor fusion",
              {"dof": 9, "rate_hz": 100, "onboard_fusion": True}),
    Component("VectorNav VN-100", "imu", 0.015, 0.2, 800.0, "VectorNav", "industrial AHRS/IMU",
              {"dof": 9, "rate_hz": 800, "onboard_fusion": True, "grade": "industrial"}),
    Component("Robotiq FT 300-S", "force_torque", 0.300, 2.0, 4500.0, "Robotiq", "6-axis wrist F/T sensor",
              {"axes": 6, "force_range_n": 300, "torque_range_nm": 30, "rate_hz": 100}),
    Component("FLIR Boson 640", "thermal", 0.0075, 0.5, 1200.0, "Teledyne FLIR", "640x512 LWIR thermal core",
              {"modality": "thermal", "resolution": "640x512", "fps": 60, "spectral_band": "LWIR"}),
    Component("u-blox ZED-F9P RTK GPS", "gps", 0.030, 0.5, 220.0, "u-blox", "cm-level RTK GNSS (outdoor)",
              {"type": "RTK-GNSS", "accuracy_m": 0.01, "rate_hz": 8}),
    Component("ReSpeaker Mic Array v2", "microphone", 0.030, 1.0, 70.0, "Seeed", "4-mic far-field array",
              {"mics": 4, "type": "far-field", "onboard_aec": True}),
    # compute — AI throughput (TOPS) + RAM
    Component("Raspberry Pi 5 (8GB)", "compute", 0.046, 8.0, 80.0, "Raspberry Pi", "quad A76, no GPU AI",
              {"cpu": "quad Cortex-A76", "ai_tops": 0, "ram_gb": 8}),
    Component("NVIDIA Jetson Orin Nano 8GB", "compute", 0.176, 15.0, 249.0, "NVIDIA", "40 TOPS edge AI",
              {"ai_tops": 40, "ram_gb": 8, "gpu": "Ampere 1024-core"}),
    Component("NVIDIA Jetson AGX Orin 64GB", "compute", 0.700, 60.0, 1999.0, "NVIDIA", "275 TOPS edge AI",
              {"ai_tops": 275, "ram_gb": 64, "gpu": "Ampere 2048-core"}),
    # power — usable energy (Wh) + rail voltage; tethered PSUs vs onboard packs
    Component("LiPo 4S 5200mAh (14.8V)", "power", 0.440, 0.0, 45.0, "Turnigy", "77Wh, ~12V rail",
              {"type": "LiPo", "capacity_wh": 77, "voltage_v": 14.8, "tethered": False}),
    Component("LiPo 6S 8000mAh (22.2V)", "power", 1.05, 0.0, 110.0, "Turnigy", "178Wh, 24V rail",
              {"type": "LiPo", "capacity_wh": 178, "voltage_v": 22.2, "tethered": False}),
    Component("Li-ion 48V 12Ah pack", "power", 2.6, 0.0, 320.0, "custom", "576Wh, 48V rail (legged/humanoid)",
              {"type": "Li-ion", "capacity_wh": 576, "voltage_v": 48, "tethered": False}),
    # power — socketed (wall) PSUs: the DEFAULT supply for tethered/benchtop robots, sized above the draw
    Component("Mean Well RSP-150-24", "power", 0.65, 0.0, 55.0, "Mean Well", "150W 24V enclosed PSU (socketed/wall)",
              {"type": "PSU", "output_w": 150, "voltage_v": 24, "tethered": True}),
    Component("Mean Well RSP-320-48", "power", 1.30, 0.0, 78.0, "Mean Well", "320W 48V enclosed PSU (socketed/wall)",
              {"type": "PSU", "output_w": 320, "voltage_v": 48, "tethered": True}),
    Component("Mean Well RSP-750-48", "power", 2.40, 0.0, 140.0, "Mean Well", "750W 48V enclosed PSU (socketed/wall)",
              {"type": "PSU", "output_w": 750, "voltage_v": 48, "tethered": True}),
    Component("Mean Well RSP-1500-48", "power", 4.20, 0.0, 320.0, "Mean Well", "1500W 48V enclosed PSU (socketed/wall)",
              {"type": "PSU", "output_w": 1500, "voltage_v": 48, "tethered": True}),
    # mobile drive
    Component("Pololu 37Dx70L 12V gearmotor + encoder", "drive_motor", 0.215, 11.0, 40.0, "Pololu", "12V, 0.8Nm, 64CPR",
              {"voltage_v": 12, "stall_torque_nm": 0.8, "no_load_rpm": 100, "encoder_cpr": 64}),
    Component("100mm rubber drive wheel + hub", "wheel", 0.150, 0.0, 15.0, "Pololu", "100mm dia, 6mm hex",
              {"diameter_mm": 100, "type": "rubber", "driven": True}),
    Component("100mm Mecanum wheel", "wheel", 0.350, 0.0, 42.0, "Nexus", "100mm omni/strafe",
              {"diameter_mm": 100, "type": "mecanum", "driven": True}),
    Component("Caster wheel 2in", "wheel", 0.080, 0.0, 8.0, "generic", "passive support caster",
              {"diameter_mm": 50, "type": "caster", "driven": False}),
    # end effectors
    Component("Robotiq 2F-85 gripper", "gripper", 0.900, 6.0, 5500.0, "Robotiq", "85mm 2-finger adaptive",
              {"fingers": 2, "stroke_mm": 85, "grip_force_n": 235, "type": "adaptive"}),
    Component("Dynamixel-driven 2-finger gripper", "gripper", 0.250, 4.0, 380.0, "ROBOTIS", "XM430 parallel jaw",
              {"fingers": 2, "stroke_mm": 60, "type": "parallel"}),
    # aerial propulsion — a quadcopter's defining parts (one motor+prop and one ESC PER rotor, one flight controller)
    Component("T-Motor MN3110 700KV + 9x4.5 prop", "rotor", 0.110, 180.0, 33.0, "T-Motor",
              "700KV brushless outrunner + 9in prop, ~0.9 kg thrust",
              {"kv": 700, "prop_in": 9.0, "max_thrust_kg": 0.9, "voltage_v": 14.8}),
    Component("BLHeli-32 40A ESC", "esc", 0.028, 0.0, 14.0, "generic", "40A brushless ESC w/ BEC (one per rotor)",
              {"current_a": 40, "protocol": "DShot600"}),
    Component("Holybro Pixhawk 6C flight controller", "flight_controller", 0.060, 3.0, 210.0, "Holybro",
              "PX4/ArduPilot autopilot, dual IMU + barometer + failsafe",
              {"imus": 2, "baro": True, "stack": "PX4/ArduPilot"}),
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


# categories the auto-selector fills; "actuator" is the ladder above. Used to validate a pin's category.
PART_CATEGORIES = ("actuator", "camera", "lidar", "imu", "force_torque", "thermal", "gps", "microphone",
                   "compute", "power", "drive_motor", "wheel", "gripper")


def resolve_part(name: str):
    """Find a part by exact or case-insensitive substring name across BOTH the actuator ladder and the component
    catalog. Returns the Actuator/Component object, or None. Lets a user say 'AK80-9' or 'Ouster OS1'."""
    if not name:
        return None
    pool = list(ACTUATORS) + list(COMPONENTS)
    for p in pool:                                             # exact first
        if p.name == name:
            return p
    low = name.lower()
    hits = [p for p in pool if low in p.name.lower()]
    return hits[0] if len(hits) == 1 else next((p for p in hits if p.name.lower() == low), hits[0] if hits else None)


def part_specs(name: str) -> dict | None:
    """The real STRUCTURED specs of one part (torque/rpm for a motor; resolution/range/channels/fps for a sensor)."""
    p = resolve_part(name)
    return p.to_spec() if p is not None else None


def list_parts(category: str | None = None) -> list[dict]:
    """Every catalog part (or one category) with its real structured specs — the queryable options for a build or a
    part swap. category='actuator' lists the motor ladder; otherwise the sensor/compute/power/drive/gripper parts."""
    out: list[dict] = []
    if category in (None, "actuator"):
        out += [a.to_spec() for a in sorted(ACTUATORS, key=lambda a: a.peak_torque_nm)]
    for c in COMPONENTS:
        if category in (None, c.category):
            out.append(c.to_spec())
    return out


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
