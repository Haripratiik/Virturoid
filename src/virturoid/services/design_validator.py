"""Physical-feasibility gates for a generated design (#7).

Answers "can this robot actually perform this task" before training: reach
envelope vs task distance, actuator torque feasibility, joint limits, mass
plausibility, training-scene coverage, and (with MuJoCo) a collision-free
spawn. A failing hard gate marks the design impossible as specified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from virturoid.schemas.components import Component
from virturoid.schemas.design_validation import DesignGate, DesignValidationReport
from virturoid.schemas.requirements import RequirementsRecord
from virturoid.schemas.robot import RobotGenome
from virturoid.schemas.scenes import SceneSet
from virturoid.services.robot_kinematics import TABLE_TOP_Z, compute_arm_layout, iter_links

_G = 9.81


def validate_design(
    requirements: RequirementsRecord,
    robot_genome: RobotGenome,
    cad_models,
    components: list[Component],
    scene_sets: list[SceneSet],
    package_dir: Path | None = None,
) -> DesignValidationReport:
    layout = compute_arm_layout(robot_genome, cad_models)
    chain = _linearize(layout)
    shoulder = _first_pitch(chain)

    gates = [
        _gate_joint_limits(robot_genome),
        _gate_mass(chain, components),
        _gate_reach(requirements, chain, shoulder, scene_sets),
        _gate_actuators(requirements, chain, shoulder, components),
        _gate_scene_coverage(scene_sets, chain, shoulder),
        _gate_collision_spawn(package_dir),
    ]

    report = DesignValidationReport(
        id=f"design_validation_{robot_genome.id}",
        robot_genome_id=robot_genome.id,
        task_graph_id="",
        gates=gates,
        notes=[
            "Gates are estimates from the kinematic layout, part specs, and scenes.",
            "A failing gate means the design cannot perform the task as specified.",
        ],
    )
    return report.recompute_ok()


def write_design_validation_report(
    report: DesignValidationReport,
    package_dir: Path,
    uri: str = "reports/design_validation_report.json",
) -> Path:
    path = package_dir / uri
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


# --- gates -----------------------------------------------------------------


def _gate_joint_limits(robot: RobotGenome) -> DesignGate:
    missing = [
        joint.name
        for joint in robot.joints
        if joint.joint_type == "revolute" and not (joint.limit and joint.limit.lower is not None and joint.limit.upper is not None)
    ]
    if missing:
        return DesignGate("joint_limits", "fail", f"Revolute joints missing limits: {', '.join(missing)}.")
    return DesignGate("joint_limits", "pass", "All revolute joints define position limits.")


def _gate_mass(chain: list[dict], components: list[Component]) -> DesignGate:
    link_mass = sum(item["length_mass"][1] for item in chain)
    component_mass = sum(c.mass_kg for c in components if c.mass_kg)
    total = round(link_mass + component_mass, 3)
    zero_links = [item["name"] for item in chain if item["length_mass"][1] <= 0]
    if zero_links:
        return DesignGate("mass_plausible", "fail", f"Links with non-positive mass: {', '.join(zero_links)}.", {"total_mass_kg": total})
    status = "pass" if 0.2 <= total <= 50.0 else "warn"
    return DesignGate("mass_plausible", status, f"Estimated total mass {total} kg.", {"total_mass_kg": total})


def _gate_reach(requirements, chain, shoulder, scene_sets) -> DesignGate:
    if shoulder is None:
        return DesignGate("reach", "skipped", "No pitch joint found to estimate reach.")
    envelope = round(shoulder["bendable"], 3)
    shoulder_z = shoulder["proximal_z"]
    required = requirements.reach_m or 0.0
    targets = _target_distances(scene_sets, shoulder_z)
    max_target = round(max(targets), 3) if targets else 0.0
    measured = {"reach_envelope_m": envelope, "shoulder_height_m": round(shoulder_z, 3), "required_reach_m": required, "max_target_distance_m": max_target}
    if max_target and envelope < max_target:
        return DesignGate("reach", "fail", f"Reach envelope {envelope} m < farthest task target {max_target} m.", measured)
    if required and envelope < required:
        return DesignGate("reach", "warn", f"Reach envelope {envelope} m below requested reach {required} m.", measured)
    return DesignGate("reach", "pass", f"Reach envelope {envelope} m covers task targets.", measured)


def _gate_actuators(requirements, chain, shoulder, components) -> DesignGate:
    actuator = next((c for c in components if c.actuator), None)
    if actuator is None or shoulder is None:
        return DesignGate("actuator_torque", "skipped", "No actuator spec or pitch joint available.")
    # Worst-case static torque at the shoulder with the arm extended horizontally.
    required_torque = _static_shoulder_torque(chain, shoulder, requirements.payload_kg or 0.0)
    rated = actuator.actuator.rated_torque_nm
    stall = actuator.actuator.stall_torque_nm
    measured = {"required_torque_nm": round(required_torque, 3), "rated_torque_nm": rated, "stall_torque_nm": stall}
    if stall and required_torque > stall:
        return DesignGate("actuator_torque", "fail", f"Required {required_torque:.2f} Nm exceeds stall torque {stall} Nm.", measured)
    if rated and required_torque > rated:
        return DesignGate("actuator_torque", "warn", f"Required {required_torque:.2f} Nm exceeds continuous rated {rated} Nm (within stall).", measured)
    return DesignGate("actuator_torque", "pass", f"Shoulder torque {required_torque:.2f} Nm within rated {rated} Nm.", measured)


def _gate_scene_coverage(scene_sets, chain, shoulder) -> DesignGate:
    purposes = {s.purpose: len(s.scenes) for s in scene_sets}
    total = sum(purposes.values())
    unreachable = 0
    if shoulder is not None:
        for distance in _target_distances(scene_sets, shoulder["proximal_z"]):
            if distance > shoulder["bendable"]:
                unreachable += 1
    measured = {"scenes_per_purpose": purposes, "total_scenes": total, "unreachable_targets": unreachable}
    if unreachable:
        return DesignGate("scene_coverage", "fail", f"{unreachable} scene target(s) outside the reach envelope.", measured)
    if total < 10:
        return DesignGate("scene_coverage", "warn", f"Only {total} training scenes generated.", measured)
    return DesignGate("scene_coverage", "pass", f"{total} scenes across {len(purposes)} purposes, all reachable.", measured)


def _gate_collision_spawn(package_dir: Path | None) -> DesignGate:
    import importlib.util

    if package_dir is None or importlib.util.find_spec("mujoco") is None:
        return DesignGate("collision_free_spawn", "skipped", "MuJoCo not available; spawn collision check skipped.")
    try:
        import mujoco

        index_path = Path(package_dir) / "simulation" / "mujoco" / "compiled_scene_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        xml_uri = index["scenes"][0]["mujoco_xml"]
        model = mujoco.MjModel.from_xml_path(str(Path(package_dir) / xml_uri))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        deepest = min((float(data.contact[i].dist) for i in range(data.ncon)), default=0.0)
        measured = {"contacts_at_spawn": int(data.ncon), "deepest_penetration_m": round(deepest, 5)}
        if deepest < -0.005:
            return DesignGate("collision_free_spawn", "fail", f"Robot spawns penetrating by {abs(deepest):.4f} m.", measured)
        return DesignGate("collision_free_spawn", "pass", "Robot spawns without deep penetration.", measured)
    except Exception as exc:  # noqa: BLE001
        return DesignGate("collision_free_spawn", "skipped", f"Spawn check could not run: {exc}.")


# --- helpers ---------------------------------------------------------------


def _linearize(root) -> list[dict]:
    """Flatten the chain with cumulative proximal heights and bendable lengths."""
    ordered = list(iter_links(root))
    by_name = {link.name: link for link in ordered}
    proximal_z = {}
    for link in ordered:
        if link.parent is None:
            proximal_z[link.name] = TABLE_TOP_Z
        else:
            parent = by_name[link.parent]
            proximal_z[link.name] = proximal_z[parent.name] + parent.length_m
    chain = []
    for link in ordered:
        chain.append(
            {
                "name": link.name,
                "length_mass": (link.length_m, link.mass_kg),
                "proximal_z": proximal_z[link.name],
                "axis": (link.joint.axis_xyz if link.joint else None),
            }
        )
    # Bendable length from each link to the tip (single chain).
    total_from = 0.0
    cumulative = {}
    for item in reversed(chain):
        total_from += item["length_mass"][0]
        cumulative[item["name"]] = total_from
    for item in chain:
        item["bendable"] = cumulative[item["name"]]
    return chain


def _first_pitch(chain: list[dict]):
    for item in chain:
        axis = item["axis"]
        if axis and abs(axis[1]) > 0.5:  # rotation about Y == pitch
            return item
    return None


def _target_distances(scene_sets, shoulder_z: float) -> list[float]:
    distances = []
    for scene_set in scene_sets:
        for scene in scene_set.scenes:
            for obj in scene.objects:
                if obj.object_type in {"cube", "box"}:
                    x, y, z = obj.pose_xyz_rpy[0], obj.pose_xyz_rpy[1], obj.pose_xyz_rpy[2]
                    horizontal = math.hypot(x, y)
                    distances.append(math.hypot(horizontal, z - shoulder_z))
    return distances


def _static_shoulder_torque(chain: list[dict], shoulder: dict, payload_kg: float) -> float:
    """Gravity torque at the shoulder with distal links extended horizontally."""
    shoulder_index = next(i for i, item in enumerate(chain) if item["name"] == shoulder["name"])
    lever = 0.0
    torque = 0.0
    for item in chain[shoulder_index:]:
        length, mass = item["length_mass"]
        com_lever = lever + length / 2.0
        torque += mass * _G * com_lever
        lever += length
    torque += payload_kg * _G * lever  # payload at the tip
    return torque
