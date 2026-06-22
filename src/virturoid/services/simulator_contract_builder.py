from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree

from virturoid.schemas.base import ArtifactRef
from virturoid.schemas.simulator_contract import (
    SimulatorBackendCapability,
    SimulatorRunContract,
    SimulatorSceneContract,
)

SIMULATOR_CONTRACT_URI = "simulation/simulator_contract.json"


def write_simulator_contract_from_export(
    package_dir: Path,
    config_uri: str = "training/training_run_config.json",
) -> Path:
    contract = build_simulator_contract_from_export(package_dir, config_uri=config_uri)
    validation = contract.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{contract.id} failed validation: {issues}")

    path = package_dir / SIMULATOR_CONTRACT_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), indent=2), encoding="utf-8")
    return path


def build_simulator_contract_from_export(
    package_dir: Path,
    config_uri: str = "training/training_run_config.json",
) -> SimulatorRunContract:
    config = _read_json(package_dir / config_uri)
    robot = _read_json(package_dir / "robot" / "robot_genome.json")
    compiled_index = _read_json(package_dir / config["compiled_scene_artifact"]["uri"])
    scene_contracts = [
        _scene_contract(package_dir, entry)
        for entry in compiled_index.get("scenes", [])
    ]
    mujoco_available = importlib.util.find_spec("mujoco") is not None
    blocking_gaps = []
    if not mujoco_available:
        blocking_gaps.append("Python package 'mujoco' is not installed; dry-run adapter remains the active runner.")

    return SimulatorRunContract(
        id=f"sim_contract_{config['id']}",
        backend=config["backend"],
        robot_genome_id=config["robot_genome_id"],
        training_run_config_uri=config_uri,
        compiled_scene_index_uri=config["compiled_scene_artifact"]["uri"],
        required_inputs=[
            ArtifactRef(uri=config["robot_model"]["uri"], media_type=config["robot_model"].get("media_type")),
            ArtifactRef(uri=config["policy_artifact"]["uri"], media_type=config["policy_artifact"].get("media_type")),
            ArtifactRef(uri=config["perception_artifact"]["uri"], media_type=config["perception_artifact"].get("media_type")),
            ArtifactRef(uri=config["objective_artifact"]["uri"], media_type=config["objective_artifact"].get("media_type")),
            ArtifactRef(uri=config["curriculum_artifact"]["uri"], media_type=config["curriculum_artifact"].get("media_type")),
            ArtifactRef(uri=config["compiled_scene_artifact"]["uri"], media_type=config["compiled_scene_artifact"].get("media_type")),
        ],
        backend_capabilities=[
            SimulatorBackendCapability(
                backend="dry_run",
                adapter_name="dry_run.synthetic_v0",
                status="contract_runner_available",
                available=True,
                reason="Deterministic adapter can consume scene/config artifacts and write replay traces.",
            ),
            SimulatorBackendCapability(
                backend="mujoco",
                adapter_name="mujoco.python",
                status="real_backend_available" if mujoco_available else "not_installed",
                available=mujoco_available,
                reason="Detected importable Python package 'mujoco'." if mujoco_available else "Install MuJoCo Python bindings to execute physics rollouts.",
            ),
            SimulatorBackendCapability(
                backend="isaac",
                adapter_name="isaac.sim",
                status="planned",
                available=False,
                reason="Isaac adapter contract is planned after the MuJoCo backend path is proven.",
            ),
        ],
        scene_contracts=scene_contracts,
        runner_requirements=[
            "load robot model and compiled scene XML",
            "bind policy action trace or learned policy to robot joints",
            "emit observation streams declared in perception_config.json",
            "score reward terms declared in training_objective.json",
            "write replay index and episode_trace.jsonl",
        ],
        blocking_gaps=blocking_gaps,
        notes=[
            f"Robot {robot['id']} has {len(robot.get('joints', []))} joint(s) and {len(robot.get('sensors', []))} sensor(s).",
            "This artifact is the handoff from package-shaped MVP to real simulator execution.",
        ],
    )


def _scene_contract(package_dir: Path, entry: dict) -> SimulatorSceneContract:
    xml_uri = entry["mujoco_xml"]
    try:
        root = ElementTree.parse(package_dir / xml_uri).getroot()
        status = "pass" if root.tag == "mujoco" else f"fail:root={root.tag}"
    except Exception as exc:  # noqa: BLE001
        status = f"fail:{exc}"
    return SimulatorSceneContract(
        scene_id=entry["scene_id"],
        purpose=entry["purpose"],
        compiled_scene_xml_uri=xml_uri,
        xml_parse_status=status,
        object_count=entry.get("object_count", 0),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
