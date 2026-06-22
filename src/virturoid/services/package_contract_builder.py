from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

from virturoid.schemas.package_contract import PackageArtifactCheck, RobotPackageContract

PACKAGE_CONTRACT_URI = "reports/robot_package_contract.json"


def write_robot_package_contract(
    package_dir: Path,
    *,
    package_type: str,
    robot_class: str,
    species: str,
    morphology_template_id: str,
    task_type: str,
    artifacts: dict[str, str],
    training: dict | None = None,
) -> RobotPackageContract:
    contract = build_robot_package_contract(
        package_dir,
        package_type=package_type,
        robot_class=robot_class,
        species=species,
        morphology_template_id=morphology_template_id,
        task_type=task_type,
        artifacts=artifacts,
        training=training,
    )
    path = Path(package_dir) / PACKAGE_CONTRACT_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), indent=2), encoding="utf-8")
    return contract


def build_robot_package_contract(
    package_dir: Path,
    *,
    package_type: str,
    robot_class: str,
    species: str,
    morphology_template_id: str,
    task_type: str,
    artifacts: dict[str, str],
    training: dict | None = None,
) -> RobotPackageContract:
    package_dir = Path(package_dir)
    checks = [_check_artifact(package_dir, key, uri) for key, uri in artifacts.items()]
    capabilities = _capabilities(artifacts, training)
    ok = all(check.exists and check.parse_status != "fail" for check in checks)
    contract = RobotPackageContract(
        id=f"package_contract_{morphology_template_id}",
        package_type=package_type,
        robot_class=robot_class,
        species=species,
        morphology_template_id=morphology_template_id,
        task_type=task_type,
        capabilities=capabilities,
        artifact_checks=checks,
        ok=ok,
        notes=[
            "Common contract for robot-class package outputs.",
            "Future robot builders should emit this contract even when they do not yet support full training.",
        ],
    )
    validation = contract.validate()
    if not validation.ok:
        issues = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"{contract.id} failed validation: {issues}")
    return contract


def _check_artifact(package_dir: Path, key: str, uri: str) -> PackageArtifactCheck:
    path = package_dir / uri
    exists = path.exists()
    media_type = _media_type(uri)
    parse_status = "not_checked"
    detail = ""
    if not exists:
        parse_status = "missing"
        detail = "Artifact does not exist."
    elif uri.endswith(".json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            parse_status = "pass"
            detail = "JSON parses."
        except Exception as exc:  # noqa: BLE001
            parse_status = "fail"
            detail = f"JSON parse failed: {exc}"
    elif uri.endswith(".xml") or uri.endswith(".urdf"):
        try:
            ElementTree.parse(path)
            parse_status = "pass"
            detail = "XML parses."
        except Exception as exc:  # noqa: BLE001
            parse_status = "fail"
            detail = f"XML parse failed: {exc}"
    else:
        detail = "Presence checked."
    return PackageArtifactCheck(
        key=key,
        uri=uri,
        exists=exists,
        parse_status=parse_status,
        media_type=media_type,
        detail=detail,
    )


def _capabilities(artifacts: dict[str, str], training: dict | None) -> list[str]:
    capabilities = ["robot_genome", "robot_model_export"]
    if "scene_set" in artifacts:
        capabilities.append("scene_generation")
    if "compiled_scene_index" in artifacts:
        capabilities.append("compiled_simulation_scenes")
    if "perception_config" in artifacts:
        capabilities.append("cv_perception_contract")
    if "world_model_contract" in artifacts:
        capabilities.append("world_model_contract")
    if "synthetic_observation_manifest" in artifacts:
        capabilities.append("synthetic_observation_dataset")
    if "synthetic_observation_index" in artifacts:
        capabilities.append("generated_cv_annotations")
    if "world_state_index" in artifacts:
        capabilities.append("generated_world_state_snapshots")
    if "simulator_contract" in artifacts:
        capabilities.append("simulator_contract")
    if "training_run_config" in artifacts:
        capabilities.append("training_run_config")
    if training and training.get("ran"):
        capabilities.extend(["policy_training", "controller_export"])
    return capabilities


def _media_type(uri: str) -> str | None:
    if uri.endswith(".json"):
        return "application/json"
    if uri.endswith(".xml") or uri.endswith(".urdf"):
        return "application/xml"
    if uri.endswith(".py"):
        return "text/x-python"
    return None
