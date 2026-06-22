from __future__ import annotations

import re

from virturoid.schemas.requirements import RequirementsRecord


def build_requirements_from_prompt(
    prompt: str,
    *,
    requirement_id: str | None = None,
    payload_kg: float | None = None,
    reach_m: float | None = None,
    sensor: str | None = None,
) -> RequirementsRecord:
    """Build deterministic requirements from a user prompt.

    This is the MVP fallback before real language-model extraction. It keeps
    custom prompt support inspectable and easy to improve.
    """
    normalized_prompt = prompt.strip()
    inferred_payload = payload_kg if payload_kg is not None else _extract_payload_kg(normalized_prompt)
    inferred_reach = reach_m if reach_m is not None else _extract_reach_m(normalized_prompt)
    environment = _infer_environment(normalized_prompt)
    sensor_requirements = [_infer_sensor(normalized_prompt, sensor)]

    return RequirementsRecord(
        id=requirement_id or f"req_{_slugify(normalized_prompt)[:48]}",
        prompt=normalized_prompt,
        task_summary=_summarize_task(normalized_prompt),
        environment=environment,
        payload_kg=inferred_payload,
        reach_m=inferred_reach,
        precision_m=0.005,
        manufacturing_constraints=["3d_printable_mounts", "off_the_shelf_actuators"],
        sensor_requirements=sensor_requirements,
    )


def _extract_payload_kg(prompt: str) -> float:
    kg_match = re.search(r"(\d+(?:\.\d+)?)\s*kg", prompt, re.IGNORECASE)
    if kg_match:
        return float(kg_match.group(1))
    gram_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|gram|grams)\b", prompt, re.IGNORECASE)
    if gram_match:
        return float(gram_match.group(1)) / 1000.0
    return 0.25


def _extract_reach_m(prompt: str) -> float:
    meter_match = re.search(r"(\d+(?:\.\d+)?)\s*m(?:eter|etre|eters|etres)?\b", prompt, re.IGNORECASE)
    if meter_match:
        return float(meter_match.group(1))
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm\b", prompt, re.IGNORECASE)
    if cm_match:
        return float(cm_match.group(1)) / 100.0
    return 0.65


def _infer_environment(prompt: str) -> str:
    lowered = prompt.lower()
    if "warehouse" in lowered or "conveyor" in lowered:
        return "warehouse"
    if "table" in lowered or "desk" in lowered or "bench" in lowered:
        return "tabletop"
    return "tabletop"


def _infer_sensor(prompt: str, sensor: str | None) -> str:
    if sensor:
        return sensor
    lowered = prompt.lower()
    if "lidar" in lowered:
        return "lidar"
    if "depth" in lowered or "3d" in lowered or "rgbd" in lowered or "rgb-d" in lowered:
        return "rgbd_camera"
    return "rgbd_camera"


def _summarize_task(prompt: str) -> str:
    cleaned = " ".join(prompt.split())
    return cleaned[:120] if cleaned else "Autonomously build and train/evaluate a robot arm."


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return slug or "mvp_robot_arm"

