"""Prompt Input Compiler — Phase 0 of the Input Ingestion Research Plan.

Today prompt parsing is split across ``requirements_builder`` (payload/reach/environment/sensor) and
``spec_parser`` (height / self-weight budget / payload / pinned parts), and neither records *provenance*: the
build cannot say which fields the user actually stated vs which were inferred or defaulted, and it cannot flag
when the two parsers disagree (e.g. ``requirements_builder`` reads a *self-weight budget* or a *stature* as if
it were a payload or an arm reach).

This service merges both into one :class:`InputInterpretation` where every field carries an
:class:`InputEvidence` (source type + confidence + conflicts), raises a *high-value* intake question only when
a genuine conflict is detected, and writes ``input/interpretation.json`` into every package. A downstream
compliance report can then be linked back to that evidence, so "did the build honor my spec?" is answered
against what the user actually asked for.

Deterministic and offline (AGENTS.md): no LLM, no backend imports.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from virturoid.schemas.input_bundle import (
    InputEvidence,
    InputInterpretation,
    InputSourceType,
    IntakeQuestion,
)
from virturoid.services.requirements_builder import build_requirements_from_prompt
from virturoid.services.spec_parser import parse_prompt_spec

# Presence detectors used only to decide a field's *source* (parsed vs defaulted). Value extraction stays in the
# existing parsers; these just answer "did the user mention this at all?".
_MASS_TOKEN = re.compile(r"\d+(?:\.\d+)?\s*(?:kg|g|gram|grams)\b", re.IGNORECASE)
_LENGTH_TOKEN = re.compile(r"\d+(?:\.\d+)?\s*(?:m|meter|metre|meters|metres|cm|centimet\w*)\b", re.IGNORECASE)
_ENV_TOKEN = re.compile(
    r"\b(warehouse|conveyor|table|desk|bench|tabletop|outdoor|indoor|field|kitchen|shelf|dock|floor)\b",
    re.IGNORECASE,
)
_SENSOR_TOKEN = re.compile(
    r"\b(lidar|depth|rgb-?d|3d|camera|thermal|force[- ]?torque|f/t|imu|gps|gnss|encoder|tactile)\b",
    re.IGNORECASE,
)

# Confidence priors per source type — deterministic so tests and UI stay stable.
_CONF = {
    InputSourceType.EXPLICIT: 1.0,
    InputSourceType.USER_CONFIRMED: 0.95,
    InputSourceType.CALIBRATED: 0.9,
    InputSourceType.PARSED: 0.85,
    InputSourceType.INFERRED: 0.4,
    InputSourceType.DEFAULTED: 0.3,
}


def _ev(field_path: str, value: Any, source: InputSourceType, unit: str | None = None,
        *, confidence: float | None = None, conflicts: list[str] | None = None,
        note: str | None = None) -> InputEvidence:
    return InputEvidence(
        field_path=field_path,
        value=value,
        source_type=source,
        unit=unit,
        confidence=_CONF[source] if confidence is None else confidence,
        conflicts=list(conflicts or []),
        note=note,
    )


def interpret_prompt(
    prompt: str,
    *,
    requirement_id: str | None = None,
    payload_kg: float | None = None,
    reach_m: float | None = None,
    sensor: str | None = None,
) -> InputInterpretation:
    """Merge ``requirements_builder`` + ``spec_parser`` into one provenance-tracked interpretation.

    Explicit CLI/UI arguments (``payload_kg``/``reach_m``/``sensor``) are recorded as EXPLICIT evidence;
    prompt-derived values are PARSED; genuine parser disagreements become INFERRED evidence with a conflict
    note plus a targeted intake question; everything else is DEFAULTED. The build still consumes the same
    ``RequirementsRecord`` — this layer makes the interpretation inspectable and honest.
    """
    normalized = (prompt or "").strip()
    spec = parse_prompt_spec(normalized)
    req = build_requirements_from_prompt(
        normalized,
        requirement_id=requirement_id,
        payload_kg=payload_kg,
        reach_m=reach_m,
        sensor=sensor,
    )

    evidence: list[InputEvidence] = []
    questions: list[IntakeQuestion] = []

    # task summary is always a derived restatement of the prompt.
    evidence.append(_ev("task_summary", req.task_summary, InputSourceType.PARSED, confidence=0.7))

    # ---- payload ---------------------------------------------------------------
    if payload_kg is not None:
        evidence.append(_ev("payload_kg", req.payload_kg, InputSourceType.EXPLICIT, "kg"))
    elif spec.payload_kg is not None:
        evidence.append(_ev("payload_kg", spec.payload_kg, InputSourceType.PARSED, "kg", confidence=0.9))
    elif spec.weight_budget_kg is not None and _MASS_TOKEN.search(normalized):
        # requirements_builder took the *self-weight budget* as if it were a carry payload — surface it.
        evidence.append(_ev(
            "payload_kg", req.payload_kg, InputSourceType.INFERRED, "kg", confidence=0.3,
            conflicts=["the only mass in the prompt is a self-weight budget, not a carry/lift payload"],
            note="payload was not explicitly stated; the value shown may be the weight budget mis-read as payload",
        ))
        questions.append(IntakeQuestion(
            id="q_payload", field_path="payload_kg",
            question="You gave a weight budget but no carry payload. What should the robot actually carry?",
            reason="A weight-budget number was present; a payload was not.", blocks_training=True,
        ))
    else:
        evidence.append(_ev("payload_kg", req.payload_kg, InputSourceType.DEFAULTED, "kg", note="system default"))

    # ---- reach -----------------------------------------------------------------
    height = spec.target_height_m
    if reach_m is not None:
        evidence.append(_ev("reach_m", req.reach_m, InputSourceType.EXPLICIT, "m"))
    elif height is not None and req.reach_m is not None and abs(req.reach_m - height) <= 0.05:
        # the only length in the prompt described stature — reach was mis-read from the height token.
        evidence.append(_ev(
            "reach_m", req.reach_m, InputSourceType.INFERRED, "m", confidence=0.3,
            conflicts=["the length in the prompt describes stature/height, not an arm reach"],
            note="reach was not explicitly stated; the value shown was taken from the height token",
        ))
        questions.append(IntakeQuestion(
            id="q_reach", field_path="reach_m",
            question="You gave a height but no arm reach. What reach should the arm have?",
            reason="A length token matched the stated stature, so reach is ambiguous.",
        ))
    elif _LENGTH_TOKEN.search(normalized):
        evidence.append(_ev("reach_m", req.reach_m, InputSourceType.PARSED, "m", confidence=0.75))
    else:
        evidence.append(_ev("reach_m", req.reach_m, InputSourceType.DEFAULTED, "m", note="system default"))

    # ---- environment / sensor / precision --------------------------------------
    env_source = InputSourceType.PARSED if _ENV_TOKEN.search(normalized) else InputSourceType.DEFAULTED
    evidence.append(_ev("environment", req.environment, env_source))

    if sensor is not None:
        evidence.append(_ev("sensor", req.sensor_requirements[0] if req.sensor_requirements else sensor,
                            InputSourceType.EXPLICIT))
    else:
        sensor_val = req.sensor_requirements[0] if req.sensor_requirements else None
        sensor_source = InputSourceType.PARSED if _SENSOR_TOKEN.search(normalized) else InputSourceType.DEFAULTED
        evidence.append(_ev("sensor", sensor_val, sensor_source))

    evidence.append(_ev("precision_m", req.precision_m, InputSourceType.DEFAULTED, "m", note="system default"))

    # ---- quantitative constraints only present when the prompt stated them ------
    if height is not None:
        evidence.append(_ev("target_height_m", height, InputSourceType.PARSED, "m", confidence=0.9))
    if spec.weight_budget_kg is not None:
        evidence.append(_ev("weight_budget_kg", spec.weight_budget_kg, InputSourceType.PARSED, "kg", confidence=0.9))
    if spec.pinned_parts:
        evidence.append(_ev("pinned_parts", list(spec.pinned_parts), InputSourceType.PARSED, confidence=0.9))

    return InputInterpretation(
        id=f"interp_{req.id}",
        prompt=normalized,
        requirement_id=req.id,
        evidence=evidence,
        intake_questions=questions,
        notes=["Merged from requirements_builder + spec_parser (Input Compiler Phase 0)."],
    )


def write_interpretation(interpretation: InputInterpretation, package_dir: str) -> str:
    """Write ``<package_dir>/input/interpretation.json`` (creating ``input/``) and return its path.

    Called by the package writer so every prompt build carries an inspectable, provenance-rich interpretation.
    """
    input_dir = os.path.join(package_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    path = os.path.join(input_dir, "interpretation.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(interpretation.to_dict(), handle, indent=2, ensure_ascii=False)
    return path


# Maps a spec_compliance_report constraint name -> the interpretation field it was derived from.
_CONSTRAINT_TO_FIELD = {
    "height_m": "target_height_m",
    "self_weight_kg": "weight_budget_kg",
    "payload_kg": "payload_kg",
    "pinned_parts": "pinned_parts",
}


def link_compliance_to_evidence(interpretation: InputInterpretation, compliance_report: dict) -> dict:
    """Annotate each compliance constraint with the input evidence it traces back to.

    Acceptance criterion from the plan: "compliance report points back to input evidence." Returns a shallow
    copy of the report where every constraint gains an ``evidence`` block: ``{source_type, confidence,
    conflicts}`` (or ``{"source_type": "unknown"}`` when no field maps).
    """
    linked = dict(compliance_report)
    out_constraints: list[dict] = []
    for constraint in compliance_report.get("constraints", []):
        item = dict(constraint)
        field_path = _CONSTRAINT_TO_FIELD.get(constraint.get("constraint"))
        ev = interpretation.field(field_path) if field_path else None
        if ev is not None:
            item["evidence"] = {
                "field_path": ev.field_path,
                "source_type": ev.source_type.value,
                "confidence": ev.confidence,
                "conflicts": ev.conflicts,
            }
        else:
            item["evidence"] = {"source_type": "unknown"}
        out_constraints.append(item)
    linked["constraints"] = out_constraints
    return linked
