"""ONE derivation of "what may we honestly claim about this robot package?".

Before this module the answer was computed independently in four places, and they contradicted each
other on screen:

===========================  ==========================================  ==========================
surface                      derived from                                said
===========================  ==========================================  ==========================
header chip / library card   ``reports/robot_package_contract.json`` ok   ``VALID`` (green)
status bar (footer)          readiness ledger ``safe_to_export``          ``unverified``
Verify tab                   readiness ledger ``safe_to_export``          ``EXPORT BLOCKED``
build console                ``reports/buildability_report.json``         ``Buildable=False``
===========================  ==========================================  ==========================

All four are real facts, but three of them were rendered as a *generic* verdict, so a customer could
read a green "VALID" chip sitting directly above "EXPORT BLOCKED". :func:`package_status` computes
the headline verdict ONCE (served on ``/api/packages``; every surface renders it), and reports the
other facts under their OWN names -- ``contract_ok``, ``buildable`` -- because they genuinely mean
different things:

* ``contract_ok``      -- every artifact the package declares exists and parses (file integrity).
* ``safe_to_export``   -- every REQUIRED readiness stage attained a provably-real status. THE gate.
* ``buildable``        -- a real actuator in the parts library covers every joint, and the structure
                          survives its loads. Says nothing about whether the package is complete.

The headline is export-readiness, worded exactly as the Verify tab words it, because that is the
strongest claim the product makes to a customer.
"""

from __future__ import annotations

import json
from pathlib import Path

CONTRACT_URI = "reports/robot_package_contract.json"
LEDGER_URI = "reports/product_readiness_ledger.json"
BUILDABILITY_URI = "reports/buildability_report.json"

# The headline vocabulary. Every word is scoped -- none of them is a bare "valid".
LABEL_EXPORT_READY = "EXPORT-READY"
LABEL_EXPORT_BLOCKED = "EXPORT BLOCKED"
LABEL_PACKAGE_INCOMPLETE = "PACKAGE INCOMPLETE"
LABEL_UNVERIFIED = "UNVERIFIED"
LABEL_NO_ROBOT = "NO ROBOT"

# Pill kinds shared with the frontend (components/ui.tsx VerdictKind).
KIND_OK = "ok"
KIND_BAD = "bad"
KIND_WARN = "warn"
KIND_MUTED = "muted"


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def has_robot_package(package_dir) -> bool:
    """True when the directory holds something the studio can actually show: a URDF to render, or
    compiled scenes to replay. THE predicate for "a robot was produced" -- used by the package list
    and by the job registry, so 'listed in the library' and 'the build succeeded' cannot disagree."""
    d = Path(package_dir)
    return (d / "robot" / "robot.urdf").exists() or (
        d / "simulation" / "mujoco" / "compiled_scene_index.json").exists()


def _blocking_stages(ledger: dict) -> list[str]:
    required = {str(s) for s in (ledger.get("required") or [])}
    blocking = []
    for stage in ledger.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        name = str(stage.get("stage") or "")
        status = str(stage.get("status") or "")
        if name in required and status not in ("attained", "not_required"):
            blocking.append(f"{name}={status}" if status else name)
    # A required stage that was never probed blocks too, and only the ledger's own issues see it.
    probed = {str(s.get("stage")) for s in (ledger.get("stages") or []) if isinstance(s, dict)}
    blocking.extend(f"{name}=never probed" for name in sorted(required - probed))
    return blocking


def package_status(package_dir) -> dict:
    """The one status object. Never raises: a missing/unreadable report is reported as unknown
    (``None``), never silently upgraded to a pass."""
    d = Path(package_dir)
    contract = _read_json(d / CONTRACT_URI)
    ledger = _read_json(d / LEDGER_URI)
    build = _read_json(d / BUILDABILITY_URI)

    contract_ok = None if contract is None else bool(contract.get("ok"))
    safe_to_export = None if ledger is None else bool(ledger.get("safe_to_export"))
    buildable = None if build is None else bool(build.get("buildable"))
    highest = None if ledger is None else (ledger.get("highest_attained") or None)

    notes: list[str] = []
    if buildable is False:
        issues = [str(i) for i in (build.get("issues") or [])]
        notes.append("Not buildable from real parts: " + (issues[0] if issues else
                     "an actuator or structural gate failed"))
    elif buildable is True:
        notes.append("Every actuated joint is covered by a real actuator in the parts library.")

    # Ladder, most severe first. A package can only be called export-ready when nothing below it
    # is broken -- and a broken contract outranks the ledger (a ledger over missing files is moot).
    if not has_robot_package(d):
        label, kind = LABEL_NO_ROBOT, KIND_BAD
        detail = "No robot.urdf and no compiled scenes — this directory holds no robot to show."
    elif contract_ok is False:
        label, kind = LABEL_PACKAGE_INCOMPLETE, KIND_BAD
        missing = [str(c.get("uri")) for c in (contract.get("artifact_checks") or [])
                   if isinstance(c, dict) and (not c.get("exists") or c.get("parse_status") == "fail")]
        detail = ("Declared artifacts are missing or unparseable: " + ", ".join(missing[:3])
                  if missing else "The package contract failed its artifact checks.")
    elif ledger is None:
        label, kind = LABEL_UNVERIFIED, KIND_MUTED
        detail = "No readiness ledger — nothing about this robot has been verified yet."
    elif safe_to_export:
        label, kind = LABEL_EXPORT_READY, KIND_OK
        detail = f"Every required readiness gate attained a real result (highest: {highest or 'none'})."
    else:
        label, kind = LABEL_EXPORT_BLOCKED, KIND_BAD
        blocking = _blocking_stages(ledger) or [str(i) for i in (ledger.get("issues") or [])]
        detail = ("Export blocked by required gate(s): " + ", ".join(blocking[:3])
                  if blocking else "Export blocked by an unmet required readiness gate.")

    return {
        "label": label,
        "kind": kind,
        "detail": detail,
        "notes": notes,
        # The individual facts, each under its own name -- so a surface with room can show them
        # without any of them masquerading as the headline verdict.
        "contract_ok": contract_ok,
        "safe_to_export": safe_to_export,
        "buildable": buildable,
        "highest_attained": highest,
    }
