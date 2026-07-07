"""Enterprise robot import report — Input Ingestion plan, Phase 1.

Composes the two existing import lanes into one durable :class:`RobotModelImportReport`:
  * FAITHFUL native lane (``model_import.import_model``): normalized MJCF that runs as-is (accuracy);
  * inferred RobotGene lane (``robot_import.import_robot``): editable, lossy (Virturoid iteration).
It classifies every gene-lane warning by severity + a concrete fix action, scores import readiness on
independent axes, and writes ``input/import_report.json``. Deterministic and offline (AGENTS.md).
"""

from __future__ import annotations

import json
import os

from virturoid.schemas.import_report import ImportLane, ImportWarning, RobotModelImportReport

# substring (lowercased) -> (code, severity, fix_action). Ordered: first match wins.
_WARNING_RULES: tuple[tuple[str, str, str, str], ...] = (
    ("root bodies attach to the world", "multiple_roots", "error",
     "Select a single world-rooted base frame for simulation."),
    ("zero/negative mass", "missing_mass", "warning",
     "Provide link inertial/mass (import a BOM or a CAD density estimate)."),
    ("synthesized a welded base", "synthesized_base", "info",
     "Confirm the synthesized world-rooted base frame is correct."),
    ("no body found to mark as the end-effector", "no_end_effector", "warning",
     "Label the end-effector body so manipulation tasks have a tool frame."),
    ("multiple end-effector candidates", "ambiguous_end_effector", "warning",
     "Choose which body is the end-effector."),
    ("a gene segment models one", "multi_joint_body_lossy", "warning",
     "The RobotGene lane is lossy here (a multi-joint body); use the faithful native lane for accuracy."),
    ("unsupported joint type", "unsupported_joint", "warning",
     "The faithful lane keeps it; the RobotGene lane approximates it as a hinge."),
    ("no limits (continuous)", "missing_joint_limits", "warning",
     "Set joint_lower/joint_upper, import ros2_control limits, or mark the joint continuous."),
)

_EXT_FORMAT = {".urdf": "urdf", ".xacro": "urdf", ".mjcf": "mjcf", ".xml": "mjcf", ".sdf": "sdf"}


def classify_warning(message: str) -> ImportWarning:
    """Map a raw importer warning string to a typed, fixable :class:`ImportWarning`."""
    lowered = message.lower()
    for needle, code, severity, fix in _WARNING_RULES:
        if needle in lowered:
            return ImportWarning(code=code, message=message, severity=severity, fix_action=fix)
    return ImportWarning(code="import_warning", message=message, severity="warning",
                         fix_action="Review this warning before training.")


def _confidence(faithful: dict, gene: dict, warnings: list[ImportWarning], actuated: int) -> dict:
    codes = {w.code for w in warnings}
    parse_ok = 1.0 if faithful.get("ok") else 0.0
    kinematic = 1.0
    if "multiple_roots" in codes:
        kinematic -= 0.5
    if codes & {"no_end_effector", "ambiguous_end_effector"}:
        kinematic -= 0.2
    if "multi_joint_body_lossy" in codes:
        kinematic -= 0.15
    kinematic = round(max(0.0, kinematic), 3)
    dynamics = round(1.0 - (0.4 if "missing_mass" in codes else 0.0), 3)
    control = 1.0 if actuated > 0 else 0.3
    has_error = any(w.severity == "error" for w in warnings)
    training_readiness = round(min(parse_ok, kinematic, dynamics, control) - (0.3 if has_error else 0.0), 3)
    return {
        "file_integrity_score": parse_ok,
        "model_parse_score": parse_ok,
        "kinematic_confidence": kinematic,
        "dynamics_confidence": dynamics,
        "control_interface_confidence": round(control, 3),
        "training_readiness_score": round(max(0.0, training_readiness), 3),
    }


def build_import_report(source: str, *, robot_id: str | None = None,
                        species: str | None = None) -> RobotModelImportReport:
    """Run both import lanes on ``source`` and return a durable :class:`RobotModelImportReport`."""
    from virturoid.services.model_import import import_model
    from virturoid.services.robot_import import import_robot

    ext = os.path.splitext(source.lower())[1]
    source_format = _EXT_FORMAT.get(ext, ext.lstrip(".") or "unknown")

    faithful = import_model(source)
    faithful_lane = ImportLane(
        name="faithful_native",
        ok=bool(faithful.get("ok")),
        summary=(f"Compiles in MuJoCo as-is: {faithful.get('parts', 0)} bodies, "
                 f"{faithful.get('actuated', 0)} actuators, free_base={faithful.get('free_base')}."
                 if faithful.get("ok") else f"Native compile failed: {faithful.get('note')}"),
        detail={k: faithful.get(k) for k in ("name", "parts", "actuated", "free_base", "note")},
    )

    warnings: list[ImportWarning] = []
    gene_lane: ImportLane | None = None
    backend_support: dict = {}
    try:
        gene = import_robot(source, robot_id=robot_id, species=species)
        warnings = [classify_warning(w) for w in gene.get("warnings", [])]
        backend_support = gene.get("backend_support", {}) or {}
        g = gene.get("gene")
        gene_lane = ImportLane(
            name="inferred_robot_gene",
            ok=bool(gene.get("valid")),
            summary=(f"Inferred an editable RobotGene (class={gene.get('robot_class')}, "
                     f"species={gene.get('species')}); {len(warnings)} warning(s), lossy where flagged."),
            detail={"robot_class": gene.get("robot_class"), "species": gene.get("species"),
                    "valid": gene.get("valid"), "gene_id": getattr(g, "id", None)},
        )
    except Exception as exc:  # noqa: BLE001 - the gene lane is best-effort; the faithful lane still stands
        gene_lane = ImportLane(name="inferred_robot_gene", ok=False,
                               summary=f"RobotGene inference failed: {exc}", detail={})

    actuated = int(faithful.get("actuated", 0) or 0)
    confidence = _confidence(faithful, {}, warnings, actuated)
    return RobotModelImportReport(
        id=f"import_{(robot_id or faithful.get('name') or 'model')}",
        source=source,
        source_format=source_format,
        faithful_lane=faithful_lane,
        gene_lane=gene_lane,
        warnings=warnings,
        body_count=int(faithful.get("parts", 0) or 0),
        actuated_joint_count=actuated,
        free_base=bool(faithful.get("free_base")),
        backend_support=backend_support,
        confidence=confidence,
        first_runnable_sim=("faithful_native_mjcf" if faithful.get("ok") else None),
    )


def write_import_report(report: RobotModelImportReport, package_dir: str) -> str:
    """Write ``<package_dir>/input/import_report.json`` (creating ``input/``). Returns the path."""
    input_dir = os.path.join(package_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    path = os.path.join(input_dir, "import_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, ensure_ascii=False)
    return path


def format_import_report(report: RobotModelImportReport) -> str:
    """Human-readable one-screen rendering (the migration-dashboard summary for a dropped model)."""
    lines = [
        f"Import: {report.source}  (format={report.source_format})",
        f"  faithful native lane: {'OK' if report.faithful_lane and report.faithful_lane.ok else 'FAIL'}"
        f"  — {report.faithful_lane.summary if report.faithful_lane else '(none)'}",
        f"  inferred RobotGene:   {'OK' if report.gene_lane and report.gene_lane.ok else 'FAIL'}"
        f"  — {report.gene_lane.summary if report.gene_lane else '(none)'}",
        f"  bodies={report.body_count}  actuators={report.actuated_joint_count}  free_base={report.free_base}",
        f"  readiness: sim_parse={report.confidence.get('model_parse_score')}  "
        f"kinematic={report.confidence.get('kinematic_confidence')}  "
        f"training={report.confidence.get('training_readiness_score')}",
    ]
    if report.warnings:
        lines.append("  warnings (each with a fix):")
        for w in report.warnings:
            lines.append(f"    [{w.severity}] {w.code}: {w.message}")
            lines.append(f"        fix: {w.fix_action}")
    return "\n".join(lines)
