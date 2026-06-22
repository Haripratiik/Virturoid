from __future__ import annotations

from pathlib import Path

from virturoid.schemas.cad import CadAssembly, CadModel


def write_placeholder_cad_files(cad_models: list[CadModel], assembly: CadAssembly, output_dir: Path) -> None:
    """Materialize minimal CAD placeholders for the MVP export.

    These are not manufacturable CAD yet. They are intentionally explicit text
    placeholders at the exact artifact paths so downstream code has real files
    to consume while the CAD kernel integration is built.
    """
    for model in cad_models:
        if model.artifact is None:
            continue
        path = output_dir / model.artifact.uri
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_step_placeholder_for_model(model), encoding="utf-8")
        _write_stl_preview(model, output_dir)

    assembly_path = output_dir / "cad" / "exact" / "robot_assembly.step"
    assembly_path.parent.mkdir(parents=True, exist_ok=True)
    assembly_path.write_text(_step_placeholder_for_assembly(assembly, cad_models), encoding="utf-8")
    parametric_path = output_dir / "cad" / "parametric" / "robot_arm.py"
    parametric_path.parent.mkdir(parents=True, exist_ok=True)
    parametric_path.write_text(_parametric_source_for_assembly(assembly, cad_models), encoding="utf-8")


def _step_placeholder_for_model(model: CadModel) -> str:
    bbox = model.bounding_box_mm or (0.0, 0.0, 0.0)
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        f"/* Virturoid MVP placeholder STEP for {model.name} */\n"
        "ENDSEC;\n"
        "DATA;\n"
        f"/* model_id={model.id} */\n"
        f"/* bbox_mm={bbox[0]},{bbox[1]},{bbox[2]} */\n"
        f"/* mass_kg={model.mass_kg} */\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )


def _step_placeholder_for_assembly(assembly: CadAssembly, models: list[CadModel]) -> str:
    model_ids = ", ".join(model.id for model in models)
    instance_ids = ", ".join(instance.instance_id for instance in assembly.instances)
    joint_names = ", ".join(joint.name for joint in assembly.joints)
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        f"/* Virturoid MVP placeholder STEP assembly for {assembly.name} */\n"
        "ENDSEC;\n"
        "DATA;\n"
        f"/* assembly_id={assembly.id} */\n"
        f"/* models={model_ids} */\n"
        f"/* instances={instance_ids} */\n"
        f"/* joints={joint_names} */\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )


def _parametric_source_for_assembly(assembly: CadAssembly, models: list[CadModel]) -> str:
    model_rows = []
    for model in models:
        width, depth, height = model.bounding_box_mm or (10.0, 10.0, 10.0)
        params = dict(model.editable_parameters)
        model_rows.append(
            f"    {model.id!r}: {{'name': {model.name!r}, 'bbox_mm': ({width}, {depth}, {height}), 'mass_kg': {model.mass_kg}, 'parameters': {params!r}}},"
        )
    instance_rows = [
        f"    {{'instance_id': {item.instance_id!r}, 'cad_model_id': {item.cad_model_id!r}, 'role': {item.role!r}, 'parent': {item.parent_instance_id!r}}},"
        for item in assembly.instances
    ]
    joint_rows = [
        f"    {{'name': {joint.name!r}, 'type': {joint.joint_type!r}, 'parent': {joint.parent_instance_id!r}, 'child': {joint.child_instance_id!r}, 'axis': {joint.axis_xyz!r}, 'limit': {joint.limit!r}}},"
        for joint in assembly.joints
    ]
    return (
        '"""Parametric source for the Virturoid MVP robot arm.\n\n'
        "This file is intentionally dependency-free. A future CAD backend can\n"
        "import MODEL_PARAMETERS, ASSEMBLY_INSTANCES, and ASSEMBLY_JOINTS, then\n"
        "replace the placeholder STEP/STL emitters with real CadQuery/OpenCascade\n"
        "geometry generation.\n"
        '"""\n\n'
        f"ASSEMBLY_ID = {assembly.id!r}\n"
        f"ASSEMBLY_NAME = {assembly.name!r}\n\n"
        "MODEL_PARAMETERS = {\n"
        + "\n".join(model_rows)
        + "\n}\n\n"
        "ASSEMBLY_INSTANCES = [\n"
        + "\n".join(instance_rows)
        + "\n]\n\n"
        "ASSEMBLY_JOINTS = [\n"
        + "\n".join(joint_rows)
        + "\n]\n\n"
        "def describe_build():\n"
        "    return {\n"
        "        'assembly_id': ASSEMBLY_ID,\n"
        "        'model_count': len(MODEL_PARAMETERS),\n"
        "        'instance_count': len(ASSEMBLY_INSTANCES),\n"
        "        'joint_count': len(ASSEMBLY_JOINTS),\n"
        "    }\n"
    )


def _write_stl_preview(model: CadModel, output_dir: Path) -> None:
    path = output_dir / "cad" / "mesh" / "visual" / f"{model.id}.stl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_ascii_box_stl(model), encoding="utf-8")


def _ascii_box_stl(model: CadModel) -> str:
    width, depth, height = model.bounding_box_mm or (10.0, 10.0, 10.0)
    x = width / 2000.0
    y = depth / 2000.0
    z = height / 2000.0
    vertices = {
        "000": (-x, -y, -z),
        "100": (x, -y, -z),
        "110": (x, y, -z),
        "010": (-x, y, -z),
        "001": (-x, -y, z),
        "101": (x, -y, z),
        "111": (x, y, z),
        "011": (-x, y, z),
    }
    triangles = [
        ("000", "100", "110"),
        ("000", "110", "010"),
        ("001", "011", "111"),
        ("001", "111", "101"),
        ("000", "001", "101"),
        ("000", "101", "100"),
        ("100", "101", "111"),
        ("100", "111", "110"),
        ("110", "111", "011"),
        ("110", "011", "010"),
        ("010", "011", "001"),
        ("010", "001", "000"),
    ]
    lines = [f"solid {model.id}"]
    for a, b, c in triangles:
        lines.extend(
            [
                "  facet normal 0 0 0",
                "    outer loop",
                f"      vertex {_fmt_vertex(vertices[a])}",
                f"      vertex {_fmt_vertex(vertices[b])}",
                f"      vertex {_fmt_vertex(vertices[c])}",
                "    endloop",
                "  endfacet",
            ]
        )
    lines.append(f"endsolid {model.id}")
    return "\n".join(lines) + "\n"


def _fmt_vertex(vertex: tuple[float, float, float]) -> str:
    return f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}"
