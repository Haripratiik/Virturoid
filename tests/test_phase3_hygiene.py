"""Phase 3 (2026-07-24 MVP-readiness audit) — correctness + honesty + security polish:
  * BOM total_mass_kg is the sum of the per-line mass column (no qty^2 double-count).
  * the spec-sheet one-liner never prints the self-contradiction "walks (0% task success)".
  * imported-URDF mesh resolution is CONFINED to the project dir (a hostile URDF can't read/convert host files).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def test_bom_total_mass_is_the_sum_of_the_mass_column():
    from virturoid.services.bom_builder import build_bom_from_genome
    from virturoid.services.gene_build import _gene_to_genome
    from virturoid.services.morphology_composer import compose_robot

    bom = build_bom_from_genome(_gene_to_genome(compose_robot("a quadruped robot dog", llm=None)))
    col_sum = round(sum(l["mass_kg"] for l in bom["lines"]), 3)
    assert abs(bom["totals"]["mass_kg"] - col_sum) < 0.01, (
        f"total {bom['totals']['mass_kg']} != mass-column sum {col_sum}")
    # every multi-unit line's mass column is qty * unit (not qty^2 * unit)
    for l in bom["lines"]:
        if l.get("qty", 1) >= 2 and "unit_mass_kg" in l:
            assert abs(l["mass_kg"] - l["qty"] * l["unit_mass_kg"]) < 1e-3, l


def test_spec_sheet_never_says_walks_with_zero_success():
    from virturoid.services import spec_sheet

    def spec(sr):
        return {"physical": {"total_mass_kg": 5}, "power_and_cost": {}, "actuation": {"dof": 8},
                "performance": {"task": "locomotion", "cadence_hz": 1.5, "success_rate": sr},
                "class": "quadruped", "dof": 8}

    assert "walks" not in spec_sheet._summary(spec(0.0)), spec_sheet._summary(spec(0.0))
    assert "does not reach a credible walk" in spec_sheet._summary(spec(0.0))
    assert "walks (60% of the distance target)" in spec_sheet._summary(spec(0.6))
    assert "0% task success" not in spec_sheet._summary(spec(0.0))


def test_imported_mesh_resolution_is_confined_to_the_project():
    """A hostile URDF must not resolve a 'mesh' to an absolute path or a ../ escape outside the imported dir."""
    from virturoid.services.model_import import _resolve_mesh_path

    root = Path(tempfile.mkdtemp())
    (root / "meshes").mkdir()
    (root / "meshes" / "arm.stl").write_text("solid", encoding="utf-8")
    assert _resolve_mesh_path("meshes/arm.stl", root) is not None            # legit relative mesh resolves

    secret = Path(tempfile.mkdtemp()) / "secret.stl"
    secret.write_text("x", encoding="utf-8")
    assert _resolve_mesh_path(str(secret), root) is None                     # absolute-outside rejected
    assert _resolve_mesh_path("file://" + str(secret), root) is None         # file:// escape rejected

    (root.parent / "evil.stl").write_text("x", encoding="utf-8")
    assert _resolve_mesh_path("../evil.stl", root) is None                   # ../ escape rejected


@pytest.mark.skipif(not _MUJOCO, reason="ingest compiles the URDF")
def test_ingest_folds_notes_md_into_the_nlp_description():
    """A customer who drops a folder with a notes.md ("aluminum body, 5 kg payload") shouldn't have to re-type
    it -- ingest reads the notes into the NLP payload and applies the parsed properties."""
    from virturoid.services import session_state as S  # noqa: F401 - ensures session dir env is honored
    from virturoid.services.input_training_tools import _ingest_project

    os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="notes_")
    d = tempfile.mkdtemp(prefix="proj_")
    (Path(d) / "robot").mkdir()
    (Path(d) / "robot" / "q.urdf").write_text(
        '<?xml version="1.0"?>\n<robot name="q"><link name="b"><inertial><mass value="2"/>'
        '<inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>',
        encoding="utf-8")
    (Path(d) / "notes.md").write_text("Aluminum body, carbon-fiber legs.", encoding="utf-8")
    r = _ingest_project({"project_path": d})                 # NO description arg -> must come from notes.md
    assert any("notes.md" in n for n in r.get("notes", [])), r.get("notes")
    assert any(m.get("material") == "aluminum" for m in r.get("materials_applied", [])), r.get("materials_applied")


@pytest.mark.skipif(not _MUJOCO, reason="URDF import needs MuJoCo")
def test_confinement_does_not_break_a_legit_relative_mesh_import():
    """A normal URDF referencing an in-tree mesh still imports (confinement only rejects escapes)."""
    from virturoid.services.robot_import import import_robot

    d = Path(tempfile.mkdtemp())
    (d / "meshes").mkdir()
    # a trivial but valid STL (ascii) so the mesh actually loads
    (d / "meshes" / "link.stl").write_text(
        "solid s\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n"
        "  endloop\n endfacet\nendsolid s\n", encoding="utf-8")
    urdf = ('<?xml version="1.0"?>\n<robot name="r">\n<link name="base"><inertial><mass value="1"/>'
            '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>'
            '<visual><geometry><mesh filename="meshes/link.stl"/></geometry></visual></link>\n</robot>')
    p = d / "r.urdf"
    p.write_text(urdf, encoding="utf-8")
    assert import_robot(str(p))["gene"] is not None
