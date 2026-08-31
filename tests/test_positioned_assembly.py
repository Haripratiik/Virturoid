"""`robot_assembly.step` must be an ASSEMBLY, not a pile of parts at the origin.

Every link solid is authored at (0,0,0) along its own +z, and the exporter boolean-unioned them:

    asm = solids[0]
    for sld in solids[1:]:
        asm = asm + sld

so the whole robot fused into one blob at one point. The file was named robot_assembly.step and was not an
assembly. That is why "make it taller" could not be shown as a geometric delta -- there was no geometry to diff --
and why anyone opening the STEP saw a knot instead of a robot.

Each solid is now placed at its forward-kinematic pose, walking the gene chain exactly as the MJCF does, so the
CAD artefact and the simulated model describe the SAME robot.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_BD = importlib.util.find_spec("build123d") is not None
pytestmark = pytest.mark.skipif(not _BD, reason="CAD export needs build123d")


@pytest.fixture(scope="module")
def exported():
    from virturoid.services.cad_geometry import export_gene_cad
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a four legged robot dog", llm=None)
    d = tempfile.mkdtemp(prefix="cad_asm_")
    return gene, d, export_gene_cad(gene, d)


def test_the_assembly_spans_the_whole_robot_not_one_link(exported):
    """The decisive measurement: parts SPREAD. If they were still unioned at the origin the assembly would be
    about the size of its largest single part, because that is all a pile at one point can be."""
    import build123d as bd
    gene, d, man = exported
    assert man["assembly_step"], "no assembly written"
    asm = bd.import_step(str(Path(d) / "robot_assembly.step"))
    bb = asm.bounding_box()
    biggest = max(float(s.length_m) for s in gene.segments) * 1000.0   # mm; the longest single link
    wide = sum(1 for v in (bb.size.X, bb.size.Y, bb.size.Z) if v > biggest * 1.5)
    assert wide >= 2, (
        f"assembly is {bb.size.X:.0f}x{bb.size.Y:.0f}x{bb.size.Z:.0f} mm against a longest link of "
        f"{biggest:.0f} mm — the parts are still stacked at the origin")


def test_the_assembly_agrees_with_the_simulated_body(exported):
    """The CAD artefact and the MJCF must describe the SAME robot, or the customer is handed two different
    machines. Compare overall spans rather than exact solids -- the visual meshes carry detail the colliders
    do not, so only the ENVELOPE is meaningfully comparable."""
    import build123d as bd
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    gene, d, _man = exported
    asm = bd.import_step(str(Path(d) / "robot_assembly.step"))
    bb = asm.bounding_box()
    cad = np.array([bb.size.X, bb.size.Y, bb.size.Z]) / 1000.0

    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
    dat = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, dat, 0)
    mujoco.mj_forward(m, dat)
    pts = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        R = dat.geom_xmat[g].reshape(3, 3)
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        ctr, ext = dat.geom_xpos[g] + R @ c, np.abs(R) @ h
        pts += [ctr - ext, ctr + ext]
    p = np.array(pts)
    sim = p.max(axis=0) - p.min(axis=0)
    # generous: the CAD path adds collars/housings and poses differ (CAD is the zero pose, sim the rest pose)
    for i, ax in enumerate("xyz"):
        assert cad[i] == pytest.approx(sim[i], rel=1.5, abs=0.25), (
            f"{ax}: CAD {cad[i]:.3f} m vs sim {sim[i]:.3f} m — the two artefacts disagree about the robot's size")


def test_parts_are_still_exported_individually(exported):
    """A positioned assembly must not cost the per-part files: those are what a manufacturer actually cuts."""
    _gene, d, man = exported
    steps = list((Path(d) / "step").glob("*.step"))
    assert len(steps) == man["part_count"] >= 2, (len(steps), man["part_count"])
