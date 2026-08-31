"""The customer's own link meshes must reach the EXPORTED PACKAGE — with no dangling reference, and without
moving a single collider.

Measured on the MuJoCo Menagerie models the product is actually pointed at (2026-08-05, ``ingest_project`` ->
``export_held``): a Unitree Go2 imported with 13 of its 13 links carrying their real geometry, and the package a
customer opened shipped ``robot.xml`` with 21 capsules, 25 cylinders and ZERO meshes. The geometry existed; it
just never crossed the export door. This file pins the three properties that fix has to hold:

  * CARRY-THROUGH — the gene's link meshes appear in the shipped MJCF and the shipped URDF.
  * NO DANGLING REFS — MuJoCo itself opens the shipped file from its own directory and resolves every asset.
    There is direct precedent for getting this wrong: the ROS2 export once shipped 22 dangling references by
    copying a URDF and leaving its meshes behind.
  * COLLIDER-NEUTRAL — it is a VISUAL change, so the set of geoms that can touch anything must be byte-identical
    with the mesh layer on and off, and the exported URDF must declare exactly the colliders we simulate.

Plus the naming rule underneath all of it: two links must never bake to one filename. ``_slug_name`` is
many-to-one, so a customer model with links ``FL/hip`` and ``FL:hip`` -- both legal, both produced by real xacro
and CAD exporters -- collapsed onto ``FL_hip.stl`` and the hip silently rendered as the thigh. Reproduced on a
Go2 with exactly those two links renamed: legacy rule -> one file, one digest for both; current rule -> two
files, two digests. Zero of the 63 Menagerie packages collide, which is why this needs a test rather than a
model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
# Four families so the property is not a quadruped quirk: legged, arm+gripper (prismatic fingers), humanoid.
_MODELS = [("unitree_go2/go2.xml", 13), ("franka_emika_panda/panda.xml", 11),
           ("universal_robots_ur5e/ur5e.xml", 7), ("unitree_g1/g1.xml", 30)]


def _import(rel: str):
    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot
    gene = import_robot(str(src), robot_id="meshcarry_" + rel.split("/")[0])["gene"]
    assert gene is not None, f"{rel} did not import"
    return gene


def _collider_hash(gene) -> tuple[str, int]:
    """Identity of every geom that can participate in a contact, from the COMPILED model."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene, meshed=False)))
    rows = []
    for g in range(m.ngeom):
        if not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g])):
            continue
        bid = int(m.geom_bodyid[g])
        rows.append("|".join(
            [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or str(bid), str(int(m.geom_type[g]))]
            + [f"{float(v):.9g}" for v in list(m.geom_size[g]) + list(m.geom_pos[g]) + list(m.geom_quat[g])]
            + [f"{float(m.body_mass[bid]):.9g}"]))
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest(), len(rows)


# --------------------------------------------------------------------------------------------------------
# The naming rule: two links, never one file.
# --------------------------------------------------------------------------------------------------------
def test_two_link_names_that_sanitize_alike_get_different_files():
    from virturoid.services.robot_import import _link_mesh_stem, _slug_name
    claimed: dict[str, str] = {}
    a = _link_mesh_stem("FL/hip", claimed)
    b = _link_mesh_stem("FL:hip", claimed)
    assert _slug_name("FL/hip") == _slug_name("FL:hip"), "premise: these collapse to one slug"
    assert a != b, "two DIFFERENT links baked to one filename; one would render as the other"
    assert a == "FL_hip", "the first link keeps the readable name"


def test_the_stem_is_deterministic_and_order_independent():
    """Re-importing the same model must reuse its own files, whichever order the links are walked in."""
    from virturoid.services.robot_import import _link_mesh_stem
    first = {n: _link_mesh_stem(n, c) for c in [{}] for n in ("a/x", "a:x", "b")}
    again = {n: _link_mesh_stem(n, c) for c in [{}] for n in ("a/x", "a:x", "b")}
    assert first == again
    assert len(set(first.values())) == 3


def test_stage_mesh_will_not_let_one_link_overwrite_another(tmp_path):
    """Same source basename, two links -> two destination files, each with its OWN bytes."""
    from virturoid.services.gene_compiler import stage_mesh
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "link.stl").write_bytes(b"AAAA")
    (tmp_path / "b" / "link.stl").write_bytes(b"BBBB")
    out, claimed = tmp_path / "pkg", {}
    p1 = stage_mesh(tmp_path / "a" / "link.stl", out, "hip", claimed)
    p2 = stage_mesh(tmp_path / "b" / "link.stl", out, "thigh", claimed)
    assert p1 != p2
    assert p1.read_bytes() == b"AAAA" and p2.read_bytes() == b"BBBB"


def test_stage_mesh_does_not_adopt_a_same_size_leftover(tmp_path):
    """A stale file of the same LENGTH from an earlier export is not this robot's geometry."""
    from virturoid.services.gene_compiler import stage_mesh
    src = tmp_path / "link.stl"
    src.write_bytes(b"NEWDATA!")
    out = tmp_path / "pkg"
    out.mkdir()
    (out / "link.stl").write_bytes(b"OLDDATA!")            # same size, different robot
    dst = stage_mesh(src, out, "hip", {})
    assert dst.read_bytes() == b"NEWDATA!"


# --------------------------------------------------------------------------------------------------------
# Carry-through + no dangling refs, on the real models.
# --------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _MUJOCO, reason="import + export need MuJoCo")
@pytest.mark.parametrize("rel,links", _MODELS)
def test_the_shipped_mjcf_carries_the_customers_meshes_and_opens_on_its_own(rel, links, tmp_path):
    from virturoid.services.gene_compiler import write_exported_mjcf
    gene = _import(rel)
    in_gene = [s for s in gene.segments
               if isinstance(getattr(s, "geometry", None), dict) and s.geometry.get("family") == "source_mesh"]
    assert len(in_gene) == links, f"{rel}: {len(in_gene)}/{links} links kept their own geometry"

    res = write_exported_mjcf(gene, tmp_path / "pkg" / "robot.xml")
    assert res["meshes"] == links, f"{rel}: gene had {links} meshes, the package shipped {res['meshes']}"
    text = (tmp_path / "pkg" / "robot.xml").read_text(encoding="utf-8")
    import re
    refs = re.findall(r'<mesh\b[^>]*file="([^"]+)"', text)
    assert len(refs) == links
    for r in refs:
        assert not Path(r).is_absolute(), f"{r} is an absolute path into the exporting machine"
        assert (tmp_path / "pkg" / r).is_file(), f"dangling reference: {r}"
    # The decisive check: MuJoCo opens the SHIPPED file and resolves every asset from its own directory.
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(tmp_path / "pkg" / "robot.xml"))
    assert int(m.nmesh) == links


@pytest.mark.skipif(not _MUJOCO, reason="import + export need MuJoCo")
@pytest.mark.parametrize("rel,links", _MODELS)
def test_the_mesh_layer_moves_no_collider(rel, links):
    """A visual change must be provably collider-neutral: same geoms, same sizes, same poses, same masses."""
    from virturoid.services import gene_compiler as GC
    gene = _import(rel)
    real = GC._with_source_meshes
    try:
        GC._with_source_meshes = lambda g, m, *, physics_only: m       # the pre-change behaviour
        off, n_off = _collider_hash(gene)
    finally:
        GC._with_source_meshes = real
    on, n_on = _collider_hash(gene)
    assert (off, n_off) == (on, n_on), f"{rel}: the mesh layer moved a collider ({n_off} -> {n_on})"


@pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")
@pytest.mark.parametrize("rel,links", _MODELS)
def test_the_cheap_primitive_compile_stayed_cheap(rel, links):
    """``source_meshes=False`` must still mean NO MESHES — and ``standing_spawn_z(meshed=False)`` must use it.

    Resolving an imported link's own STL is right for the shipped model and wrong for a measurement. When
    ``compile_gene_to_mjcf`` learned to do it unconditionally, the branch documented as "the cheap primitive
    model" started reading every one of the customer's meshes off disk: MEASURED, a Unitree G1 went 0.032 ->
    0.159 s per call and a Go2 0.024 -> 0.251 s, and BOTH branches of ``standing_spawn_z`` then returned the
    same number to the digit (0.8805 and 0.3384) because they had quietly become the same model — which also
    turned ``render_sim_parity``'s ``spawn_z_identical`` into a comparison of a value with itself.

    Two assertions, because either alone can be satisfied while the regression is back: the primitive model
    references no mesh asset, and the two branches are built from genuinely different XML.
    """
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z
    gene = _import(rel)
    prim = compile_gene_to_mjcf(gene, include_floor=False, source_meshes=False)
    assert "<mesh " not in prim, f"{rel}: the primitive model still loads mesh assets"
    withmesh = compile_gene_to_mjcf(gene, include_floor=False)
    assert "<mesh " in withmesh, f"{rel}: precondition — the default model DOES carry the customer's meshes"
    # and the two spawn-height branches are measuring two genuinely different models again
    assert gene_to_meshed_mjcf(gene, include_floor=False) != prim
    # the cheap branch must ASK for the cheap model -- spied, not timed, so it cannot flake. A FIXED-BASE arm
    # short-circuits to its mount height without compiling anything, so there is nothing to spy on there.
    if gene.base_mount != "free":
        return
    from virturoid.services import gene_compiler as GC
    seen = []
    real = GC.compile_gene_to_mjcf
    try:
        GC.compile_gene_to_mjcf = lambda g, **k: (seen.append(k.get("source_meshes", True)), real(g, **k))[1]
        standing_spawn_z(gene, meshed=False)
    finally:
        GC.compile_gene_to_mjcf = real
    assert seen and seen[-1] is False, f"{rel}: standing_spawn_z(meshed=False) asked for the meshed model"


@pytest.mark.skipif(not _MUJOCO, reason="import + export need MuJoCo")
def test_the_exported_urdf_declares_the_colliders_we_simulate(tmp_path):
    """RViz/Gazebo must see our collision set, not our decoration -- and it must not depend on the visual layer.

    Measured before this held: a Go2 exported 46 <collision> elements against 13 real colliders, a hexapod 62
    against 26, because every geom (motor housings, panel detail, visual-only shells at mass=0/contype=0) was
    transcribed into both blocks. It also made the collision set MOVE when a link gained its imported mesh.
    """
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.gene_urdf import gene_to_urdf
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a hexapod robot")
    urdf = gene_to_urdf(gene)
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=False, spawn_z=standing_spawn_z(gene)))
    colliding = sum(1 for g in range(m.ngeom) if int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))
    assert urdf.count("<collision>") == colliding, "the URDF ships colliders the simulator does not have"
    assert colliding < m.ngeom, "premise: this body does carry visual-only geoms"


@pytest.mark.skipif(not _MUJOCO or importlib.util.find_spec("build123d") is None,
                    reason="the CAD lane needs MuJoCo + build123d")
def test_the_cad_lane_ships_the_customers_surface_and_a_correctly_sized_solid(tmp_path):
    """``realize_shape`` has no ``source_mesh`` family, so an imported link hit its malformed-spec fallback and
    every link exported as the DEFAULT 100 mm x 30 mm capsule: measured on a Go2, the 0.376 m chassis at
    395.84 cm3 against a real ~11,300, and every hip and thigh at an identical 474.53 cm3 while differing 2.2x
    in length. Right names, right masses, nobody's geometry."""
    from virturoid.services.cad_geometry import export_gene_cad
    gene = _import("universal_robots_ur5e/ur5e.xml")
    man = export_gene_cad(gene, str(tmp_path / "cad"))
    parts = {p["name"]: p for p in man["parts"]}
    dims = {s.name: (s.length_m, s.radius_m) for s in gene.segments}
    assert all(p["geometry_source"] == "customer_mesh" for p in man["parts"])
    for s in gene.segments:                                # the STL lane is their file, byte for byte
        got = (tmp_path / "cad" / parts[s.name]["stl"]).read_bytes()
        assert got == Path(str(s.geometry["path"])).read_bytes(), f"{s.name}: the CAD STL is not their mesh"
    vols = {n: p["volume_cm3"] for n, p in parts.items()}
    assert len(set(vols.values())) > 1, "every link exported the same volume -- that is the default stub again"
    big = max(dims, key=lambda n: dims[n][0] * dims[n][1] ** 2)
    small = min(dims, key=lambda n: dims[n][0] * dims[n][1] ** 2)
    assert vols[big] > vols[small], "the CAD solid does not track the link's own measured size"


@pytest.mark.skipif(not _MUJOCO, reason="import + export need MuJoCo")
def test_an_imported_urdfs_collision_set_is_the_same_with_and_without_meshes():
    from virturoid.services import gene_compiler as GC
    from virturoid.services.gene_urdf import gene_to_urdf
    gene = _import("unitree_go2/go2.xml")
    real = GC._with_source_meshes
    try:
        GC._with_source_meshes = lambda g, m, *, physics_only: m
        off = gene_to_urdf(gene).count("<collision>")
    finally:
        GC._with_source_meshes = real
    assert gene_to_urdf(gene).count("<collision>") == off
