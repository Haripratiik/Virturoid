"""A package rebuilt into a REUSED directory must describe ONE robot, and its documents must agree.

Three measured defects in what the export door hands a customer, all of the same family -- an artifact that
misleads whoever is holding it -- each pinned here against the production behaviour that fixes it:

1. STALE GEOMETRY SURVIVED A REBUILD. ``autonomous_build`` re-runs ``build_gene_package`` into the directory
   it already used once a redesign is accepted, and ``export_held`` reuses ``build/agent_exports/<robot_id>``
   on every re-export. The geometry writers only ever ADDED files. Measured before the fix: after a second
   build produced a 9-part arm, ``cad/step/`` held 27 STEP files, 18 of them a DISCARDED quadruped's
   (``leg0_0``, ``torso``, ``head_0``, ...), and the arm's ROS 2 package installed 25 meshes of which its own
   URDF referenced 7. Every index we write named only the current robot, so a manifest reader was safe and
   ``glob("cad/step/*.step")`` -- not an unreasonable way to open a folder of STEP files -- assembled a
   chimera out of two robots.

2. THE DEPLOYMENT GUIDE'S ASSEMBLY TABLE HEADED ITS COLUMN "Joint" AND PRINTED SEGMENT NAMES. It said
   ``shoulder`` where ``config/hardware_interface.yaml`` in the same package says ``shoulder_joint``:
   measured 0 of 8 arm rows and 0 of 12 quadruped rows were joint names of the shipped model.

3. ``export_held`` PICKED ITS ROS 2 PACKAGE BY UNORDERED GLOB (``pkgs[0]``), so a sibling written by an
   earlier export could be returned as this one. That substitution has already produced a false finding
   during a review of this very export.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

_B123D = importlib.util.find_spec("build123d") is not None
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_QUAD = "a quadruped walking robot"
_ARM = "a 6-dof robotic arm"

_YAML_ROW = re.compile(r"^  (\S+):\n    actuator: \"([^\"]+)\"\n    bom_key: \"([^\"]+)\"", re.M)


def _compose(prompt):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt, llm=None)


# --------------------------------------------------------------------------- 1. stale geometry

@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="real CAD needs build123d; composition needs MuJoCo")
def test_rebuilt_cad_directory_holds_only_the_robot_its_manifest_describes(tmp_path):
    """``glob("cad/step/*.step")`` and the manifest must return the SAME robot after a rebuild.

    Fails on the pre-fix code with 27 STEPs on disk against a 9-part manifest.
    """
    from virturoid.services.cad_geometry import export_gene_cad

    out = tmp_path / "cad"
    first = export_gene_cad(_compose(_QUAD), str(out))
    assert first["part_count"] > 0
    quad_stems = {Path(p["step"]).stem for p in first["parts"]}

    second = export_gene_cad(_compose(_ARM), str(out))          # SAME directory, DIFFERENT robot
    listed = {Path(p["step"]).name for p in second["parts"]}
    on_disk = {p.name for p in (out / "step").glob("*.step")}
    assert on_disk == listed, (
        f"a customer globbing cad/step/ gets {len(on_disk)} files for a {len(listed)}-part robot; "
        f"not in the manifest: {sorted(on_disk - listed)}")

    stl_listed = {Path(p["stl"]).name for p in second["parts"]}
    assert {p.name for p in (out / "stl").glob("*.stl")} == stl_listed
    assert not ({Path(n).stem for n in on_disk} & quad_stems), "a discarded robot's links are still shipped"
    assert second["removed_stale"], "the export must SAY what stale geometry it removed, not do it silently"


@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d; URDF needs MuJoCo")
def test_rebuilt_ros2_package_installs_only_the_meshes_its_own_urdf_references(tmp_path):
    """``setup.py`` installs ``meshes/*`` wholesale, so an unreferenced mesh is another robot's geometry
    shipped and installed as this one's. Fails on the pre-fix code with 25 meshes against 7 references."""
    from virturoid.services.gene_build import _write_genome_and_urdf

    out = tmp_path / "pkg"
    _write_genome_and_urdf(_compose(_QUAD), out, task=_QUAD)
    pkg = _write_genome_and_urdf(_compose(_ARM), out, task=_ARM)   # SAME directory, DIFFERENT robot
    assert pkg is not None and pkg.is_dir()

    urdf = (pkg / "urdf" / "robot.urdf").read_text(encoding="utf-8")
    referenced = set(re.findall(r'filename="package://[^/]+/meshes/([^"]+)"', urdf))
    from virturoid.services.gene_compiler import STAGE_LEDGER_NAME
    installed = {p.name for p in (pkg / "meshes").glob("*")}
    # the ledger is the ONLY thing in meshes/ that is not geometry; naming it here rather than filtering by
    # suffix means a future non-mesh file cannot slip past this assertion
    assert installed - referenced <= {STAGE_LEDGER_NAME}, f"unexpected non-mesh files: {installed - referenced}"
    shipped = installed - {STAGE_LEDGER_NAME}
    assert shipped == referenced, (
        f"the package installs {len(shipped)} meshes but its URDF references {len(referenced)}; "
        f"orphans: {sorted(shipped - referenced)[:6]}")

    # ...and the source directory the package copies FROM must be this robot's too.
    src_referenced = set(re.findall(r'filename="meshes/([^"]+)"',
                                    (out / "robot" / "robot.urdf").read_text(encoding="utf-8")))
    assert {p.name for p in (out / "robot" / "meshes").glob("*.stl")} == src_referenced


@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d and MuJoCo")
def test_rebuilt_package_viewer_assets_hold_only_what_its_mesh_index_names(tmp_path):
    """``simulation/viewer_assets/`` is the package's OWN asset directory, not the shared bake cache, and it
    accumulated the same way: a second build shipped the first robot's 18 link STLs beside this one's."""
    from virturoid.services.gene_compiler import write_packaged_visual_mjcf

    out = tmp_path / "pkg"
    assert write_packaged_visual_mjcf(_compose(_QUAD), out) is not None
    assert write_packaged_visual_mjcf(_compose(_ARM), out) is not None   # SAME package, DIFFERENT robot

    index = json.loads((out / "simulation" / "viewer_mesh_index.json").read_text(encoding="utf-8"))
    named = {Path(m["uri"]).name for m in index["meshes"].values()}
    on_disk = {p.name for p in (out / "simulation" / "viewer_assets").glob("*.stl")}
    assert on_disk == named, f"viewer_assets holds {len(on_disk)} STLs; the index names {len(named)}"


# --------------------------------------------------------------------------- 2. the guide agrees with the yaml

@pytest.mark.skipif(not _MUJOCO, reason="the package's genome/URDF need MuJoCo")
@pytest.mark.parametrize("prompt", [_ARM, _QUAD])
def test_deployment_guide_assembly_table_names_the_joints_the_hardware_interface_names(tmp_path, prompt):
    """Every row of the guide's assembly table must be findable in ``config/hardware_interface.yaml`` under
    the SAME joint name, with the SAME actuator and the SAME parts-list key.

    Fails on the pre-fix guide, whose "Joint" column carried segment names: 0 of 8 (arm) / 0 of 12 (quad)
    of its values were joint names of the shipped model.
    """
    from virturoid.services.deployment_guide import build_deployment_guide
    from virturoid.services.gene_build import _emit_bom, _write_genome_and_urdf

    out = tmp_path / "pkg"
    gene = _compose(prompt)
    pkg = _write_genome_and_urdf(gene, out, task=prompt)
    _emit_bom(gene, out, task=prompt)
    assert pkg is not None

    yaml = (pkg / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
    yrows = {j: (act, key) for j, act, key in _YAML_ROW.findall(yaml)}
    assert yrows, "the hardware interface must name joints for this body"

    section = build_deployment_guide(out).split("## 2. Assemble")[1].split("## 3.")[0]
    rows = [tuple(c.strip() for c in r)
            for r in re.findall(r"^\| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|$", section, re.M)]
    rows = [r for r in rows if r[0] not in ("Joint", "---")]
    assert rows, "the guide must still print an assembly table"

    joint_names = {str(j["name"]) for j in
                   json.loads((out / "robot" / "robot_genome.json").read_text(encoding="utf-8"))["joints"]}
    for joint, key, actuator in rows:
        assert joint in joint_names, f"guide row '{joint}' is not a joint of the shipped model"
        assert joint in yrows, f"the guide names joint '{joint}'; hardware_interface.yaml does not"
        assert yrows[joint] == (actuator, key), (
            f"guide says {joint}: {actuator!r} via key {key!r}; the yaml says {yrows[joint]!r}")
    assert len(rows) == len(yrows), "the two documents must describe the same set of joints"


def test_assembly_table_does_not_claim_a_joint_column_without_a_genome(tmp_path):
    """A package with no genome on disk has no joint names to print, so the guide must label the column for
    what it actually holds rather than heading a list of parts-list keys "Joint"."""
    from virturoid.services.deployment_guide import _assembly_rows, build_deployment_guide

    (tmp_path / "robot").mkdir(parents=True)
    (tmp_path / "robot" / "bill_of_materials.json").write_text(json.dumps(
        {"robot_class": "legacy", "totals": {}, "lines": [],
         "actuator_map": {"shoulder": "T-Motor AK10-9"}}), encoding="utf-8")

    asm = _assembly_rows(tmp_path, {"shoulder": "T-Motor AK10-9"})
    assert asm["namespace"] is None and asm["rows"] == [(None, "shoulder", "T-Motor AK10-9")]
    section = build_deployment_guide(tmp_path).split("## 2. Assemble")[1]
    assert "| Parts-list key | Actuator |" in section
    assert "| Joint | Actuator |" not in section


def test_a_foreign_parts_list_resolves_nothing_and_the_guide_says_so(tmp_path):
    """A parts list whose keys match neither this robot's joint names nor its segment names is UNDETERMINED --
    the same verdict ``hardware_interface.yaml`` prints. The guide must not resolve rows the yaml refuses to."""
    from virturoid.services.deployment_guide import _assembly_rows

    (tmp_path / "robot").mkdir(parents=True)
    (tmp_path / "robot" / "robot_genome.json").write_text(json.dumps(
        {"joints": [{"name": "shoulder_joint", "child_link": "shoulder"},
                    {"name": "j1_joint", "child_link": "j1"}],
         "links": [{"name": "shoulder"}, {"name": "j1"}]}), encoding="utf-8")

    asm = _assembly_rows(tmp_path, {"someone_elses_leg": "X", "someone_elses_torso": "Y"})
    assert asm["namespace"]["namespace"] == "undetermined"
    assert [a for _, _, a in asm["rows"]] == [None, None]
    assert asm["unused_keys"] == ["someone_elses_leg", "someone_elses_torso"]


# --------------------------------------------------------------------------- 3. which package was chosen

@pytest.mark.skipif(not _MUJOCO, reason="the export door needs MuJoCo")
def test_export_held_returns_the_package_it_wrote_not_an_unordered_sibling(tmp_path, monkeypatch):
    """``pkgs[0]`` over ``glob("*")`` returned ``aaa_stale`` when a sibling sorted ahead of the real package.
    The writer names its own package now, and the result says which one it chose."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool

    monkeypatch.chdir(tmp_path)                       # safe_build_path is CWD-relative, deliberately
    monkeypatch.setenv("VIRTUROID_SESSION_DIR", tempfile.mkdtemp(prefix="one_robot_sess_"))
    rid = S.put_robot(_compose(_ARM), robot_id="one_robot_arm")

    stale = tmp_path / "build" / "agent_exports" / rid / "export" / "ros2" / "aaa_stale"
    stale.mkdir(parents=True)                         # a sibling from an earlier export, sorting FIRST
    (stale / "package.xml").write_text("<package/>", encoding="utf-8")

    res = call_tool("export_held", {"robot_id": rid, "formats": ["ros2"]}).get("result", {})
    arts = res.get("artifacts", {})
    assert "ros2" in arts, f"no ROS 2 package returned: {arts}"
    chosen = Path(arts["ros2"])
    assert chosen.name != "aaa_stale", "export_held handed back a sibling package it did not write"
    assert arts.get("ros2_package") == chosen.name, "the export must SAY which package it chose"
    assert (chosen / "config" / "hardware_interface.yaml").exists(), "the chosen package is this export's"


# ============================================================================================================
# The prune that fixed (1) above was itself wrong in four ways. Each is pinned here against the behaviour that
# fixes it, and each assertion below FAILS on the first version of the prune.
# ============================================================================================================

# --------------------------------------------------------------- 4. it deleted files we did not write

_PLANTED_STEP = "ISO-10303-21;\n" + "#1 = CARTESIAN_POINT(('',(0.,0.,0.)));\n" * 40
_PLANTED_STL = "solid customer\nendsolid customer\n"


@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="real CAD needs build123d; composition needs MuJoCo")
def test_re_exporting_the_same_robot_does_not_delete_the_customers_own_files(tmp_path):
    """``output_dir`` IS THE CUSTOMER'S (``compose.py`` exposes ``--build OUTPUT_DIR`` and hands it straight to
    ``build_gene_package``), so "delete everything of this shape that I did not just write" is not ours to do.

    Measured on the first version of the prune, re-exporting THE SAME ROBOT into a directory with files planted
    in it: ``cad/reference_fixture.step`` (the CAD root, not ``step/``), ``cad/step/my_custom_bracket.step`` and
    ``cad/stl/customer_scan.STL`` (matched case-insensitively) were all deleted -- three of the five prunes; the
    mesh directories are the test below. Removal has to be by PROVENANCE: only what a previous run of ours
    wrote, proved by this directory's staging ledger or by our own last index for it.
    """
    from virturoid.services.cad_geometry import export_gene_cad

    out = tmp_path / "cad"
    gene = _compose(_ARM)
    export_gene_cad(gene, str(out))
    planted = {out / "reference_fixture.step": _PLANTED_STEP,
               out / "step" / "my_custom_bracket.step": _PLANTED_STEP,
               out / "stl" / "customer_scan.STL": _PLANTED_STL}
    for p, body in planted.items():
        p.write_text(body, encoding="utf-8")

    res = export_gene_cad(gene, str(out))             # the SAME robot, into the SAME directory
    for p, body in planted.items():
        assert p.is_file(), f"the export deleted {p.relative_to(out).as_posix()}, which it never wrote"
        assert p.read_text(encoding="utf-8") == body, f"the export rewrote {p.name}"
    assert res["removed_stale"] == [], "re-exporting the same robot removed something"
    # ...and it SAYS what it left alone, so a directory fuller than the manifest has a written explanation
    assert set(res["left_not_ours"]) == {"reference_fixture.step", "step/my_custom_bracket.step",
                                         "stl/customer_scan.STL"}


@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d; URDF needs MuJoCo")
def test_a_customers_own_mesh_survives_a_rebuild_that_removes_the_previous_robots(tmp_path):
    """The other two of the five: ``robot/meshes/`` and the ROS 2 package's ``meshes/``, both of which deleted a
    planted ``customer_endeffector.stl`` SILENTLY -- no ``removed_stale``, no note, nothing.

    Both halves are asserted together on purpose: a prune that keeps the customer's file by simply not pruning
    at all would pass the first half and fail the second.
    """
    from virturoid.services.gene_build import _write_genome_and_urdf

    out = tmp_path / "pkg"
    _write_genome_and_urdf(_compose(_QUAD), out, task=_QUAD)
    quad_meshes = {p.name for p in (out / "robot" / "meshes").glob("*.stl")}
    assert quad_meshes, "premise gone: the quadruped staged no meshes"
    customer = out / "robot" / "meshes" / "customer_endeffector.stl"
    customer.write_text(_PLANTED_STL, encoding="utf-8")

    pkg = _write_genome_and_urdf(_compose(_ARM), out, task=_ARM)
    assert pkg is not None
    assert customer.is_file() and customer.read_text(encoding="utf-8") == _PLANTED_STL, (
        "the rebuild deleted a mesh the customer put in robot/meshes/")
    left = {p.name for p in (out / "robot" / "meshes").glob("*.stl")}
    assert not (left & quad_meshes), f"the discarded quadruped's meshes are still shipped: {sorted(left & quad_meshes)}"

    # and the removal is DISCLOSED -- three of the five prunes said nothing at all before.
    log = json.loads((out / "reports" / "stale_removed.json").read_text(encoding="utf-8"))
    assert set(log["robot/meshes"]["removed_stale"]) == quad_meshes
    assert log["robot/meshes"]["left_not_ours"] == ["customer_endeffector.stl"]
    urdf = (pkg / "urdf" / "robot.urdf").read_text(encoding="utf-8")
    assert "NOTE: this export removed" in urdf, (
        "setup.py installs meshes/* wholesale, so what this export took out of meshes/ belongs in the shipped file")


@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d and MuJoCo")
def test_a_file_we_wrote_but_the_customer_then_edited_is_theirs_now(tmp_path):
    """Provenance is not "we wrote this name once". A staged file the customer has since changed is no longer
    ours to delete, and the ledger's digest is what tells the two apart."""
    from virturoid.services.gene_compiler import prune_staged_dir, write_packaged_visual_mjcf

    out = tmp_path / "pkg"
    assert write_packaged_visual_mjcf(_compose(_QUAD), out) is not None
    assets = out / "simulation" / "viewer_assets"
    ours = sorted(p for p in assets.glob("*.stl"))
    assert len(ours) >= 2, "premise gone: fewer than two staged meshes"
    edited, untouched = ours[0], ours[1]
    edited.write_text("solid edited_by_the_customer\nendsolid edited_by_the_customer\n", encoding="utf-8")

    res = prune_staged_dir(assets, keep=set())        # ask it to clear everything it owns
    assert untouched.name in res["removed"], "a file we staged and nobody touched should be ours to remove"
    assert edited.name in res["kept_modified"] and edited.is_file(), (
        "a file we staged but the customer then edited must be left alone")


@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d and MuJoCo")
def test_a_later_export_does_not_re_adopt_a_file_the_customer_edited(tmp_path):
    """The same rule, through a REAL rebuild rather than a direct prune call.

    The ledger is written by two helpers -- one after a prune, one as files are staged -- and the second is
    where this can go wrong: if it re-stamps the digest of every name it finds instead of only the ones this run
    wrote, a file the customer edited is silently re-adopted and the NEXT prune deletes it. Nothing about the
    edited file is visible from the outside, so this has to be asserted against the rebuild, not the helper.
    """
    from virturoid.services.gene_compiler import write_packaged_visual_mjcf

    out = tmp_path / "pkg"
    assert write_packaged_visual_mjcf(_compose(_QUAD), out) is not None
    assets = out / "simulation" / "viewer_assets"
    quad = {p.name for p in assets.glob("*.stl")}
    edited = sorted(assets.glob("*.stl"))[0]
    body = "solid edited_by_the_customer\nendsolid edited_by_the_customer\n"
    edited.write_text(body, encoding="utf-8")

    res = write_packaged_visual_mjcf(_compose(_ARM), out)     # SAME package, DIFFERENT robot
    assert res is not None
    assert edited.is_file() and edited.read_text(encoding="utf-8") == body, (
        "the rebuild deleted a file the customer had edited")
    assert edited.name in res["left_modified"] and edited.name not in res["removed_stale"], (
        "the rebuild must say it left the edited file alone, not pass over it in silence")
    left = {p.name for p in assets.glob("*.stl")}
    assert (quad - {edited.name}) & left == set(), "the untouched quadruped meshes should still have gone"


# --------------------------------------------------------------- 5. prune-before-write broke the package

@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d and MuJoCo")
def test_a_failed_visual_write_still_leaves_the_package_loadable(tmp_path, monkeypatch):
    """``write_packaged_visual_mjcf`` is documented fail-open: missing CAD tooling "leaves the normal primitive
    replay intact" and returns None. Pruning at the top of the function turned that into a BROKEN PACKAGE.

    Measured with ``compile_gene_to_mjcf`` forced to raise between the prune and the write: the function still
    returned None, while on disk ``viewer_mesh_index.json`` named 18 meshes of which ZERO existed and
    ``MjModel.from_xml_path`` failed outright -- on a package that loaded a moment earlier.
    """
    import mujoco

    from virturoid.services import gene_compiler as GC

    out = tmp_path / "pkg"
    assert GC.write_packaged_visual_mjcf(_compose(_QUAD), out) is not None
    sim = out / "simulation"
    mujoco.MjModel.from_xml_path(str(sim / "robot_visual.xml"))       # premise: it loads

    def _boom(*a, **k):
        raise RuntimeError("injected compiler failure")

    monkeypatch.setattr(GC, "compile_gene_to_mjcf", _boom)
    assert GC.write_packaged_visual_mjcf(_compose(_ARM), out) is None, "the documented fail-open is gone"

    index = json.loads((sim / "viewer_mesh_index.json").read_text(encoding="utf-8"))
    named = {Path(m["uri"]).name for m in index["meshes"].values()}
    on_disk = {p.name for p in (sim / "viewer_assets").glob("*.stl")}
    assert not (named - on_disk), f"the index names {len(named - on_disk)} meshes that no longer exist"
    mujoco.MjModel.from_xml_path(str(sim / "robot_visual.xml"))       # ...and it still loads


@pytest.mark.skipif(not _MUJOCO, reason="the exported MJCF needs MuJoCo")
def test_a_failed_mjcf_export_does_not_strip_the_model_already_on_disk(tmp_path, monkeypatch):
    """``write_exported_mjcf`` has the same ordering and, unlike the visual writer, lets the failure propagate
    -- so the caller is told, and the package it is told about must still be the one that was there before."""
    import mujoco

    from virturoid.services import gene_compiler as GC

    src = _MENAGERIE / "unitree_go2" / "go2.xml"
    if not src.is_file():
        pytest.skip("unitree_go2 is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot

    # AN IMPORTED BODY, NOT A COMPOSED ONE, AND THAT IS THE WHOLE POINT. This test read
    # `set(meshes_before) <= set(after)` over a composed quadruped, which carries ZERO source meshes -- so
    # `meshes_before` was [] and the assertion was `set() <= set(...)`, true whatever the prune did. Measured:
    # moving the prune back above `compile_gene_to_mjcf` left it PASSING in 0.53 s while a real export was
    # broken. On an imported Go2 the same mutation leaves robot.xml naming meshes that no longer exist and
    # `MjModel.from_xml_path` fails with "Error opening file 'meshes/base.stl'" on a model that loaded a
    # moment earlier.
    gene = import_robot(str(src), robot_id="ordering_go2")["gene"]
    assert gene is not None, "go2 did not import"

    xml = tmp_path / "exp" / "robot.xml"
    GC.write_exported_mjcf(gene, xml)
    before = xml.read_text(encoding="utf-8")
    mesh_dir = xml.parent / "meshes"
    meshes_before = sorted(p.name for p in mesh_dir.glob("*")) if mesh_dir.is_dir() else []
    assert meshes_before, ("this body must ship source meshes or the assertion below is vacuous -- "
                           "that is the defect this test was rewritten to fix")

    def _boom(*a, **k):
        raise RuntimeError("injected compiler failure")

    monkeypatch.setattr(GC, "compile_gene_to_mjcf", _boom)
    with pytest.raises(RuntimeError):
        GC.write_exported_mjcf(_compose(_ARM), xml)

    assert xml.read_text(encoding="utf-8") == before, "the failed export replaced the model on disk"
    after = sorted(p.name for p in mesh_dir.glob("*")) if mesh_dir.is_dir() else []
    assert set(meshes_before) <= set(after), "the failed export removed geometry the surviving model references"
    mujoco.MjModel.from_xml_path(str(xml))


# --------------------------------------------------------------- 6. the manifest/glob authority can invert

@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="real CAD needs build123d; composition needs MuJoCo")
def test_a_zero_part_export_never_leaves_the_manifest_describing_what_is_not_there(tmp_path):
    """This fix DESIGNATES ``cad_manifest.json`` as the authority on what belongs in ``cad/``. A zero-part
    export made it the wrong one: ``export_gene_cad`` pruned with empty keep-sets and ``_export_real_cad``
    returned None WITHOUT rewriting the manifest, so (measured) the manifest claimed 9 parts naming
    ``step/base.step`` while ``cad/step/`` held zero files.

    An export that produced nothing has no authority to remove anything, so the two keep describing each other.
    """
    import copy

    from virturoid.services.gene_build import _export_real_cad

    out = tmp_path / "pkg"
    good = _export_real_cad(_compose(_ARM), out)
    assert good and good["part_count"] > 0
    cad = out / "cad"
    listed = {Path(p["step"]).name for p in json.loads(
        (cad / "cad_manifest.json").read_text(encoding="utf-8"))["parts"]}
    assert listed == {p.name for p in (cad / "step").glob("*.step")}

    empty = copy.deepcopy(_compose(_ARM))
    empty.segments = []
    assert _export_real_cad(empty, out) is None, "a zero-part export must still report no real CAD"

    manifest = json.loads((cad / "cad_manifest.json").read_text(encoding="utf-8"))
    on_disk = {p.name for p in (cad / "step").glob("*.step")}
    assert {Path(p["step"]).name for p in manifest["parts"]} == on_disk, (
        f"the manifest claims {len(manifest['parts'])} parts over a directory holding {len(on_disk)} files")
    assert on_disk == listed, "the zero-part export deleted the previous export's CAD"


# --------------------------------------------------------------- 7. the preview directory is shared

@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="the preview bakes real meshes")
def test_two_previews_do_not_delete_each_others_assets(tmp_path, monkeypatch):
    """``project_dir/.preview`` was one directory shared by every ``/api/preview`` call. Harmless while the
    visual writer only ADDED files; destructive the moment it began removing what the current robot does not
    use. Measured on the shared directory: preview A handed the browser 18 mesh URIs and all 18 404'd once
    preview B had run, and run concurrently one request lost 7 of its own 7 assets and the other 5 of 18.

    A lock would not have fixed it -- the browser fetches the meshes AFTER the response returns -- so each
    request gets its own directory.
    """
    from fastapi.testclient import TestClient

    from virturoid.services import robot_factory
    from virturoid.services.agent import VirturoidAgent
    from virturoid.webapp import create_app

    monkeypatch.setattr(robot_factory, "build_robot",
                        lambda prompt, **k: {"gene": _compose(prompt), "intent": {}})
    client = TestClient(create_app(tmp_path))
    project_dir = VirturoidAgent(tmp_path, target_success=0.8).project_dir
    assert project_dir  # the endpoint writes under it; the URIs below are relative to it

    def preview(prompt):
        r = client.post("/api/preview", json={"prompt": prompt})
        assert r.status_code == 200, r.text[:300]
        return sorted({g["mesh_uri"] for g in r.json()["geoms"] if g.get("mesh_uri")})

    first = preview(_QUAD)
    second = preview(_ARM)
    assert first and second, "premise gone: the preview shipped no meshes"
    for tag, uris in (("first", first), ("second", second)):
        gone = [u for u in uris
                if client.get("/api/artifact-binary", params={"path": u}).status_code != 200]
        assert not gone, f"the {tag} preview's own assets 404 after the other ran: {gone[:4]}"
    assert not set(first) & set(second), "the two previews wrote into the same directory"


# --------------------------------------------------------------- 8. one directory, ONE KIND of robot

@pytest.mark.skipif(not _MUJOCO, reason="a real build needs MuJoCo")
def test_building_a_different_class_into_a_used_directory_is_refused_not_mixed(tmp_path):
    """THE PRUNE MADE THIS WORSE, NOT BETTER, AND THIS IS THE CLOSE.

    Build a quadruped then an arm into one directory and the customer used to get, measured:
    ``export/ros2/virturoid_robot/config/robot.yaml`` listing the ARM's 8 joints beside
    ``policy_type: "trot_cpg_gait"`` and ``has_controller: true``; an installed ``controller.py`` that is
    verbatim the quadruped trot CPG; a ``policy_params.json`` keyed to ``genome_built_quadruped_18seg``; plus a
    stale ``viewer_mesh_index.json``, ``robot_visual.xml``, ``locomotion_qpos.json``, ``gait_controller.py`` and
    ``fusion/config/ekf.yaml``. Before the prune, 36 obviously-quadruped STLs told a customer at a glance that
    something was wrong. After it, every surface a human checks first is a correct arm and only the thing that
    COMMANDS THE HARDWARE is wrong.

    The prune cannot reach these: the writers are BRANCH-CONDITIONAL (``write_packaged_visual_mjcf`` has one
    caller, on the locomotion path), so a non-legged rebuild never runs them, and the indexes are stale too, so
    "follow the manifest" saves nobody. The build refuses instead, and says so in the package.
    """
    from virturoid.services.gene_build import MixedRobotPackageError, build_gene_package

    out = tmp_path / "pkg"
    build_gene_package(_compose(_QUAD), _QUAD, out, scene_count=2)
    yaml_before = (out / "export" / "ros2" / "virturoid_robot" / "config" / "robot.yaml").read_text(
        encoding="utf-8")
    assert "trot_cpg_gait" in yaml_before, "premise gone: the quadruped shipped no gait controller"

    with pytest.raises(MixedRobotPackageError) as exc:
        build_gene_package(_compose(_ARM), _ARM, out, scene_count=2)
    assert "locomotion" in str(exc.value) and "manipulation" in str(exc.value)

    # the refusal is an ARTIFACT, not only a traceback
    conflict = json.loads((out / "reports" / "package_conflict.json").read_text(encoding="utf-8"))
    assert conflict["existing_package_path"] == "locomotion"
    assert conflict["requested_package_path"] == "manipulation"
    assert "software/gait_controller.py" in conflict["stale_artifacts"]
    assert (out / "PACKAGE_CONFLICT.md").is_file()

    # ...and the package that was already there is untouched and still describes ONE robot
    assert (out / "export" / "ros2" / "virturoid_robot" / "config" / "robot.yaml").read_text(
        encoding="utf-8") == yaml_before
    genome_id = json.loads((out / "robot" / "robot_genome.json").read_text(encoding="utf-8"))["id"]
    params = json.loads((out / "software" / "controller" / "policy_params.json").read_text(encoding="utf-8"))
    assert params["robot_genome_id"] == genome_id, "the shipped brain is not this package's robot's"


@pytest.mark.skipif(not _MUJOCO, reason="a real build needs MuJoCo")
def test_rebuilding_the_same_kind_of_robot_into_its_own_directory_still_works(tmp_path):
    """The guard must not break the flow it exists inside: ``autonomous_build`` re-runs ``build_gene_package``
    into the directory it already used once a redesign is accepted, and that rebuild is the SAME kind of robot
    -- every artifact on its path is rewritten, so there is nothing to refuse."""
    from virturoid.services.gene_build import build_gene_package

    out = tmp_path / "pkg"
    build_gene_package(_compose(_QUAD), _QUAD, out, scene_count=2)
    (out / "PACKAGE_CONFLICT.md").write_text("stale note from an earlier refusal\n", encoding="utf-8")
    summary = build_gene_package(_compose(_QUAD), _QUAD, out, scene_count=2)
    assert summary["task_type"] == "locomotion"
    assert not (out / "PACKAGE_CONFLICT.md").exists(), "a compatible rebuild must clear its own stale warning"


# --------------------------------------------------------------- 9. the mimic branch, on a body that has one

_MENAGERIE = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
#: Real robots whose grippers are one DOF driven through a linkage, so the URDF we ship carries a ``<mimic>``.
#: Both are used because they reach the branch by different routes: the Robotiq IS the gripper, the Panda
#: carries one at the end of a 7-DOF arm.
_MIMIC_MODELS = ["robotiq_2f85/2f85.xml", "franka_emika_panda/panda.xml"]


@pytest.mark.skipif(not _MUJOCO, reason="importing a real description needs MuJoCo")
@pytest.mark.parametrize("rel", _MIMIC_MODELS)
def test_the_assembly_table_skips_mimic_joints_and_says_why(tmp_path, rel):
    """A MIMIC JOINT HAS NO MOTOR TO MOUNT, AND NOTHING IN THIS SUITE HAD ONE.

    Mutating ``_assembly_rows`` to iterate ``joints`` instead of ``commandable`` -- dropping the mimic
    exclusion entirely -- passed all 8 tests above, because every body they compose is a plain tree. So this
    runs on a REAL description that declares the coupling: a Robotiq 2F-85 (one ``<equality><joint>`` between
    its two drivers) and a Franka Panda (its two fingers). Measured on the imported Robotiq: 8 joints, 1 of
    them a mimic, so the table must have 7 rows and the yaml 7 entries -- the mutation gives 8 and 8, and asks
    the customer to mount a motor on a joint that is driven through a gear.
    """
    src = _MENAGERIE / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")

    from virturoid.services.deployment_guide import _assembly_rows, build_deployment_guide
    from virturoid.services.gene_build import _emit_bom, _write_genome_and_urdf
    from virturoid.services.robot_import import import_robot

    gene = import_robot(str(src), robot_id=f"mimic_{Path(rel).parent.name}")["gene"]
    assert gene is not None, f"{rel} did not import"
    out = tmp_path / "pkg"
    pkg = _write_genome_and_urdf(gene, out, task="")
    bom = _emit_bom(gene, out, task="")
    assert pkg is not None

    urdf = (out / "robot" / "robot.urdf").read_text(encoding="utf-8")
    assert "<mimic joint=" in urdf, f"premise gone: the URDF we ship for {rel} declares no coupling"
    actuator_map = (bom or {}).get("actuator_map") or json.loads(
        (out / "robot" / "bill_of_materials.json").read_text(encoding="utf-8"))["actuator_map"]

    asm = _assembly_rows(out, actuator_map)
    joints = [str(j["name"]) for j in
              json.loads((out / "robot" / "robot_genome.json").read_text(encoding="utf-8"))["joints"]]
    assert asm["mimic"], "the guide did not see the coupling its own URDF declares"
    assert len(asm["rows"]) == len(joints) - len(asm["mimic"]), (
        f"{len(asm['rows'])} rows for {len(joints)} joints of which {len(asm['mimic'])} are driven")
    for driven in asm["mimic"]:
        assert driven not in {j for j, _k, _a in asm["rows"]}, (
            f"{driven} is driven through a transmission; the table asks for a motor to mount on it")

    section = build_deployment_guide(out).split("## 2. Assemble")[1].split("## 3.")[0]
    rows = [tuple(c.strip() for c in r)
            for r in re.findall(r"^\| ([^|]+?) \| ([^|]+?) \| ([^|]+?) \|$", section, re.M)]
    rows = [r for r in rows if r[0] not in ("Joint", "---")]
    assert len(rows) == len(asm["rows"])
    for driven, driver in asm["mimic"].items():
        assert driven not in {r[0] for r in rows}
        assert f"`{driven}`" in section and f"`{driver['joint']}`" in section, (
            "a joint left out of the assembly table must be accounted for, not silently missing")

    # ...and the hardware interface in the same package must describe the same set of joints.
    yaml = (pkg / "config" / "hardware_interface.yaml").read_text(encoding="utf-8")
    yrows = {j for j, _act, _key in _YAML_ROW.findall(yaml)}
    assert yrows == {r[0] for r in rows}, "the guide and hardware_interface.yaml disagree about what to wire"


# --------------------------------------------------------------- 10. the ledger is not a single point of truth

@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="real CAD needs build123d; composition needs MuJoCo")
def test_a_deleted_staging_ledger_costs_the_record_not_the_customers_files(tmp_path):
    """The ledger lives in the directory it describes, so the customer can delete it. Two things must then
    hold, and they pull in opposite directions:

      * the rebuild still cleans up, because the LAST ``cad_manifest.json`` we wrote names the same files from
        the other side -- that redundancy is what makes a provenance scheme survivable at all;
      * with BOTH gone we know of nothing here that is ours, so nothing is removed. The safe direction is to
        leave files alone and SAY the directory holds more than the manifest, not to fall back to deleting
        everything of a shape -- that fallback is the regression this whole design exists to undo.
    """
    from virturoid.services.cad_geometry import export_gene_cad
    from virturoid.services.gene_compiler import STAGE_LEDGER_NAME

    def _strip_ledgers(cad):
        for d in (cad, cad / "step", cad / "stl"):
            (d / STAGE_LEDGER_NAME).unlink(missing_ok=True)

    # (a) ledger gone, manifest intact -> the manifest carries the provenance
    with_manifest = tmp_path / "a"
    quad = {Path(p["step"]).name for p in export_gene_cad(_compose(_QUAD), str(with_manifest))["parts"]}
    _strip_ledgers(with_manifest)
    planted = with_manifest / "step" / "my_custom_bracket.step"
    planted.write_text(_PLANTED_STEP, encoding="utf-8")
    res = export_gene_cad(_compose(_ARM), str(with_manifest))
    on_disk = {p.name for p in (with_manifest / "step").glob("*.step")}
    assert not (quad & on_disk), "with the manifest still there, the previous robot's CAD must go"
    assert planted.is_file(), "and the customer's file must not"
    assert res["left_not_ours"] == ["step/my_custom_bracket.step"]

    # (b) ledger AND manifest gone -> nothing here is provably ours
    naked = tmp_path / "b"
    quad2 = {Path(p["step"]).name for p in export_gene_cad(_compose(_QUAD), str(naked))["parts"]}
    _strip_ledgers(naked)
    (naked / "cad_manifest.json").unlink()
    res2 = export_gene_cad(_compose(_ARM), str(naked))
    assert res2["removed_stale"] == [], "with no provenance at all, nothing may be deleted"
    assert quad2 <= {p.name for p in (naked / "step").glob("*.step")}
    assert len(res2["left_not_ours"]) >= len(quad2), (
        "an export that cannot clean up must at least SAY the directory holds more than its manifest")


@pytest.mark.skipif(not _MUJOCO, reason="composition needs MuJoCo")
def test_the_mixed_package_override_still_writes_the_warning_into_the_package(tmp_path, monkeypatch):
    """``VIRTUROID_ALLOW_MIXED_PACKAGE=1`` exists for a scratch directory, and it downgrades the refusal to a
    warning -- it does not make the package honest. The written artifact is the same either way, because the
    resulting package really does ship a controller for a different robot."""
    from virturoid.services.gene_build import _guard_package_is_one_robot

    out = tmp_path / "pkg"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "robot_package_contract.json").write_text(
        json.dumps({"package_type": "gene_locomotion_package"}), encoding="utf-8")
    (out / "software").mkdir(parents=True)
    (out / "software" / "gait_controller.py").write_text("# the previous robot's brain\n", encoding="utf-8")

    monkeypatch.setenv("VIRTUROID_ALLOW_MIXED_PACKAGE", "1")
    with pytest.warns(RuntimeWarning, match="mixed package"):
        _guard_package_is_one_robot(_compose(_ARM), out)          # allowed, not silent
    assert (out / "PACKAGE_CONFLICT.md").is_file()
    conflict = json.loads((out / "reports" / "package_conflict.json").read_text(encoding="utf-8"))
    assert conflict["allowed_anyway"] is True
    assert "software/gait_controller.py" in conflict["stale_artifacts"]


# ------------------------------------- 10. the SAME-ROBOT rebuild, which is where the re-adoption really lives

@pytest.mark.skipif(not (_B123D and _MUJOCO), reason="visual meshes need build123d and MuJoCo")
def test_a_same_robot_rebuild_does_not_adopt_the_customers_edit(tmp_path):
    """THE SIBLING TEST ABOVE GOES QUAD -> ARM AND MISSES THIS ENTIRELY.

    The re-adoption needs a rebuild of the SAME body in between, because that is the only case where the mesh
    cache SERVES the customer's edited file instead of overwriting it: ``build_visual_meshes`` reuses a
    generated mesh on ``fp.exists()`` alone. Two separate bugs conspired, and each was invisible without this
    sequence -- the writer handed ``note_staged`` every name it returned rather than the ones it wrote, and
    ``prune_staged_dir`` finished with ``_write_stage_ledger(d, keep)``, re-digesting every surviving file with
    no carry-over.

    MEASURED before the fix, through this exact sequence: the ledger swore the customer's bytes were ours, the
    package shipped their geometry attributed to us, and the next rebuild of a different body DELETED it and
    recorded it in ``removed_stale`` as our own stale geometry -- a written claim that it was never theirs.
    That is strictly worse than the stale meshes the prune exists to remove: stale geometry is visibly wrong,
    this looks correct and then vanishes.

    Note what is NOT asserted: that the edit survives a rebuild is not a feature being promised, it is a
    consequence of the cache. What is promised is that we never CLAIM a file we did not write, and never delete
    one while saying it was ours.
    """
    from virturoid.services import gene_compiler as GC

    pkg = tmp_path / "pkg"
    gene = _compose(_QUAD)
    assert GC.write_packaged_visual_mjcf(gene, pkg) is not None
    assets = pkg / "simulation" / "viewer_assets"
    victim = sorted(assets.glob("*.stl"))[0]
    recorded = dict((GC.read_stage_ledger(assets) or {}).get(victim.name) or {})
    assert recorded, "precondition: the first export claimed this file"

    victim.write_bytes(victim.read_bytes() + b"\n# customer hand-edit\n")

    GC.write_packaged_visual_mjcf(gene, pkg)                      # SAME robot: the permitted rebuild
    after = dict((GC.read_stage_ledger(assets) or {}).get(victim.name) or {})
    assert after == recorded, (
        "a rebuild re-digested a file it did not write; the ledger now claims the customer's edit as ours")

    out = GC.write_packaged_visual_mjcf(_compose(_ARM), pkg) or {}  # a DIFFERENT body: the deleting step
    assert victim.exists(), "the rebuild deleted a file the customer had edited"
    assert victim.name not in set(out.get("removed_stale") or []), (
        "the export claimed a customer-edited file as its own stale geometry")
    assert victim.name in (set(out.get("left_modified") or []) | set(out.get("left_not_ours") or [])), (
        "a file we declined to remove must be reported, or the customer cannot tell it was left behind")
    assert out.get("removed_stale"), "the prune must still clear the genuinely stale meshes of the old body"
