"""A folder a customer points us at is THEIRS. We read it; we never write in it.

Measured through ``agent_tools.call_tool`` on a real Menagerie Unitree Go2, an ingest put TWO files inside
``~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2``::

    go2.__mjcfprep__.xml    model_import.import_model      a prep copy, compiled then unlinked
    ingestion_report.json   input_training_tools           our report, left behind

Only the second survived to be seen in a before/after listing, which is exactly why the tests below do not
rely on a listing alone. ``test_the_mjcf_prep_copy_is_gone`` is the regression for the one that cleaned up
after itself, and it is the harder half: a write that deletes its own evidence still fails on a read-only
mount, a network share, a vendor drop, or a CI checkout with a dirty-tree gate.

The enforcement is ``source_guard.read_only``, armed at the ingest door rather than at each writer, because
an ingest fans out through the classifier, three importers, the grounding pass and the edit operators --
"every write reachable from an ingest" is not a list anyone keeps current by reading.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_URDF = """<?xml version="1.0"?>
<robot name="arm2">
  <link name="base_link"><inertial><mass value="1.0"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="upper"><inertial><mass value="0.6"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.04 0.04 0.3"/></geometry></collision></link>
  <joint name="pan" type="revolute"><parent link="base_link"/><child link="upper"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/></joint>
</robot>"""

_MJCF = """<mujoco model="probe">
  <worldbody>
    <body name="trunk" pos="0 0 0.3">
      <freejoint/>
      <geom name="t" type="box" size="0.1 0.05 0.03" mass="1"/>
      <body name="leg" pos="0.08 0 0">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom name="l" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hip" ctrlrange="-1 1" gear="10"/></actuator>
</mujoco>"""


def _snapshot(root: Path) -> dict:
    """Name -> (mtime_ns, size, bytes-digest). Content AND stamp, because `shutil.copy2` moves an mtime
    without moving a byte and an mtime-only check would miss a re-stamp."""
    import hashlib
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = Path(dirpath) / fn
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_mtime_ns, st.st_size,
                                             hashlib.sha1(p.read_bytes(), usedforsecurity=False).hexdigest())
    return out


@pytest.fixture()
def project(tmp_path):
    d = tmp_path / "customers_robot"
    (d / "meshes").mkdir(parents=True)
    (d / "arm.urdf").write_text(_URDF, encoding="utf-8")
    (d / "robot.xml").write_text(_MJCF, encoding="utf-8")
    (d / "notes.md").write_text("aluminum body, 1 kg payload\n", encoding="utf-8")
    (d / "meshes" / "placeholder.txt").write_text("not a real mesh\n", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- the guarantee, end to end
@pytest.mark.skipif(not _MUJOCO, reason="ingesting a model needs MuJoCo")
def test_an_ingest_leaves_the_customers_folder_byte_identical_AND_still_imports(project, monkeypatch):
    """The headline -- and it needs BOTH halves, which the first version of this test did not have.

    As written on 2026-08-12 it asserted only ``robot_id`` plus an unchanged folder. Both of those are ALSO
    true when the guard does its job and the import dies: refusing the write makes ``_load_model`` raise, the
    ingest falls back to composing a proxy from the text description, and the customer gets back a truthy
    ``robot_id`` for a robot that is not theirs -- over a folder that is, indeed, byte-identical. That is
    exactly what happened (``robot_import._compile`` writes a temp file NEXT TO the real meshes so relative
    paths resolve; the guard refused it and #215 reopened), and THIS TEST STAYED GREEN THROUGH IT. It measured
    that we did not write, never that the import still worked.

    So containment is asserted together with the thing containment must not cost. MEASURED on this fixture:
    ``lane_used='faithful'``, ``lanes_attempted=['faithful','gene']``, ``substituted=None``,
    ``source_folder_protection=None``.

    ``source_folder_protection is None`` is the sharper of the two. A non-None report means the guard BLOCKED
    something -- i.e. the folder survived because a write was refused, not because none was attempted. Passing
    on refusals would let a new writer land inside the customer's folder and be papered over by the guard,
    which is the failure mode one layer up from the one this file exists for.
    """
    from virturoid.services.agent_tools import call_tool
    monkeypatch.chdir(project.parent)                       # build/ lands beside, never inside, the project
    before = _snapshot(project)
    env = call_tool("ingest_project", {"project_path": str(project)})
    r = env["result"]
    assert r.get("robot_id"), env
    after = _snapshot(project)
    assert after == before, {
        "added": sorted(set(after) - set(before)),
        "removed": sorted(set(before) - set(after)),
        "changed": sorted(k for k in set(after) & set(before) if after[k] != before[k]),
    }
    # ...and the customer's own model is what we actually loaded, not a proxy composed after a refusal.
    assert r.get("lane_used") == "faithful", (
        f"the folder is intact but the import did not use the customer's model: lane_used="
        f"{r.get('lane_used')!r}, lanes_attempted={r.get('lanes_attempted')!r} -- containment must not be "
        f"bought by failing the import (#215)")
    assert not r.get("substituted"), f"a body was substituted for the customer's: {r.get('substituted')!r}"
    assert r.get("source_folder_protection") is None, (
        f"the folder survived because writes were REFUSED, not because none were attempted: "
        f"{r.get('source_folder_protection')!r} -- find the writer and stage it out-of-tree instead")


@pytest.mark.skipif(not _MUJOCO, reason="ingesting a model needs MuJoCo")
def test_the_mjcf_prep_copy_is_gone(project, monkeypatch):
    """The write that DELETED ITS OWN EVIDENCE. `import_model` wrote `<stem>.__mjcfprep__.xml` beside the
    customer's model, compiled it, then unlinked it -- invisible to any before/after listing. Catching it
    needs an observer at the write, not at the directory."""
    import builtins
    import io
    monkeypatch.chdir(project.parent)
    from virturoid.services.model_import import import_model

    root = os.path.normcase(str(project))
    writes: list[str] = []
    for module, name in ((builtins, "open"), (io, "open")):
        original = getattr(module, name)

        def spy(file, mode="r", *a, _o=original, **k):
            if any(c in str(mode) for c in "wxa+") and os.path.normcase(
                    os.path.abspath(os.fspath(file))).startswith(root):
                writes.append(str(file))
            return _o(file, mode, *a, **k)
        monkeypatch.setattr(module, name, spy)

    assert import_model(str(project / "robot.xml"))["ok"]
    assert import_model(str(project / "arm.urdf"))["ok"]
    assert writes == [], f"import wrote inside the customer's folder: {writes}"


# --------------------------------------------------------------------------- the guard itself
def test_the_guard_refuses_a_write_and_records_it(project):
    """A guard whose violations are invisible is a guard nobody knows has fired -- the report writer's own
    `except Exception: pass` would have swallowed the refusal silently."""
    from virturoid.services import source_guard

    with source_guard.read_only(project) as guard:
        # BOTH doors. `builtins.open` and `io.open` are two names for one function object, so patching one
        # leaves the other live -- and `pathlib.Path.write_text` reaches for `io.open`, which is precisely
        # how the MJCF prep write got past. This assertion is the reason both are patched.
        with pytest.raises(PermissionError):
            (project / "via_pathlib.json").write_text("{}", encoding="utf-8")
        with pytest.raises(PermissionError):
            open(project / "via_builtins.json", "w", encoding="utf-8").close()
        with pytest.raises(PermissionError):
            os.remove(project / "arm.urdf")
        # ...and an mtime re-stamp, which writes no bytes and would pass a content check
        with pytest.raises(PermissionError):
            os.utime(project / "arm.urdf", None)
        assert (project / "arm.urdf").read_text(encoding="utf-8")     # reading is untouched
    assert len(guard.blocked) == 4
    assert {b["operation"] for b in guard.blocked} == {"io.open(write)", "open(write)",
                                                       "os.remove", "os.utime"}
    assert guard.report()["n_blocked_writes"] == 4
    assert not (project / "via_pathlib.json").exists() and not (project / "via_builtins.json").exists()


def test_the_guard_only_protects_its_own_root(project, tmp_path):
    """Arming for one project must not turn the whole filesystem read-only -- build/ is written throughout
    an ingest, and a guard that blocked it would be worse than the defect."""
    from virturoid.services import source_guard
    elsewhere = tmp_path / "build"
    with source_guard.read_only(project) as guard:
        elsewhere.mkdir()
        (elsewhere / "artifact.json").write_text("{}", encoding="utf-8")
        # a sibling whose name merely PREFIXES the root is not inside it
        sibling = project.parent / (project.name + "_backup")
        sibling.mkdir()
        (sibling / "f.txt").write_text("x", encoding="utf-8")
    assert guard.blocked == []


def test_the_patches_are_removed_when_the_block_exits(project):
    """A leaked patch would make every later write in the process pay for a guard nobody armed."""
    import builtins
    import io
    from virturoid.services import source_guard
    before = (builtins.open, io.open, os.remove, os.utime)
    with source_guard.read_only(project):
        assert builtins.open is not before[0], "the guard never installed"
    assert (builtins.open, io.open, os.remove, os.utime) == before


def test_nesting_on_the_SAME_root_does_not_release_it_early(project):
    """The nastier half of nesting, and the one a two-different-roots test cannot see.

    An ingest can re-enter on the same folder (a sub-project under a parent already being ingested). If the
    inner exit dropped the root from the active set, the patches would still be installed -- armed, and
    guarding nothing -- which reads as protection and is not."""
    from virturoid.services import source_guard
    with source_guard.read_only(project) as outer:
        with source_guard.read_only(project) as inner:
            with pytest.raises(PermissionError):
                (project / "a").write_text("x", encoding="utf-8")
        assert inner.blocked is outer.blocked, "one root, one ledger"
        with pytest.raises(PermissionError):
            (project / "b").write_text("x", encoding="utf-8")
    assert len(outer.blocked) == 2
    (project / "c").write_text("x", encoding="utf-8")           # released once the last holder exits


def test_nesting_does_not_unpatch_the_outer_guard(project, tmp_path):
    """An ingest can call another ingest (a bundled sub-project); the inner exit must not disarm the outer.
    Checked by BEHAVIOUR -- a write that must still be refused -- not by inspecting the patch."""
    from virturoid.services import source_guard
    other = tmp_path / "second"
    other.mkdir()
    with source_guard.read_only(project) as outer:
        with source_guard.read_only(other):
            pass
        with pytest.raises(PermissionError):
            (project / "x").write_text("y", encoding="utf-8")
    assert len(outer.blocked) == 1
    (project / "x").write_text("y", encoding="utf-8")       # ...and writable again once the block exits


def test_a_missing_or_file_root_does_not_pretend_to_protect(tmp_path):
    """A .zip ingest and a description-only ingest have no directory to guard. Disarm honestly rather than
    arming on a path that does not exist -- and a FILE root protects exactly itself, never its parent."""
    from virturoid.services import source_guard
    with source_guard.read_only(tmp_path / "nope") as g:
        (tmp_path / "free.txt").write_text("x", encoding="utf-8")
    assert g.blocked == [] and g.report() is None

    zipped = tmp_path / "robot.zip"
    zipped.write_bytes(b"PK\x03\x04")
    with source_guard.read_only(zipped) as g:
        (tmp_path / "beside.txt").write_text("x", encoding="utf-8")   # the parent stays writable
        with pytest.raises(PermissionError):
            zipped.write_bytes(b"clobbered")
    assert len(g.blocked) == 1
    assert zipped.read_bytes() == b"PK\x03\x04"


# --------------------------------------------------------------------------- where our artifacts go instead
def test_staging_keys_two_projects_with_the_same_name_apart(tmp_path):
    """Every customer has a folder called `robot/`. Sharing a staging dir would let the second import read
    the first one's prepped file."""
    from virturoid.services import source_guard
    a = tmp_path / "alice" / "robot"
    b = tmp_path / "bob" / "robot"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    da, db = source_guard.staging_dir(a), source_guard.staging_dir(b)
    assert da != db
    assert da.name.startswith("robot-") and db.name.startswith("robot-")
    assert source_guard.staging_dir(a) == da, "the same source must map to the same staging dir"


@pytest.mark.skipif(not _MUJOCO, reason="ingesting a model needs MuJoCo")
def test_the_ingestion_report_lands_under_build_and_is_named(project, monkeypatch):
    """Moving the report out of the customer's folder only helps if the customer is told where it went --
    otherwise one honesty gap is traded for another."""
    from virturoid.services.agent_tools import call_tool
    monkeypatch.chdir(project.parent)
    r = call_tool("ingest_project", {"project_path": str(project)})["result"]
    written = r.get("ingestion_report_path")
    assert written, "the ingest must say where it wrote the report"
    p = Path(written)
    assert p.is_file() and p.name == "ingestion_report.json"
    assert "build" in p.parts, p
    assert not str(p).startswith(str(project)), "the report is still inside the customer's folder"
    assert not (project / "ingestion_report.json").exists()
