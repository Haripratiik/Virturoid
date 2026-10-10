"""The two imported-mesh caches must HIT, and 148 GB says what happens when they do not.

Two caches, the same flaw: each keyed on something that changes on every run, so neither ever hit and nothing
evicted the misses.

``build/_viewmesh``, MEASURED 2026-09-09: 132,555 STL files totalling 134.92 GB, of which only 1,449 were
distinct meshes. One copy of each is 1.02 GB, so 133.89 GB was duplicates, 99.2% waste, 128.5 GB of it written
in a single month. ``FR_hip_src`` had 1,648 copies. The key was ``path | size | mtime``, sound in isolation
(re-baking an import writes a new mtime, so a re-imported robot can never read a stale copy) and wrong about
what sits upstream: ``robot_import`` re-bakes link meshes into ``build/_importmesh`` on every import, so the
source mtime moved on every run and the key never repeated. It now keys on the bytes.

``build/_importmesh``, MEASURED 2026-10-10: 1,892 directories holding 81 distinct robots, 16.85 GB of which
13.47 GB was duplicates. The Menagerie Go2 alone had been baked into 73 directories. The key preferred the
caller's ``robot_id``, and every probe or test picked a fresh one, though nothing the bake produces depends on
that id. It now keys on the model.

Each test asserts both directions where it can, because the two failure modes are opposite and equally bad: a
key that always hits is cheap and serves the wrong geometry, and a key that never hits is what filled the disk.
"""

import os
import pathlib
import shutil
import time
import types

import pytest

_MENAGERIE = pathlib.Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
_GO2 = _MENAGERIE / "unitree_go2" / "go2.xml"
_HAVE_GO2 = _GO2.is_file()


def _seg(name="FR_hip"):
    return types.SimpleNamespace(name=name)


@pytest.fixture
def isolated_build(tmp_path, monkeypatch):
    """Point every ``anchored`` build path at ``tmp_path`` so a test cannot add to the real ``build/``.

    The real-robot test below used to import with ``robot_id=f"cache_{i}"`` straight into the checkout's
    ``build/_importmesh``, so every run of the test that guards against duplication added three directories of
    duplicates. ``robot_import`` reads ``anchored`` at call time, so patching the module attribute reaches it.
    """
    from virturoid.services import install_paths

    monkeypatch.setattr(install_paths, "anchored", lambda rel: tmp_path / rel)
    return tmp_path


# ------------------------------------------------------------------ build/_viewmesh: key on the mesh's bytes

def test_the_same_mesh_re_materialised_elsewhere_hits_the_cache(tmp_path):
    """The regression itself: identical bytes at a new path with a new mtime must resolve to one cache entry."""
    from virturoid.services.cad_geometry import _src_mesh_fp

    a = tmp_path / "src_a" / "hip.stl"
    a.parent.mkdir()
    a.write_bytes(b"solid hip\n" + b"x" * 4096 + b"\nendsolid\n")
    first = _src_mesh_fp(tmp_path / "cache", _seg(), a)

    time.sleep(1.1)                                   # a new mtime, which is what used to mint a new name
    b = tmp_path / "src_b" / "hip.stl"
    b.parent.mkdir()
    shutil.copyfile(a, b)
    assert int(a.stat().st_mtime) != int(b.stat().st_mtime), "precondition: the mtime really did move"

    assert _src_mesh_fp(tmp_path / "cache", _seg(), b).name == first.name, (
        "a re-materialised copy of the same mesh minted a second cache entry; this is the 134 GB defect")


def test_a_different_mesh_never_reuses_another_ones_entry(tmp_path):
    """The other half. A cache that always hits is worse than one that never does: it ships wrong geometry."""
    from virturoid.services.cad_geometry import _src_mesh_fp

    a = tmp_path / "hip.stl"
    a.write_bytes(b"solid hip\n" + b"x" * 4096 + b"\nendsolid\n")
    b = tmp_path / "hip_other.stl"
    b.write_bytes(b"solid hip\n" + b"y" * 4096 + b"\nendsolid\n")   # same size, different bytes

    assert a.stat().st_size == b.stat().st_size, "precondition: size alone cannot tell these apart"
    assert _src_mesh_fp(tmp_path, _seg(), a).name != _src_mesh_fp(tmp_path, _seg(), b).name


def test_an_unreadable_source_still_produces_a_name(tmp_path):
    """A cache miss costs a wasted copy. Refusing to render a link the customer shipped costs their robot."""
    from virturoid.services.cad_geometry import _src_mesh_fp

    missing = tmp_path / "gone.stl"
    missing.write_bytes(b"x")
    fp = _src_mesh_fp(tmp_path, _seg(), missing)
    assert fp.name.startswith("FR_hip_src_") and fp.suffix == ".stl"


@pytest.mark.skipif(not _HAVE_GO2, reason="unitree_go2 is not cached locally (robot_descriptions fetches on demand)")
def test_re_importing_a_real_robot_adds_nothing_to_the_cache(isolated_build):
    """End to end on the body that produced the duplicates, because the unit tests above cannot see the churn.

    The source mtime moved because ``robot_import`` re-bakes every link on every import. Only a real import
    reproduces that, which is why this drives ``import_robot`` three times rather than copying a file about.
    """
    from virturoid.services.cad_geometry import build_visual_meshes
    from virturoid.services.robot_import import import_robot

    cache = isolated_build / "viewmesh"
    counts = []
    for i in range(3):
        gene = import_robot(str(_GO2), robot_id=f"cache_{i}")["gene"]
        assert gene is not None
        build_visual_meshes(gene, str(cache), cache=True, actuator_in_mesh=False)
        counts.append(len(list(cache.glob("*.stl"))))

    assert counts[0] > 0, "precondition: the first import populated the cache"
    assert counts == [counts[0]] * 3, (
        f"re-importing the same robot grew the mesh cache {counts}; each repeat used to add a full set")


# ------------------------------------------------------------- build/_importmesh: key on the model, not the id

@pytest.mark.skipif(not _HAVE_GO2, reason="unitree_go2 is not cached locally (robot_descriptions fetches on demand)")
def test_naming_the_robot_does_not_give_the_same_model_a_new_directory(isolated_build):
    """The 13.47 GB defect, reproduced with the ids that produced it.

    These three names are real directories from the checkout that measured it: probes and verifiers each picked a
    fresh id for the same ``go2.xml``, and the key followed the id.
    """
    from virturoid.services.robot_import import import_robot

    for rid in ("abl_unitree_go2", "advfinA_unitree_go2", "cache_2"):
        assert import_robot(str(_GO2), robot_id=rid)["gene"] is not None

    dirs = sorted(d.name for d in (isolated_build / "build" / "_importmesh").iterdir() if d.is_dir())
    assert len(dirs) == 1, f"three imports of one model made {len(dirs)} mesh directories: {dirs}"


def test_the_mesh_directory_follows_the_model(tmp_path):
    """Both directions, on fixtures, so this runs on a machine without the Menagerie.

    Two models with the SAME basename must not share: that is the collision an earlier key was written against,
    where two ``go2.xml`` files overwrote each other's links and one half-written STL made MuJoCo refuse the whole
    robot. And an EDITED model must move to a new directory, so a saved robot that points at its meshes by
    absolute path keeps the geometry it was imported with.
    """
    from virturoid.services.robot_import import _import_mesh_key

    a = tmp_path / "pkg_a" / "robot.urdf"
    a.parent.mkdir()
    a.write_text("<robot name='a'/>")
    b = tmp_path / "pkg_b" / "robot.urdf"
    b.parent.mkdir()
    b.write_text("<robot name='b'/>")

    assert _import_mesh_key(str(a)) != _import_mesh_key(str(b)), "two different models shared a directory"
    assert _import_mesh_key(str(a)) == _import_mesh_key(str(a)), "the same model must land in the same place"

    before = _import_mesh_key(str(a))
    a.write_text("<robot name='a_edited'/>")
    assert _import_mesh_key(str(a)) != before, "an edited model reused the directory a saved robot points at"


def test_inline_xml_gets_a_short_honest_name():
    """An XML string has no basename. The old key used whatever text followed its last slash."""
    from virturoid.services.robot_import import _import_mesh_key

    key = _import_mesh_key("<mujoco><worldbody><body name='b'/></worldbody></mujoco>")
    assert key.startswith("inline_") and len(key) < 64, key


def test_the_import_cache_still_separates_robot_ids(tmp_path, monkeypatch):
    """The asymmetry is deliberate, and this pins it so nobody "fixes" one key to match the other.

    A robot id changes what an import RETURNS (the gene carries it), so the in-process import cache must keep
    two ids apart. It changes nothing about the geometry, so the mesh directory must not. Both read the same
    ``_source_identity``; only the cache adds the caller's arguments to it.
    """
    from virturoid.services.robot_import import _import_cache_key, _import_mesh_key

    monkeypatch.setenv("VIRTUROID_IMPORT_CACHE", "1")
    a = tmp_path / "robot.urdf"
    a.write_text("<robot name='a'/>")

    assert _import_cache_key(str(a), "one", None) != _import_cache_key(str(a), "two", None)
    assert _import_cache_key(str(a), "one", None)[:-1] == _import_cache_key(str(a), "two", None)[:-1], (
        "the two keys must differ ONLY in the caller's arguments, not in what they think the model is")
    assert _import_mesh_key(str(a)) == _import_mesh_key(str(a))
