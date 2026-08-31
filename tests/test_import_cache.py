"""Importing the same unchanged robot 26 times should cost one import.

MEASURED on this checkout (2026-08-06, one CPU), a single ``robot_import.import_robot`` call on a real
Menagerie package: Unitree G1 13.88 s, Go2 11.29 s, Talos 7.45 s, Booster T1 4.93 s, Panda 4.07 s, Spot 3.84 s,
UR5e 3.49 s, Cassie 2.79 s. The suite names ~115 Menagerie model references across nine files -- go2.xml 26
times, g1.xml 13, ur5e.xml 11 -- and every one re-parsed a file that had not changed. On those eight files
alone that is ~563 s (9.4 min) of a two-hour suite.

``VIRTUROID_IMPORT_CACHE=1`` memoizes the import for the life of the process, default-off exactly like
``VIRTUROID_GAIT_FIT_CACHE``. It loses no coverage: every distinct model still runs the real importer, only the
Nth identical re-import is free.

The two things a cache like this gets wrong, both asserted below:

  * HANDING BACK THE SAME OBJECT. Callers mutate what they are given -- ``ingest_project`` re-grounds masses,
    amend operators rewrite segments, tests set ``gene.loop_closures = []``. One caller would silently rewrite
    the next caller's robot.
  * A KEY THAT MISSES A CHANGE. Keying on path+mtime is not enough: a test that writes a fixture, imports it,
    rewrites it and imports again is ordinary, and Windows file timestamps do not reliably separate two writes
    milliseconds apart. The model file is keyed by CONTENT DIGEST, its directory (meshes, ``<include>``s,
    keyframes) by a bounded fingerprint, and the two options that alter the result by value.
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")

_TWO_LINK = """
<mujoco model="two_link">
  <worldbody>
    <body name="upper" pos="0 0 1">
      <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.03" mass="1"/>
      <body name="lower" pos="0 0 -0.30">
        <joint name="j" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.03" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

_THREE_LINK = _TWO_LINK.replace(
    '      </body>\n    </body>',
    """        <body name="tip" pos="0 0 -0.30">
          <joint name="k" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.03" mass="1"/>
        </body>
      </body>
    </body>""")


@pytest.fixture()
def cache_on(monkeypatch):
    monkeypatch.setenv("VIRTUROID_IMPORT_CACHE", "1")


def _write(tmp_path: Path, text: str, name: str = "m.xml") -> str:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _imp():
    from virturoid.services import robot_import
    return robot_import


# ------------------------------------------------------------------------------------ default-off
def test_the_cache_is_off_unless_the_variable_says_so(monkeypatch, tmp_path):
    """A product run must be byte-identical to one with no cache in the module at all — a customer who edits a
    file and re-imports it can never be handed yesterday's robot."""
    R = _imp()
    monkeypatch.delenv("VIRTUROID_IMPORT_CACHE", raising=False)
    src = _write(tmp_path, _TWO_LINK)
    assert R._import_cache_key(src, None, None) is None
    before = len(R._IMPORT_CACHE)
    R.import_robot(src)
    R.import_robot(src)
    assert len(R._IMPORT_CACHE) == before, "the cache was written to while it was switched off"


def test_the_suite_turns_it_on():
    """Conftest owns the switch; if it stops setting it, nothing gets faster and this test says so out loud."""
    conf = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    assert "VIRTUROID_IMPORT_CACHE" in conf
    val = os.environ.get("VIRTUROID_IMPORT_CACHE")
    if val != "1":
        pytest.skip(f"cache deliberately disabled for this run (VIRTUROID_IMPORT_CACHE={val!r})")


# ------------------------------------------------------------------------------------ it is actually faster
def test_a_repeat_import_of_a_real_robot_is_nearly_free(monkeypatch):
    """Measured against the uncached path in the same process, so it never clears state other files rely on."""
    R = _imp()
    src = _MEN / "universal_robots_ur5e/ur5e.xml"
    if not src.is_file():
        pytest.skip("MuJoCo Menagerie not cached locally")

    monkeypatch.delenv("VIRTUROID_IMPORT_CACHE", raising=False)
    t0 = time.perf_counter(); R.import_robot(str(src)); cold = time.perf_counter() - t0
    if cold < 0.3:
        pytest.skip(f"this machine imports a UR5e in {cold:.3f}s; the saving is not measurable here")

    monkeypatch.setenv("VIRTUROID_IMPORT_CACHE", "1")
    R.import_robot(str(src))                                  # populate
    t0 = time.perf_counter(); R.import_robot(str(src)); warm = time.perf_counter() - t0
    assert warm < cold / 10.0, f"cold {cold:.3f}s vs memoized {warm:.3f}s — barely a saving"


def test_the_memoized_result_is_the_SAME_robot_not_merely_a_fast_one(cache_on):
    """A cache that returns a different robot is worse than no cache. Compared through the serializer AND
    through the compiled MJCF, because the MJCF is what every downstream consumer actually sees."""
    R = _imp()
    src = _MEN / "agility_cassie/cassie.xml"
    if not src.is_file():
        pytest.skip("MuJoCo Menagerie not cached locally")
    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    R.clear_import_cache()
    first = R.import_robot(str(src), robot_id="cache_same")
    second = R.import_robot(str(src), robot_id="cache_same")
    assert first["gene"] is not second["gene"]
    assert second["gene"].to_dict() == first["gene"].to_dict()
    assert second["warnings"] == first["warnings"]
    assert (second["valid"], second["robot_class"], second["species"]) == \
           (first["valid"], first["robot_class"], first["species"])
    assert second["backend_support"] == first["backend_support"]
    assert compile_gene_to_mjcf(second["gene"]) == compile_gene_to_mjcf(first["gene"])
    # and the loops came with it (a closed-loop robot is the case a lossy snapshot would quietly ruin)
    assert len(second["gene"].loop_closures) == 4


# ------------------------------------------------------------------------------------ mutation safety
def test_a_caller_may_mutate_what_it_is_given(cache_on, tmp_path):
    """Every level: the result dict, the warnings list, the gene, a segment, a NESTED metadata dict, and the
    loop list. ``to_dict``'s ``dict(self.metadata)`` is shallow, so a nested dict is the one that bites."""
    R = _imp()
    R.clear_import_cache()
    src = _write(tmp_path, _TWO_LINK)
    a = R.import_robot(src)
    a["warnings"].append("scribble")
    a["backend_support"]["mujoco"]["status"] = "wrecked"
    a["gene"].id = "renamed"
    a["gene"].segments[0].mass_kg = 999.0
    a["gene"].segments.pop()
    a["gene"].loop_closures.append({"a": "upper", "b": "lower"})
    a["gene"].metadata["mass_source"] = "vandalised"
    a["gene"].metadata.setdefault("rest_pose", {})["j"] = 1.234

    b = R.import_robot(src)
    assert "scribble" not in b["warnings"]
    assert b["backend_support"]["mujoco"]["status"] == "supported"
    assert b["gene"].id != "renamed"
    assert b["gene"].loop_closures == []
    assert b["gene"].metadata.get("mass_source") == "source_model"
    assert b["gene"].metadata.get("rest_pose", {}).get("j") is None
    assert len(b["gene"].segments) == len(a["gene"].segments) + 1
    assert all(s.mass_kg != 999.0 for s in b["gene"].segments)


# ------------------------------------------------------------------------------------ the key
def test_rewriting_the_same_path_gives_the_new_robot(cache_on, tmp_path):
    """The mtime trap. Two writes to one path, milliseconds apart — a content digest separates them and a
    filesystem timestamp may not."""
    R = _imp()
    R.clear_import_cache()
    src = _write(tmp_path, _TWO_LINK)
    n_before = len(R.import_robot(src)["gene"].segments)
    _write(tmp_path, _THREE_LINK)                      # same path, new robot, immediately
    n_after = len(R.import_robot(src)["gene"].segments)
    assert n_after == n_before + 1, f"the edit was not seen: {n_before} -> {n_after} segments"


def test_a_changed_sibling_asset_busts_the_key(cache_on, tmp_path):
    """An MJCF is not self-contained: meshes, ``<include>``s and keyframes live beside it and change the model
    while its own bytes do not."""
    R = _imp()
    src = _write(tmp_path, _TWO_LINK)
    k1 = R._import_cache_key(src, None, None)
    (tmp_path / "assets.xml").write_text("<mujoco/>", encoding="utf-8")
    k2 = R._import_cache_key(src, None, None)
    assert k1 is not None and k1 != k2, "a new file beside the model left the key unchanged"


def test_the_options_that_change_the_answer_are_in_the_key(cache_on, tmp_path):
    """``robot_id`` names the gene AND picks the directory the customer's meshes are baked into; ``species``
    names the species-tree node. A key without them hands the second caller the first caller's identity."""
    R = _imp()
    R.clear_import_cache()
    src = _write(tmp_path, _TWO_LINK)
    assert R.import_robot(src, robot_id="alpha")["gene"].id == "alpha"
    assert R.import_robot(src, robot_id="beta")["gene"].id == "beta"
    assert R.import_robot(src, species="x.y")["gene"].species == "x.y"
    assert R.import_robot(src, species="p.q")["gene"].species == "p.q"


def test_an_xml_string_is_keyed_by_its_own_text(cache_on):
    """A source with no path at all still has an exact identity."""
    R = _imp()
    R.clear_import_cache()
    assert len(R.import_robot(_TWO_LINK)["gene"].segments) + 1 == \
           len(R.import_robot(_THREE_LINK)["gene"].segments)
    k = R._import_cache_key(_TWO_LINK, None, None)
    assert k is not None and k[0] == "text" and k == R._import_cache_key(_TWO_LINK, None, None)


# ------------------------------------------------------------------------------------ failures and clearing
def test_a_failed_import_is_never_cached(cache_on, tmp_path):
    """Same rule as ``gait_flywheel._remember``: a parse error is not an answer about the model, so the next
    caller gets a real attempt rather than a memoized failure."""
    R = _imp()
    R.clear_import_cache()
    src = _write(tmp_path, "<mujoco><worldbody><body name='b'><geom type='nonsense'/></body></worldbody></mujoco>")
    with pytest.raises(Exception):
        R.import_robot(src)
    assert len(R._IMPORT_CACHE) == 0


def test_clear_import_cache_makes_the_parser_run_again(cache_on, tmp_path):
    R = _imp()
    R.clear_import_cache()
    R.import_robot(_write(tmp_path, _TWO_LINK))
    assert len(R._IMPORT_CACHE) == 1
    R.clear_import_cache()
    assert len(R._IMPORT_CACHE) == 0


def test_the_directory_fingerprint_is_bounded(cache_on, tmp_path):
    """A customer can drop a model into a monorepo. The walk is capped, and truncation is recorded IN the
    fingerprint so a truncated key never compares equal to a complete one."""
    R = _imp()
    assert R._DIR_FINGERPRINT_CAP > 0
    for i in range(12):
        (tmp_path / f"f{i}.bin").write_bytes(b"x" * (i + 1))
    n, total, newest, truncated = R._dir_fingerprint(tmp_path)
    assert n == 12 and total == sum(range(1, 13)) and newest > 0 and truncated is False
    t0 = time.perf_counter()
    R._dir_fingerprint(tmp_path)
    assert time.perf_counter() - t0 < 1.0, "fingerprinting must be cheap next to a multi-second import"
