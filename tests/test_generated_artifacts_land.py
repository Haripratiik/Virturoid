"""#289: a tool that GENERATES something must hand it over, and a tool that does not touch your robot must say so.

Three defects, found by the promise-vs-code sweep and pinned here:

1. ``generate_control_scripts`` compiled the scripts, wrote them into ``tempfile.mkdtemp()``, validated them,
   returned the manifest + verdict and ABANDONED the directory. An agent that asked for control scripts got
   filenames and a pass/fail and no code and no path. (``generate_fusion`` popped ``_files_content`` the same
   way.) The files did reach the customer eventually through ``export_held``, so this was the weakest of the
   three -- but "eventually, via a different tool" is not what the caller asked for.
2. ``start_training`` accepted a ``robot_id``, read the PROMPT off it, and had ``autonomous_build`` compose a
   BRAND-NEW body. The held robot was untouched and nothing in the reply said so.
3. SECURITY (the only non-cosmetic one): the same handler built its workspace as
   ``Path(args.get("build_root") or "build/agent_builds")`` RAW, skipping the ``safe_build_path`` containment
   every other ``build_root`` caller uses. Measured before the fix, ``build_root="../../../../../../pwned"``
   put a real file six directories above the process's build root -- and, because
   ``job_registry._run_autonomous_build`` also derives ``memory_dir = build_root / "memory"``, it steered the
   flywheel bank out of the tree with it.

Every assertion goes through ``agent_tools.call_tool`` -- the surface a customer's agent actually uses.
"""
from __future__ import annotations

import importlib.util
import json
import time
import types
import uuid
from pathlib import Path

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")


def _held(prompt="a robot dog"):
    """A held robot, the way a customer's agent would have one after create_robot/ingest_project."""
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    return S.put_robot(compose_robot(prompt, llm=None), prompt=prompt)


def _under(child: Path, root: Path) -> bool:
    child, root = Path(child).resolve(), Path(root).resolve()
    return child == root or root in child.parents


# --------------------------------------------------------------------- 1. the generators hand over the goods
def test_control_scripts_come_back_as_files_on_disk_not_just_a_verdict(tmp_path, monkeypatch):
    """The defect: manifest + validation and nothing usable. Now: a KEPT directory, an absolute path per file,
    and the per-robot config inline -- and the validation verdict is about the very files being handed over."""
    monkeypatch.chdir(tmp_path)
    from virturoid.services.agent_tools import call_tool
    rid = _held()

    env = call_tool("generate_control_scripts", {"robot_id": rid, "task": "patrol"})
    assert env["ok"], env
    r = env["result"]

    sdir = Path(r["scripts_dir"])
    assert sdir.is_dir(), "the directory must be KEPT, not a mkdtemp the handler walks away from"
    assert _under(sdir, tmp_path / "build"), f"scripts landed outside build/: {sdir}"

    # every advertised file exists at the absolute path advertised, with the advertised size
    assert len(r["files"]) == 8, r["files"].keys()            # 6 .py + control_config.json + script_manifest
    for rel, info in r["files"].items():
        p = Path(info["path"])
        assert p.is_file(), f"{rel} was advertised at {p} and is not there"
        assert p.name == Path(rel).name
        assert len(p.read_text(encoding="utf-8").encode("utf-8")) == info["bytes"]

    # the part that is DERIVED FROM THIS ROBOT comes back inline, always
    cfg = json.loads(r["config"]["control_config.json"])
    assert cfg["torque_ceilings_nm"] and cfg["obs_dim"] > 0
    assert cfg["n_joints"] == r["manifest"]["n_joints"]

    # the verdict is about these files (validate_scripts ran in this directory)
    assert r["validation"]["n_scripts"] == 6 and r["validation"]["all_pass"] is True, r["validation"]
    assert not (sdir / "__pycache__").exists(), "the dry-run's bytecode is not part of the deliverable"

    # ...and the 9.6 KB of shared template is not re-sent unless asked for
    assert "source" not in r


def test_include_source_returns_every_byte_for_an_agent_that_cannot_read_the_path(tmp_path, monkeypatch):
    """An agent whose MCP server is not on its own filesystem cannot open ``scripts_dir``. One flag, every byte,
    and the inline source must be identical to what was written."""
    monkeypatch.chdir(tmp_path)
    from virturoid.services.agent_tools import call_tool
    rid = _held()

    r = call_tool("generate_control_scripts", {"robot_id": rid, "include_source": True})["result"]
    assert set(r["source"]) == set(r["files"])
    for rel, text in r["source"].items():
        assert text == Path(r["files"][rel]["path"]).read_text(encoding="utf-8")
    assert "def clamp" in r["source"]["safety_filter.py"] or "torque" in r["source"]["safety_filter.py"]


def test_fusion_stack_is_written_and_returned_whole(tmp_path, monkeypatch):
    """Every fusion file is derived from THIS robot's BOM, so all of it is inline -- and on disk under
    ``fusion_dir``, which the handler used to drop on the floor with ``out.pop('_files_content')``."""
    monkeypatch.chdir(tmp_path)
    from virturoid.services.agent_tools import call_tool
    rid = _held()

    env = call_tool("generate_fusion", {"robot_id": rid, "task": "patrol"})
    assert env["ok"], env
    r = env["result"]

    fdir = Path(r["fusion_dir"])
    assert fdir.is_dir() and _under(fdir, tmp_path / "build")
    # one contract across both generators: files = {rel: {path, bytes}}, source = {rel: text}
    assert set(r["files"]) == set(r["source"]) and "config/fusion_manifest.json" in r["files"]
    for rel, info in r["files"].items():
        p = Path(info["path"])
        assert p.is_file() and p.read_text(encoding="utf-8") == r["source"][rel]
    # the EKF config is a real one, keyed to this robot's frame
    assert "ekf_filter_node" in r["source"]["config/ekf.yaml"]
    assert r["source"]["config/ekf.yaml"].count("base_link_frame") == 1


def test_generators_refuse_an_unheld_robot_without_writing_anything(tmp_path, monkeypatch):
    """A refusal must stay a refusal: no directory is created for a robot that does not exist."""
    monkeypatch.chdir(tmp_path)
    from virturoid.services.agent_tools import call_tool
    for tool in ("generate_control_scripts", "generate_fusion"):
        env = call_tool(tool, {"robot_id": "robot_does_not_exist"})
        assert env["ok"] is False and "no held robot" in env["error"]
    assert not (tmp_path / "build" / "agent_builds" / "robot_does_not_exist").exists()


# ------------------------------------------------------------- 2. start_training says whose body it trains
def _stub_build(monkeypatch, seen: dict):
    """Replace the real (minutes-long) build with a recorder that still WRITES where it was pointed."""
    import virturoid.services.autonomous_build as AB

    def fake(prompt, output_dir, **kw):
        output_dir = Path(output_dir)
        seen["output_dir"] = output_dir
        seen["memory_dir"] = Path(kw.get("memory_dir") or "")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "LANDED.txt").write_text("a real write landed here", encoding="utf-8")
        return types.SimpleNamespace(decisions=[], notes=["stub"], robot_class="quadruped", succeeded=True)

    monkeypatch.setattr(AB, "autonomous_build", fake)


def _await_job(jid, timeout=30.0):
    from virturoid.services import job_registry as J
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = J.get(jid)
        if v and v["status"] not in ("queued", "running"):
            return v
        time.sleep(0.02)
    raise AssertionError(f"job {jid} never finished")


def test_start_training_discloses_that_it_does_not_train_your_held_robot(tmp_path, monkeypatch):
    """The customer ingested their Go2 and called start_training on it. A different robot got trained. The
    reply now names that, and names the door that trains the body they hold."""
    monkeypatch.chdir(tmp_path)
    seen = {}
    _stub_build(monkeypatch, seen)
    from virturoid.services.agent_tools import call_tool
    rid = _held("a quadruped inspection robot")

    env = call_tool("start_training", {"robot_id": rid, "train": False})
    assert env["ok"], env
    trains = env["result"]["trains"]
    assert trains["held_robot"] is False
    assert trains["robot_id"] == rid
    assert "PROMPT only" in trains["reason"] and "train_held" in trains["reason"]
    assert trains["prompt_used"] == "a quadruped inspection robot"
    _await_job(env["result"]["job_id"])

    # and the description an agent reads BEFORE calling says it too
    from virturoid.services.agent_tools import TOOLS
    assert "BUILDS A NEW BODY" in TOOLS["start_training"]["description"]
    assert "train_held" in TOOLS["start_training"]["description"]


def test_start_training_without_a_robot_id_still_states_what_it_builds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}
    _stub_build(monkeypatch, seen)
    from virturoid.services.agent_tools import call_tool
    env = call_tool("start_training", {"prompt": "a walking dog", "train": False})
    trains = env["result"]["trains"]
    assert trains["held_robot"] is False and "robot_id" not in trains
    assert "holds nothing" in trains["reason"]
    _await_job(env["result"]["job_id"])


# ------------------------------------------------------------------------- 3. the traversal, refused
@pytest.mark.parametrize("prefix", [
    "../../../../../..",                                      # relative escape (the measured one)
    "..\\..\\..\\..\\..\\..",                                 # windows separators
    "build/../..",                                            # escape through the root it is meant to stay in
])
def test_start_training_refuses_a_build_root_that_escapes_the_build_tree(tmp_path, monkeypatch, prefix):
    """Before the fix this wrote ``LANDED.txt`` six directories above the build root (measured 2026-08-09).
    ``build_root`` is agent-supplied and is where the whole package -- and the memory dir -- lands.

    The target directory name is UNIQUE per run on purpose: a fixed name would let a leftover directory from
    some earlier run mask (or fake) the escape, and this assertion is the whole point of the test."""
    monkeypatch.chdir(tmp_path)
    seen = {}
    _stub_build(monkeypatch, seen)
    from virturoid.services.agent_tools import call_tool

    target = f"escape_probe_{uuid.uuid4().hex[:10]}"
    env = call_tool("start_training", {"prompt": "a walking dog", "train": False,
                                       "build_root": f"{prefix}/{target}"})
    assert env["ok"], env
    _await_job(env["result"]["job_id"])

    build_root = tmp_path / "build"
    out = seen["output_dir"]
    assert _under(out, build_root), f"escaped to {out}"
    assert _under(seen["memory_dir"], build_root), f"memory escaped to {seen['memory_dir']}"
    assert (out / "LANDED.txt").is_file()                      # the build really ran; it was only redirected
    # nothing named for the payload exists anywhere at or above the temp root
    for probe in (tmp_path, *tmp_path.parents):
        assert not (probe / target).exists(), f"traversal landed at {probe / target}"


def test_start_training_refuses_an_absolute_build_root_outside_the_tree(tmp_path, monkeypatch):
    """The blunter attack: not a traversal, just an absolute path somewhere else on the disk."""
    monkeypatch.chdir(tmp_path)
    seen = {}
    _stub_build(monkeypatch, seen)
    from virturoid.services.agent_tools import call_tool

    outside = tmp_path / "not_the_build_tree"
    env = call_tool("start_training", {"prompt": "a walking dog", "train": False, "build_root": str(outside)})
    _await_job(env["result"]["job_id"])
    assert _under(seen["output_dir"], tmp_path / "build")
    assert not outside.exists()


def test_a_build_root_inside_the_tree_is_still_honored(tmp_path, monkeypatch):
    """Containment is not a lobotomy: a legitimate sub-directory of build/ must still be used."""
    monkeypatch.chdir(tmp_path)
    seen = {}
    _stub_build(monkeypatch, seen)
    from virturoid.services.agent_tools import call_tool

    env = call_tool("start_training", {"prompt": "a walking dog", "train": False,
                                       "build_root": "agent_builds/customer_acme"})
    _await_job(env["result"]["job_id"])
    assert seen["output_dir"].parent == (tmp_path / "build" / "agent_builds" / "customer_acme").resolve()
