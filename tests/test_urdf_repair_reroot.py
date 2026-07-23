"""P0 agentic-ingestion: a real customer URDF must LOAD (repair pass) and be able to LOCOMOTE (free-base
reroot) — with every change reported, never silent.

Grounded on the AS-PUBLISHED Unitree Go2 URDF, which (measured 2026-07-22) fails MuJoCo as-is on two exporter
artifacts at once: the same material re-defined in block after block, and MANY <material> elements stuffed into
one <visual>. Before P0 it hard-failed both import lanes; the generic proxy that ingest_project built instead
discarded the customer's real FL_hip/FL_thigh/FL_calf geometry (#215), and fixed-base URDFs imported bolted to
the world so verify read "WHEELS OFF GROUND" (#214).
"""
from __future__ import annotations

import importlib.util

import pytest

from virturoid.services.model_import import repair_urdf_text, reroot_free_base

_MUJOCO = importlib.util.find_spec("mujoco") is not None


# --- the repair pass is pure string work; no MuJoCo needed for these -------------------------------------
def test_repair_reports_every_change_and_is_a_noop_on_clean_input():
    clean = '<robot name="x"><link name="a"/></robot>'
    out, repairs = repair_urdf_text(clean)
    assert out == clean and repairs == []                      # nothing to fix -> untouched, nothing claimed


def test_repair_collapses_duplicate_and_multi_material_visuals():
    urdf = (
        '<robot name="d">'
        '<link name="l1"><visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="red"><color rgba="1 0 0 1"/></material>'
        '<material name="blue"><color rgba="0 0 1 1"/></material></visual></link>'
        '<link name="l2"><visual><geometry><box size="1 1 1"/></geometry>'
        '<material name="red"><color rgba="1 0 0 1"/></material></visual></link>'
        '</robot>'
    )
    out, repairs = repair_urdf_text(urdf)
    kinds = {r["kind"] for r in repairs}
    assert "material_normalize" in repairs[0]["kind"] or "material_normalize" in kinds
    # exactly one FULL definition survives per name, and no <visual> keeps two materials
    import re
    assert len(re.findall(r"<material[^>]*>\s*<color", out)) == 2     # red + blue, each defined once
    for vis in re.findall(r"<visual>.*?</visual>", out, re.S):
        assert vis.count("<material") <= 1


def test_repair_strips_gazebo_and_transmission():
    urdf = '<robot name="d"><link name="a"/><gazebo reference="a"><mu1>1</mu1></gazebo>' \
           '<transmission name="t"><type>x</type></transmission></robot>'
    out, repairs = repair_urdf_text(urdf)
    assert "<gazebo" not in out and "<transmission" not in out
    assert {"strip_gazebo", "strip_transmission"} <= {r["kind"] for r in repairs}


# --- the Go2 acceptance properties (need MuJoCo + the file) ----------------------------------------------
def _go2_path():
    import os
    p = r"C:\Users\harie\AppData\Local\Temp\go2_description.urdf"
    return p if os.path.exists(p) else None


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_go2_loads_faithfully_with_customer_names_preserved():
    """#215 core: the as-published Go2 URDF loads via the repair pass and keeps the CUSTOMER'S link names —
    it is NOT silently swapped for our generic skeleton."""
    path = _go2_path()
    if path is None:
        pytest.skip("Go2 URDF fixture not present on this machine")
    from virturoid.services.model_import import import_model
    r = import_model(path)
    assert r["ok"], r.get("note")
    assert any(r["kind"] == "material_normalize" for r in r["repairs"]), "the repair must be reported"
    names = " ".join(r["body_names"])
    assert "FL_hip" in names and "FL_thigh" in names and "FL_calf" in names
    assert r["actuated"] >= 12                                   # its 12 leg motors are drivable


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_fixed_base_import_reroots_to_standing_not_bolted():
    """#214 core: a fixed-base import gets a free base and settles ON ITS FEET (base z > 0 after 500 steps),
    instead of the bolted/collapsed body a naive import leaves (measured -0.835 m)."""
    import mujoco
    path = _go2_path()
    if path is None:
        pytest.skip("Go2 URDF fixture not present on this machine")
    from virturoid.services.model_import import import_model
    r = import_model(path)
    assert r["ok"] and not r["free_base"]                        # imported bolted
    mjcf2, added = reroot_free_base(r["mjcf"])
    assert added, "a fixed-base legged import must be rerootable"
    m = mujoco.MjModel.from_xml_string(mjcf2)
    assert any(int(m.jnt_type[j]) == 0 for j in range(m.njnt))   # now free-based
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert float(d.qpos[2]) > 0.0                                # standing, not sunk through the floor


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_reroot_is_idempotent_on_an_already_free_model():
    from virturoid.services.morph_policy import robot_mjcf
    from virturoid.services.morphology_composer import compose_robot
    mjcf = robot_mjcf(compose_robot("a quadruped robot", llm=None))
    _, added = reroot_free_base(mjcf)
    assert added is False                                        # our composed bodies are already free-based


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_ingest_go2_uses_faithful_lane_and_discloses_it(tmp_path):
    """#215 end-to-end: dropping the Go2 folder holds the FAITHFUL model (FL_* names), names the lane it used,
    and emits an understood/guessed/dropped ingestion report — no silent generic-proxy swap."""
    import shutil

    src = _go2_path()
    if src is None:
        pytest.skip("Go2 URDF fixture not present")
    from virturoid.services.input_training_tools import _ingest_project
    (tmp_path / "robot").mkdir()
    shutil.copy(src, tmp_path / "robot" / "go2.urdf")
    r = _ingest_project({"project_path": str(tmp_path), "description": "a Unitree Go2 quadruped robot dog"})

    assert r.get("robot_id")
    assert r["lane_used"] == "faithful", r.get("lanes_attempted")   # disclosure is schema-required
    f = r["faithful"]
    assert f["ok"] and f["actuated"] >= 12
    kept = " ".join(f["link_names_preserved"])
    assert "FL_hip" in kept and "FL_thigh" in kept                  # customer geometry, not our skeleton
    assert any(rp["kind"] == "material_normalize" for rp in f["repairs"])
    rep = r["ingestion_report"]
    assert rep["understood"] and set(rep) >= {"understood", "guessed", "dropped", "lane_used"}
    assert (tmp_path / "ingestion_report.json").exists()


# --- P0 exposure/honesty: ingestion discoverable + amend reaches hip links -------------------------------
def test_ingestion_is_discoverable_in_the_mcp_view():
    """BYOK model: a customer's own agent must find ingestion from tools/list (it could not before P0)."""
    from virturoid.services.agent_tools import tool_specs
    view = tool_specs(view="mcp")
    names = [t["name"] for t in view]
    assert "ingest_project" in names
    gw = next(t for t in view if t["name"] == "ingest_project")
    for sib in ("import_onnx_policy", "import_bom", "adopt_control_script", "import_dataset"):
        assert sib in gw["description"], f"{sib} must be advertised on the gateway so agents can call it"


def test_scale_group_legs_reaches_hip_links_but_not_ship():
    """#216-adjacent: `scale_group legs` on a Unitree-style FL_hip/FR_hip chain must include the hips, and the
    word-boundary guard must never match unrelated words like 'battleship'."""
    from virturoid.services.edit_operators import _GROUP_WORDS, _name_in_group
    legs = _GROUP_WORDS["legs"]
    assert _name_in_group("FL_hip", legs) and _name_in_group("RR_hip_joint", legs)
    assert not _name_in_group("battleship", legs) and not _name_in_group("microchip", legs)
