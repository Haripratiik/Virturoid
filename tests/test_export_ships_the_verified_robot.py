"""The package export_held writes is the robot that was verified: the same body AND the same controller.

MEASURED 2026-10-11 by driving the whole loop through the MCP server: create_robot -> edit_robot set_payload 2 kg
-> train_held (a CREDIBLE WALK, 10.5 m) -> export_held. The certificate in the package said DEPLOY != MEASURE on
both halves, and it was right:

* the BODY: the export door's grounding re-derived the 2 kg payload link from its volume x density, so the package
  carried 3.925 kg there (36.98 -> 38.905 kg), although set_payload had recorded the load and every amend keeps it;
* the CONTROLLER: the export never read the operating point the robot carries. It ran its own tuning grid, found
  no credible crawl, and shipped the fallback trot starter, which crouched backwards (-0.254 m) while the trained
  crawl walked. With both fixed, the same robot exports a crawl frozen from its own point that walks when the
  shipped file is run, and the certificate reads deploy == measure on both halves.
"""
from __future__ import annotations

import copy
import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")


def _walker():
    """The design schema's quadruped: its shipped default crawl point walks, so it can carry an op-point."""
    from virturoid.services import session_state as S
    from virturoid.services.agent_design_tools import get_design_schema, submit_design
    rid = submit_design({"graph": get_design_schema({})["examples"]["quadruped"]})["robot_id"]
    return rid, S.get_robot(rid)


def test_a_stated_payload_keeps_its_mass_through_the_export_door():
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gene_build import ground_and_repair
    rid, _ = _walker()
    env = call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": "set_payload", "args": {"payload_kg": 2.0}}]})
    assert env["ok"], env
    held = S.get_robot(rid)
    stated = (held.metadata.get("embodied_mass") or {}).get("payload_kg") or {}
    assert stated, "set_payload must record the load it put on the robot"
    exported = copy.deepcopy(held)
    ground_and_repair(exported)
    for link, kg in stated.items():
        mass = next(s.mass_kg for s in exported.segments if s.name == link)
        assert mass == pytest.approx(kg), f"the export re-derived the stated {kg} kg on {link!r} as {mass} kg"
    # Only the stated load is asserted. On this body the export's housing-fit pass also thickens the legs around
    # the larger motors the payload called for (r 0.013-0.019 -> 0.027 m, +1.8 kg), a separate difference that
    # export_held and the certificate already report as body parity; it is not this fix's to hide.


def test_the_export_freezes_the_operating_point_the_body_carries():
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    from virturoid.services.morph_policy import extract_crawl_gait_params
    _, gene = _walker()
    own = {k: float(_DEFAULT_GAIT[k]) for k in ("freq", "hip_amp", "knee_amp", "kp", "kd")}
    gene.metadata = {**(gene.metadata or {}), "gait_params": dict(own)}
    gait = extract_crawl_gait_params(gene)
    assert gait is not None, "a body whose own operating point walks must export a crawl"
    assert gait["operating_point_source"] == "held"
    assert gait["operating_point"] == pytest.approx(own)


def test_the_certificate_compares_the_frozen_point_not_the_probed_frequency():
    """The crawl freezes the frequency its probe picked from a ladder around ``freq`` (a 2.49 Hz point shipped a
    4.23 Hz wave), so ``frequency_hz`` can never equal the point it came from; the program records that point."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.verdict_certificate import build_certificate
    gene = compose_robot("a small quadruped robot dog")
    point = {"freq": 2.4858, "hip_amp": 0.693, "knee_amp": 1.326, "kp": 140.68, "kd": 12.69}
    gene.metadata = {**(gene.metadata or {}), "gait_params": dict(point)}
    v = {"verdict": "CREDIBLE WALK", "survived": True, "forward_m": 0.9, "gait_source": "tuned_for_this_body"}
    shipped = {"policy_type": "crawl_wave_gait", "entrypoint": "software/gait_controller.py",
               "frequency_hz": 4.2258, "hip_amp": 0.693, "knee_amp": 1.326, "kp": 140.68, "kd": 12.69,
               "operating_point": dict(point), "verified_walk": True, "sim_forward_m": 1.6}
    same = build_certificate(gene, v, robot_id="r_same", shipped_controller=shipped)["controller_parity"]
    assert same["operating_point_match"] is True and same["same"] is True, same
    moved = build_certificate(gene, v, robot_id="r_moved", shipped_controller={
        **shipped, "operating_point": {**point, "freq": 2.0}})["controller_parity"]
    assert moved["operating_point_match"] is False and moved["same"] is False, moved
