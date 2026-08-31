"""The robot the customer SEES, the robot we VERIFY, and the robot we SHIP must be ONE robot.

Three measured ways they had come apart, each pinned here against real MuJoCo Menagerie models (skipped
cleanly when the corpus is absent, so CI stays hermetic):

A. AN AMEND DISMEMBERED THE ROBOT AND RETURNED ``ok: true``. ``edit_robot {op: set_height, target_m: 1.5}`` on
   an imported Unitree G1 scaled only the SIX ``*_hip_*_link`` bodies -- ``segments_for_group(gene, "legs")``
   matches a word list that knows nothing about ``knee``, ``ankle`` or ``pelvis`` -- landed the body at
   1.117 m instead of 1.5 m, printed the miss in ``standing_height_m`` and still reported success. The render
   showed the thighs floating off the pelvis and the shins and feet hanging below with a visible gap.

B. WHAT YOU SEE VS WHAT GETS VERIFIED. ``render_view`` builds ``gene_to_meshed_mjcf`` while the gait sim
   builds ``compile_gene_to_mjcf``, and the geom census looks alarming (held Go2: 42 geoms / 13 meshes versus
   47 geoms / 21 capsules + 25 cylinders). Measured, the two share ONE set of colliders, masses and spawn
   height -- the extra geoms on both sides are ``mass=0 contype=0`` cosmetics. Nothing enforced that, and
   nothing disclosed it. These tests make it enforced and checkable.

C. EXPORT MUTATED THE VERIFIED GENE. ``ground_and_repair`` hardcoded ``carbon_fiber``/0.25 and ``ground_gene``
   re-derives every mass from geometry x density x fill, so a body grounded at aluminium lost 41% of its
   structural mass on the way out (a Go2 verified at 27.362 kg exported at 19.151 kg, 1600/2700 = 0.593 on all
   13 links), and an IMPORTED robot had its manufacturer's real masses replaced by our primitive-volume
   estimate. The certificate travelling in the same package still printed "deploy==measure", and ``bom.json``
   listed different motors than the certificate's own ``margins.bom``.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_ALLOW_HEURISTIC_FALLBACK", "1")

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="every check compiles a real MuJoCo model")

_MENAGERIE = Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie"


def _model(rel: str) -> Path:
    """A real Menagerie model, or skip -- the corpus is a developer convenience, never a CI dependency."""
    p = _MENAGERIE / rel
    if not p.is_file():
        pytest.skip(f"MuJoCo Menagerie model {rel} not present at {_MENAGERIE}")
    return p


def _imported(rel: str):
    from virturoid.services.robot_import import import_robot
    gene = import_robot(str(_model(rel)))["gene"]
    assert gene is not None and gene.segments, f"{rel} imported as an empty body"
    return gene


def _mass(gene) -> float:
    return round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 3)


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUROID_SESSIONS_DIR", str(tmp_path / "sessions"))
    from virturoid.services import session_state as S
    return S


# --------------------------------------------------------------------- A. the amend must not lie
def test_set_height_scales_the_whole_height_bearing_chain_not_one_joint_family(session):
    """The selection is STRUCTURAL now (ancestors of the ground-contacting links), so it reaches a G1's
    ``left_knee_link`` / ``left_ankle_roll_link`` -- which no word in the ``legs`` group matches."""
    from virturoid.services.edit_operators import height_bearing_segments

    chain = set(height_bearing_segments(_imported("unitree_g1/g1.xml")))
    assert len(chain) >= 12, f"only {len(chain)} height-bearing links found on a G1: {sorted(chain)}"
    for token in ("knee", "ankle"):
        assert any(token in n for n in chain), (
            f"no {token} link in the height chain {sorted(chain)} -- set_height is back to scaling the hip "
            "family alone, which is what left the body in pieces")


@pytest.mark.parametrize("rel", ["unitree_g1/g1.xml", "unitree_go2/go2.xml"])
def test_a_height_edit_either_hits_its_target_or_reports_ok_false(session, rel):
    """The invariant: never ``ok: true`` with a miss. Asking for a height the body can reach must LAND there;
    asking for one it cannot must fail loudly and leave the robot untouched."""
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.edit_operators import _standing_height

    rid = session.put_robot(_imported(rel), robot_id="h_target")
    start = _standing_height(session.get_robot(rid))

    for target in (round(start * 1.25, 3), 4.9):          # one reachable, one physically out of range
        before = _standing_height(session.get_robot(rid))
        res = call_tool("edit_robot",
                        {"robot_id": rid, "ops": [{"op": "set_height", "args": {"target_m": target}}]})["result"]
        after = _standing_height(session.get_robot(rid))
        if res.get("ok"):
            assert abs(after - target) <= max(0.02 * target, 0.005) + 1e-6, (
                f"{rel}: edit_robot returned ok=True for a {target} m request but the body stands at "
                f"{after:.3f} m -- a {after - target:+.3f} m miss reported as success")
        else:
            assert abs(after - before) < 1e-6, (
                f"{rel}: the edit failed but still mutated the body ({before:.4f} -> {after:.4f} m)")
            assert res.get("error"), "a failed edit must say why"


def test_the_reported_miss_is_the_measured_miss(session):
    """The diff has to carry BOTH numbers, so 'did it do what I asked' is answerable without re-measuring."""
    from virturoid.services.edit_operators import _standing_height, apply_ops

    gene = _imported("unitree_g1/g1.xml")
    target = round(_standing_height(gene) * 1.25, 3)
    new, diffs = apply_ops(copy.deepcopy(gene), [{"op": "set_height", "args": {"target_m": target}}])
    d = diffs[0]
    assert d["op"] == "set_height" and d["target_m"] == pytest.approx(target)
    assert d["achieved_m"] == pytest.approx(_standing_height(new), abs=1e-3)
    assert d["miss_m"] == pytest.approx(d["achieved_m"] - target, abs=1e-3)


def test_an_edit_that_detaches_links_is_rejected_not_reported_as_success():
    """The connectivity gate. ``evaluate_structural_assertions`` was already a HARD gate on ``submit_design``
    but the amend path never called it, so an edit could hand back floating chunks with ``ok: true``."""
    from virturoid.services.ai_native_tools import _newly_detached_seams

    gene = _imported("unitree_go2/go2.xml")
    torn = copy.deepcopy(gene)
    moved = 0
    for s in torn.segments:                                # rip every child a clear half-metre off its parent
        if s.parent is not None:
            mo = list(getattr(s, "mount_offset", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
            s.mount_offset = (mo[0], mo[1], mo[2] + 0.5)
            moved += 1
    assert moved, "fixture built no detachable seams"

    assert _newly_detached_seams(gene, gene) == [], "an unchanged body must not report new detachments"
    seams = _newly_detached_seams(gene, torn)
    assert seams, "a body torn apart by 0.5 m per seam was not detected as detached"
    assert all(s["gap_m"] > 0.006 for s in seams)


def test_edit_robot_refuses_to_commit_a_dismembered_body(session):
    """End-to-end through the tool: the gate has to be WIRED, not merely present."""
    from virturoid.services import ai_native_tools as AN
    from virturoid.services.agent_tools import call_tool

    rid = session.put_robot(_imported("unitree_go2/go2.xml"), robot_id="tear")
    before = _mass(session.get_robot(rid))
    real = AN._newly_detached_seams
    AN._newly_detached_seams = lambda *_a, **_k: [{"seam": "base->FL_hip", "gap_m": 0.42, "was_m": 0.0}]
    try:
        res = call_tool("edit_robot", {"robot_id": rid,
                                       "ops": [{"op": "scale_group",
                                                "args": {"group": "legs", "dims": "length",
                                                         "factor": 1.1}}]})["result"]
    finally:
        AN._newly_detached_seams = real
    assert res.get("ok") is False, "a dismembering edit was committed as a success"
    assert "detach" in str(res.get("error", "")).lower()
    assert res.get("detached_seams")
    assert _mass(session.get_robot(rid)) == pytest.approx(before), "the rejected edit still mutated the body"


def test_cloning_a_gene_does_not_alias_its_geometry():
    """``RobotGene.to_dict`` passes ``geometry`` through by reference, and ``scale_group`` mutates geometry in
    place -- so an edit leaked back into the caller's gene and repeated probes compounded on each other."""
    from virturoid.services.edit_operators import _clone, scale_group

    gene = _imported("unitree_go2/go2.xml")
    geo_before = copy.deepcopy([getattr(s, "geometry", None) for s in gene.segments])
    lens_before = [float(s.length_m) for s in gene.segments]
    scale_group(gene, group="legs", dims="both", factor=1.4)
    assert [float(s.length_m) for s in gene.segments] == lens_before, "scale_group mutated the caller's lengths"
    assert [getattr(s, "geometry", None) for s in gene.segments] == geo_before, (
        "scale_group mutated the caller's VISUAL geometry through a shared dict")

    twin = _clone(gene)
    for s in twin.segments:
        if isinstance(s.geometry, dict):
            s.geometry["__probe__"] = 1
    assert not any("__probe__" in (s.geometry or {}) for s in gene.segments
                   if isinstance(s.geometry, dict)), "_clone aliases the original's geometry dicts"


# ------------------------------------------------- B. what you see IS what gets verified
@pytest.mark.parametrize("rel", ["unitree_go2/go2.xml", "unitree_g1/g1.xml",
                                 "boston_dynamics_spot/spot.xml"])
def test_the_rendered_body_and_the_simulated_body_share_one_physics(rel):
    """``render_view`` draws ``gene_to_meshed_mjcf``; the verdict runs ``compile_gene_to_mjcf``. The geom
    counts differ (cosmetic cans and collars on one side, surface meshes on the other) but every geom that
    COLLIDES, every mass and the spawn height must be identical -- otherwise the picture is not the robot."""
    from virturoid.services.ai_native_tools import render_parity

    p = render_parity(_imported(rel))
    assert p.get("same_physics") is True, f"{rel}: render/sim parity broken -> {p}"
    assert p["colliders_identical"] and p["mass_identical"] and p["spawn_z_identical"]
    assert p["n_colliders"][0] == p["n_colliders"][1] > 0


@pytest.mark.parametrize("prompt", ["a robot dog", "a hexapod robot"])
def test_render_sim_parity_holds_for_composed_bodies_too(prompt):
    from virturoid.services.ai_native_tools import render_parity
    from virturoid.services.morphology_composer import compose_robot

    p = render_parity(compose_robot(prompt, llm=None))
    assert p.get("same_physics") is True, f"{prompt!r}: render/sim parity broken -> {p}"


def test_render_view_discloses_which_body_it_drew_and_can_show_the_simulated_one(session):
    """Disclosure, not trust: the payload states the parity, and ``view='collision'`` renders the exact
    bodies the verdict is computed on so a customer can LOOK at them."""
    from virturoid.services.agent_tools import call_tool

    rid = session.put_robot(_imported("unitree_go2/go2.xml"), robot_id="parity_view")
    vis = call_tool("render_view", {"robot_id": rid})["result"]
    assert vis["view"] == "visual"
    assert vis["parity"]["same_physics"] is True, vis["parity"]
    assert "collision" in vis["parity"]["note"]

    coll = call_tool("render_view", {"robot_id": rid, "view": "collision"})["result"]
    assert coll["ok"] and coll["view"] == "collision"
    assert coll["artifacts"] and Path(coll["artifacts"][0]).is_file()
    assert Path(coll["artifacts"][0]) != Path(vis["artifacts"][0]), "both views wrote the same file"

    bad = call_tool("render_view", {"robot_id": rid, "view": "xray"})["result"]
    assert bad["ok"] is False and "xray" in bad["error"]


# ------------------------------------------------------- C. export ships the verified body
@pytest.mark.parametrize("rel", ["unitree_go2/go2.xml", "unitree_g1/g1.xml"])
def test_export_grounding_preserves_an_imported_robots_real_masses(rel):
    """An imported robot's per-link masses are the MANUFACTURER'S. Re-deriving them from a primitive-volume
    estimate replaces the customer's robot with our guess at it (Go2: 15.206 -> 13.235 kg, base 6.921 ->
    6.107)."""
    from virturoid.services.gene_build import ground_and_repair, grounding_config

    gene = _imported(rel)
    assert grounding_config(gene)["preserve_mass"] is True, "an imported body is not marked authoritative"
    before = {s.name: float(s.mass_kg) for s in gene.segments}
    exported = copy.deepcopy(gene)
    ground_and_repair(exported)
    after = {s.name: float(s.mass_kg) for s in exported.segments}
    assert sum(after.values()) == pytest.approx(sum(before.values()), abs=0.01), (
        f"{rel}: export re-derived the mass {sum(before.values()):.3f} -> {sum(after.values()):.3f} kg")
    worst = max(((abs(after[n] - before[n]), n) for n in before), default=(0.0, ""))
    assert worst[0] < 0.01, f"{rel}: link {worst[1]} changed by {worst[0]:.3f} kg on the export path"


@pytest.mark.parametrize("material", ["steel", "aluminum"])
def test_export_grounding_does_not_switch_an_amended_bodys_material(session, material):
    """The reported 27.362 -> 19.151 kg shape. ``ground_and_repair`` hardcoded carbon fibre (1600 kg/m3) while
    the amend path grounds at the body's own material, so every link silently lost mass at the door."""
    from virturoid.services.agent_tools import call_tool
    from virturoid.services.gene_build import ground_and_repair

    rid = call_tool("create_robot", {"prompt": "a robot dog"})["result"]["robot_id"]
    r = call_tool("edit_robot", {"robot_id": rid,
                                 "ops": [{"op": "set_material",
                                          "args": {"group": "all", "material": material}}]})["result"]
    assert r.get("ok"), r
    held = _mass(session.get_robot(rid))
    exported = copy.deepcopy(session.get_robot(rid))
    ground_and_repair(exported)
    assert _mass(exported) == pytest.approx(held, abs=0.05), (
        f"a {material} body was verified at {held} kg and exported at {_mass(exported)} kg "
        f"({_mass(exported) / max(held, 1e-9):.3f}x) -- the verdict was earned on a different robot")


def test_the_exported_package_carries_the_body_that_was_verified(session):
    """End-to-end through the real export door, on a real customer robot."""
    from virturoid.services.agent_tools import call_tool

    gene = _imported("unitree_go2/go2.xml")
    rid = session.put_robot(gene, robot_id="ship_go2", prompt="a quadruped robot dog")
    held = _mass(session.get_robot(rid))

    res = call_tool("export_held", {"robot_id": rid,
                                    "formats": ["mjcf", "urdf", "bom", "certificate"]})["result"]
    assert res.get("ok"), res
    assert res["body_parity"]["same_as_verified"] is True, res["body_parity"]
    assert not res.get("warnings"), res.get("warnings")

    urdf = Path(res["out_dir"]) / "robot" / "robot.urdf"
    assert urdf.is_file(), f"no URDF written under {res['out_dir']}"
    shipped = round(sum(float(m.get("value", 0.0)) for m in ET.parse(urdf).getroot().iter("mass")), 3)
    assert shipped == pytest.approx(held, abs=0.05), (
        f"held/verified at {held} kg but the shipped URDF weighs {shipped} kg")


def test_one_package_ships_exactly_one_motor_selection(session):
    """``bom.json`` said 8x AK80-64 + 4x XM540-W270-T while the certificate's ``margins.bom`` said 8x AK10-9 +
    4x XM430-W350-T -- ``build_bom`` re-applied a 1.3x margin to ``actuator_torque_nm``, which grounding had
    already overwritten with the SELECTED motor's peak. A buyer got contradictory part numbers."""
    from virturoid.services.agent_tools import call_tool

    rid = session.put_robot(_imported("unitree_go2/go2.xml"), robot_id="bom_go2",
                            prompt="a quadruped robot dog")
    res = call_tool("export_held", {"robot_id": rid, "formats": ["bom", "certificate"]})["result"]
    bom_p = Path(res["out_dir"]) / "bom.json"
    cert_p = Path(res["out_dir"]) / "verification_certificate.json"
    if not (bom_p.is_file() and cert_p.is_file()):
        pytest.skip(f"export produced no bom/certificate to compare: {res.get('artifacts')}")

    bom = json.loads(bom_p.read_text(encoding="utf-8"))
    cert = json.loads(cert_p.read_text(encoding="utf-8"))
    cert_bom = ((cert.get("margins") or {}).get("bom")) or []
    if not cert_bom:
        pytest.skip("certificate carried no margins.bom on this body")

    from_bom = sorted((str(li["part"]), int(li["qty"]))
                      for li in bom.get("lines", []) if li.get("category") == "actuator")
    from_cert = sorted((str(b.get("part")), int(b.get("qty", 0))) for b in cert_bom)
    assert from_bom == from_cert, (
        f"one package, two motor selections: bom.json says {from_bom} while the certificate's margins.bom "
        f"says {from_cert}")


def test_the_certificate_only_claims_deploy_equals_measure_when_it_is_true():
    """The claim used to be a hardcoded string printed unconditionally."""
    from virturoid.services.grounded_physics import fingerprint_delta, physical_fingerprint
    from virturoid.services.verdict_certificate import build_certificate

    gene = _imported("unitree_go2/go2.xml")
    verdict = {"verdict": "CREDIBLE WALK", "credible": True, "forward_m": 0.9, "survived": True,
               "gait_source": "tuned_for_this_body"}
    # The CONTROLLER half of the same claim. "The SAME rollout that deploys" is about a rollout, and a rollout is
    # a body and a controller; passing only the body left the other half asserted rather than checked. The
    # OPERATING POINT counts as part of the controller — measured, the export's own gait search lands somewhere
    # else than the point the deploy path reads, and matching only the law would call that deploy==measure.
    _op = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5}
    gene.metadata = {**(getattr(gene, "metadata", None) or {}), "gait_params": _op}
    shipped = {"policy_type": "crawl_wave_gait", "parameters_file": "software/control_program.json",
               "program_fingerprint": "cafe1234", "sim_forward_m": 0.79, "sim_verdict": "CREDIBLE WALK",
               "frequency_hz": _op["freq"], "hip_amp": _op["hip_amp"], "knee_amp": _op["knee_amp"],
               "kp": _op["kp"], "kd": _op["kd"]}

    same = fingerprint_delta(physical_fingerprint(gene), physical_fingerprint(gene))
    cert = build_certificate(gene, verdict, body_parity=same, shipped_controller=shipped)
    assert cert["deploy_is_measure"] is True
    assert "deploy==measure" in cert["verified_with"]

    lighter = copy.deepcopy(gene)
    for s in lighter.segments:
        s.mass_kg = round(float(s.mass_kg) * 0.7, 3)
    drifted = fingerprint_delta(physical_fingerprint(gene), physical_fingerprint(lighter))
    assert drifted["same"] is False
    cert2 = build_certificate(lighter, verdict, body_parity=drifted, shipped_controller=shipped)
    assert cert2["deploy_is_measure"] is False
    assert "deploy==measure" not in cert2["verified_with"], (
        "the certificate still claims deploy==measure while shipping a different body")
    assert "DEPLOY != MEASURE" in cert2["verified_with"]

    # ...and the mirror case: the right body, the WRONG controller. This is the one a real Go2 export hit.
    cert3 = build_certificate(gene, verdict, body_parity=same,
                              shipped_controller={**shipped, "policy_type": "trot_cpg_gait",
                                                  "sim_forward_m": 0.311, "sim_verdict": "CROUCH"})
    assert cert3["deploy_is_measure"] is False
    assert cert3["deploy_is_measure_parts"] == {"same_body": True, "same_controller": False}
    assert "deploy==measure" not in cert3["verified_with"], (
        "the certificate claims deploy==measure while shipping a controller it never measured")
