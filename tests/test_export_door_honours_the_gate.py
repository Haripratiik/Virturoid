"""The export door must honour the simulability gate that commit 9fbdcdc promised it did.

``robot_import`` and ``robot_import_report`` both tell the customer that when the editable twin cannot be
stepped, "*no verdict, certificate, BOM, spec sheet or calibration number can be produced from this twin*".
``export_held`` never read the flag. MEASURED through ``agent_tools.call_tool`` on MuJoCo Menagerie's
``flybody`` — the one package in 63 whose twin genuinely does not compile — the envelope came back ``ok: True``
and the package contained ``bom.json``: **66 Dynamixel actuators, $5,598.00, 5.146 kg, 213.6 W**, closing with
"verify exact specs against the live datasheet before procurement", with no occurrence of "simulab",
"unverified", "warning" or "cannot" anywhere in the file. The body those motors were sized for weighs
**0.000985 kg**. A customer could procure from that.

WHERE THE LINE IS, and what these tests pin on BOTH sides of it:

  * ``bom`` / ``spec`` / ``certificate`` assert a number about physics that was never simulated → refused.
  * ``mjcf`` / ``urdf`` / ``ros2`` / ``cad`` / ``usd`` / ``isaac_lab`` transcribe the customer's OWN geometry →
    still written, still valid XML, and carrying the refusal INSIDE the file, because an artifact that leaves
    the package leaves the envelope behind.
  * a SIMULABLE body is completely unaffected — the gate must not become a second way to lose a good export.

Plus the two smaller untruths found in the artifacts that were otherwise honest: a certificate claiming
``deploy_is_measure``/``body_parity.same`` for a rollout that never ran, and a spec sheet saying "no
verification certificate in this package" while its own ``sources`` block said there was one.
"""
from __future__ import annotations

import importlib.util
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")

_GEOMETRY_ASKED = ["mjcf", "urdf", "ros2", "cad"]
_PHYSICS_ASKED = ["bom", "spec", "certificate"]


def _menagerie(pkg: str, model: str) -> str:
    p = os.path.join(_MENAGERIE, pkg, model)
    if not os.path.exists(p):
        pytest.skip(f"MuJoCo Menagerie not cached at {p}")
    return p


def _export(gene, label: str, formats) -> dict:
    """The REAL agent door, not the handler — the envelope is half of what was wrong.

    No ``out_dir``: ``safe_build_path`` (H2) confines every agent-supplied write path under ``build/`` and
    silently falls back to the default on escape, so a ``tmp_path`` here would be ignored and the assertions
    would be reading a directory nothing was written to. The package lands under ``build/agent_exports/<rid>``,
    and ``rid`` is unique per call.
    """
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    rid = S.put_robot(gene, prompt=label, label=label)
    try:
        return call_tool("export_held", {"robot_id": rid, "formats": list(formats)})
    finally:
        S.forget_robot(rid)


@pytest.fixture(scope="module")
def flybody_export():
    """flybody exported for real, once, asking for both sides of the line."""
    if not _MUJOCO:
        pytest.skip("the gate is a MuJoCo probe")
    from virturoid.services.robot_import import import_robot
    imp = import_robot(_menagerie("flybody", "fruitfly.xml"), robot_id="flybody_gate")
    assert imp.get("simulable") is False, (
        "flybody is the live unsimulable test case; if it now imports as simulable this test needs a new one")
    env = _export(imp["gene"], "fruit fly", _GEOMETRY_ASKED + _PHYSICS_ASKED)
    res = env.get("result") or {}
    return env, res, Path(res["out_dir"])


# ------------------------------------------------------------------ the refused half: nothing asserted
def test_no_bom_no_spec_sheet_no_certificate_is_written_for_an_unsimulable_twin(flybody_export):
    _env, _res, out = flybody_export
    for name in ("bom.json", "bom.md", "verification_certificate.json",
                 "reports/spec_sheet.json", "reports/spec_sheet.md"):
        assert not (out / name).exists(), (
            f"{name} was written for a twin that cannot be stepped — this is the file a customer procures from")


def test_no_price_mass_or_power_figure_survives_anywhere_in_the_package(flybody_export):
    """The specific measured payload: $5,598.00 / 5.146 kg / 213.6 W, sized for a 0.000985 kg body."""
    _env, _res, out = flybody_export
    offenders = []
    for f in out.rglob("*"):
        if f.is_file() and f.suffix.lower() in (".json", ".md", ".xml", ".urdf", ".usda", ".yaml", ".py"):
            text = f.read_text(encoding="utf-8", errors="replace")
            offenders += [f"{f.relative_to(out)}:{n}" for n in ("5598", "5,598", "5.146", "213.6") if n in text]
    assert not offenders, f"a confident procurement number survived the refusal: {offenders}"


def test_the_envelope_refuses_rather_than_reporting_success(flybody_export):
    env, res, out = flybody_export
    assert env["ok"] is False, "a scripted caller whose next line is `if ok: order_parts()` must stop"
    assert res["simulable"] is False
    assert set(res["refused"]["refused"]) == set(_PHYSICS_ASKED)
    assert "NOT SIMULABLE" in env["error"] and "mjMINVAL" in env["error"], env["error"]
    for fmt in _PHYSICS_ASKED:                       # the withheld artifacts are NAMED, not silently missing
        assert Path(res["artifacts"][f"{fmt}_refused"]).exists()


def test_a_refusal_is_never_parked_under_an_evidence_filename(flybody_export):
    """``spec_sheet`` globs for ``bom.json``/``verification_certificate.json`` and reports
    ``sources.verification_certificate: true`` for anything it finds there. A refusal written under an evidence
    filename becomes evidence one directory later, so the refusal documents get their own names."""
    _env, _res, out = flybody_export
    from virturoid.services.spec_sheet import build_spec_sheet
    for name in ("bom.json", "verification_certificate.json", "spec_sheet.json"):
        assert not list(out.rglob(name)), f"a refusal is sitting at {name}"
    spec = build_spec_sheet(out)
    assert spec.get("refused") is True
    assert spec.get("robot_class") is None and spec.get("dof") is None, (
        "the sheet still described a robot in a package that declares itself unsimulable")


# ------------------------------------------------------------------ the shipped half: the customer's geometry
def test_the_customers_own_geometry_still_ships(flybody_export):
    """Refusing everything would withhold the customer's own data at the moment it is most useful — the MJCF
    and the URDF are the files an engineer needs to FIND the degenerate link."""
    _env, res, out = flybody_export
    assert (out / "robot.xml").exists() and (out / "robot" / "robot.urdf").exists()
    assert set(res["refused"]["still_shipped"]) >= {"mjcf", "urdf", "cad", "ros2"}
    assert res["artifacts"].get("mjcf") and res["artifacts"].get("urdf")


def test_every_shipped_artifact_carries_the_refusal_and_still_parses(flybody_export):
    """IN THE ARTIFACT, not only in the envelope: a URDF handed to a colleague has left the envelope behind.
    And the stamp must not break the file — a comment containing a MuJoCo error string can carry ``--``,
    which is illegal inside an XML comment."""
    _env, _res, out = flybody_export
    shipped = [out / "robot.xml", out / "robot" / "robot.urdf", *sorted((out / "export").rglob("*.urdf"))]
    for f in shipped:
        assert f.exists(), f
        text = f.read_text(encoding="utf-8")
        assert "NOT SIMULABLE" in text[:2000], f"{f.name} ships with no refusal in it"
        assert "Do not size motors" in text[:2000], f"{f.name} does not say what not to do"
        ET.parse(f)                                   # raises if the stamp produced illegal XML
    assert (out / "NOT_SIMULABLE.md").exists() and (out / "export_refusal.json").exists()
    assert (out / "cad" / "NOT_SIMULABLE.md").exists()


def test_the_stamp_survives_a_reason_containing_a_double_hyphen():
    """``--`` may not appear inside an XML comment, and a real compile error can contain one."""
    from virturoid.services import export_gate
    comment = export_gate._xml_comment("boom -- because of --flag-- and a trailing -")
    root = ET.fromstring("<r>" + comment + "</r>")    # raises XML syntax error if the escaping is wrong
    assert root.tag == "r"
    assert "because of" in comment and "flag" in comment, "the reason must survive the escaping"


# ------------------------------------------------------------------ the gate must not cost a good export
@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_a_simulable_body_still_gets_the_whole_confident_package():
    from virturoid.services.robot_import import import_robot
    gene = import_robot(_menagerie("unitree_go2", "go2.xml"), robot_id="go2_gate")["gene"]
    env = _export(gene, "a quadruped", ["mjcf", "bom", "spec", "certificate"])
    res = env["result"]
    assert env["ok"] is True and res["simulable"] is True and "refused" not in res
    out = Path(res["out_dir"])
    assert (out / "bom.json").exists() and (out / "verification_certificate.json").exists()
    assert (out / "reports" / "spec_sheet.json").exists()
    assert not (out / "NOT_SIMULABLE.md").exists()
    assert json.loads((out / "bom.json").read_text(encoding="utf-8"))["totals"]["price_usd"] > 0
    assert not (out / "robot.xml").read_text(encoding="utf-8").startswith("<!--")


def test_the_gate_does_not_run_at_all_when_no_physics_format_was_asked_for(monkeypatch):
    """A geometry-only export pays nothing for the gate — the probe is a settle+excitation rollout and there is
    nothing for it to protect when no requested format asserts a physical number."""
    from virturoid.services import export_gate
    from virturoid.services.morphology_composer import compose_robot
    called = []
    monkeypatch.setattr(export_gate, "check_simulable", lambda g: called.append(g) or {"ok": True})
    env = _export(compose_robot("a small quadruped robot dog"), "dog", ["mjcf"])
    assert env["ok"] is True and not called


# ------------------------------------------------------------------ untruth 2: a verdict that never happened
def test_a_certificate_does_not_claim_deploy_equals_measure_when_no_rollout_ran():
    """MEASURED on flybody: ``verify_robot`` returned ``could not simulate (ValueError)`` with no checks, and
    the certificate answered ``deploy_is_measure: true``, ``body_parity.same: true``, and "the verdict is
    signed by the SAME rollout that deploys". Three claims about a rollout that never ran."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.verdict_certificate import build_certificate
    gene = compose_robot("a small quadruped robot dog")
    same = {"same": True, "n_links_changed": 0, "delta_mass_kg": 0.0, "total_mass_kg": [1.0, 1.0]}

    dead = build_certificate(gene, {"verdict": "could not simulate (ValueError)", "credible": False,
                                    "error": "mass and inertia of moving bodies must be larger than mjMINVAL"},
                             body_parity=same)
    assert dead["rollout_ran"] is False
    assert dead["deploy_is_measure"] is None, "tri-state: 'never measured' is not 'measured on this body'"
    assert dead["body_parity"]["same"] is None
    assert dead["body_parity"]["held_vs_shipped"] == same, "the grounding delta is still reported, correctly"
    # the token is the claim, and it is checked by substring elsewhere in this suite — it must not appear even
    # inside a sentence denying it.
    assert "deploy==measure" not in dead["verified_with"]
    assert "NOTHING WAS MEASURED" in dead["verified_with"]

    # ...and a real rollout is untouched: the claim it was always entitled to still reads exactly as before.
    live = build_certificate(gene, {"verdict": "CREDIBLE (trot, 0.83 m)", "survived": True, "forward_m": 0.83,
                                    "cadence": 2.1}, body_parity=same)
    assert live["rollout_ran"] is True and live["deploy_is_measure"] is True
    assert live["body_parity"] == same and "deploy==measure" in live["verified_with"]


@pytest.mark.skipif(not _MUJOCO, reason="certificate v2 compiles the model")
def test_certificate_v2_carries_the_rollout_flag_through():
    from virturoid.services.certificate_v2 import build_certificate_v2
    from virturoid.services.morphology_composer import compose_robot
    cert = build_certificate_v2(compose_robot("a small quadruped robot dog"),
                                {"verdict": "could not simulate (ValueError)", "error": "boom"},
                                run_dr=False, run_margins=False)
    assert cert["verdict"]["rollout_ran"] is False
    assert cert["verdict"]["deploy_is_measure"] is None


# ------------------------------------------------------------------ untruth 3: the sheet contradicting itself
def test_the_spec_sheet_does_not_deny_a_certificate_it_is_reading():
    """``sources.verification_certificate: true`` and "no verification certificate in this package" shipped in
    the SAME file. The cert was there, VOID, and its model_sanity had no mass because nothing compiled."""
    from virturoid.services.spec_sheet import _mass_breakdown
    totals = {"mass_kg": 5.146}
    facts = {"structure_mass_kg": 0.001}

    absent = _mass_breakdown(totals, facts, {})
    assert "no verification certificate in this package" in absent["note"]

    void = _mass_breakdown(totals, facts, {
        "valid": False, "voided_reason": "model_sanity failed: model did not compile",
        "model_sanity": {"ok": False, "issues": ["model did not compile"]}})
    assert "no verification certificate in this package" not in void["note"], (
        "the sheet is denying the certificate it just read")
    assert "DOES carry a verification certificate" in void["note"]
    assert "VOID" in void["note"] and void["certificate_valid"] is False

    good = _mass_breakdown(totals, facts, {"valid": True, "model_sanity": {"ok": True, "total_mass_kg": 2.0}})
    assert good["as_built_over_simulated"] == 2.57 and "SIMULATED body" in good["note"]
