"""Five surfaces that stated something their own data did not support.

All five were found the same way -- through ``agent_tools.call_tool`` on a real Menagerie Unitree Go2, reading
what a customer reads -- and none of them was a physics bug. In every case the number was right and the
sentence around it was wrong, which is the failure mode a unit test on the computation cannot see:

  1. ``probe_robot.torque`` cannot answer "can it carry the arm?" and did not say so. Mounting 2.207 kg on the
     Go2's base left all 12 leg joints BYTE-IDENTICAL, because ``distal_mass_kg`` walks DOWN the kinematic
     tree. ``verify_robot`` already carried the caveat; ``probe_robot`` -- the tool reached for first -- did not.
  2. ``scope_amend`` on a purely ADDITIVE ``add_limb`` said ``touches: "a new chain plus its mount"`` and in
     the same object listed all 13 existing parts as ``editable`` with ``preserved: []``.
  3. ``get_robot.end_effector: "none"`` on a robot carrying a part named ``arm_gripper``.
  4. ``get_robot.design_source: "unknown"`` for a robot ingested from a named file.
  5. ``probe_robot.torque`` reported ``arm_0 margin: 15041983.01``. Fifteen million is a division by something
     near zero, not a measurement.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="measuring a body needs MuJoCo")


@pytest.fixture(scope="module")
def quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four legged robot dog", llm=None)


def _armed(gene, **kw):
    """The same body with a 3-link chain grown on its root -- the amend that produced gaps 1, 3 and 5."""
    from virturoid.services.edit_operators import add_limb
    root = next(s.name for s in gene.segments if not s.parent)
    g, _diff = add_limb(gene, parent=root, segments=3, name="arm", end_effector="gripper", **kw)
    return g, root


# ------------------------------------------------------------------ 1: the caveat that lived on one surface
def test_torque_says_what_it_cannot_answer(quad):
    """`verify_robot` carries "Ground-reaction load is not in this number". The same quantity on the tool a
    customer reaches for FIRST carried nothing at all."""
    from virturoid.services.robot_probe import probe
    rep = probe(quad, {"fields": ["torque"]})
    note = rep["torque_note"]
    assert "distal" in note["what_this_measures"].lower()
    cannot = note["what_it_cannot_answer"].lower()
    assert "carry" in cannot and "proximal" in cannot and "ground-reaction" in cannot
    assert "verify_robot" in note["ask_that_instead_with"]
    assert note["posed_from"] == rep["posed_from"]


def test_mass_added_proximally_moves_no_distal_joint_and_the_note_owns_it(quad):
    """The measurement behind gap 1, kept as a test so the caveat can never drift away from the fact.

    The load is put on the ROOT LINK DIRECTLY rather than through `add_limb`, because `add_limb` also
    re-grounds the body and every link's mass moves -- which would make this test pass or fail for reasons
    unrelated to the load path. Adding mass to the root is the cleanest statement of the claim: 2.2 kg
    bolted to the trunk is proximal to every leg joint, so none of them sees it."""
    from virturoid.services.robot_probe import probe
    before = probe(quad, {"fields": ["torque", "mass"]})

    loaded = type(quad).from_dict(quad.to_dict())
    root = next(s for s in loaded.segments if not s.parent)
    root.mass_kg = float(root.mass_kg) + 2.2
    after = probe(loaded, {"fields": ["torque", "mass"]})

    assert after["mass"]["total_kg"] - before["mass"]["total_kg"] == pytest.approx(2.2, abs=1e-3), (
        "the payload has to be really on the robot for this to mean anything")

    shared = [k for k in before["torque"] if k in after["torque"]]
    assert shared, "expected joints to compare"
    moved = [k for k in shared if after["torque"][k] != before["torque"][k]]
    assert moved == [], (
        f"if a proximal load ever DOES reach these numbers, the note has to change with it (moved: {moved})")
    assert "proximal" in after["torque_note"]["what_it_cannot_answer"].lower()
    assert "2.2 kg" in after["torque_note"]["what_it_cannot_answer"]


# ------------------------------------------------------------------ 5: fifteen million is not a margin
def test_a_gravity_neutral_joint_reports_no_margin_and_says_why(quad):
    """A chain hanging straight down carries no gravity torque about its own axis -- correct physics, and
    `rated / ~0` printed as `margin: 707771.93` reads as "enormously safe" when the truth is "unanswerable
    in this pose". Measured on the real Go2: arm_0 15041983.01, arm_1 707771.93, arm_2 1663922.19."""
    from virturoid.services.robot_probe import probe
    armed, _root = _armed(quad)
    rows = probe(armed, {"fields": ["torque"]})["torque"]
    arm = {k: v for k, v in rows.items() if k.startswith("arm")}
    assert arm, "the amend should have added actuated joints"

    neutral = [k for k, v in arm.items() if v["static_hold_nm"] < 1e-3]
    assert neutral, "a straight-down chain should have at least one gravity-neutral joint"
    for k in neutral:
        assert arm[k]["margin"] is None, f"{k} still reports {arm[k]['margin']} against ~0 N.m"
        why = arm[k]["margin_omitted_because"]
        assert "gravity-neutral" in why and "not a margin" in why
        assert arm[k]["rated_nm"] is not None, "the rating is still reported — only the ratio is withheld"

    # ...and the guard is NARROW: a joint under real load still gets its number.
    loaded = {k: v for k, v in rows.items()
              if v["rated_nm"] and v["static_hold_nm"] > 1e-2 and not k.startswith("arm")}
    assert loaded, "expected genuinely loaded joints on a quadruped"
    for k, v in loaded.items():
        assert isinstance(v["margin"], float), f"{k} lost a margin it should still have"
        assert v["margin"] == pytest.approx(v["rated_nm"] / v["static_hold_nm"], rel=0.02)
        assert "margin_omitted_because" not in v


# ------------------------------------------------------------------ 2: prose and lists disagreeing
def _as_imported(gene):
    """The same body, labelled the way an ingested customer robot is — masses are the manufacturer's."""
    g = type(gene).from_dict(gene.to_dict())
    g.metadata = {**(g.metadata or {}), "mass_source": "source_model"}
    return g


def test_an_additive_amend_preserves_what_it_says_it_preserves(quad):
    """`touches: "a new chain plus its mount"` alongside `editable: [all 13 parts]`, `preserved: []`. A
    customer reading that would conclude bolting an arm on reshapes their whole robot.

    Measured on the mass-preserved regime the Go2 is in: `verify_preserved` finds 12 of 12 existing links
    untouched after one `add_limb`, mount included."""
    from virturoid.services.change_impact import scope
    imported = _as_imported(quad)
    root = next(s.name for s in imported.segments if not s.parent)
    sc = scope(imported, [{"op": "add_limb", "args": {"parent": root, "segments": 3, "name": "arm"}}])

    assert sc["editable"] == [root], sc["editable"]
    others = sorted(s.name for s in imported.segments if s.name != root)
    assert sc["preserved"] == others
    op = sc["per_op"][0]
    assert op["additive"] is True and op["mount"] == [root] and op["existing_parts_reshaped"] == []
    assert "only ADDS" in sc["scope_note"]
    # the rechecks are NOT softened -- a robot that walked may not walk carrying this
    for k in ("gait", "stability", "torque", "mass"):
        assert k in sc["invalidates"]


def test_the_preserved_promise_names_the_fields_it_does_not_cover(quad):
    """The trap in fixing gap 2: "additive" does NOT mean "no field on any other part moved".

    Measured with `verify_preserved` after one `add_limb` -- geometry and placement held every time, mass and
    actuator rating did not, and WHICH of them moves depends on the body's grounding state, not on anything
    `scope` can see beforehand:

        real imported Go2, already ground through ingest   0 of 12 existing links moved at all
        freshly composed dog, masses derived by us         19 of 19 moved mass_kg + actuator_torque_nm
        the same dog labelled mass_source='source_model'   19 of 19 moved actuator_torque_nm (5.64 -> 18.0)

    An unqualified `preserved` would be a new over-claim in place of the old one, so the word states its own
    scope."""
    from virturoid.services.change_impact import scope, verify_preserved
    root = next(s.name for s in quad.segments if not s.parent)
    sc = scope(quad, [{"op": "add_limb", "args": {"parent": root, "segments": 3, "name": "arm"}}])

    assert sc["preserved_covers"] == "shape, size, placement, joint type and joint limits"
    assert "mass_kg" in sc["preserved_does_not_cover"]
    assert "actuator_torque_nm" in sc["preserved_does_not_cover"]
    assert "verify_preserved" in sc["preserved_does_not_cover"]

    # ...and what it DOES cover is checked against the edit, on both regimes, field by field.
    for label, body in (("derived", quad), ("source_model", _as_imported(quad))):
        armed, _ = _armed(body)
        got = verify_preserved(body, armed, preserved=sc["preserved"])
        moved = {f for d in got["changed"].values() for f in d}
        assert moved <= {"mass_kg", "actuator_torque_nm"}, (
            f"[{label}] preserved_covers claims geometry holds, and {moved - {'mass_kg', 'actuator_torque_nm'}} "
            f"moved")
        assert any(n.startswith("arm") for n in got["added"])


def test_a_reshaping_amend_still_takes_everything_below_it(quad):
    """The additive carve-out must not leak into the ops that genuinely propagate down a chain."""
    from virturoid.services.change_impact import scope
    mid = next(s.name for s in quad.segments if s.name.endswith("_1"))
    sc = scope(quad, [{"op": "scale_group", "args": {"group": mid, "factor": 1.3}}])
    assert not sc["per_op"][0].get("additive")
    below = [s.name for s in quad.segments if s.parent == mid]
    for n in below:
        assert n in sc["editable"], f"{n} hangs below {mid} and still moves with it"


def test_end_effector_reports_what_the_body_actually_carries(quad):
    """`end_effector_type` is a single declared field and nothing keeps it in step with the parts list."""
    from virturoid.services.ai_native_tools import _end_effector_report
    armed, _root = _armed(quad)
    armed.end_effector_type = "none"
    rep = _end_effector_report(armed)

    assert any(f["part"].endswith("_gripper") for f in rep["end_effectors_on_body"]), rep
    assert rep["end_effector"] == "gripper", rep
    assert "measured from the body" in rep["end_effector_source"]
    assert "not updated" in rep["end_effector_source"]


def test_a_declared_end_effector_is_not_overwritten_by_an_inference(quad):
    """Reporting the truth must not mean silently replacing what the gene declares."""
    from virturoid.services.ai_native_tools import _end_effector_report
    armed, _root = _armed(quad)
    armed.end_effector_type = "suction"
    rep = _end_effector_report(armed)
    assert rep["end_effector"] == "suction"
    assert rep["end_effector_source"].startswith("declared on the gene")


def test_a_flagged_part_that_names_no_tool_does_not_invent_one(quad):
    """The imported Go2 flags `RR_calf` as its end effector. Answering "its kind is 'none' because the gene
    says 'none'" is the circular non-answer this report exists to remove -- unknown is reported as unknown."""
    from virturoid.services.ai_native_tools import _end_effector_report
    g = type(quad).from_dict(quad.to_dict())
    g.end_effector_type = "none"
    for s in g.segments:
        s.is_end_effector = False
    leaf = next(s for s in reversed(g.segments) if not any(x.parent == s.name for x in g.segments))
    leaf.is_end_effector = True

    rep = _end_effector_report(g)
    assert rep["end_effector"] == "none", "nothing on this body names a tool"
    flagged = [f for f in rep["end_effectors_on_body"] if f["evidence"] == "flagged"]
    assert flagged and flagged[0]["kind"] is None, flagged
    assert "names no tool type" in rep["end_effector_source"]


# ------------------------------------------------------------------ 4: "unknown" for a named file
def test_an_imported_robot_says_it_was_imported(tmp_path, monkeypatch):
    """`design_source` defaulted to "unknown" and the import path never overwrote it -- so the one fact about
    provenance we most certainly had was reported as the one thing we did not know."""
    from virturoid.services.agent_tools import call_tool
    project = tmp_path / "customer"
    project.mkdir()
    (project / "robot.xml").write_text("""<mujoco model="probe">
  <worldbody><body name="trunk" pos="0 0 0.3"><freejoint/>
    <geom name="t" type="box" size="0.1 0.05 0.03" mass="1"/>
    <body name="leg" pos="0.08 0 0"><joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom name="l" type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.3"/></body>
  </body></worldbody>
  <actuator><motor joint="hip" ctrlrange="-1 1" gear="10"/></actuator></mujoco>""", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    rid = call_tool("ingest_project", {"project_path": str(project)})["result"]["robot_id"]
    got = call_tool("get_robot", {"robot_id": rid})["result"]
    assert got["design_source"] == "imported", got["design_source"]


def test_the_stamp_cannot_blind_the_substitution_gate():
    """`_ingest_project` reads `design_source` to catch a COMPOSED body wearing a "faithful" lane label.
    Stamping "imported" on a body that has no `imported_from` would have disabled exactly that gate."""
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a four legged robot dog", llm=None)
    assert not str((getattr(g, "metadata", None) or {}).get("imported_from") or ""), (
        "a composed body must carry no imported_from — the stamp is keyed on it")
    assert not str(getattr(g, "design_source", "")).startswith("imported")
