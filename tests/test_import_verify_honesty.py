"""B2 + #214 (2026-07-24 audit): ingesting a customer robot must (1) keep the customer's OWN geometry as the
held robot, (2) report ITS honest verdict -- never a canonical template's walk reported as the customer's,
(3) offer any walkable template as an explicit opt-in with undo, and (4) classify a fixed-base multi-limb body
as legged so verify uses the right rubric (not the arm rubric).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import + eval need MuJoCo")


def _fixed_base_quad_urdf(tmp) -> str:
    urdf = ('<?xml version="1.0"?>\n<robot name="customer_quad">\n'
            '<link name="trunk"><inertial><mass value="6.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" '
            'ixz="0" iyz="0"/></inertial><visual><geometry><box size="0.5 0.25 0.1"/></geometry></visual></link>\n')
    for nm, x, y in [("FL", 0.2, 0.12), ("FR", 0.2, -0.12), ("HL", -0.2, 0.12), ("HR", -0.2, -0.12)]:
        urdf += (f'<link name="{nm}_leg"><inertial><mass value="0.8"/><inertia ixx="0.01" iyy="0.01" izz="0.01" '
                 f'ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><cylinder radius="0.03" length="0.3"/>'
                 f'</geometry></visual></link>\n'
                 f'<joint name="{nm}_hip" type="revolute"><parent link="trunk"/><child link="{nm}_leg"/>'
                 f'<axis xyz="0 1 0"/><limit lower="-1.2" upper="1.2" effort="20" velocity="3"/>'
                 f'<origin xyz="{x} {y} 0"/></joint>\n')
    urdf += "</robot>"
    p = os.path.join(tmp, "q.urdf")
    open(p, "w").write(urdf)
    return tmp


def test_fixed_base_quad_classifies_legged_not_manipulator():
    """#214: a fixed-base body with >=3 symmetric revolute limbs off the root is legged, not an arm."""
    from virturoid.services.robot_import import import_robot
    from virturoid.services.task_matched_eval import robot_kind
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    g = import_robot(os.path.join(tmp, "q.urdf"))["gene"]
    assert robot_kind(g) == "legged", f"fixed-base quad classified {robot_kind(g)}"
    # a single-chain arm must NOT be caught by the new rule
    from virturoid.services.morphology_composer import compose_robot
    assert robot_kind(compose_robot("a 6-axis robot arm with a gripper", llm=None)) == "manipulator"


def test_ingest_keeps_customer_body_and_reports_its_own_verdict():
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    r = call_tool("ingest_project", {"path": tmp, "description": "my patrol quadruped"}).get("result", {})
    g = S.get_robot(r["robot_id"])
    names = [s.name for s in g.segments]
    assert any(n.endswith("_leg") for n in names), f"customer link names not preserved: {names}"
    assert "torso" not in names and "neck" not in names, "held body is a composed template, not the import"
    iv = r.get("imported_verdict")
    assert iv is not None and "own imported geometry" in iv["body"]
    # verify runs the legged rubric on the customer's body (not the arm rubric)
    v = call_tool("verify_robot", {"robot_id": r["robot_id"], "mode": "quick"})["result"]
    assert v.get("kind") == "legged"


def test_base_link_inertial_survives_import_and_reroot():
    """M3: MuJoCo fuses a URDF's static root link by default, silently dropping its <inertial>. We inject
    <compiler fusestatic="false"> so a declared 2 kg base is still 2 kg after import AND after we bolt on a
    free joint -- the imported+rerooted mass stays within tolerance of the URDF's stated total."""
    import mujoco

    from virturoid.services.model_import import import_model, reroot_free_base

    tmp = tempfile.mkdtemp()
    urdf = '<?xml version="1.0"?>\n<robot name="massquad">\n'
    urdf += ('<link name="trunk"><inertial><mass value="2.0"/><inertia ixx="0.05" iyy="0.05" izz="0.05" '
             'ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><box size="0.4 0.2 0.1"/></geometry>'
             '</collision></link>\n')
    for nm, x, y in [("FL", 0.18, 0.1), ("FR", 0.18, -0.1), ("HL", -0.18, 0.1), ("HR", -0.18, -0.1)]:
        urdf += (f'<link name="{nm}"><inertial><mass value="0.8"/><inertia ixx="0.01" iyy="0.01" izz="0.01" '
                 f'ixy="0" ixz="0" iyz="0"/></inertial><collision><geometry><cylinder radius="0.03" '
                 f'length="0.25"/></geometry></collision></link>\n'
                 f'<joint name="{nm}_j" type="revolute"><parent link="trunk"/><child link="{nm}"/>'
                 f'<axis xyz="0 1 0"/><origin xyz="{x} {y} 0"/>'
                 f'<limit lower="-1" upper="1" effort="20" velocity="5"/></joint>\n')
    urdf += "</robot>"
    p = os.path.join(tmp, "massquad.urdf")
    open(p, "w").write(urdf)
    declared = 2.0 + 4 * 0.8

    imp = import_model(p)
    assert imp["ok"], imp.get("note")
    m = mujoco.MjModel.from_xml_string(imp["mjcf"])
    assert abs(sum(m.body_mass) - declared) / declared < 0.05, (
        f"base inertial lost on import: {sum(m.body_mass):.3f} vs declared {declared}")
    mjcf2, rer = reroot_free_base(imp["mjcf"])
    assert rer
    m2 = mujoco.MjModel.from_xml_string(mjcf2)
    assert abs(sum(m2.body_mass) - declared) / declared < 0.05, (
        f"base inertial lost on reroot: {sum(m2.body_mass):.3f} vs declared {declared}")


def _geom_sig(gene):
    """The operator's own geometry signature -- imported, not restated, so the test cannot drift from it."""
    from virturoid.services.edit_operators import _geometry_signature
    return _geometry_signature(gene)


def _verdict(gene):
    """(distance_m, gait_quality verdict) for a body, from ONE rollout -- the pair the product judges by."""
    from virturoid.services import gait_quality as gq
    from virturoid.services.task_matched_eval import evaluate_robot
    ev = evaluate_robot(gene)
    return float(ev.get("value", 0.0)), gq.classify(ev.get("detail") or {})


def test_walkable_template_is_opt_in_and_undoable():
    """Opt-in, and undoable -- ON A BODY WHERE THE ADOPT ACTUALLY FIRES.

    THIS TEST USED TO PASS VACUOUSLY. It asserted only that ``undo`` restored the original body, and never that
    the adopt had changed anything, so it was green whether or not the template was ever applied -- an undo of a
    no-op restores the original body trivially. MEASURED 2026-08-12 on this file's own ``_fixed_base_quad_urdf``
    fixture: the adopt DOES fire here (5 customer links -> 20 template segments, 3.201 -> 0.771 kg), so the
    "did something" half is now asserted first and the undo half is no longer testing a no-op.
    """
    from virturoid.services import session_state as S
    from virturoid.services.agent_tools import call_tool
    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    r = call_tool("ingest_project", {"path": tmp, "description": "patrol quad"}).get("result", {})
    rid = r["robot_id"]
    before = [s.name for s in S.get_robot(rid).segments]
    ed = call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": "adopt_walkable_template"}]}).get("result", {})
    after = [s.name for s in S.get_robot(rid).segments]

    # THE ADOPT MUST HAVE DONE SOMETHING, or "undo restores it" proves nothing.
    assert after != before, (
        f"adopt_walkable_template changed nothing on a body it is measured to change -- this test cannot say "
        f"anything about undo. segments {before}")
    assert len(after) > len(before), f"expected the template's segments, got {after}"
    # ...and the op's OWN diff must say so, in the numbers the customer reads. Measured through
    # ``call_tool("edit_robot", ...)``: result['diffs'][0] = {op, applied: True, segments_before 5,
    # segments_after 20, mass{total_mass_kg [3.201, 0.771], n_existing_links_dropped 5, dropped [...]}}.
    _diff = next((d for d in (ed.get("diffs") or [])
                  if isinstance(d, dict) and d.get("op") == "adopt_walkable_template"), None)
    assert _diff is not None, f"edit_robot reported no diff for the op it applied: {ed.get('diffs')}"
    assert _diff.get("applied") is True, f"adopt reported applied={_diff.get('applied')}: {_diff}"
    assert int((_diff.get("mass") or {}).get("n_existing_links_dropped") or 0) == len(before), (
        f"the mass ledger must disclose all {len(before)} dropped customer links: {_diff.get('mass')}")

    call_tool("edit_robot", {"robot_id": rid, "ops": [{"op": "undo"}]})
    restored = [s.name for s in S.get_robot(rid).segments]
    assert restored == before, "undo did not restore the customer's original body"


def test_walkable_template_actually_helps_a_body_that_cannot_walk():
    """GAP 1: does the opt-in HELP? Nothing anywhere proved it did. MEASURED 2026-08-12, it does.

    On this file's ``_fixed_base_quad_urdf`` fixture, through ``edit_operators.adopt_walkable_template``:

        before   0.000 m   SLIDE (feet barely lift / no real stepping)   5 links, 3.201 kg
        after    1.356 m   CREDIBLE WALK                                20 segments, 0.771 kg
        diff     applied=True, n_existing_links_dropped=5, total_mass_kg [3.201, 0.771]

    and the ingest's own offer for the same body quoted ``template_distance_m 1.356`` -- the number the adopt
    then actually delivered, so the offer does not promise a walk it cannot produce. The same op on a real
    Menagerie Unitree Go2 goes 0.000 m / CROUCH -> 1.998 m / CREDIBLE WALK (13 links, 15.206 kg -> 20 segments,
    9.794 kg); see ``test_walkable_template_helps_a_real_menagerie_go2``.

    SO THE ANSWER IS "IT WORKS" -- at the price of every one of the customer's links, which is why it may only
    ever be an explicit, disclosed, undoable choice (#215/B2). The sibling assertion that a body which ALREADY
    walks survives this op byte-identical lives in
    ``tests/test_customer_ingest.py::test_ingested_quadruped_is_honestly_verified_and_never_silently_swapped``.

    IS THIS STILL ADVERSARIAL? Only while the ``before`` half still fails. If the fixture ever starts walking
    as imported, this test stops testing the opt-in and starts testing nothing -- which is why ``before`` is
    asserted to fail explicitly instead of being assumed.
    """
    from virturoid.services import session_state as S
    from virturoid.services.edit_operators import adopt_walkable_template
    from virturoid.services.input_training_tools import _ingest_project

    tmp = _fixed_base_quad_urdf(tempfile.mkdtemp())
    r = _ingest_project({"project_path": tmp, "description": "patrol quad"})
    gene = S.get_robot(r["robot_id"])
    before_names = [s.name for s in gene.segments]
    before_m, before_verdict = _verdict(gene)

    # 1. the premise: this body genuinely FAILS the gait gate as imported (0.000 m / SLIDE when measured)
    assert before_m < 0.5 and before_verdict != "CREDIBLE WALK", (
        f"PREMISE GONE: the fixture now walks as imported ({before_m:.3f} m, {before_verdict!r}), so this test no "
        f"longer measures what the opt-in does to a body that cannot walk. Re-anchor it on a body that fails.")
    # ...and the ingest says so, and offers the template rather than applying it
    iv = r.get("imported_verdict") or {}
    assert iv.get("walks_under_our_scripted_gait") is False, iv
    assert [s.name for s in S.get_robot(r["robot_id"]).segments] == before_names, (
        "the ingest substituted the template by itself -- that is the #215/B2 defect")
    offer = r.get("walkable_template_offer") or {}
    assert offer.get("available") is True, f"a failing quadruped must be OFFERED the template: {offer}"

    # 2. the customer opts in
    new, diff = adopt_walkable_template(gene)
    after_names = [s.name for s in new.segments]
    after_m, after_verdict = _verdict(new)

    # 3. it fires, and the body it hands back WALKS
    assert diff.get("applied") is True, f"the opt-in declined on a body that cannot walk: {diff}"
    assert after_verdict == "CREDIBLE WALK", (
        f"the opt-in applied but the result still does not walk: {after_m:.3f} m, {after_verdict!r}. If this is "
        f"the new truth, keep the assertion and change the OFFER's wording -- it must not promise a walk it "
        f"cannot deliver.")
    assert after_m > before_m + 0.5, f"{before_m:.3f} -> {after_m:.3f} m is not a material improvement"

    # 4. and it charges the customer every link, in the open
    assert after_names != before_names and "torso" in after_names, after_names
    ledger = diff.get("mass") or {}
    assert int(ledger.get("n_existing_links_dropped") or 0) == len(before_names), (
        f"a wholesale swap must be visible in the mass ledger: {ledger}")
    assert "undo restores your original body" in (diff.get("note") or ""), diff.get("note")

    # 5. the offer did not over-promise: it quoted what the adopt actually delivered
    quoted = float(offer.get("template_distance_m") or 0.0)
    assert quoted <= after_m + 0.25, (
        f"the ingest offered {quoted:.3f} m but adopting it delivered {after_m:.3f} m -- the offer promises a "
        f"walk the op does not produce")


def test_walkable_template_decline_names_its_reason():
    """A REFUSAL IS AN ANSWER AND MUST NAME ITSELF -- the decline used to lie by catch-all.

    MEASURED 2026-08-12, before the fix, through ``edit_operators.adopt_walkable_template``: a real Menagerie
    ``unitree_g1`` (30 links, 33.341 kg, 0.000 m, CROUCH), the already-walking ingest fixture quad (1.604 m,
    CREDIBLE WALK) and a composed 6-axis arm ALL came back with the byte-identical note "the original body
    already walks or no better template was found -- unchanged". For the humanoid the first disjunct is FALSE
    and the second implies we measured a template against their body when we never left the robot-class check;
    for the arm the sentence is meaningless. The ingest surface tells the SAME humanoid the true reason in the
    same session -- one question, two framings, the #215/#218 shape.

    Now each decline reports the branch that actually ran (measured, after the fix):
        composed humanoid  (0.000 m, CROUCH)  -> reason 'not_a_quadruped',   robot_class 'humanoid'
        composed 6-axis arm                   -> reason 'not_a_legged_body', robot_kind  'manipulator'

    ``geometry_unchanged`` is asserted too, because "nothing was changed" is itself a claim: ``applied`` reads
    one metadata key, and ``anatomy_compiler._splay_before_substituting`` can hand back a body whose legs have
    been rotated outward while that key stays unset. The operator compares a geometry signature rather than
    inferring stillness from the flag.
    """
    from virturoid.services.edit_operators import adopt_walkable_template
    from virturoid.services.morphology_composer import compose_robot

    biped = compose_robot("a bipedal humanoid robot that walks", llm=None)
    dist, verdict = _verdict(biped)
    assert verdict != "CREDIBLE WALK", (
        f"PREMISE GONE: the composed biped now walks ({dist:.3f} m, {verdict!r}), so a decline that says 'already "
        f"walks' would no longer be false. Re-anchor on a legged body that fails.")
    biped_before = _geom_sig(biped)
    new_biped, diff = adopt_walkable_template(biped)
    assert diff.get("applied") is False, diff
    # the "unchanged" half of the note must be MEASURED, not inferred from the applied flag
    assert diff.get("geometry_unchanged") is (_geom_sig(new_biped) == biped_before), (
        f"geometry_unchanged={diff.get('geometry_unchanged')} disagrees with the body that came back")
    if diff.get("geometry_unchanged"):
        assert "nothing was changed" in (diff.get("note") or ""), diff.get("note")
    else:
        assert "geometry WAS adjusted" in (diff.get("note") or ""), (
            f"the body came back modified and the note called it unchanged: {diff.get('note')!r}")
    declined = diff.get("declined") or {}
    assert declined.get("reason"), f"the op declined and said nothing about why: {diff}"
    assert declined["reason"] != "unreported", f"the decline reason went unreported: {diff}"
    # the specific lie: telling a body that measurably does not walk that it already walks
    assert "already walk" not in (diff.get("note") or "").lower(), (
        f"the decline claims this body walks; it measured {dist:.3f} m / {verdict!r}. note={diff.get('note')!r}")
    assert "quadruped" in (declined.get("detail") or "").lower(), (
        f"the reason must name the real one -- the template is a quadruped recipe: {declined}")

    # a body that is not legged at all must not be told anything about walking either
    arm = compose_robot("a 6-axis robot arm with a gripper", llm=None)
    _, arm_diff = adopt_walkable_template(arm)
    assert arm_diff.get("applied") is False, arm_diff
    assert (arm_diff.get("declined") or {}).get("reason") == "not_a_legged_body", arm_diff.get("declined")
    assert "already walk" not in (arm_diff.get("note") or "").lower(), arm_diff.get("note")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie",
                                    "unitree_go2", "go2.xml")),
    reason="needs the MuJoCo Menagerie cache (a real robot, not a fixture)")
def test_walkable_template_helps_a_real_menagerie_go2():
    """The same question as the fixture test, asked of a REAL robot -- fixtures have lied in this repo.

    MEASURED 2026-08-12 on ``~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2/go2.xml``, imported and
    then run through ``edit_operators.adopt_walkable_template``:

        before  0.000 m  CROUCH (low/unstable stance)  13 links,     15.206 kg
        after   1.998 m  CREDIBLE WALK                 20 segments,   9.794 kg
        metadata.walkability_fallback  {applied: True, from_distance_m 0.0, to_distance_m 1.998,
                                        gait_tuning {freq 1.305, kp 247.9, verdict 'CREDIBLE WALK'}}

    The opt-in works on a real customer robot -- and costs that customer all 13 of their links, which the mass
    ledger and ``composition_notes`` both state.
    """
    from virturoid.services.edit_operators import adopt_walkable_template
    from virturoid.services.robot_import import import_robot

    p = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie",
                     "unitree_go2", "go2.xml")
    gene = import_robot(p)["gene"]
    before_names = [s.name for s in gene.segments]
    before_m, before_verdict = _verdict(gene)
    assert before_m < 0.5 and before_verdict != "CREDIBLE WALK", (
        f"PREMISE GONE: the imported Go2 now walks under our scripted gait ({before_m:.3f} m, {before_verdict!r}) "
        f"-- pick a body that still fails, or this measures nothing about the opt-in")

    new, diff = adopt_walkable_template(gene)
    after_m, after_verdict = _verdict(new)
    assert diff.get("applied") is True, f"the opt-in declined on a real Go2 that cannot walk: {diff}"
    assert after_verdict == "CREDIBLE WALK", f"{after_m:.3f} m, {after_verdict!r} after adopting the template"
    assert after_m > before_m + 0.5, f"{before_m:.3f} -> {after_m:.3f} m"
    # ...and the swap is disclosed in numbers, not just in prose
    ledger = diff.get("mass") or {}
    assert int(ledger.get("n_existing_links_dropped") or 0) == len(before_names), ledger
    assert ledger["total_mass_kg"][1] < ledger["total_mass_kg"][0], ledger
    notes = " ".join(getattr(new, "composition_notes", None) or [])
    assert "substituted" in notes.lower(), f"the substitution must be stated in composition_notes: {notes!r}"
