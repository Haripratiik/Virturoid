"""An amend should say what it will touch BEFORE it commits, and prove it afterwards.

An edit currently just happens: the customer asks for a taller robot and gets a new robot back, with no statement
of what was meant to change, what was meant to stay, or which established facts stop being true. That is the
difference between editing a file and amending an engineering design.

The pair matters more than either half. `scope()` is a PROMISE; `verify_preserved()` is the check that turns it
into a fact -- and it is needed precisely because mount offsets, mass and inertia propagate along a chain, so
"I only changed the calf" is easy to believe and often false.
"""
from __future__ import annotations

import copy
import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composing a body needs MuJoCo")


@pytest.fixture(scope="module")
def dog():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four legged robot dog", llm=None)


def test_scoping_a_link_edit_claims_the_link_AND_everything_below_it(dog):
    """A descendant's placement is defined relative to its parent, so it moves whether or not it was named.
    Scoping that omitted descendants would promise more than the edit can keep."""
    from virturoid.services.change_impact import scope
    mid = next(s.name for s in dog.segments if s.name.endswith("_1"))
    sc = scope(dog, [{"op": "scale_group", "args": {"group": mid, "field": "length", "factor": 1.3}}])
    assert mid in sc["editable"]
    below = [s.name for s in dog.segments if s.name.startswith(mid[:-1]) and s.name > mid]
    for n in below:
        assert n in sc["editable"], f"{n} hangs below {mid} and must be in scope"
    assert mid not in sc["preserved"]


def test_a_scope_names_the_consequences_not_just_tags(dog):
    """'invalidates: [gait]' is only useful if it says what that means to re-check."""
    from virturoid.services.change_impact import scope
    sc = scope(dog, [{"op": "set_height", "args": {"height_m": 0.5}}])
    assert "gait" in sc["invalidates"] and "stability" in sc["invalidates"]
    assert sc["invalidates_meaning"]["gait"].startswith("the locomotion verdict")


def test_an_unknown_operator_is_assumed_to_touch_everything(dog):
    """A missed recheck is a silent wrong answer; an extra one costs seconds of simulation. Conservative by
    construction, and the unknown op is reported rather than quietly treated as harmless."""
    from virturoid.services.change_impact import scope
    sc = scope(dog, [{"op": "some_new_op_nobody_classified", "args": {}}])
    assert sc["unknown_ops"] == ["some_new_op_nobody_classified"]
    for k in ("gait", "stability", "mass", "torque", "self_collision"):
        assert k in sc["invalidates"], sc["invalidates"]


def test_verify_preserved_catches_a_part_that_moved_when_it_should_not_have(dog):
    """The load-bearing one. A promise nothing else moved is worth exactly the check that confirms it."""
    from virturoid.services.change_impact import scope, verify_preserved
    mid = next(s.name for s in dog.segments if s.name.endswith("_1"))
    sc = scope(dog, [{"op": "scale_group", "args": {"group": mid, "field": "length", "factor": 1.3}}])

    honest = copy.deepcopy(dog)
    for s in honest.segments:
        if s.name == mid:
            s.length_m = round(float(s.length_m) * 1.3, 4)
    assert verify_preserved(dog, honest, sc["preserved"])["held"] is True

    sloppy = copy.deepcopy(honest)
    victim = sc["preserved"][0]
    for s in sloppy.segments:
        if s.name == victim:
            s.mass_kg = float(s.mass_kg or 0.1) * 2.0        # an out-of-scope part quietly drifts
    out = verify_preserved(dog, sloppy, sc["preserved"])
    assert out["held"] is False and victim in out["changed"], out
    assert "mass_kg" in out["changed"][victim], out["changed"][victim]


def test_an_op_that_names_no_part_says_its_scope_is_UNBOUNDED(dog):
    """The most dangerous possible answer is an empty `editable` reading as 'nothing changes'. set_height
    reshapes every ground-reaching chain while naming no part, so the report has to say so out loud."""
    from virturoid.services.change_impact import scope
    sc = scope(dog, [{"op": "set_height", "args": {"height_m": 0.5}}])
    assert sc["editable"] == [] and sc["scope_is_bounded"] is False
    assert "may touch ANY" in sc["scope_note"]
    assert sc["preserved"] == [], "nothing may be claimed preserved when the scope is unbounded"

    bounded = scope(dog, [{"op": "scale_group",
                           "args": {"group": next(s.name for s in dog.segments if s.name.endswith("_1"))}}])
    assert bounded["scope_is_bounded"] is True and bounded["preserved"]


def test_a_length_edit_propagates_to_everything_that_depends_on_it(dog):
    """"Taller" must mean LONGER LINKS AND A RE-DERIVED STANCE, never a spacer block wedged into the legs.

    That requires an edit to carry through everything downstream of it: child mount offsets (or the chain
    detaches), mass and inertia, the actuator demand each joint now sees, and the height the body stands at.
    Measured rather than assumed -- this was believed to be pending until it was checked, and the value of the
    test is that it stays true.
    """
    from virturoid.services.edit_operators import OPERATORS
    from virturoid.services.gene_compiler import standing_spawn_z
    before = {s.name: (s.length_m, s.mass_kg, s.actuator_torque_nm, tuple(s.mount_offset or ()))
              for s in dog.segments}
    z0 = standing_spawn_z(dog)
    res = OPERATORS["scale_group"](dog, group="legs", dims="length", factor=1.3)
    after = res[0] if isinstance(res, tuple) else res

    moved = {k: [] for k in ("length", "mass", "torque", "mount")}
    for s in after.segments:
        b = before.get(s.name)
        if not b:
            continue
        for lbl, now, was in (("length", s.length_m, b[0]), ("mass", s.mass_kg, b[1]),
                              ("torque", s.actuator_torque_nm, b[2])):
            if abs(float(now or 0) - float(was or 0)) > 1e-6:
                moved[lbl].append(s.name)
        if tuple(s.mount_offset or ()) != b[3]:
            moved["mount"].append(s.name)

    assert moved["length"], "the legs did not get longer"
    assert moved["mass"], "mass was not re-derived from the new geometry"
    assert moved["torque"], "actuator demand was not recomputed for the new lever arms"
    z1 = standing_spawn_z(after)
    assert z1 > z0 + 1e-3, f"standing height did not follow the longer legs: {z0:.4f} -> {z1:.4f}"
    # NOTE mount offsets deliberately do NOT move here, and asserting they did was wrong: every segment in a leg
    # chain mounts at its parent's TIP with offset (0,0,0), so the tip travels when the parent lengthens and the
    # offset has nothing to correct. #216 is the OTHER case — a parent scaling while its children are outside the
    # edited set — which the next test covers.


def test_scaling_a_parent_drags_its_out_of_scope_children_with_it(dog):
    """#216: a child attaches at (mount_x, mount_y, parent.length + mount_z), so the offset BAKES IN the parent's
    old length. Scale the parent without scaling that offset and the child drifts off its anchor -- which is
    exactly how an amended arm lost its gripper."""
    from virturoid.services.edit_operators import OPERATORS
    before = {s.name: tuple(s.mount_offset or ()) for s in dog.segments}
    res = OPERATORS["scale_group"](dog, group="torso", dims="length", factor=1.4)
    after = res[0] if isinstance(res, tuple) else res
    torso_kids = [s.name for s in after.segments if s.parent == "torso"]
    if not torso_kids:
        pytest.skip("this body has nothing mounted on the torso")
    followed = [n for n in torso_kids
                if tuple(next(s for s in after.segments if s.name == n).mount_offset or ()) != before.get(n)]
    assert followed, (
        f"torso scaled 1.4x but none of its children {torso_kids} moved their mount offset — they are now "
        "anchored to a length the torso no longer has")


def test_added_and_removed_parts_are_reported_separately(dog):
    """A structural amend adds and removes parts; conflating that with 'changed' would hide topology edits."""
    from virturoid.services.change_impact import verify_preserved
    fewer = copy.deepcopy(dog)
    dropped = fewer.segments[-1].name
    fewer.segments = [s for s in fewer.segments if s.name != dropped]
    out = verify_preserved(dog, fewer)
    assert dropped in out["removed"] and out["added"] == []
