"""The part language must express MACHINES, not only animals.

`agent_design_tools._ROLES` is 23 animal roles (body/neck/head/tail/leg/arm/tentacle/... /beak/snout) and
`submit_design` rejected anything else outright -- before geometry was even read, so no amount of shape authoring
helped. Combined with `joint` being carried as a TYPE ONLY, a whole family of ordinary industrial machines was
inexpressible: a gantry axis, a SCARA elbow, a linear rail carriage, a turret yaw, an excavator boom. Every one of
those is defined BY its joint axis and travel, and half of them slide rather than rotate.

`GeneSegment` and `gene_compiler` have always carried `joint_type="prismatic"`, `joint_axis`, `joint_lower` and
`joint_upper`. Only the anatomy path never let a caller say them, and derived all three from the animal role
instead (abduction-vs-stride for a walking leg, unlimited spin for a wheel, else a +-2.6 catch-all).

Three things are pinned here, each stated as the defect it would let back in.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="compiling a design needs MuJoCo")

_RAIL = {
    "robot_class": "manipulator", "name": "rail_carriage",
    "parts": [
        {"name": "rail", "role": "body", "size": 1.2, "girth": 0.05},
        {"name": "carriage", "role": "slider", "like": "arm", "parent": "rail", "attach": "tip",
         "aim": "forward", "size": 0.12, "girth": 0.05,
         "joint": "prismatic", "axis": [1, 0, 0], "lower": 0.0, "upper": 1.0},
    ],
}


def _joints(gene):
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
    kinds = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"j{j}"
        kinds[nm] = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[int(m.jnt_type[j])]
    return m, kinds


def test_a_sliding_joint_can_be_asked_for_and_arrives_as_a_slide():
    """The smallest possible probe: one prismatic axis with a stroke. Nothing about animals needs it, which is
    exactly why it was missing -- and a gantry, telescoping mast and rail carriage are all just this."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    gene = build_from_anatomy(_RAIL)
    m, kinds = _joints(gene)
    assert kinds.get("carriage_joint") == "slide", kinds
    seg = next(s for s in gene.segments if s.name == "carriage")
    assert seg.joint_type == "prismatic"
    assert (seg.joint_lower, seg.joint_upper) == (0.0, 1.0), (seg.joint_lower, seg.joint_upper)


def test_a_declared_joint_axis_beats_the_animal_default():
    """A SCARA turns both elbows about Z. The role-derived default is (0,1,0), so without an override every
    industrial arm would be built with the wrong axis and silently 'work'."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    gene = build_from_anatomy({
        "robot_class": "manipulator", "name": "scara",
        "parts": [
            {"name": "column", "role": "body", "size": 0.30, "girth": 0.06},
            {"name": "link1", "role": "arm", "parent": "column", "attach": "front_top", "aim": "forward",
             "size": 0.30, "girth": 0.04, "joint": "revolute", "axis": [0, 0, 1], "lower": -2.6, "upper": 2.6},
            {"name": "zram", "role": "ram", "like": "arm", "parent": "link1", "attach": "tip", "aim": "down",
             "size": 0.15, "girth": 0.025, "joint": "prismatic", "axis": [0, 0, 1],
             "lower": -0.12, "upper": 0.0},
        ]})
    by = {s.name: s for s in gene.segments}
    ax = by["link1"].joint_axis
    assert abs(abs(ax[2]) - 1.0) < 1e-6, f"declared Z axis was overridden by the animal default: {ax}"
    _m, kinds = _joints(gene)
    assert kinds.get("zram_joint") == "slide", kinds
    assert by["zram"].joint_lower == -0.12 and by["zram"].joint_upper == 0.0


def test_an_unfamiliar_role_is_accepted_but_must_say_what_it_is():
    """Open vocabulary, nothing guessed. The old guard was right that an unknown role must not become a limb by
    default -- so the rule is that an unfamiliar name is allowed and has to declare `like`."""
    from virturoid.services.agent_design_tools import submit_design
    bare = {"robot_class": "manipulator", "name": "g",
            "parts": [{"name": "base", "role": "body", "size": 0.3, "girth": 0.08},
                      {"name": "col", "role": "gantry_column", "parent": "base", "attach": "front_top",
                       "aim": "up", "size": 0.5, "girth": 0.04, "joint": "prismatic"}]}
    out = submit_design({"graph": bare})
    assert not out.get("ok"), "an undeclared novel role must be refused, not guessed at"
    assert "like" in (out.get("error") or ""), out.get("error")
    assert "gantry_column" in (out.get("error") or ""), out.get("error")


def test_a_bolted_down_machine_does_not_get_a_floating_base():
    """Every class defaulted to base_mount='free', so an arm -- and now a gantry, rail or turret, which are all
    bolted down -- got a 6-DOF floating base and fell over, making any verdict on it meaningless."""
    from virturoid.services.anatomy_compiler import build_from_anatomy
    assert build_from_anatomy(_RAIL).base_mount == "table"
    legged = build_from_anatomy({"robot_class": "quadruped", "name": "q", "parts": [
        {"name": "torso", "role": "body", "size": 0.4, "girth": 0.1},
        {"name": "leg", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out",
         "size": 0.3, "girth": 0.02, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]})
    assert legged.base_mount == "free", "a body that locomotes still needs a floating base"
    # an explicit choice always wins over the class default
    forced = build_from_anatomy({**_RAIL, "base_mount": "free"})
    assert forced.base_mount == "free"


@pytest.mark.parametrize("named,param", [
    ("front_top", {"along": 1.0, "height": 0.70}),
    ("rear_bottom", {"along": 0.0, "height": 0.40}),
    ("mid_bottom", {"along": 0.5, "height": 0.40}),
    ("front_bottom", {"along": 1.0, "height": 0.40}),
    ("rear_top", {"along": 0.0, "height": 0.70}),
    ("front_mid_bottom", {"along": 0.75, "height": 0.40}),
    ("rear_mid_bottom", {"along": 0.25, "height": 0.40}),
])
def test_a_parametric_attach_point_generalises_the_named_sites_exactly(named, param):
    """The 8 named sites are an animal's anchors (shoulder, hip, withers, tail root) and cannot say "60% along the
    deck" -- a turret mid-chassis, a carriage partway down its rail, a mast at 70% of the body.

    The dict form must be a strict GENERALISATION, not a parallel code path: if every named site is a point in
    this space then there is one anchor rule, and the named tokens are shorthand. Asserting the equivalence is
    what makes that claim checkable rather than hopeful."""
    from virturoid.services.anatomy_compiler import _anchor_on_body
    HL, HW, H, E = 0.30, 0.10, 0.12, 0.6
    a = _anchor_on_body(named, HL, HW, H, edge=E)
    b = _anchor_on_body(param, HL, HW, H, edge=E)
    assert a == pytest.approx(b, abs=1e-9), f"{named} -> {a} but {param} -> {b}"


def test_a_part_can_mount_somewhere_no_named_site_reaches():
    """The point of the exercise: an anchor the token vocabulary simply cannot express, measured on the compiled
    body rather than asserted from the graph."""
    import mujoco

    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    xs = {}
    for along in (0.30, 0.60):
        gene = build_from_anatomy({"robot_class": "mobile_base", "name": f"deck_{along}", "parts": [
            {"name": "deck", "role": "body", "aspect": "deck", "size": 0.5, "girth": 0.16},
            {"name": "mast", "role": "sensor_mast", "like": "neck", "parent": "deck",
             "attach": {"along": along, "height": 0.7}, "aim": "up", "size": 0.2, "girth": 0.03}]})
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mast")
        assert b > 0, "the mast did not survive compilation"
        xs[along] = float(m.body_pos[b][0])
    assert xs[0.30] < xs[0.60], f"a larger `along` must sit further forward: {xs}"
    assert abs(xs[0.60] - xs[0.30]) > 1e-3, f"`along` had no effect on placement: {xs}"


def test_every_existing_animal_role_still_compiles_unchanged():
    """The 23 roles are the regression surface: opening the vocabulary must not perturb any of them."""
    from virturoid.services.agent_design_tools import _ROLES
    from virturoid.services.anatomy_compiler import build_from_anatomy
    for role in _ROLES:
        if role == "body":
            continue
        gene = build_from_anatomy({"robot_class": "quadruped", "name": f"t_{role}", "parts": [
            {"name": "torso", "role": "body", "size": 0.4, "girth": 0.1},
            {"name": "p", "role": role, "parent": "torso", "attach": "front_top", "aim": "forward",
             "size": 0.15, "girth": 0.03}]})
        assert gene is not None and len(gene.segments) >= 2, role
