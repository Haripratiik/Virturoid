"""A part can be joined to a SECOND part, so a gantry's bridge rests on both its columns.

`segments` is a strict tree: one parent each. A gantry's bridge is supported at BOTH columns and a delta's three
arms meet at ONE platform, so neither was expressible — and the failure was quiet rather than loud. Our gantry
compiled with the right 3 prismatic DOF, passed every structural check, rendered convincingly, and its second
column carried no load at all. Driving the bridge along its rail walked it clean off that column.

Modelled as a separate `RobotGene.loop_closures` list rather than a second `parent`, because MuJoCo's own
`<equality><connect>` does not touch the body tree either — it is a top-level constraint naming two bodies that
already exist. Keeping the tree a tree is what lets the ~40 places that walk parent->child keep working; several
of them have no visited-set guard and would hang outright on a cyclic parent.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="compiling a body needs MuJoCo")

_GANTRY = {"robot_class": "gantry", "name": "gantry", "parts": [
    {"name": "bed", "role": "body", "aspect": "deck", "size": 1.20, "girth": 0.90},
    {"name": "column_l", "role": "column", "like": "arm", "parent": "bed", "joint": "fixed",
     "attach": {"along": 0.5, "lateral": -1.0, "height": 1.0}, "aim": "up", "size": 0.90, "girth": 0.05},
    {"name": "column_r", "role": "column", "like": "arm", "parent": "bed", "joint": "fixed",
     "attach": {"along": 0.5, "lateral": 1.0, "height": 1.0}, "aim": "up", "size": 0.90, "girth": 0.05},
    {"name": "bridge", "role": "bridge", "like": "arm", "parent": "column_l", "aim": "out",
     "size": 0.738, "girth": 0.05,
     "joint": "prismatic", "axis": [0, 1, 0], "lower": -0.45, "upper": 0.45, "rest": 0.0}]}


def _gantry(with_loop: bool):
    from virturoid.services.anatomy_compiler import build_from_anatomy
    gene = build_from_anatomy(_GANTRY)
    if with_loop:
        gene.loop_closures = [{"a": "bridge", "b": "column_r"}]
    return gene


def test_a_loop_closure_reaches_the_compiled_model():
    """The whole chain: gene -> MJCF <equality><connect> -> a constraint MuJoCo actually holds."""
    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    xml = robot_mjcf(_gantry(True))
    assert "<equality>" in xml and "connect" in xml, "no equality block was emitted"
    m = compiled_model(xml)
    assert m.neq >= 1, f"the model carries {m.neq} equality constraints; the loop did not survive compilation"


def test_the_bridge_is_HELD_to_its_far_column_when_driven():
    """THE measurement, and the reason this exists. Without the loop the bridge is a cantilever off one column:
    drive the rail and it leaves the other behind, while every DOF and joint-type check still passes.

    Measured by STEPPING, not by mj_forward. A `connect` is a soft constraint the solver satisfies with FORCES
    during integration — it does not project qpos — so a forward pass shows the unconstrained pose and would
    report the constraint doing nothing at all. That distinction is the whole character of this feature: it
    buys expressiveness, not rigidity."""
    import mujoco
    import numpy as np
    from virturoid.services.morph_policy import compiled_model, robot_mjcf

    def _drift(gene):
        """How far the bridge's far end travels away from the far column while the rail is driven."""
        m = compiled_model(robot_mjcf(gene))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "bridge_joint")
        col = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "column_r")
        br = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bridge")
        assert j >= 0 and col >= 0 and br >= 0
        L = float(next(s.length_m for s in gene.segments if s.name == "bridge"))

        def _gap():
            R = d.xmat[br].reshape(3, 3)
            far = np.array(d.xpos[br], dtype=float) + R @ np.array([0.0, 0.0, L])
            return float(np.linalg.norm(far - np.array(d.xpos[col], dtype=float)))

        start = _gap()
        act = next((a for a in range(m.nu)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith("bridge")), None)
        hi = float(m.jnt_range[j][1])
        worst = 0.0
        for _ in range(400):
            if act is not None:
                d.ctrl[act] = hi                       # drive the carriage to the end of its rail
            mujoco.mj_step(m, d)
            worst = max(worst, abs(_gap() - start))
        return worst

    free = _drift(_gantry(False))
    tied = _drift(_gantry(True))
    assert free > 0.05, (
        f"the un-tied bridge barely moves away from its far column ({free:.4f} m) — this test cannot show the "
        "constraint doing anything, so it would prove nothing")
    assert tied < free * 0.6, (
        f"the loop closure did not hold the bridge: it drifted {tied:.4f} m tied vs {free:.4f} m free")


def test_a_loop_closure_round_trips_through_to_dict():
    """The species tree persists genes via to_dict, so a field that vanishes there is a field the flywheel
    silently loses — and the body would come back out of memory as a cantilever again."""
    import json
    from virturoid.schemas.gene import RobotGene
    gene = _gantry(True)
    back = RobotGene.from_dict(json.loads(json.dumps(gene.to_dict())))
    assert back.loop_closures == [{"a": "bridge", "b": "column_r"}]


def test_a_loop_that_names_a_missing_part_is_reported():
    """It must fail as a stated design error, not as a mysteriously absent constraint."""
    gene = _gantry(False)
    gene.loop_closures = [{"a": "bridge", "b": "column_that_does_not_exist"}]
    issues = gene.validate()
    assert any("loop_closure" in i for i in issues), issues


def test_a_loop_may_not_restate_an_edge_the_tree_already_has():
    """A parent and child are already rigidly connected. Asking the solver to also constrain them is a
    contradiction dressed as a loop, and it would quietly stiffen a joint that is supposed to move."""
    gene = _gantry(False)
    gene.loop_closures = [{"a": "bridge", "b": "column_l"}]        # bridge's actual parent
    issues = gene.validate()
    assert any("already parent and child" in i for i in issues), issues


def test_the_loop_pair_stops_colliding_with_itself():
    """The co-change. The two joined bodies are not ancestor and descendant, so the exclude walk never reached
    them — leaving the contact solver pushing them apart while the equality constraint pulls them together, at
    the one joint the design cares most about."""
    from virturoid.services.morph_policy import robot_mjcf
    xml = robot_mjcf(_gantry(True))
    assert ('body1="bridge" body2="column_r"' in xml or 'body1="column_r" body2="bridge"' in xml), \
        "the loop-joined pair is still contact-enabled"


def test_a_body_with_no_loops_is_unchanged():
    """The regression surface is every robot ever built: no loops must mean no equality block at all."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.morph_policy import robot_mjcf
    xml = robot_mjcf(compose_robot("a four legged robot dog", llm=None))
    assert "<equality>" not in xml


def test_the_URDF_export_says_it_cannot_carry_the_loop():
    """URDF is a strict tree, so a closed loop genuinely cannot be represented. Dropping it SILENTLY is the
    worst available option: the file looks complete and ships the exact cantilever the loop was declared to fix,
    under a green export. Say it in the file, where whoever opens it will see it."""
    from virturoid.services.gene_urdf import gene_to_urdf
    urdf = gene_to_urdf(_gantry(True))
    assert "closed kinematic loop" in urdf.lower(), "the URDF drops the loop without a word"
    assert "bridge" in urdf and "column_r" in urdf, "the warning does not name which join was lost"
    clean = gene_to_urdf(_gantry(False))
    assert "closed kinematic loop" not in clean.lower(), "a body with no loops must not carry the warning"


def test_a_braced_gantry_and_a_cantilever_do_not_share_a_memory_KEY():
    """The moat depends on this. Our memory is keyed by MORPHOLOGY, so two robots that index to the same
    fingerprint get each other's banked control hints — and a gantry braced at both columns and one cantilevering
    off a single column are genuinely different machines with different load paths.

    They hashed IDENTICALLY, because the fingerprint's neighbour graph was built from parent edges alone. A loop
    closure is a neighbour relation, which is exactly what a Weisfeiler-Lehman fingerprint exists to capture."""
    from virturoid.services.morph_wl_fingerprint import wl_fingerprint
    free = wl_fingerprint(_gantry(False))
    tied = wl_fingerprint(_gantry(True))
    assert free != tied, "a braced gantry and a cantilever still index to the same morphology key"
    cos = sum(x * y for x, y in zip(free, tied))
    assert cos < 0.95, f"the two are still near-identical in the embedding (cosine {cos:.4f})"


def test_a_body_with_no_loops_fingerprints_exactly_as_before():
    """The regression surface is every robot already banked: adding a loop channel must not re-key them, or the
    entire existing corpus silently stops matching."""
    from virturoid.services.morph_wl_fingerprint import wl_fingerprint
    from virturoid.services.morphology_composer import compose_robot
    dog = compose_robot("a four legged robot dog", llm=None)
    assert dog.loop_closures == []
    assert wl_fingerprint(dog) == wl_fingerprint(dog)
    plain = _gantry(False)
    assert wl_fingerprint(plain) == wl_fingerprint(plain)


def test_a_fixed_base_can_be_ELEVATED():
    """`base_mount` says what the robot is bolted TO; it could not say how high that is. The three named heights
    are 0.025 / 0 / 0, so an overhead-mounted machine compiled BELOW THE FLOOR — a delta's platform measured
    z = -0.069 m, its whole mechanism underground. Overhead gantries, ceiling rails and bench arms share the
    shape of the problem."""
    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf

    def _lowest(gene):
        m = compiled_model(robot_mjcf(gene))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        return min(float(d.xpos[b][2]) for b in range(1, m.nbody)), max(
            float(d.xpos[b][2]) for b in range(1, m.nbody))

    ground = _gantry(True)
    lo0, hi0 = _lowest(ground)
    raised = _gantry(True)
    raised.base_height_m = 1.40
    lo1, hi1 = _lowest(raised)
    assert hi1 > hi0 + 1.0, f"declaring base_height_m 1.40 did not raise the body ({hi0:.3f} -> {hi1:.3f} m)"
    assert (hi1 - lo1) == pytest.approx(hi0 - lo0, abs=1e-6), "raising the base changed the body's own shape"


def test_base_height_round_trips_and_defaults_to_the_named_mount():
    """None must mean the named height, or every body ever banked silently moves."""
    import json
    from virturoid.schemas.gene import RobotGene
    plain = _gantry(False)
    assert plain.base_height_m is None
    raised = _gantry(False)
    raised.base_height_m = 0.85
    assert RobotGene.from_dict(json.loads(json.dumps(raised.to_dict()))).base_height_m == 0.85
    assert RobotGene.from_dict(json.loads(json.dumps(plain.to_dict()))).base_height_m is None
