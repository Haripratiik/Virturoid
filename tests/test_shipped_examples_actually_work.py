"""An example shipped as PRECEDENT has to do what it claims, because agents copy it.

`get_design_schema` hands these graphs to the agent as worked examples. A broken one does not merely fail — it
teaches the failure, and it teaches it to every design that adapts it.

The SCARA shipped with `axis: [0, 0, 1]` on both elbows, copied from the schema's own description of that field.
`axis` is in the SEGMENT'S OWN frame, where local +z runs ALONG the part, so for a forward-aimed link that is a
ROLL about the arm's own length. Both "elbows" were roll joints: the arm could not fold, and the `rest: 0.9`
added to demonstrate the rest angle moved the tip 0.0000 m. It compiled, passed every gate in `submit_design`,
and RENDERED as a convincing SCARA — which is why nothing caught it.

So these assert BEHAVIOUR, not structure. Counting joints is what missed it.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="compiling a body needs MuJoCo")


def _model(graph):
    import mujoco
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    gene = build_from_anatomy(graph)
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return gene, m, d


def _world_axis(m, d, joint_name):
    import mujoco
    import numpy as np
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert j >= 0, f"no joint {joint_name!r}"
    R = d.xmat[int(m.jnt_bodyid[j])].reshape(3, 3)
    return R @ np.asarray(m.jnt_axis[j], dtype=float)


def _tip_travel(graph, joint_name, angle):
    """How far the far end of the robot moves when one joint is driven. The only question that matters."""
    import mujoco
    import numpy as np
    _, m, d = _model(graph)
    tip = m.nbody - 1
    before = np.array(d.xpos[tip], dtype=float)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    d.qpos[m.jnt_qposadr[j]] = angle
    mujoco.mj_forward(m, d)
    return float(np.linalg.norm(np.array(d.xpos[tip], dtype=float) - before))


def test_the_shipped_scara_elbows_turn_about_the_VERTICAL():
    """A SCARA is defined by this: two revolutes in a horizontal plane, i.e. about the world vertical. Getting
    it wrong is invisible to a joint-type check — a roll joint is still `revolute`."""
    import numpy as np
    from virturoid.services.agent_design_tools import _EXAMPLE_SCARA
    _, m, d = _model(_EXAMPLE_SCARA)
    for name in ("link1_joint", "link2_joint"):
        ax = _world_axis(m, d, name)
        assert abs(abs(float(ax[2])) - 1.0) < 0.05, (
            f"{name} turns about {np.round(ax, 3)} — a SCARA elbow must be vertical, and a horizontal one is a "
            "roll about the arm's own length that cannot fold the arm")


def test_the_shipped_scara_can_actually_fold():
    """The behavioural check. The broken version moved its tip 0.0000 m through 1.2 rad of 'elbow'."""
    from virturoid.services.agent_design_tools import _EXAMPLE_SCARA
    travel = _tip_travel(_EXAMPLE_SCARA, "link2_joint", 1.2)
    assert travel > 0.03, f"driving the elbow 1.2 rad moved the tip {travel:.4f} m — this arm does not articulate"


def test_the_shipped_scara_quill_slides_vertically():
    """The third axis: a SCARA's Z is a vertical prismatic at the wrist, not a diagonal one."""
    import numpy as np
    from virturoid.services.agent_design_tools import _EXAMPLE_SCARA
    _, m, d = _model(_EXAMPLE_SCARA)
    ax = _world_axis(m, d, "quill_joint")
    assert abs(abs(float(ax[2])) - 1.0) < 0.05, f"the quill slides along {np.round(ax, 3)}, not the vertical"


def test_the_shipped_excavator_arm_articulates():
    """Same standard for the other machine example: its boom must actually swing."""
    from virturoid.services.agent_design_tools import _EXAMPLE_EXCAVATOR
    travel = _tip_travel(_EXAMPLE_EXCAVATOR, "boom_joint", 0.8)
    assert travel > 0.10, f"driving the boom 0.8 rad moved the bucket {travel:.4f} m"


@pytest.mark.parametrize("name", ["quadruped", "hexapod", "rover", "scara_arm", "excavator"])
def test_every_shipped_example_compiles_and_every_actuator_does_something(name):
    """Breadth: no example may ship with a dead actuator. A joint that moves nothing is a lie in the DOF count,
    and the DOF count is what our gates read."""
    import mujoco
    import numpy as np
    from virturoid.services.agent_design_tools import get_design_schema
    graph = (get_design_schema({}) or {}).get("examples", {}).get(name)
    if not graph:
        pytest.skip(f"no shipped example named {name}")
    gene, m, d = _model(graph)
    assert m.nu >= 1, f"{name} compiled with no actuators"
    dead = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (2, 3):
            continue
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = (float(v) for v in m.jnt_range[j])
        drive = 0.6 if lo == hi == 0.0 else max(min(0.6, hi if hi else 0.6), lo if lo else -0.6)
        # POSE, not position. A joint sits at its own body's origin, so a bucket curling or a wheel spinning
        # about its axle changes nothing about that origin's location — measuring xpos alone calls those dead
        # when they are working exactly as intended. What a genuinely dead joint cannot do is change any body's
        # orientation either.
        def _pose():
            return np.concatenate([
                np.array([d.xpos[b] for b in range(1, m.nbody)], dtype=float).ravel(),
                np.array([d.xmat[b] for b in range(1, m.nbody)], dtype=float).ravel()])

        before = _pose()
        adr = m.jnt_qposadr[j]
        keep = float(d.qpos[adr])
        d.qpos[adr] = drive
        mujoco.mj_forward(m, d)
        moved = float(np.abs(_pose() - before).max())
        d.qpos[adr] = keep
        mujoco.mj_forward(m, d)
        if moved < 1e-4:
            dead.append((jn, round(moved, 6)))
    assert not dead, f"{name} has joints that move nothing when driven: {dead}"


def test_the_shipped_delta_is_a_real_PARALLEL_mechanism():
    """The one shape a tree cannot express alone, so it is the example that proves loop_closures reach an agent.

    Three arms drive ONE shared platform. The trap it exists to teach: MuJoCo's `connect` locks in whatever
    offset the two parts have AT BUILD TIME, so declaring a loop between parts that are apart WELDS THE GAP. An
    earlier version had arm tips 0.5045 m from the platform before stepping and 0.5045 m after 2000 steps — with
    nu=3, neq=2 and a clean validate the whole time."""
    import numpy as np
    from virturoid.services.agent_design_tools import _EXAMPLE_DELTA
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.gene_validation import validate_gene_design

    gene = build_from_anatomy(_EXAMPLE_DELTA)
    assert len(gene.loop_closures) == 2, (
        f"the graph declares 2 loops and the gene carries {len(gene.loop_closures)} — the anatomy compiler is "
        "dropping them, so this example builds a tree with two dangling arms")
    assert gene.base_height_m == 1.05, "the overhead plate height did not reach the gene"

    _, m, d = _model(_EXAMPLE_DELTA)
    assert m.neq >= 2, f"the compiled model carries {m.neq} equality constraints, not the 2 declared"
    assert m.nu == 3, f"a delta has 3 actuated arms; this one has {m.nu}"

    # the platform must sit ON the machine's axis — an off-centre one is three arms dangling, not a delta
    import mujoco
    pl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "platform")
    assert pl >= 0
    assert float(np.linalg.norm(np.asarray(d.xpos[pl], dtype=float)[:2])) < 0.02, (
        f"the platform sits {np.round(d.xpos[pl][:2], 4)} off the axis")

    # and every declared loop must actually MEET, not hold a gap
    checks = (validate_gene_design(gene) or {}).get("checks", {})
    assert checks.get("loop_closures_compiled") is not False, checks
    assert checks.get("loop_closures_meet") is not False, (
        "the shipped delta declares loops between parts that are not touching — it would teach an agent to weld "
        "gaps")
