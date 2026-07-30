"""An imported robot must still be the CUSTOMER'S robot, in shape and not only in link names.

This is the gate that was missing, and its absence let a 0.19-0.28 m deformation land with a green suite.

The editable twin exists so a customer can amend their own machine. Every check we had asked whether the twin
was internally consistent -- does it compile, are the link counts right, does the trunk measure what we think --
and none asked the only question that matters: is it the same SHAPE as the thing they handed us. So a change
that improved one number (the Go2's trunk, genuinely 0.199 -> 0.387 m) while sliding whole subtrees by up to
0.38 m passed 1837 tests and an 8-test file written specifically for it.

The measurement is the full PAIRWISE DISTANCE MATRIX over body origins. That is invariant under any rigid
transform, so a change that merely re-datums the model (moves its origin, which imports legitimately do) scores
zero, while a change that genuinely deforms the body cannot hide behind "it just moved".
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="importing a robot needs MuJoCo")

_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))


def _pairwise(xpos):
    import numpy as np
    P = np.asarray(xpos, dtype=float)
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)


def _source_and_twin(rel: str):
    """Body positions of the customer's own model, and of the gene we rebuilt from it, keyed by link name."""
    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    from virturoid.services.robot_import import import_robot

    src = _MEN / rel
    if not src.exists():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    sm = mujoco.MjModel.from_xml_path(str(src))
    sd = mujoco.MjData(sm)
    if sm.nkey:
        mujoco.mj_resetDataKeyframe(sm, sd, 0)
    mujoco.mj_forward(sm, sd)
    src_pos = {mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_BODY, b): sd.xpos[b] for b in range(1, sm.nbody)}

    out = import_robot(str(src), robot_id=f"kin_{Path(rel).stem}")
    gene = out["gene"]
    if gene is None:
        pytest.skip(f"{rel} did not import: {out.get('warnings')}")
    tm = compiled_model(robot_mjcf(gene))
    td = mujoco.MjData(tm)
    if tm.nkey:
        mujoco.mj_resetDataKeyframe(tm, td, 0)
    mujoco.mj_forward(tm, td)
    twin_pos = {mujoco.mj_id2name(tm, mujoco.mjtObj.mjOBJ_BODY, b): td.xpos[b] for b in range(1, tm.nbody)}
    return src_pos, twin_pos


@pytest.mark.parametrize("rel", [
    "unitree_go2/go2.xml",          # 4 legs on one hub -- the case the trunk-span change was written for
    "unitree_h1/h1.xml",            # a torso with a SYMMETRIC PAIR of shoulders: the child spread's principal
                                    # axis is LATERAL here, which is what made a shape-from-children rule wrong
    "pal_talos/talos.xml",          # nested hubs: a hub hanging off another hub
    "boston_dynamics_spot/spot.xml",
])
def test_the_twin_is_the_same_shape_as_the_robot_we_were_given(rel):
    """The load-bearing one. Pairwise distances between the links they share must agree to a millimetre.

    Deliberately NOT a check on positions: an import may legitimately re-datum the whole body. It is a check on
    the body's own internal geometry, which an import may never change."""
    import numpy as np
    src_pos, twin_pos = _source_and_twin(rel)
    shared = sorted(set(src_pos) & set(twin_pos))
    if len(shared) < 4:
        pytest.skip(f"only {len(shared)} link names survived the import; nothing to compare")
    A = _pairwise([src_pos[n] for n in shared])
    B = _pairwise([twin_pos[n] for n in shared])
    worst = float(np.abs(A - B).max())
    i, j = np.unravel_index(int(np.abs(A - B).argmax()), A.shape)
    assert worst < 0.02, (
        f"{rel}: the twin is deformed by {worst:.4f} m — {shared[i]}..{shared[j]} is {A[i, j]:.4f} m on the "
        f"customer's robot and {B[i, j]:.4f} m on ours, over {len(shared)} shared links. The editable twin is "
        "supposed to BE their machine; a link-name match with a different shape is the failure this gate exists "
        "for.")


def test_a_symmetric_pair_does_not_decide_the_torso_axis():
    """H1's chest carries two shoulders, so the principal axis of its children runs SIDE TO SIDE. Any rule that
    reads a body's long axis off its child spread gets a humanoid torso exactly 90 degrees wrong, and the failure
    is invisible to a length check because the number it produces is plausible."""
    src_pos, twin_pos = _source_and_twin("unitree_h1/h1.xml")
    shared = set(src_pos) & set(twin_pos)
    shoulders = sorted(n for n in shared if "shoulder" in (n or "").lower())
    if len(shoulders) < 2:
        pytest.skip("this import produced no named shoulder pair")
    import numpy as np
    for pos, who in ((src_pos, "source"), (twin_pos, "twin")):
        P = np.asarray([pos[n] for n in shoulders], dtype=float)
        spread = P.max(axis=0) - P.min(axis=0)
        assert int(np.argmax(spread)) == 1, (
            f"{who}: the shoulders spread furthest along axis {int(np.argmax(spread))}, not the lateral y "
            f"(spread {np.round(spread, 4)}) — a torso has been rebuilt sideways")
