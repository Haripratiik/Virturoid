"""A hub's length is its OWN span, not the distance to one of its children.

`_link_vector` returns the vector to the farthest child. For a serial link that is exactly right and measures the
Go2's thigh at 0.2130 m against a real 0.213. For a TORSO with four legs on it, it is close to meaningless: it
picks one hip and calls the distance to it the trunk's length. Measured across the cached Menagerie corpus, 33
models have a multi-child root and 20 of them import understated by >=1.5x -- the Go2's trunk at 0.199 m against
a real 0.387, Berkeley Humanoid's at 0.070 m for a 0.25 m torso.

Three things had to be measured before this could be fixed, and each one moved the answer:

  * the AXIS is the principal axis of the child spread, not the direction to one child
  * VISUAL geometry overstates it -- the Go2's base carries five non-colliding shrouds, and measuring those
    together with the collision box returns 0.530 m for a 0.376 m structure
  * so does an outlying FEATURE -- the Go2's head is a cylinder and a sphere at x ~ +0.29, which push the
    collision union to 0.528; ANYmal C's nose pushes its 0.600 m base to 1.049

Hence the rule: the main structural geom, unioned with the points where children actually mount. A trunk spans
its own body and at least far enough to reach everything hanging off it.

The origin is the other half, and NOT a flag: off/len has median -0.504 but sd 0.263 across the corpus, with only
12 of 31 within 0.1 of centred. TIAGo++'s root starts nearly a full length behind its origin; LEAP Hand's starts
slightly ahead of it. So the shift is carried as a real number.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="importing a robot needs MuJoCo")

_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))


def _menagerie(rel: str) -> str:
    p = _MEN / rel
    if not p.exists():
        pytest.skip(f"{rel} not in the local Menagerie cache")
    return str(p)


def _seg(gene, name):
    return next((s for s in gene.segments if s.name == name), None)


def test_a_four_legged_trunk_measures_its_own_span_not_the_reach_to_one_hip():
    """The Go2's hips sit at x = +-0.1934, so the trunk is 0.387 m. Importing it at 0.199 m -- the distance to a
    single hip -- describes a robot half the length of the one the customer handed us."""
    from virturoid.services.robot_import import import_robot
    out = import_robot(_menagerie("unitree_go2/go2.xml"), robot_id="go2_trunk_span")
    gene = out["gene"]
    assert gene is not None, out.get("warnings")
    root = gene.root()
    assert root.length_m == pytest.approx(0.387, abs=0.02), (
        f"trunk imported at {root.length_m:.3f} m; the hips are 0.387 m apart")


def test_the_trunk_does_not_swallow_the_head():
    """The maximal collision extent is 0.528 m, because the Go2's head is a cylinder and a sphere at x ~ +0.29.
    A trunk drawn that long sticks 0.14 m out past its own front legs. A feature mounted ON the body is not the
    body's length -- ANYmal C fails this harder still, 1.049 m of nose-inclusive extent over a 0.600 m base."""
    from virturoid.services.robot_import import import_robot
    root = import_robot(_menagerie("unitree_go2/go2.xml"), robot_id="go2_no_head")["gene"].root()
    assert root.length_m < 0.46, f"trunk {root.length_m:.3f} m has swallowed the head bump"


def test_the_legs_still_land_where_the_customer_put_them():
    """THE regression this fix could plausibly cause, and the reason the origin shift exists at all.

    A child mounts at `parent.length + mount_z`, so lengthening the trunk moves every mount unless the offset
    absorbs it. With the hips at local z = +-0.1934 and a 0.387 m trunk drawn from 0, the rear pair would land at
    -0.193 -- BEHIND the trunk's own start. Measured in the compiled model rather than from the gene, because the
    gene is what is being questioned."""
    import numpy as np

    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    from virturoid.services.robot_import import import_robot
    gene = import_robot(_menagerie("unitree_go2/go2.xml"), robot_id="go2_legs_land")["gene"]
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    hips = [b for b in range(1, m.nbody)
            if "hip" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()]
    if len(hips) < 4:
        pytest.skip("this import did not produce four named hips")
    root_i = next(b for b in range(1, m.nbody) if int(m.body_parentid[b]) == 0)
    org = np.asarray(d.xpos[root_i], dtype=float)
    R = d.xmat[root_i].reshape(3, 3)
    # Spread of the hips in the trunk's own frame, along whichever axis carries it. Deliberately NOT hardcoded to
    # x: a segment's link axis is its local +z, so the fore-aft spread lands there, and asserting on x measured
    # 0.000 and read as a total failure when the mounts were in fact exact.
    L = np.array([R.T @ (np.asarray(d.xpos[h], dtype=float) - org) for h in hips])
    per_axis = L.max(axis=0) - L.min(axis=0)
    spread = float(per_axis.max())
    assert spread == pytest.approx(0.387, abs=0.05), (
        f"hips spread {spread:.3f} m along the trunk (per-axis {np.round(per_axis, 4)}); the customer's are "
        "0.387 m apart — the mounts did not follow the corrected trunk length")
    lateral = float(sorted(per_axis)[-2])
    assert lateral == pytest.approx(0.093, abs=0.03), (
        f"stance width {lateral:.3f} m; the Go2's hips sit at +-0.0465 — length must not leak into width")


def test_a_serial_link_is_untouched():
    """The hub rule must not disturb the case that already worked. The Go2's thigh measures 0.2130 m against a
    real 0.213, and that number was hard-won -- it comes from the joint span, because the visual mesh over-reads
    it at 0.373 m and drove grounded mass to 39.9 kg for a 15.2 kg robot."""
    from virturoid.services.robot_import import import_robot
    gene = import_robot(_menagerie("unitree_go2/go2.xml"), robot_id="go2_thigh")["gene"]
    thigh = next((s for s in gene.segments if "thigh" in s.name.lower()), None)
    assert thigh is not None, [s.name for s in gene.segments]
    assert thigh.length_m == pytest.approx(0.213, abs=0.005), f"thigh drifted to {thigh.length_m:.4f} m"


def test_the_whole_body_is_the_right_size_end_to_end():
    """Length is only worth correcting if it survives into the compiled robot. A Go2 is ~0.70 m nose-to-tail
    standing; the pre-fix twin measured barely half that, which is the sort of error a customer sees instantly
    and no scalar in a report would have surfaced."""
    import numpy as np

    import mujoco
    from virturoid.services.morph_policy import compiled_model, robot_mjcf
    from virturoid.services.robot_import import import_robot
    gene = import_robot(_menagerie("unitree_go2/go2.xml"), robot_id="go2_extent")["gene"]
    m = compiled_model(robot_mjcf(gene))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    xs = [float(d.xpos[b][0]) for b in range(1, m.nbody)]
    assert (max(xs) - min(xs)) > 0.30, f"body spans only {max(xs) - min(xs):.3f} m fore-aft"


@pytest.mark.parametrize("rel,name,lo,hi", [
    ("unitree_go2/go2.xml", "go2", 0.34, 0.46),                # hips +-0.1934, head excluded
    ("anybotics_anymal_c/anymal_c.xml", "anymal_c", 0.52, 0.70),   # 0.600 base, 1.049 with the nose
    ("boston_dynamics_spot/spot.xml", "spot", 0.55, 0.95),     # chassis genuinely overhangs its hips
])
def test_the_rule_holds_across_real_quadrupeds(rel, name, lo, hi):
    """One model can be fitted by accident. These three disagree with each other about whether the chassis is
    longer or shorter than the hip spread, so a rule that satisfies all three is measuring something real."""
    from virturoid.services.robot_import import import_robot
    out = import_robot(_menagerie(rel), robot_id=f"{name}_span")
    gene = out["gene"]
    if gene is None:
        pytest.skip(f"{name} did not import: {out.get('warnings')}")
    assert lo <= gene.root().length_m <= hi, (
        f"{name} trunk imported at {gene.root().length_m:.3f} m, outside [{lo}, {hi}]")
