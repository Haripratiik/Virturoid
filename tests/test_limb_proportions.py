"""Legged bodies read as toys because of PROPORTION, not polygon count -- these pin the four defects behind it.

Measured on our own dog before the fix: a 416 mm leg was cut into FOUR EQUAL 105 mm stubs, so an 82 mm motor can
sat on a 105 mm segment (can/length 0.78 -- the actuator IS the silhouette) where a Unitree Go2 hangs the same
size motor on a 213 mm thigh (0.38, a detail on a limb). That is the bead-chain look. Real legged robots put a
SHORT proximal abduction block, then two LONG load-bearing links, then a small contact pad:

    Unitree Go2       hip/abad ~0.08, thigh 0.213, calf 0.213, foot = a 22 mm sphere   (leg 0.426 m)
    MIT Mini Cheetah  abad 0.062, thigh 0.209, calf 0.195, foot radius 0.02
    ANYmal C          hip 0.09, thigh 0.285, shank 0.33

Three more defects fell out of chasing that one, each pinned below: the joint motor was drawn TWICE (mesh + MJCF
geom) and z-fought into a sawtooth seam; the foot solid was long in two axes at once, so a "sole" was a brick
standing on its end; and the rubber boot pad hung a full foot-radius BELOW the surface the robot stands on.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composition needs MuJoCo")


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _legs(gene):
    """Group the segments of each leg chain, in proximal->distal order."""
    import re
    out: dict[str, list] = {}
    for s in gene.segments:
        m = re.match(r"((?:(?:front|hind)_)?leg\d*(?:_[lr])?)_(\d+)$", (s.name or "").lower())
        if m:
            out.setdefault(m.group(1), []).append((int(m.group(2)), s))
    return [[s for _, s in sorted(v)] for v in out.values() if len(v) >= 3]


@pytest.mark.xfail(reason="REVERTED, deliberately. The anatomical split measurably works as geometry -- it took "
                          "stride-link aspect from 2.27 to 4.33/5.28, the band morphology_priors itself asks "
                          "for -- but it moves every legged body off the operating point the SCRIPTED crawl "
                          "was co-tuned with, and 12 locomotion tests went red. Landing it means re-fitting "
                          "the gait layer to non-uniform limbs (default_gait_for's Froude scaling is the first "
                          "half of that). Kept as the executable spec of the target.", strict=True)
def test_a_leg_is_not_cut_into_equal_stubs():
    """The load-bearing links must dominate the limb, and the proximal block must be short."""
    from virturoid.services.morphology_composer import compose_robot

    for prompt in ("a robot dog", "a robot horse"):
        chains = _legs(compose_robot(prompt))
        assert chains, f"{prompt!r}: no leg chains found"
        for ch in chains[:1]:
            lens = [float(s.length_m) for s in ch]
            total = sum(lens)
            assert lens[0] / total < 0.25, (
                f"{prompt!r}: the proximal block is {lens[0] / total:.0%} of the leg -- a real abduction block is "
                f"~15% (Go2 80 mm of 426). Split: {[round(x * 1000) for x in lens]} mm")
            assert max(lens[1:]) / total > 0.30, (
                f"{prompt!r}: no link carries even 30% of the leg; it is back to equal stubs. "
                f"Split: {[round(x * 1000) for x in lens]} mm")


@pytest.mark.xfail(reason="Same reverted change: with equal-length links the can/stride-link ratio stays at the "
                          "measured 0.78 (Go2 is 0.38). This is the number that makes a limb read as beads on "
                          "a string, and it is the payoff for re-fitting the gait layer.", strict=True)
def test_the_motor_does_not_dominate_the_link_it_drives():
    """can/segment-length on the STRIDE links. Go2 = 0.38; ours was 0.78 when legs were equal stubs."""
    from virturoid.services.component_geometry import actuator_for_joint
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morphology_composer import compose_robot

    gene = compose_robot("a robot dog")
    ground_and_repair(gene)                       # the housing-fit pass runs here; measure the real numbers
    worst, worst_name = 0.0, ""
    for ch in _legs(gene)[:1]:
        for s in ch[1:]:                          # skip the proximal block: a real one IS shorter than its motor
            ds = actuator_for_joint(s)
            if ds is None:
                continue
            can = max(float(ds["envelope_m"][0]), float(ds["envelope_m"][1]))
            ratio = can / max(1e-9, float(s.length_m))
            if ratio > worst:
                worst, worst_name = ratio, s.name
    assert worst <= 0.62, (
        f"{worst_name}: the motor can is {worst:.2f}x the length of the link it drives (Go2 is 0.38). At this "
        "ratio the actuator is the silhouette and the limb reads as beads on a string.")


def test_limb_shares_sum_to_one_and_leave_arms_alone():
    from virturoid.services.anatomy_compiler import _limb_length_shares

    for n in range(1, 7):
        for role in ("leg", "arm"):
            for foot in (True, False):
                sh = _limb_length_shares(role, n, has_foot=foot and n > 1)
                assert len(sh) == n and abs(sum(sh) - 1.0) < 1e-9, (role, n, foot, sh)
    # An arm's links really are near-equal (a UR5 is 0.425 / 0.392) and the arm is the one archetype that already
    # renders convincingly -- it must stay uniform.
    assert _limb_length_shares("arm", 3, has_foot=False) == pytest.approx([1 / 3] * 3)
    legs = _limb_length_shares("leg", 4, has_foot=True)
    assert legs[0] < legs[1] and legs[1] == pytest.approx(legs[2]) and legs[3] < legs[0]


@pytest.mark.parametrize("prompt", ["a robot dog", "a hexapod robot"])
def test_the_joint_motor_is_drawn_exactly_once(prompt):
    """The visual mesh USED to fuse the actuator envelope AND the compiler emitted a datasheet ``_act`` geom for
    the same joint. Two coincident surfaces z-fight, ringing every joint with a sawtooth seam."""
    import mujoco

    from virturoid.services.cad_geometry import build_visual_meshes
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.morphology_composer import compose_robot

    gene = compose_robot(prompt)
    ground_and_repair(gene)
    m = mujoco.MjModel.from_xml_string(
        gene_to_meshed_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    acts = [i for i in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_act")]
    assert acts, f"{prompt!r}: the render path stopped showing real actuators at all"

    # The render path must ask for meshes WITHOUT the fused housing, and the flag must actually reach the bake:
    # exactly the segments that HAVE a datasheet actuator change, and nothing else does.
    import tempfile

    from virturoid.services.component_geometry import actuator_for_joint
    with tempfile.TemporaryDirectory() as td:
        withf = build_visual_meshes(gene, td + "/a", actuator_in_mesh=True)
        without = build_visual_meshes(gene, td + "/b", actuator_in_mesh=False)
        assert withf and without
        changed, unchanged_actuated = [], []
        for s in gene.segments:
            if s.name not in withf:
                continue
            same = _read(withf[s.name]) == _read(without[s.name])
            if actuator_for_joint(s) is None:
                assert same, f"{s.name} has no actuator, yet the flag changed its mesh"
            elif same:
                unchanged_actuated.append(s.name)
            else:
                changed.append(s.name)
    assert changed and not unchanged_actuated, (
        f"actuator_in_mesh did not reach the bake for {unchanged_actuated or 'any segment'} -- the render path is "
        "still fusing a second copy of the motor into the mesh, on top of the _act geom the compiler emits")


def test_the_foot_pad_is_a_sole_not_a_brick():
    """``foot_pad`` laid a heel->toe polyline in XY and THEN extruded 0.9L along z, so the part was long in x AND
    long in z: a ~174 x 114 x 144 mm cube for a humanoid foot. A sole is long forward, wide across, THIN."""
    pytest.importorskip("build123d")
    from virturoid.services.cad_geometry import build_anatomy

    part = build_anatomy("foot_pad", 0.16, 0.06)          # the humanoid's own foot dims
    bb = part.bounding_box()
    dx, dy, dz = bb.size.X, bb.size.Y, bb.size.Z          # x = thickness, y = width, z = forward (the link axis)
    assert dz > 1.4 * dx, f"foot is {dz:.0f} mm forward but {dx:.0f} mm thick -- that is a block, not a sole"
    assert dy > 1.4 * dx, f"foot is {dy:.0f} mm wide but {dx:.0f} mm thick -- that is a block, not a sole"


@pytest.mark.parametrize("prompt", ["a quadruped walking robot", "a humanoid robot"])
def test_nothing_visual_hangs_below_the_surface_the_robot_stands_on(prompt):
    """The rubber boot was placed assuming a CAPSULE foot (distal cap at L + R), but the composer emits BOX feet
    (flat at L), so every walking body wore a pad protruding a full foot-radius -- 38 mm on a quadruped -- below
    its own contact surface. It read as the robot sinking into the floor, and it hid inside the spawn height."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.morphology_composer import compose_robot

    gene = compose_robot(prompt)
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lowest, name = 1e9, ""
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c, h = m.geom_aabb[i, :3], m.geom_aabb[i, 3:]
        R = d.geom_xmat[i].reshape(3, 3)
        v = float(d.geom_xpos[i, 2] + R[2] @ c - np.abs(R[2]) @ h)
        if v < lowest:
            lowest, name = v, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    assert lowest > -0.03, f"{prompt!r}: {name} sits {-lowest * 1000:.0f} mm below the floor at spawn"


def test_the_prior_gate_accepts_a_real_quadruped_leg():
    """The stride-link aspect band (>= 2.2 L/D) was applied to EVERY leg segment including the proximal
    abduction block -- which is a stub on every real quadruped (a Go2's is ~1.0). It rejected our dog the moment
    the legs stopped being equal stubs, even though the same change took the stride links from 2.27 (barely
    clearing, i.e. sausages) to 4.33/5.28 -- dead centre of the 'real quads ~4-6' the band itself cites."""
    from virturoid.services.anatomy_compiler import build_from_anatomy, _generic_legged_graph
    from virturoid.services.morphology_priors import validate_morphology

    r = validate_morphology(build_from_anatomy(_generic_legged_graph(n_pairs=2, fan=True, girth=0.18)))
    assert r.ok, f"the prior gate rejects our own fanned reference quadruped: {r.issues}"
    assert r.measured.get("min_leg_aspect", 0) >= 3.0, (
        f"stride links are back to sausages: {r.measured.get('min_leg_aspect')}")
    assert "min_hip_aspect" in r.measured, "the hip block is no longer reported separately"
