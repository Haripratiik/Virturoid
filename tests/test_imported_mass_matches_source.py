"""An imported twin must weigh what the customer's robot weighs, and must not gain parts on the way in.

Two defects, one mechanism: the importer was inventing SEGMENTS, and every invented segment carries mass.

  * THE SYNTHESIZED WELDED BASE CARRIED A COPY OF THE FIRST ROOT'S MASS. ``robot_import`` synthesizes a base
    segment whenever the model has several roots (or one actuated root) and reparents the real roots onto it.
    That base was given ``max(root_segs[0].mass_kg, 0.05)`` -- the mass of a real link, duplicated. Measured
    source-vs-twin totals at HEAD, both computed from the SOURCE model rather than a fixture:

        aloha             8.517 -> 9.486 kg   (+0.969, +11.4%)
        trossen_wxai      7.823 -> 8.080 kg   (+0.258,  +3.3%)
        shadow_dexee      4.139 -> 4.679 kg   (+0.540, +13.0%)
        apptronik_apollo 80.898 -> 88.344 kg  (+7.446,  +9.2%)

    Nothing was being recovered by the copy. When MuJoCo fuses a URDF's fixed base link into the world it
    DISCARDS that link's inertia (measured: a base link declaring 3.5 kg compiles to ``nbody == 2`` with
    ``body_mass[0] == 0``), so the source's own total does not contain it either -- and duplicating a
    different link's mass is not a reconstruction of it. The base is a mounting frame, so it now carries a
    nominal 0.001 kg, which is the smallest value ``RobotGene.validate()`` (mass > 0) and the repo's coarsest
    mass rounding (3 dp) permit. That residual is stated in the import warning WITH ITS NUMBER.

  * A COORDINATE FRAME IS NOT A LINK. Bodies with no mass, no geom and no joint exist in real models purely to
    name a pose -- shadow_dexee's ``F0/ F1/ F2/`` (the placement frames its three <attach>-ed fingers arrive
    with), apollo's ``world_link``, and a dozen sensor/attachment datums across the corpus. Emitted as
    segments they became physical stubs: 0.01 kg from the importer's zero-mass floor, plus a collision
    primitive from the compiler, i.e. parts with contact surfaces the customer never built.

    The choice was measured, not guessed. Over 100 loadable Menagerie models there are exactly 18 zero-mass
    bodies and ALL 18 are of this kind -- none has a geom, none has a joint -- so there is no "massless but
    real link" case that dropping them would damage. They are folded into their children with their
    transforms COMPOSED, so no surviving link moves; the ``F0/ F1/ F2/`` frames carry the entire finger
    placement (0.052 m of offset each, plus 161.3 degrees of yaw on two of the three), which would otherwise
    stack the three fingers on one point facing one way.

And one difference that is NOT a defect of ours and cannot be removed here, so it is REPORTED. MuJoCo's URDF
loader welds the static base into the world and discards its ``<inertial>``: a fixed-base URDF quadruped
declaring 5.200 kg imports as 3.200 kg, 38.5% light, because its 2.000 kg trunk is that base. The old base
copy half-covered that hole with an unrelated 0.8 kg, which is exactly the failure mode this repo keeps
hitting -- a fitted number standing in for a modelling difference. The number now ships in the warnings, with
the link named, cross-checked against the compiled total so an unreconcilable parse claims nothing.
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

# The four packages the defect was measured on: every model whose import synthesizes a welded base, plus
# apollo, whose only extra "root" turned out to be a coordinate frame.
MEASURED = [
    "aloha/aloha.xml",
    "trossen_wxai/trossen_ai_bimanual.xml",
    "shadow_dexee/shadow_dexee.xml",
    "apptronik_apollo/apptronik_apollo.xml",
]

# A frame body in the middle of a chain, carrying BOTH an offset and a rotation, with two children hanging off
# it and a sibling that does not. If the frame is dropped without composing its transform, the two children
# move relative to the sibling and the pairwise-distance check below catches it. `mount` is legal MJCF
# precisely because it has no joint: MuJoCo only demands inertia of bodies that can move.
_FRAMED = """
<mujoco model="framed">
  <worldbody>
    <body name="trunk" pos="0 0 0.5">
      <geom type="box" size="0.10 0.10 0.05" mass="2.0"/>
      <body name="mount" pos="0.20 0.05 -0.03" euler="0 0 40">
        <body name="armA" pos="0 0.10 -0.15">
          <joint name="ja" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.02" mass="1.0"/>
        </body>
        <body name="armB" pos="0 -0.10 -0.15">
          <joint name="jb" type="hinge" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.02" mass="1.0"/>
        </body>
      </body>
      <body name="tail" pos="-0.20 0 0">
        <joint name="jt" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 -0.15 0 0" size="0.02" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="ja" gear="10"/><motor joint="jb" gear="10"/><motor joint="jt" gear="10"/>
  </actuator>
</mujoco>
"""


def _source_total_kg(path_or_xml: str) -> float:
    """The source model's OWN total mass, computed from the source. Never a fixture constant."""
    import mujoco
    m = (mujoco.MjModel.from_xml_path(path_or_xml) if os.path.exists(path_or_xml)
         else mujoco.MjModel.from_xml_string(path_or_xml))
    return float(sum(m.body_mass[i] for i in range(1, m.nbody)))


def _twin_total_kg(gene) -> float:
    return float(sum(s.mass_kg for s in gene.segments))


def _import(rel_or_xml: str, prefix: str):
    from virturoid.services.robot_import import import_robot

    if rel_or_xml.lstrip().startswith("<"):
        return import_robot(rel_or_xml, robot_id=prefix)
    src = _MEN / rel_or_xml
    if not src.exists():
        pytest.skip(f"{rel_or_xml} is not in the local Menagerie cache")
    return import_robot(str(src), robot_id=f"{prefix}_{Path(rel_or_xml).stem}")


def _pairwise(points):
    import numpy as np
    P = np.asarray(points, dtype=float)
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)


def _shape_error_m(source_path: str, gene) -> float:
    """Largest disagreement in the PAIRWISE DISTANCE matrix over the links the source and twin share.

    Invariant under any rigid transform, so a legitimate re-datum scores zero while a deformation cannot hide.
    """
    import mujoco
    import numpy as np
    from virturoid.services.morph_policy import compiled_model, robot_mjcf

    sm = (mujoco.MjModel.from_xml_path(source_path) if os.path.exists(source_path)
          else mujoco.MjModel.from_xml_string(source_path))
    sd = mujoco.MjData(sm)
    mujoco.mj_forward(sm, sd)
    sp = {mujoco.mj_id2name(sm, mujoco.mjtObj.mjOBJ_BODY, b): sd.xpos[b].copy() for b in range(1, sm.nbody)}

    tm = compiled_model(robot_mjcf(gene))
    td = mujoco.MjData(tm)
    mujoco.mj_forward(tm, td)
    tp = {mujoco.mj_id2name(tm, mujoco.mjtObj.mjOBJ_BODY, b): td.xpos[b].copy() for b in range(1, tm.nbody)}

    shared = sorted(set(sp) & set(tp))
    assert len(shared) >= 3, f"only {shared} shared links; the comparison would be vacuous"
    return float(np.max(np.abs(_pairwise([sp[n] for n in shared]) - _pairwise([tp[n] for n in shared]))))


# ------------------------------------------------------------------ the twin weighs what the robot weighs
@pytest.mark.parametrize("rel", MEASURED, ids=[r.split("/")[0] for r in MEASURED])
def test_the_twin_weighs_what_the_source_weighs(rel):
    """The headline. Compared against the SOURCE's own total, computed here, not a recorded number."""
    from virturoid.services.robot_import import _SYNTH_BASE_MASS_KG

    src = _MEN / rel
    if not src.exists():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    want = _source_total_kg(str(src))
    out = _import(rel, "mass")
    assert out["gene"] is not None, out.get("warnings")
    got = _twin_total_kg(out["gene"])

    # The ONLY admissible difference is the synthesized base, and only when one was synthesized at all.
    n_synth = sum(1 for s in out["gene"].segments if s.mass_kg == _SYNTH_BASE_MASS_KG and s.parent is None)
    slack = _SYNTH_BASE_MASS_KG * n_synth + 5e-4          # + float/rounding noise on a per-link sum
    assert abs(got - want) <= slack, (
        f"{rel}: the customer's robot weighs {want:.3f} kg and our twin weighs {got:.3f} kg "
        f"({got - want:+.3f} kg, {100 * (got - want) / want:+.1f}%). An import may lose a link's mass to a "
        f"modelling limit and SAY so; it may not silently add mass.")


@pytest.mark.parametrize("rel", MEASURED, ids=[r.split("/")[0] for r in MEASURED])
def test_the_synthesized_base_is_not_a_copy_of_a_real_links_mass(rel):
    """The mechanism, as its own gate. The regression is specifically 'the base weighs what some link weighs'."""
    from virturoid.services.robot_import import _SYNTH_BASE_MASS_KG

    out = _import(rel, "basemass")
    gene = out["gene"]
    assert gene is not None
    root = gene.root()
    if not any("synthesized a welded base segment" in w for w in out["warnings"]):
        pytest.skip(f"{rel} keeps its own root ({root.name}); no base is synthesized")
    assert root.mass_kg == _SYNTH_BASE_MASS_KG, (
        f"{rel}: the synthesized base {root.name!r} weighs {root.mass_kg} kg. It is a mounting frame with no "
        f"counterpart in the source; it may not be given a real link's mass.")
    others = [s.mass_kg for s in gene.segments if s is not root]
    assert root.mass_kg < 0.01 * max(others), (
        f"{rel}: the base's {root.mass_kg} kg is not negligible against the body's links (max {max(others)})")


@pytest.mark.parametrize("rel", MEASURED, ids=[r.split("/")[0] for r in MEASURED])
def test_the_residual_is_disclosed_with_its_number(rel):
    """A residual we cannot remove must ship as a NUMBER the customer can read, not be tuned out of sight."""
    from virturoid.services.robot_import import _SYNTH_BASE_MASS_KG

    out = _import(rel, "disclose")
    assert out["gene"] is not None
    synth = [w for w in out["warnings"] if "synthesized a welded base segment" in w]
    if not synth:
        pytest.skip(f"{rel} keeps its own root; no base is synthesized")
    assert f"{_SYNTH_BASE_MASS_KG:g} kg" in synth[0], (
        f"the disclosure must state the residual mass itself: {synth[0]}")
    assert "MOUNTING FRAME" in synth[0], synth[0]


# ---------------------------------------------------------------- a coordinate frame is not a robot part
def test_a_massless_geomless_jointless_body_is_not_emitted_as_a_part():
    out = _import(_FRAMED, "frame_syn")
    gene = out["gene"]
    assert gene is not None, out["warnings"]
    names = [s.name for s in gene.segments]
    assert "mount" not in names, (
        f"'mount' is a coordinate frame -- no mass, no geom, no joint -- and was emitted as a segment: {names}. "
        f"Every segment gets mass and a collision primitive, so that is a part the source does not have.")
    assert {"trunk", "armA", "armB", "tail"} <= set(names), names
    kids = {s.name: s.parent for s in gene.segments}
    assert kids["armA"] == "trunk" and kids["armB"] == "trunk", (
        f"the frame's children must re-attach to the link above it, not vanish: {kids}")


def test_folding_a_frame_carries_its_transform_into_its_children():
    """The frame holds 0.20/0.05/-0.03 m of offset and 40 degrees of yaw. Dropping it without composing that
    in would slide both arms relative to the tail -- which pairwise distances detect and a re-datum does not."""
    out = _import(_FRAMED, "frame_kin")
    err = _shape_error_m(_FRAMED, out["gene"])
    assert err < 1e-3, f"folding 'mount' deformed the body by {1000 * err:.2f} mm"


def test_the_frames_mass_is_not_invented():
    """A frame weighs nothing in the source, so it may not weigh 0.01 kg in the twin."""
    want = _source_total_kg(_FRAMED)          # 2.0 + 1.0 + 1.0 + 0.5
    out = _import(_FRAMED, "frame_mass")
    got = _twin_total_kg(out["gene"])
    assert abs(got - want) < 1e-6, f"source {want:.4f} kg, twin {got:.4f} kg"


def test_a_massless_body_that_has_geometry_is_still_a_part():
    """The counter-case, so the rule cannot widen into 'drop anything light'. A body with a COLLISION GEOM is
    a real part of the machine whatever its declared inertia; it keeps the zero-mass floor and a warning."""
    xml = _FRAMED.replace(
        '<body name="mount" pos="0.20 0.05 -0.03" euler="0 0 40">',
        '<body name="mount" pos="0.20 0.05 -0.03" euler="0 0 40">'
        '<geom type="box" size="0.03 0.03 0.03" mass="0"/>')
    out = _import(xml, "frame_geom")
    gene = out["gene"]
    assert gene is not None, out["warnings"]
    seg = next((s for s in gene.segments if s.name == "mount"), None)
    assert seg is not None, [s.name for s in gene.segments]
    assert seg.mass_kg > 0
    assert any("zero/negative mass" in w and "mount" in w for w in out["warnings"]), out["warnings"]


# --------------------------------------------------- the residual we cannot remove, reported with its number
_FIXED_BASE_QUAD = """<?xml version="1.0"?>
<robot name="massquad">
  <link name="trunk"><inertial><mass value="2.0"/>
    <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><box size="0.4 0.2 0.1"/></geometry></collision></link>
  {legs}
</robot>"""
_LEG = """
  <link name="{n}"><inertial><mass value="0.8"/>
    <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><geometry><cylinder radius="0.03" length="0.25"/></geometry></collision></link>
  <joint name="{n}_j" type="revolute"><parent link="trunk"/><child link="{n}"/>
    <axis xyz="0 1 0"/><origin xyz="{x} {y} 0"/>
    <limit lower="-1" upper="1" effort="20" velocity="5"/></joint>"""


def _fixed_base_quad_urdf() -> str:
    legs = "".join(_LEG.format(n=n, x=x, y=y) for n, x, y in
                   [("FL", 0.18, 0.1), ("FR", 0.18, -0.1), ("HL", -0.18, 0.1), ("HR", -0.18, -0.1)])
    return _FIXED_BASE_QUAD.format(legs=legs)


def test_a_urdfs_static_base_mass_is_reported_not_replaced():
    """MuJoCo's URDF loader welds the static base into the world and DISCARDS its <inertial>. That is a real
    2.000 kg of a 5.200 kg declared robot -- 38.5% -- and it is not recoverable at this layer.

    It used to be half-hidden: the synthesized base was given a copy of the first root's mass (0.8 kg, a leg's)
    which covered part of the hole with an unrelated number. The requirement is the opposite -- state the
    figure. This test pins BOTH halves: the base is not padded, and the loss is disclosed with its kilograms.
    """
    import mujoco
    from virturoid.services.robot_import import _SYNTH_BASE_MASS_KG, _urdf_world_fused_links

    urdf = _fixed_base_quad_urdf()
    fused, declared = _urdf_world_fused_links(urdf)
    assert fused == {"trunk": 2.0} and abs(declared - 5.2) < 1e-9, (fused, declared)
    compiled = float(sum(mujoco.MjModel.from_xml_string(urdf).body_mass))
    assert abs(compiled - 3.2) < 1e-6, f"MuJoCo kept {compiled} kg; the scan and the compile must reconcile"

    out = _import(urdf, "fusedbase")
    gene = out["gene"]
    assert gene is not None, out["warnings"]
    assert gene.root().mass_kg == _SYNTH_BASE_MASS_KG, (
        f"the base was padded to {gene.root().mass_kg} kg to cover the fused trunk instead of disclosing it")
    assert abs(_twin_total_kg(gene) - (compiled + _SYNTH_BASE_MASS_KG)) < 1e-6

    told = [w for w in out["warnings"] if "your URDF declares" in w]
    assert told, f"the 2.000 kg loss was not disclosed at all: {out['warnings']}"
    assert "2.000 kg" in told[0] and "5.200 kg" in told[0] and "trunk" in told[0], told[0]


def test_the_fused_mass_claim_is_not_made_when_it_cannot_be_reconciled():
    """A number we cannot check is worse than no number: the text scan is only believed when
    declared == compiled + fused. An MJCF has no <robot> element at all and must never trigger the claim."""
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.fixtures.gene_library import tabletop_arm_gene

    out = _import(compile_gene_to_mjcf(tabletop_arm_gene(), include_floor=False), "nofuse")
    assert not [w for w in out["warnings"] if "your URDF declares" in w], out["warnings"]


def test_a_model_of_nothing_but_frames_is_refused_not_invented():
    """The degenerate end of the same rule, and the one that would look healthiest if we got it wrong.

    With every body folded away, the base synthesis would run on an empty segment list and produce a single
    0.05 m box reported ``valid=True, simulable=True`` — a robot that exists nowhere but in our output. Refuse
    it, the way an <include> fragment is refused, and name what the file actually is.
    """
    xml = ('<mujoco model="allframes"><worldbody>'
           '<body name="f0" pos="0 0 1"><body name="f1" pos="0.1 0 0"/></body>'
           '</worldbody></mujoco>')
    out = _import(xml, "allframes")
    assert out["gene"] is None, [s.name for s in out["gene"].segments]
    assert out["valid"] is False and out["simulable"] is False
    assert "COORDINATE FRAME" in out["warnings"][0], out["warnings"]


@pytest.mark.parametrize("rel,frames", [
    ("shadow_dexee/shadow_dexee.xml", ["F0/", "F1/", "F2/"]),
    ("apptronik_apollo/apptronik_apollo.xml", ["world_link"]),
    ("franka_emika_panda/panda_nohand.xml", ["attachment"]),
    ("hello_robot_stretch_3/stretch.xml", ["realsense", "base_imu", "d405_cam", "head_nav_cam",
                                           "link_grasp_center"]),
], ids=["shadow_dexee", "apptronik_apollo", "panda_nohand", "stretch_3"])
def test_real_models_frames_are_folded_and_the_body_is_untouched(rel, frames):
    """The same rule on real files, including the two that carried it into the mass numbers above.

    Apollo is the pure case: ``world_link`` is an empty datum with no children, so folding it costs nothing at
    all and leaves apollo with a SINGLE root -- it no longer needs a synthesized base. shadow_dexee is the
    hard case: its three frames each carry a finger's whole placement.
    """
    src = _MEN / rel
    if not src.exists():
        pytest.skip(f"{rel} is not in the local Menagerie cache")
    out = _import(rel, "realframe")
    gene = out["gene"]
    assert gene is not None, out.get("warnings")
    emitted = {s.name for s in gene.segments}
    assert not (emitted & set(frames)), f"{rel}: coordinate frames emitted as segments: {emitted & set(frames)}"
    assert any("COORDINATE FRAMES" in w for w in out["warnings"]), (
        f"{rel}: folding a frame changes the segment list and must be disclosed: {out['warnings'][:3]}")
    err = _shape_error_m(str(src), gene)
    assert err < 2e-3, f"{rel}: folding {frames} deformed the body by {1000 * err:.2f} mm"
    assert out["simulable"] is True, out.get("simulation_check")


def test_apollos_only_root_is_its_own_base_link():
    """Apollo used to import as multi-root because ``world_link`` counted as a body. With the frame folded it
    has one root and keeps the customer's own ``base_link`` as the gene root -- no base is synthesized, and its
    twin's mass matches the source exactly rather than to within the base residual."""
    rel = "apptronik_apollo/apptronik_apollo.xml"
    if not (_MEN / rel).exists():
        pytest.skip("apptronik_apollo is not in the local Menagerie cache")
    out = _import(rel, "apolloroot")
    gene = out["gene"]
    assert gene.root().name == "base_link", gene.root().name
    assert not any("synthesized a welded base segment" in w for w in out["warnings"]), out["warnings"][:3]
    want = _source_total_kg(str(_MEN / rel))
    assert abs(_twin_total_kg(gene) - want) < 5e-4, (want, _twin_total_kg(gene))


def test_an_ee_site_parked_on_a_folded_frame_is_still_found():
    """panda_nohand puts ``attachment_site`` on the empty ``attachment`` body. A frame has no joint, so the
    site is welded to the link above it; losing the hint with the frame would silently change which link the
    twin calls its end effector."""
    rel = "franka_emika_panda/panda_nohand.xml"
    if not (_MEN / rel).exists():
        pytest.skip("franka_emika_panda is not in the local Menagerie cache")
    out = _import(rel, "eeframe")
    gene = out["gene"]
    ees = [s.name for s in gene.segments if s.is_end_effector]
    assert len(ees) == 1, ees
    assert ees[0] == "link7", (
        f"the end effector should be the link `attachment` hangs off, not a fallback leaf: {ees}")
