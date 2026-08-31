"""An imported robot must stand in the stance ITS OWN DESCRIPTION ships, not at qpos 0.

A robot description carries the pose its designers intended as a named keyframe, and for a legged robot that
pose IS the design: a Unitree Go2's `home` key is base z=0.27 with every leg at (0, 0.9, -1.8) -- hip forward,
knee folded back. 45 of the 74 MuJoCo Menagerie models ship one (`home`, `stand`, `standing`, `retract`), so it
is the ecosystem convention rather than one vendor's quirk.

We ignored it and spawned every imported robot at qpos 0, which for a quadruped means STRAIGHT LEGS -- a stance
no real quadruped uses. Measured on the Go2 that read 0.601 m tall against a real 0.394 m, and it is why an
imported Go2 verified CROUCH/FELL: the verdict was judging a pose the robot was never designed to hold.

Our own composer has always baked a bent-knee rest pose (morphology_composer: thigh -0.55, calf +1.10) and
gene_compiler emits it as a MuJoCo keyframe. Only the IMPORT path never populated it.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MEN = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="import needs MuJoCo")


def _import(rel: str):
    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot
    return import_robot(str(src), robot_id="t")


def test_the_source_standing_pose_is_carried_into_the_gene():
    """The Go2's home key bends both leg joints; the imported gene must carry those angles."""
    out = _import("unitree_go2/go2.xml")
    gene = out["gene"]
    assert gene is not None, out.get("warnings")
    pose = (gene.metadata or {}).get("rest_pose") or {}
    assert pose, "no rest_pose imported — the source's own standing keyframe was dropped"
    assert "home" in str((gene.metadata or {}).get("rest_pose_source", "")).lower()
    # go2.xml: qpos "... 0 0.9 -1.8" per leg. Thigh +0.9, calf -1.8, abduction 0 (so it is absent from the pose).
    thighs = [v for k, v in pose.items() if "thigh" in k.lower()]
    calves = [v for k, v in pose.items() if "calf" in k.lower()]
    assert len(thighs) == 4 and len(calves) == 4, f"expected 4 thighs + 4 calves, got {sorted(pose)}"
    assert all(abs(v - 0.9) < 1e-3 for v in thighs), thighs
    assert all(abs(v + 1.8) < 1e-3 for v in calves), calves


def test_the_imported_body_stands_at_a_realistic_height():
    """Straight legs made the Go2 read 0.601 m tall; its real standing height is 0.394 m. The stance is what
    closes most of that, so pin the outcome rather than the mechanism."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    gene = _import("unitree_go2/go2.xml")["gene"]
    assert gene is not None
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    pts = []
    for i in range(m.ngeom):
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c, h = m.geom_aabb[i, :3], m.geom_aabb[i, 3:]
        R = d.geom_xmat[i].reshape(3, 3)
        ctr = d.geom_xpos[i] + R @ c
        e = np.abs(R) @ h
        pts += [ctr - e, ctr + e]
    p = np.array(pts)
    height = float(p.max(axis=0)[2] - p.min(axis=0)[2])
    assert height < 0.55, (
        f"the imported Go2 stands {height:.3f} m tall against a real 0.394 m — it is posed with straight legs, "
        "which is not a stance any quadruped holds")


def test_the_keyframe_agrees_with_the_orientation_baked_into_the_root_body():
    """A free base's keyframe quaternion must equal the one MuJoCo derives from the root ``<body euler=...>``.

    ``_pose_keyframe`` hardcoded identity, which is right only for a composed body (unrotated root). An imported
    root carries the rotation that aligns its reconstructed link frame, so the keyframe silently re-oriented the
    whole robot: the Go2 pitched ~19 deg and measured 0.803 m tall. Pin the INVARIANT, not the Go2's numbers --
    the two must agree for every model, however the root happens to be rotated."""
    import mujoco
    import numpy as np

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    gene = _import("unitree_go2/go2.xml")["gene"]
    assert gene is not None
    assert any(abs(v) > 1e-6 for v in (getattr(gene.root(), "mount_euler", None) or (0.0,))), \
        "this model's root is unrotated, so it cannot exercise the bug"
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    assert m.nkey, "no keyframe emitted"
    # key_qpos is what a rollout/render resets to; qpos0 is what the body tag itself declares. Both the euler
    # and the qpos are serialized with 5 decimals, so 1e-5 is the precision floor no code change can beat.
    np.testing.assert_allclose(m.key_qpos[0][3:7], m.qpos0[3:7], atol=1e-5)
    """Not a legged-only fix: an arm's `home` key is its ready pose (a Franka FR3 folds its elbow to -1.57)."""
    out = _import("franka_fr3/fr3.xml")
    gene = out["gene"]
    assert gene is not None, out.get("warnings")
    pose = (gene.metadata or {}).get("rest_pose") or {}
    assert pose, "the arm's home pose was dropped; it would render bolt-upright instead of ready"
    assert any(abs(v) > 1.0 for v in pose.values()), f"no bent joint in the imported pose: {pose}"


def test_a_model_without_a_keyframe_is_left_alone():
    """No keyframe must mean no rest_pose, not an invented one — never fabricate a stance we were not given."""
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.robot_import import import_robot

    bare = RobotGene(id="bare", species="test", robot_class="manipulator", base_mount="table", segments=[
        GeneSegment(name="base", parent=None, shape="box", length_m=0.1, radius_m=0.05, mass_kg=1.0),
        GeneSegment(name="link", parent="base", shape="capsule", length_m=0.2, radius_m=0.03, mass_kg=0.5,
                    joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.5, joint_upper=1.5,
                    actuator_torque_nm=8.0, is_end_effector=True)])
    xml = compile_gene_to_mjcf(bare, include_floor=True)
    out = import_robot(xml, robot_id="rt")
    assert out["gene"] is not None
    # Round-tripping our own keyframe-less model must not conjure angles from nowhere.
    assert not ((out["gene"].metadata or {}).get("rest_pose") or {}), \
        (out["gene"].metadata or {}).get("rest_pose")
