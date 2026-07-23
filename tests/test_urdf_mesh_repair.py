"""I1 (#219): the URDF repair must be broad enough that an AS-PUBLISHED Franka/KUKA/a1 -- relative mesh paths,
Collada .dae visuals, meshes not shipped alongside the file -- INGESTS (degraded but drivable) instead of
hard-failing on the first "Error opening file", and a xacro TEMPLATE fails with a named, fixable reason rather
than a cryptic crash. Every fallback is reported, never silent.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest

from virturoid.services.model_import import _resolve_mesh_path, repair_urdf_text

_MUJOCO = importlib.util.find_spec("mujoco") is not None

_FRANKA_LIKE = '''<?xml version="1.0"?>
<robot name="panda">
  <link name="base">
    <visual><geometry><mesh filename="meshes/link0.dae"/></geometry></visual>
    <collision><geometry><mesh filename="meshes/link0.stl"/></geometry></collision>
    <inertial><mass value="2.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="link1">
    <visual><geometry><mesh filename="package://franka/meshes/link1.stl" scale="1 1 1"/></geometry></visual>
    <inertial><mass value="1.5"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/><axis xyz="0 0 1"/>
    <limit lower="-2.9" upper="2.9" effort="87" velocity="2.6"/><origin xyz="0 0 0.14"/>
  </joint>
</robot>'''


def test_missing_meshes_become_boxes_and_are_reported():
    """A URDF referencing meshes that don't exist must have them replaced by placeholder boxes, reported."""
    fixed, repairs = repair_urdf_text(_FRANKA_LIKE, mesh_root=None)
    assert "<mesh" not in fixed and "<box" in fixed          # every mesh boxed
    mesh_rep = [r for r in repairs if r["kind"] == "mesh_resolve"]
    assert mesh_rep and mesh_rep[0]["boxed"] == 3            # link0.dae, link0.stl, link1.stl
    assert "kinematic structure imports" in mesh_rep[0]["detail"]


def test_resolve_mesh_path_handles_package_relative_and_basename(tmp_path):
    """The resolver finds a real mesh via the ROS-package-relative path, the mesh_root join, and the basename."""
    (tmp_path / "meshes").mkdir()
    real = tmp_path / "meshes" / "link0.stl"
    real.write_bytes(b"\x00" * 84)                            # a file that exists (content irrelevant to resolve)
    assert _resolve_mesh_path("package://franka/meshes/link0.stl", tmp_path) == real
    assert _resolve_mesh_path("meshes/link0.stl", tmp_path) == real
    assert _resolve_mesh_path("some/other/path/link0.stl", tmp_path) == real  # basename fallback
    assert _resolve_mesh_path("meshes/nope.stl", tmp_path) is None


def test_an_existing_supported_mesh_is_resolved_not_boxed(tmp_path):
    """A mesh that DOES exist in a supported format is kept (path rewritten), never needlessly boxed."""
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "link1.stl").write_bytes(b"\x00" * 84)
    urdf = ('<robot name="x"><link name="l"><visual><geometry>'
            '<mesh filename="meshes/link1.stl"/></geometry></visual></link></robot>')
    fixed, repairs = repair_urdf_text(urdf, mesh_root=tmp_path)
    assert "<mesh" in fixed and "<box" not in fixed          # kept as a mesh
    assert (tmp_path / "meshes" / "link1.stl").as_posix() in fixed
    assert [r for r in repairs if r["kind"] == "mesh_resolve"][0]["boxed"] == 0


def test_a_mesh_free_urdf_gets_no_mesh_repair():
    """A clean primitive-geometry URDF must be untouched by the mesh pass (no false repair reported)."""
    urdf = '<robot name="x"><link name="l"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link></robot>'
    _, repairs = repair_urdf_text(urdf, mesh_root=None)
    assert not any(r["kind"] == "mesh_resolve" for r in repairs)


@pytest.mark.skipif(not _MUJOCO, reason="the faithful lane needs MuJoCo")
def test_franka_like_urdf_now_imports_instead_of_hard_failing():
    from virturoid.services.model_import import import_model
    d = tempfile.mkdtemp()
    p = os.path.join(d, "panda.urdf")
    open(p, "w").write(_FRANKA_LIKE)
    r = import_model(p)
    assert r["ok"] is True, r.get("note")
    assert r["actuated"] >= 1                                 # the revolute joint imported + got a motor
    assert any(rp["kind"] == "mesh_resolve" for rp in r["repairs"])


@pytest.mark.skipif(not _MUJOCO, reason="the faithful lane needs MuJoCo")
def test_xacro_template_fails_with_an_actionable_reason():
    from virturoid.services.model_import import import_model
    d = tempfile.mkdtemp()
    p = os.path.join(d, "arm.urdf")
    open(p, "w").write('<robot name="a" xmlns:xacro="http://ros.org/wiki/xacro">'
                       '<xacro:property name="w" value="1"/><link name="base"/></robot>')
    r = import_model(p)
    assert r["ok"] is False
    assert "xacro" in r["note"].lower() and "expand" in r["note"].lower()
