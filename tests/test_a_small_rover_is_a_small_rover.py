"""A 0.36 m rover is built, billed and driven as a 0.36 m rover.

MEASURED 2026-10-11 through the MCP server, "a six-wheeled rover" (a 0.36 m deck on 6 cm wheels):

* it weighed 70.0 kg: four or more wheels selected the WAREHOUSE-AMR mass envelope (70-150 kg, machines
  0.8-0.9 m long), and the balance of system topped the body's own 13 kg up to the envelope's floor with 57 kg
  of estimated wiring, fasteners and covers;
* its parts list bought every wheel's drive twice (6x T-Motor AK80-9 for the wheel joints AND 6x Pololu 37D
  gearmotors) and a caster the six-wheeler does not have, $4,644 in all;
* all six motors and six hubs rode on ``wheel_0``, 4.4x the mass of each of its siblings.

Once it weighed what it is, it was fitted with 0.52 N.m servos, and the compiler's fixed wheel damping
(0.25 N.m.s) then held those wheels to 1.6 rad/s: the rover crawled 0.14 m in the drive test. A wheel's viscous
loss is now its own motor's back-EMF (stall torque over no-load speed). The same drive test also stopped torquing
a mobile manipulator's arm out of its stowed pose while it drove (it TIPPED at 70 kg and spun in place at 13).
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _grounded(prompt):
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot(prompt, llm=None, ensure_walkable=False)
    ground_and_repair(gene, task=prompt)
    return gene


def test_a_small_rover_keeps_its_own_mass():
    from virturoid.services.grounded_physics import physical_prior_for
    gene = _grounded("a six-wheeled rover")
    assert physical_prior_for(gene) is None
    assert sum(s.mass_kg for s in gene.segments) < 20.0
    assert not ((gene.metadata.get("embodied_mass") or {}).get("balance_of_system_kg") or {})


def test_each_wheel_is_driven_by_one_motor_and_a_caster_only_where_there_is_one():
    from virturoid.services.bom_builder import build_bom
    six = build_bom(_grounded("a six-wheeled rover"), task="a six-wheeled rover")
    cats = [ln["category"] for ln in six["lines"]]
    assert "drive_motor" not in cats, "the wheel joints' actuators ARE the drive motors"
    assert sum(ln["qty"] for ln in six["lines"] if ln["category"] == "actuator") == 6
    assert not any("caster" in ln["part"].lower() for ln in six["lines"])
    two = build_bom(_grounded("a two-wheeled delivery robot"), task="a two-wheeled delivery robot")
    assert sum(ln["qty"] for ln in two["lines"] if "caster" in ln["part"].lower()) == 1


def test_a_part_bought_per_wheel_rides_on_every_wheel():
    gene = _grounded("a six-wheeled rover")
    wheels = [s for s in gene.segments if s.name.startswith("wheel_")]
    assert len(wheels) == 6
    carried = (gene.metadata.get("embodied_mass") or {}).get("component_kg") or {}
    shares = {round(float(carried.get(w.name, 0.0)), 6) for w in wheels}
    assert len(shares) == 1 and shares != {0.0}, carried
    masses = {round(w.mass_kg, 6) for w in wheels}
    assert len(masses) == 1, [(w.name, w.mass_kg) for w in wheels]


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
def test_a_wheels_damping_is_its_motors_back_emf():
    import mujoco

    from virturoid.services.component_catalog import resolve_part
    from virturoid.services.bom_builder import build_bom
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    gene = _grounded("a six-wheeled rover")
    part = resolve_part(build_bom(gene)["actuator_map"]["wheel_0"])
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=True))
    dof = int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "wheel_0_joint")])
    assert float(m.dof_damping[dof]) == pytest.approx(part.peak_torque_nm / part.max_speed_radps, rel=1e-3)


@pytest.mark.skipif(not _MUJOCO, reason="needs MuJoCo")
@pytest.mark.parametrize("prompt", ["a six-wheeled rover", "a mobile manipulator"])
def test_it_drives(prompt):
    from virturoid.services.ai_native_tools import _honest_drive
    r = _honest_drive(_grounded(prompt))
    assert r["verdict"].startswith("DRIVES"), r
    assert r["upright_min"] > 0.9, r
