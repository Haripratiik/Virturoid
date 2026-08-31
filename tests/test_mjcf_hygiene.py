import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")


def test_compiled_gene_has_stable_joint_and_export_hygiene():
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a robot dog", llm=None, ensure_walkable=True)
    xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene, meshed=False))
    model = mujoco.MjModel.from_xml_string(xml)
    assert 'integrator="implicitfast"' in xml
    assert 'frictionloss="0.05"' in xml
    assert '<default class="visual">' in xml and '<default class="collision">' in xml
    assert model.nkey >= 1 and any(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) == "home" for i in range(model.nkey))
    assert model.nu == len(gene.actuated_joints())
    assert all(bool(model.actuator_ctrllimited[i]) for i in range(model.nu))
    assert model.nsensor >= 2 + 2 * len(gene.actuated_joints())
    contact_names = {s.name for s in gene.segments if s.joint_type in (None, "fixed")
                     and any(token in s.name.lower() for token in ("foot", "paw", "leg", "hoof"))}
    foot_ids = [i for i in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").removesuffix("_geom")
                in contact_names and int(model.geom_contype[i])]
    assert foot_ids
    assert all(int(model.geom_condim[i]) == 6 and int(model.geom_priority[i]) == 1 for i in foot_ids)


def test_six_axis_prompt_exports_six_arm_axes_plus_gripper_controls():
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot("a 6-axis robot arm with a gripper", llm=None)
    arm = [s for s in gene.segments if s.joint_type == "revolute" and "finger" not in s.name]
    assert len(arm) == 6
    model = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
    assert model.nu == 6 + 2  # six arm axes and two independently controlled jaws
    arm_dofs = []
    for segment in arm:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{segment.name}_joint")
        arm_dofs.append(int(model.jnt_dofadr[joint_id]))
    assert all(model.dof_armature[dof] >= 0.1 for dof in arm_dofs)
    assert all(model.dof_frictionloss[dof] >= 0.2 for dof in arm_dofs)
    assert all(model.dof_damping[dof] >= 1.0 for dof in arm_dofs)
