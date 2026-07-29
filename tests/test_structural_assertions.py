import importlib.util

import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")


def test_reference_authored_bodies_have_mated_visible_seams():
    from virturoid.services.agent_design_tools import get_design_schema
    from virturoid.services.anatomy_compiler import build_from_anatomy
    from virturoid.services.structural_assertions import evaluate_structural_assertions

    for graph in get_design_schema({})["examples"].values():
        report = evaluate_structural_assertions(build_from_anatomy(graph))
        assert report.ok, [a.detail for a in report.assertions if not a.ok]
        assert report.topology_edges


def test_detached_child_is_rejected():
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.structural_assertions import evaluate_structural_assertions

    gene = RobotGene(
        id="detached", species="test", robot_class="new_class", base_mount="table", end_effector_type="none",
        segments=[
            GeneSegment("root", shape="box", length_m=0.2, radius_m=0.1, mass_kg=1.0),
            GeneSegment("tool", parent="root", joint_type="fixed", length_m=0.1, radius_m=0.02,
                        mass_kg=0.1, mount_offset=(1.0, 0.0, 0.0), is_end_effector=True),
        ],
    )
    report = evaluate_structural_assertions(gene)
    assert not report.ok
    assert any(not assertion.ok and assertion.name == "seam:root->tool" for assertion in report.assertions)
