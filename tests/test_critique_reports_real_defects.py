"""critique_design reports defects the body has, not ones a measurement invented.

MEASURED 2026-10-11 by composing and grounding 20 prompts and reading the three critics critique_design merges
(engineering, anatomy, structural seams). Fifteen HIGH or MED findings were false, on the bodies the product
composes most:

* every composed walker (quadruped, tripod, hexapod, octopod) had its head "visibly detached" by 0.037-0.094 m,
  and the octopus four tentacles by 0.061 m: measured on the bare colliders, while the drawn body mates;
* the quadcopter's four rotors were "detached" by 0.085 m: each sits on a welded arm that never counted;
* every humanoid and biped "would topple": the support polygon was built from foot CENTRES, and two feet are a
  line;
* the snake (279%) and the fish were told their yaw joints cannot hold the body up off the floor, which no
  vertical hinge is ever asked to do;
* every wheeled base had "vestigial stubs on a monolith", and the snake was "not left/right symmetric" for the S
  it rests in.

Each test below pins the false alarm gone AND the real defect the same rule exists for still caught.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")


def _failed(report):
    return {a.name: a.value for a in report.assertions if not a.ok}


@pytest.mark.parametrize("prompt", ["a quadruped robot", "a six-legged robot", "a three-legged robot"])
def test_a_composed_walkers_head_mates_in_the_body_that_is_drawn(prompt):
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.structural_assertions import evaluate_structural_assertions
    gene = compose_robot(prompt)
    drawn = evaluate_structural_assertions(gene)
    assert drawn.measured_on == "drawn" and drawn.ok, _failed(drawn)
    # The colliders alone still show why it used to fire: the trunk collider is deliberately shorter than the
    # trunk it draws, and the head sits at the drawn nose. Asked for explicitly, that measurement is unchanged.
    bare = evaluate_structural_assertions(gene, meshed=False)
    assert bare.measured_on == "colliders" and "seam:torso->head_0" in _failed(bare)


def test_a_rotor_welded_to_an_arm_is_attached_through_the_arm():
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.structural_assertions import evaluate_structural_assertions
    gene = compose_robot("a quadcopter drone")
    for meshed in (True, False):
        report = evaluate_structural_assertions(gene, meshed=meshed)
        assert report.ok, _failed(report)
        rotor0 = next(a for a in report.assertions if a.name == "seam:hub->rotor0")
        assert "through 'arm0'" in rotor0.detail, rotor0.detail


def _bridge_gene(sibling_joint):
    """A root, a sibling touching it, and a welded part that touches only the sibling."""
    from virturoid.schemas.gene import GeneSegment, RobotGene
    return RobotGene(
        id="bridge", species="test", robot_class="new_class", base_mount="table", end_effector_type="none",
        segments=[
            GeneSegment("root", shape="box", length_m=0.1, radius_m=0.05, mass_kg=1.0),
            GeneSegment("strut", parent="root", shape="box", length_m=0.3, radius_m=0.01, mass_kg=0.1,
                        joint_type=sibling_joint, mount_euler=(0.0, 1.5707963, 0.0), joint_lower=-1.0,
                        joint_upper=1.0),
            GeneSegment("pod", parent="root", shape="box", length_m=0.04, radius_m=0.03, mass_kg=0.1,
                        joint_type="fixed", mount_offset=(0.32, 0.0, -0.05), is_end_effector=True),
        ])


def test_only_a_welded_sibling_bridges_a_seam():
    from virturoid.services.structural_assertions import evaluate_structural_assertions
    welded = evaluate_structural_assertions(_bridge_gene("fixed"), meshed=False)
    assert welded.ok, _failed(welded)
    # A JOINTED sibling moves: the part it touched would float the moment it did, so it never carries a seam.
    jointed = evaluate_structural_assertions(_bridge_gene("revolute"), meshed=False)
    assert "seam:root->pod" in _failed(jointed)


@pytest.mark.parametrize("prompt", ["a humanoid robot", "a biped robot"])
def test_a_biped_stands_on_its_foot_plates_not_on_a_line(prompt):
    from virturoid.services.gene_validation import validate_gene_design
    from virturoid.services.morphology_composer import compose_robot
    report = validate_gene_design(compose_robot(prompt))
    assert report["checks"]["stability"] is True
    assert not any(f["check"] == "stability" for f in report["risk_flags"]), report["risk_flags"]


def test_a_body_on_one_point_still_topples():
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.gene_validation import validate_gene_design
    base = GeneSegment(name="b", parent=None, shape="capsule", length_m=0.04, radius_m=0.02, mass_kg=0.2)
    pole = GeneSegment(name="p", parent="b", shape="capsule", length_m=1.2, radius_m=0.02, mass_kg=1.0,
                       joint_type=None, is_end_effector=True)
    report = validate_gene_design(RobotGene(id="pole", species="x", robot_class="legged", base_mount="free",
                                            end_effector_type="none", segments=[base, pole]))
    assert report["checks"]["stability"] is False


@pytest.mark.parametrize("prompt", ["a snake robot", "a robotic fish"])
def test_a_spine_of_vertical_hinges_is_not_asked_to_lift_itself(prompt):
    from virturoid.services.gene_validation import _gravity_free_hinges, validate_gene_design
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot(prompt)
    hinges = {s.name for s in gene.segments if s.joint_type == "revolute"}
    assert hinges and _gravity_free_hinges(gene) == hinges
    report = validate_gene_design(gene)
    assert not any(f["check"] == "torque_margin" for f in report["risk_flags"]), report["risk_flags"]


def test_a_hinge_that_gravity_loads_keeps_its_hold_check():
    from virturoid.services.gene_validation import _gravity_free_hinges
    from virturoid.services.morphology_composer import compose_robot
    arm = compose_robot("a robot arm with a gripper")
    free = _gravity_free_hinges(arm)
    hinges = [s.name for s in arm.segments if s.joint_type == "revolute"]
    # only the base yaw (vertical, nothing above it to tilt it) is exempt; every pitch joint is still sized
    assert free == {hinges[0]}, (free, hinges)
    assert _gravity_free_hinges(compose_robot("a quadruped robot")) == set()


@pytest.mark.parametrize("prompt", ["a six-wheeled rover", "a mobile base robot that navigates around obstacles",
                                    "a two-wheeled delivery robot"])
def test_a_wheeled_chassis_is_not_a_can_tower(prompt):
    from virturoid.services.anatomy_critic import critique_gene
    from virturoid.services.morphology_composer import compose_robot
    issues = critique_gene(compose_robot(prompt))["issues"]
    assert not any(i["check"] == "part_balance" and "root body" in i["detail"] for i in issues), issues


def test_a_limbed_can_tower_is_still_flagged():
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.anatomy_critic import critique_gene
    segs = [GeneSegment("torso", shape="box", length_m=0.5, radius_m=0.3, mass_kg=5.0)]
    for i, (x, y) in enumerate([(0.2, 0.2), (0.2, -0.2), (-0.2, 0.2), (-0.2, -0.2)]):
        segs.append(GeneSegment(f"leg{i}", parent="torso", length_m=0.05, radius_m=0.01, mass_kg=0.05,
                                joint_type="revolute", mount_offset=(x, y, -0.5), mount_euler=(3.14159, 0, 0),
                                is_end_effector=(i == 3)))
    issues = critique_gene(RobotGene(id="tower", species="x", robot_class="quadruped", base_mount="free",
                                     end_effector_type="none", segments=segs))["issues"]
    assert any(i["check"] == "part_balance" and "root body" in i["detail"] for i in issues), issues


def test_a_limbless_spine_is_not_judged_for_left_right_symmetry():
    from virturoid.services.anatomy_critic import critique_gene
    from virturoid.services.morphology_composer import compose_robot
    issues = critique_gene(compose_robot("a snake robot"))["issues"]
    assert not any(i["check"] == "symmetry" for i in issues), issues


def test_a_lopsided_limbed_body_is_still_flagged():
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.anatomy_critic import critique_gene
    segs = [GeneSegment("torso", shape="box", length_m=0.1, radius_m=0.12, mass_kg=2.0)]
    for i, x in enumerate((0.1, -0.1)):          # both legs on the LEFT side only
        segs.append(GeneSegment(f"leg{i}", parent="torso", length_m=0.25, radius_m=0.02, mass_kg=0.2,
                                joint_type="revolute", mount_offset=(x, 0.25, -0.1), mount_euler=(3.14159, 0, 0),
                                is_end_effector=(i == 1)))
    issues = critique_gene(RobotGene(id="lopsided", species="x", robot_class="legged", base_mount="free",
                                     end_effector_type="none", segments=segs))["issues"]
    assert any(i["check"] == "symmetry" for i in issues), issues
