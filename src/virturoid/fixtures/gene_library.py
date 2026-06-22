"""Seed robot genes — the initial trunk of the species tree, as parametric genes.

These prove the gene + compiler keystone: genuinely different morphologies expressed as
genes and compiled to MuJoCo, plus the amendment op that branches a new species from an
existing one (the flywheel move). Concrete dimensions are deliberate so the compiled
bodies differ structurally (a tabletop arm vs. a torso-mounted humanoid upper-body arm).
"""

from __future__ import annotations

import math

from virturoid.schemas.gene import GeneSegment, RobotGene, amend_gene

_PI = math.pi


def seed_genes() -> list[RobotGene]:
    """The seed genes that make up the buildable trunk of the species tree."""
    return [tabletop_arm_gene(), humanoid_upper_body_gene(), humanoid_spray_paint_gene()]


def gene_for_class(robot_class: str, prompt: str = "") -> RobotGene | None:
    """The default buildable gene for a robot class (None if the class has no gene yet).

    This is what lets a humanoid *request* build+run a real humanoid instead of tracing to
    the tabletop arm. The build then amends this gene to the customer's specifics. ``prompt`` is
    used by the legged gene to honour the requested leg count (hexapod/octopod/N-legged); the
    other classes ignore it.
    """
    if robot_class in ("quadruped", "legged"):
        return quadruped_gene(prompt)
    return {
        "manipulator": tabletop_arm_gene,
        "humanoid": humanoid_upper_body_gene,
    }.get(robot_class, lambda: None)()


def quadruped_gene(prompt: str = "") -> RobotGene:
    """The default buildable gene for a legged robot: a real free-base walker (Go1-class, 3 DOF per leg).

    Reuses the morphology composer (one source of truth — the same call the MorphPolicy tests use) rather
    than hand-authoring a kinematic tree, so the legged gene stays consistent with the rest of the system.
    The composer infers the LEG COUNT from ``prompt`` (4 by default; hexapod=6, octopod=8, 'six-legged'=6,
    …), so a hexapod request builds a real 6-legged body — not a 4-leg stand-in. The result is a free-base
    body (``base_mount='free'``, no end-effector tool) that ``robot_kind`` classifies as ``'legged'`` —
    which routes it to the locomotion eval + the policy flywheel in the autonomous build. Imported lazily
    to avoid an import cycle (the composer imports the gene schema this module also imports)."""
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements

    return compose_from_spec(morphology_from_requirements(0.65, 0.25, prompt=prompt, robot_class="quadruped"))


def tabletop_arm_gene() -> RobotGene:
    """The current MVP morphology, as a gene: a 3-DOF serial arm on a table."""
    return RobotGene(
        id="gene_fixed_arm_three_dof_tabletop",
        species="fixed_arm.three_dof.tabletop",
        robot_class="manipulator",
        parent_species="fixed_arm",
        base_mount="table",
        end_effector_type="gripper",
        # Matches the known-good MVP exporter arm exactly (so the gene path is at parity with
        # the 91%-as-built arm): BOX links (capsule end-caps overlap joints/table and break
        # the tuned control), forearm 0.357 m, light 0.04 kg gripper, all joints 12 Nm.
        segments=[
            GeneSegment("base", parent=None, shape="box", length_m=0.07, radius_m=0.06, mass_kg=0.45),
            GeneSegment("upper", parent="base", shape="box", length_m=0.07, radius_m=0.018, mass_kg=0.22,
                        joint_type="revolute", joint_axis=(0, 0, 1), joint_lower=-_PI, joint_upper=_PI, actuator_torque_nm=12),
            GeneSegment("forearm", parent="upper", shape="box", length_m=0.357, radius_m=0.016, mass_kg=0.18,
                        joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-1.57, joint_upper=1.57, actuator_torque_nm=12),
            GeneSegment("wrist", parent="forearm", shape="box", length_m=0.16, radius_m=0.02, mass_kg=0.035,
                        joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-1.57, joint_upper=1.57, actuator_torque_nm=12),
            GeneSegment("gripper", parent="wrist", shape="box", length_m=0.07, radius_m=0.03, mass_kg=0.04,
                        joint_type="fixed", is_end_effector=True),
        ],
    )


def humanoid_upper_body_gene() -> RobotGene:
    """A torso-mounted single-arm humanoid upper body — structurally distinct from the arm.

    A tall torso lifts a 4-DOF arm to human shoulder height; it reaches down to a table.
    Fixed (welded) base for the manipulation MVP — no balance/locomotion required.
    """
    return RobotGene(
        id="gene_humanoid_upper_body_single_arm",
        species="humanoid.upper_body.single_arm",
        robot_class="humanoid",
        parent_species="humanoid",
        base_mount="floor",
        end_effector_type="gripper",
        # Box limbs (same reason as the arm). A tall torso lifts a 4-DOF arm to shoulder height.
        # Limb lengths/torques + arm mount-pitch below are the values a parallel CPU co-design
        # search (scripts/search_humanoid.py) found to reach 100% on the sort task — i.e. this
        # is the *co-designed* humanoid, not a hand guess. The pitch keeps the arm task-ready
        # (forward over the table) at zero joint angles. Paired with the humanoid controller
        # defaults in gene_build.default_controller_params.
        segments=[
            GeneSegment("torso", parent=None, shape="box", length_m=0.55, radius_m=0.09, mass_kg=6.0),
            GeneSegment("shoulder", parent="torso", shape="box", length_m=0.07, radius_m=0.04, mass_kg=0.5,
                        joint_type="revolute", joint_axis=(0, 0, 1), joint_lower=-_PI, joint_upper=_PI, actuator_torque_nm=37.6),
            GeneSegment("upper_arm", parent="shoulder", shape="box", length_m=0.3268, radius_m=0.035, mass_kg=1.2,
                        joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-2.6, joint_upper=2.6,
                        mount_euler=(0.0, 0.234, 0.0), actuator_torque_nm=37.6),
            GeneSegment("forearm", parent="upper_arm", shape="box", length_m=0.3035, radius_m=0.03, mass_kg=0.9,
                        joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-2.4, joint_upper=2.4, actuator_torque_nm=28.2),
            GeneSegment("wrist", parent="forearm", shape="box", length_m=0.0934, radius_m=0.025, mass_kg=0.25,
                        joint_type="revolute", joint_axis=(1, 0, 0), joint_lower=-1.57, joint_upper=1.57, actuator_torque_nm=14.1),
            GeneSegment("hand", parent="wrist", shape="box", length_m=0.105, radius_m=0.035, mass_kg=0.4,
                        joint_type="fixed", is_end_effector=True),
        ],
    )


def humanoid_spray_paint_gene() -> RobotGene:
    """Customer 2 in the flywheel: amend the humanoid node for spray painting.

    Same body, swap the gripper hand for a longer spray-nozzle end effector + reach — the
    morphology is *reused*, only the diff changes. Demonstrates the reuse/amend loop.
    """
    base = humanoid_upper_body_gene()
    return amend_gene(
        base,
        new_id="gene_humanoid_upper_body_spray_paint",
        species="humanoid.upper_body.spray_paint",
        end_effector_type="spray_nozzle",
        segment_overrides={
            # A longer, lighter forearm for standoff distance; a slim nozzle end effector.
            "forearm": {"length_m": 0.30},
            "hand": {"name": "hand", "shape": "cylinder", "length_m": 0.14, "radius_m": 0.02, "mass_kg": 0.2},
        },
        metadata={"task_hint": "spray_paint", "reused_from": base.species},
    )
