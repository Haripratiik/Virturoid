"""An imported robot must be told apart by WHAT CARRIES IT, not by how many branches it has.

`_infer_class` counted limb-chains off the root and dispatched on the count, which made three whole families
unreachable. Measured across all 63 cached MuJoCo Menagerie models:

  * `limbs >= 4 -> quadruped` caught every HUMANOID (two legs + two arms IS four limbs) and every four-fingered
    HAND (Allegro, LEAP) for the same reason;
  * `free_root -> mobile_base` sat BEFORE the humanoid rule, and every legged robot has a free root, so the
    `limbs == 2` humanoid branch could never fire for a floating-base biped.

12 of 12 bipeds were misclassified, and the four bodies labelled `humanoid` were a bimanual station, a bimanual
arm and two grippers -- precision 0/4 AND recall 0/12.

This is not a cosmetic label. The verify rubric is chosen by kind (#218), so an imported humanoid was judged by a
wheeled-driving rubric, which is the dishonest-verdict defect at corpus scale.

The tests below pin the DISTINCTIONS the old rule could not draw, each on a real robot, so a future refactor that
reverts to branch-counting fails loudly.
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


def _klass(rel: str) -> str:
    src = _MEN / rel
    if not src.is_file():
        pytest.skip(f"{rel} is not cached locally (robot_descriptions fetches on demand)")
    from virturoid.services.robot_import import import_robot
    return import_robot(str(src), robot_id="t")["robot_class"]


# (model, expected class, what the old rule said, why it is unambiguous)
_CORPUS = [
    ("unitree_g1/g1.xml", "humanoid", "mobile_base"),
    ("unitree_h1/h1.xml", "humanoid", "mobile_base"),
    ("robotis_op3/op3.xml", "humanoid", "quadruped"),
    ("agility_cassie/cassie.xml", "humanoid", "mobile_base"),
    ("berkeley_humanoid/berkeley_humanoid.xml", "humanoid", "mobile_base"),
    ("pal_talos/talos.xml", "humanoid", "mobile_base"),
    ("booster_t1/t1.xml", "humanoid", "quadruped"),
    ("toddlerbot_2xc/toddlerbot_2xc.xml", "humanoid", "quadruped"),
    ("unitree_go2/go2.xml", "quadruped", "quadruped"),
    ("boston_dynamics_spot/spot.xml", "quadruped", "quadruped"),
    ("anybotics_anymal_c/anymal_c.xml", "quadruped", "quadruped"),
    ("wonik_allegro/left_hand.xml", "manipulator", "quadruped"),
    ("leap_hand/left_hand.xml", "manipulator", "quadruped"),
    ("universal_robots_ur5e/ur5e.xml", "manipulator", "manipulator"),
    ("franka_fr3/fr3.xml", "manipulator", "manipulator"),
    ("pal_tiago/tiago.xml", "mobile_base", "quadruped"),
    ("skydio_x2/x2.xml", "mobile_base", "mobile_base"),
    ("hello_robot_stretch/stretch.xml", "mobile_base", "mobile_base"),
]


@pytest.mark.parametrize("rel,expected,old", _CORPUS, ids=[c[0].split("/")[0] for c in _CORPUS])
def test_real_robots_are_classified_by_what_holds_them_up(rel, expected, old):
    got = _klass(rel)
    assert got == expected, (
        f"{rel} is a {expected}; imported as {got}"
        + (f" (the branch-counting rule also said {old})" if old != expected else ""))


def test_a_humanoid_is_not_a_quadruped_just_because_it_has_four_limbs():
    """The single distinction the old rule could not make: arms are limbs but they do not carry the body."""
    assert _klass("unitree_g1/g1.xml") == "humanoid"
    assert _klass("unitree_go2/go2.xml") == "quadruped"


def test_legs_are_found_even_when_they_hang_off_a_waist_rather_than_the_root():
    """Counting the ROOT's direct children is not enough, and three real humanoids prove it.

    Booster T1's legs are under `Waist` and both ToddlerBots' under `waist_gears`, so the root sees exactly ONE
    ground-reaching branch and all three read as mobile_base. The count has to follow a single branch down to the
    point where the load-bearing chains actually diverge."""
    assert _klass("booster_t1/t1.xml") == "humanoid"
    assert _klass("toddlerbot_2xc/toddlerbot_2xc.xml") == "humanoid"


def test_a_four_fingered_hand_is_not_a_quadruped():
    """Four fingers are four chains that reach the 'lowest' point; a fixed base is what says it cannot walk."""
    assert _klass("wonik_allegro/left_hand.xml") == "manipulator"


def test_casters_are_not_legs():
    """Tiago's four casters DO touch the floor and DO have two joints each, so ground contact alone is not enough.

    They are rolling elements: 2 unlimited hinges and 0 limited joints per caster, against 0 unlimited and 12
    limited in the arm. A leg joint has a range; a wheel turns without end."""
    assert _klass("pal_tiago/tiago.xml") == "mobile_base"


def test_an_include_fragment_says_so_instead_of_raising():
    """A model directory ships files meant to be pulled INTO a parent -- keyframes, assets, shared defaults -- and
    a customer dropping a folder will hand us one.

    shadow_hand/keyframes.xml is the single import failure in the whole 63-model Menagerie corpus, and it failed
    as a raw `ValueError: keyframe 'scissors': invalid qpos size, expected 0, got 24`, which tells the customer
    nothing actionable. A <mujoco> root declaring no <body> is the signature of a fragment."""
    src = _MEN / "shadow_hand/keyframes.xml"
    if not src.is_file():
        pytest.skip("shadow_hand is not cached locally")
    from virturoid.services.robot_import import import_robot
    out = import_robot(str(src), robot_id="frag")
    assert out["gene"] is None and not out["valid"]
    w = " ".join(out.get("warnings") or []).lower()
    assert "fragment" in w, w
    assert "left_hand.xml" in w, f"the report must name the real model beside it: {w}"


def test_a_classification_failure_never_breaks_the_import():
    """The class is a guess layered on top of a faithful import; it must degrade, not raise."""
    from virturoid.schemas.gene import GeneSegment, RobotGene
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.robot_import import import_robot

    g = RobotGene(id="min", species="test", robot_class="manipulator", base_mount="table", segments=[
        GeneSegment(name="base", parent=None, shape="box", length_m=0.1, radius_m=0.05, mass_kg=1.0),
        GeneSegment(name="link", parent="base", shape="capsule", length_m=0.2, radius_m=0.03, mass_kg=0.5,
                    joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.5, joint_upper=1.5,
                    actuator_torque_nm=8.0, is_end_effector=True)])
    out = import_robot(compile_gene_to_mjcf(g, include_floor=True), robot_id="rt")
    assert out["gene"] is not None
    assert out["robot_class"] in {"manipulator", "mobile_base", "humanoid", "quadruped", "hexapod"}
