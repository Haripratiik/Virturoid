"""B2: the customer's REAL actuator torque limits, and their real drivetrain, must reach the model.

Two independent losses, both measured on real MuJoCo Menagerie packages rather than on a fixture:

1. THE IMPORTER READ ONE OF THREE DECLARATION SITES. It read ``actuator/forcerange`` only, so a Unitree Go2 --
   which declares 23.7/23.7/45.43 N.m as ``ctrlrange`` on ``<motor>`` actuators with ``forcelimited=false`` --
   imported with NO torque at all, and a Unitree G1 -- which declares 88/139/50/25/5 N.m as
   ``<joint actuatorfrcrange>`` and nothing on the actuator -- did the same. Real numbers, in the file,
   discarded.

   The trap this test also pins: ``ctrlrange`` is a torque ONLY for a force-mode actuator. The G1's actuators
   are ``<position>`` servos whose ctrlrange is the joint's POSITION range in radians, so reading it as a
   torque hands the 88 N.m hip a 2.88 N.m limit and stamps it as the manufacturer's own number -- worse than
   reading nothing.

2. GROUNDING OVERWROTE WHATEVER SURVIVED. ``ground_gene`` replaced ``actuator_torque_nm`` with a catalog pick;
   ``preserve_mass`` protected mass and nothing protected torque. A Panda's 87/87/87/87/12/12/12 became
   520/520/360/120/48/1.5/0.5 and a UR5e's 150/150/150/28/28/28 became 360/360/48/4.1/1.5/0.5.

Why it is urgent beyond accuracy: ``sysid.excitation`` bounds a plan written to be RUN ON PHYSICAL HARDWARE at
a fraction of ``actuator_torque_nm``. Against the invented numbers that put 180 N.m of commanded torque on the
Panda's 87 N.m J2. Fixing this removes a hardware-damage path, so the excitation bound is asserted here too.
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


def _torques(gene) -> dict:
    return {s.name: s.actuator_torque_nm for s in gene.segments
            if s.joint_type in ("revolute", "prismatic")}


# --------------------------------------------------------------------------- the three declaration sites
def test_go2_torque_comes_from_a_force_mode_motors_ctrlrange():
    """23.7 / 23.7 / 45.43 N.m, declared as ctrlrange with forcerange left unlimited."""
    gene = _import("unitree_go2/go2.xml")["gene"]
    t = _torques(gene)
    assert t["FL_hip"] == pytest.approx(23.7, abs=0.01)
    assert t["FL_thigh"] == pytest.approx(23.7, abs=0.01)
    assert t["FL_calf"] == pytest.approx(45.43, abs=0.01)
    # THE ORDERING IS THE POINT: the knee is the STRONGEST joint on this machine. It used to be the weakest.
    assert t["FL_calf"] > t["FL_hip"]
    assert gene.metadata["torque_source"] == "source_model"
    assert "ctrlrange" in gene.metadata["source_actuator_torque_where"]["FL_calf"]


def test_g1_torque_comes_from_the_joints_actuatorfrcrange_not_its_position_ctrlrange():
    """The G1 declares torque ONLY on the joint. Its actuator ctrlrange is a POSITION range in radians."""
    gene = _import("unitree_g1/g1.xml")["gene"]
    t = _torques(gene)
    assert t["left_hip_pitch_link"] == pytest.approx(88.0, abs=0.01)
    assert t["left_knee_link"] == pytest.approx(139.0, abs=0.01)
    assert t["left_ankle_roll_link"] == pytest.approx(50.0, abs=0.01)
    assert t["left_wrist_yaw_link"] == pytest.approx(5.0, abs=0.01)
    # The trap: ctrlrange for left_hip_pitch is [-2.531, 2.880] rad. Read as a torque it gives 2.88 N.m.
    assert t["left_hip_pitch_link"] > 10.0
    assert gene.metadata["source_actuator_torque_where"]["left_knee_link"] == "joint/actuatorfrcrange"


def test_panda_and_ur5e_torque_comes_from_actuator_forcerange():
    panda = _torques(_import("franka_emika_panda/panda.xml")["gene"])
    assert [panda[f"link{i}"] for i in range(1, 8)] == pytest.approx(
        [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], abs=0.01)
    ur5e = _torques(_import("universal_robots_ur5e/ur5e.xml")["gene"])
    assert ur5e["shoulder_link"] == pytest.approx(150.0, abs=0.01)
    assert ur5e["wrist_3_link"] == pytest.approx(28.0, abs=0.01)


def test_a_joint_with_no_declared_limit_anywhere_is_reported_not_invented():
    """The Panda's two finger sliders declare nothing. Say so; do not present a catalog guess as theirs."""
    out = _import("franka_emika_panda/panda.xml")
    gene = out["gene"]
    assert "left_finger" not in gene.metadata["source_actuator_torque_nm"]
    assert any("declare NO torque limit" in w for w in out["warnings"]), out["warnings"]


# --------------------------------------------------------------------------- grounding must not overwrite it
@pytest.mark.parametrize("rel,checks", [
    ("unitree_go2/go2.xml", {"FL_hip": 23.7, "FL_calf": 45.43}),
    ("franka_emika_panda/panda.xml", {"link1": 87.0, "link7": 12.0}),
    ("unitree_g1/g1.xml", {"left_knee_link": 139.0, "left_wrist_yaw_link": 5.0}),
    ("universal_robots_ur5e/ur5e.xml", {"shoulder_link": 150.0, "wrist_3_link": 28.0}),
])
def test_grounding_preserves_the_customers_declared_torque(rel, checks):
    """The real product grounding path (``gene_build.ground_and_repair``) must leave these alone."""
    from virturoid.services.gene_build import ground_and_repair
    gene = _import(rel)["gene"]
    ground_and_repair(gene)                                    # grounds, structurally repairs, RE-grounds
    t = _torques(gene)
    for name, want in checks.items():
        assert t[name] == pytest.approx(want, abs=0.01), f"{rel}:{name} was overwritten by grounding"


def test_the_bom_still_names_a_real_part_and_labels_whose_number_the_limit_is():
    from virturoid.services.grounded_physics import ground_gene
    gene = _import("universal_robots_ur5e/ur5e.xml")["gene"]
    rep = ground_gene(gene, preserve_mass=True)
    assert rep["torque_preserved"] is True
    assert rep["n_torque_preserved"] == 6
    row = next(r for r in rep["bom"] if r["role"] == "shoulder_link")
    assert row["torque_source"] == "source_model"
    assert row["joint_torque_limit_nm"] == pytest.approx(150.0, abs=0.01)
    assert row["stall_nm"] >= 150.0                            # the catalog equivalent really covers it
    assert row["part"]


def test_preserve_torque_false_restores_the_catalog_sizing():
    """The protection is a default, not a cage: an explicit re-spec still re-sizes."""
    from virturoid.services.grounded_physics import ground_gene
    gene = _import("universal_robots_ur5e/ur5e.xml")["gene"]
    rep = ground_gene(gene, preserve_mass=True, preserve_torque=False)
    assert rep["torque_preserved"] is False
    assert gene.segment("shoulder_link").actuator_torque_nm != pytest.approx(150.0, abs=0.01)


def test_a_deliberate_capability_amend_still_upsizes_the_motor():
    """set_payload scales the joint REQUIREMENT; preservation must release, or the amend is silently reverted."""
    from virturoid.services.edit_operators import set_payload
    gene = _import("universal_robots_ur5e/ur5e.xml")["gene"]
    from virturoid.services.grounded_physics import ground_gene
    ground_gene(gene, preserve_mass=True)
    before = gene.segment("shoulder_link").actuator_torque_nm
    amended, _ = set_payload(gene, payload_kg=20.0)
    after = amended.segment("shoulder_link").actuator_torque_nm
    assert after > before, "the customer asked for more capacity and the declared limit overrode them"


def test_a_generated_body_is_unaffected():
    """No ``torque_source`` marker -> nothing preserved, so every existing path grounds exactly as before."""
    from virturoid.services.grounded_physics import ground_gene, source_declared_torques
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a quadruped walking robot")
    assert source_declared_torques(g) == {}
    rep = ground_gene(g)
    assert rep["torque_preserved"] is False
    assert all(s.actuator_torque_nm for s in g.actuated_joints())


# --------------------------------------------------------------------------- it reaches the compiled model
def test_the_compiled_model_carries_the_declared_limit():
    """``actuator_torque_nm`` becomes the emitted forcerange, so this is what the simulator actually clamps."""
    import mujoco

    from virturoid.schemas.gene import RobotGene
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    gene = _import("unitree_go2/go2.xml")["gene"]
    gene = RobotGene.from_dict(gene.to_dict())                 # the record must survive serialization
    ground_and_repair(gene)
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))
    want = gene.metadata["source_actuator_torque_nm"]
    seen = 0
    for u in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, u) or ""
        seg = name[:-6] if name.endswith("_motor") else name
        if seg not in want:
            continue
        seen += 1
        assert abs(float(m.actuator_forcerange[u][1])) == pytest.approx(want[seg], abs=0.02)
    assert seen == 12


# --------------------------------------------------------------------------- the drivetrain the fit identifies
def test_the_sources_own_armature_damping_frictionloss_survive_import():
    """The three parameters ``sysid.fit_parameters`` identifies are DECLARED by the Go2 and the Panda.

    Discarding them means a calibration run 'identifies' numbers the customer already handed us, and the
    sim-to-sim gap against their own model is one we manufactured at import.
    """
    go2 = _import("unitree_go2/go2.xml")["gene"]
    dyn = go2.metadata["source_joint_dynamics"]
    assert dyn["FL_calf"] == {"armature": 0.01, "damping": 2.0, "frictionloss": 0.2}
    assert len(dyn) == 12
    panda = _import("franka_emika_panda/panda.xml")["gene"]
    assert panda.metadata["source_joint_dynamics"]["link1"] == {
        "armature": 0.1, "damping": 1.0, "frictionloss": 0.0}


def test_the_drivetrain_record_survives_serialization():
    from virturoid.schemas.gene import RobotGene
    gene = _import("unitree_go2/go2.xml")["gene"]
    back = RobotGene.from_dict(gene.to_dict())
    assert back.metadata["source_joint_dynamics"] == gene.metadata["source_joint_dynamics"]
    assert back.metadata["source_actuator_torque_nm"] == gene.metadata["source_actuator_torque_nm"]


# --------------------------------------------------------------------------- the hardware-safety consequence
@pytest.mark.parametrize("rel", ["unitree_go2/go2.xml", "franka_emika_panda/panda.xml",
                                 "unitree_g1/g1.xml", "universal_robots_ur5e/ur5e.xml"])
def test_the_excitation_plan_never_commands_more_than_the_joint_is_rated_for(rel):
    """This plan is written to be run ON HARDWARE. Its torque budget is a fraction of the DECLARED peak, so
    the declaration being invented is a physical hazard, not a reporting defect."""
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.sysid.excitation import TORQUE_FRACTION, build_excitation
    gene = _import(rel)["gene"]
    ground_and_repair(gene)
    declared = gene.metadata["source_actuator_torque_nm"]
    plan = build_excitation(gene, budget_s=60.0)
    checked = 0
    for jp in plan["joints"]:
        real = declared.get(jp["joint"])
        if real is None:
            continue
        checked += 1
        commanded = TORQUE_FRACTION * float(jp["datasheet_peak_torque_nm"])
        assert commanded <= float(real) + 1e-6, (
            f"{rel}:{jp['joint']} plan commands {commanded:.1f} N.m on a {real} N.m joint")
    assert checked > 0
