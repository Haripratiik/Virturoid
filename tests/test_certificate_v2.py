"""WS-E — certificate v2: tiered sim-to-real evidence, NASA-STD-7009 schema (master_plan_v6 §9)."""
from __future__ import annotations

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import certificate_v2 as C2


def _legged():  # a hand-built body WITHOUT torque clamps -> actuator level L0
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.5, radius_m=0.1, mass_kg=2.0)]
    for i in range(4):
        prev = "torso"
        for j in range(3):
            nm = f"leg{i}_{j}"
            segs.append(GeneSegment(name=nm, parent=prev, joint_type="revolute", length_m=0.16, radius_m=0.02,
                                    mass_kg=0.1, joint_lower=-1.5, joint_upper=1.5))
            prev = nm
    segs[-1].is_end_effector = True
    return RobotGene(id="L", species="t.L", robot_class="quadruped", segments=segs, base_mount="free",
                     end_effector_type="none")


# ---------------------------------------------------------------- pure statistics
def test_incomplete_beta_matches_known_values():
    assert C2.reg_incomplete_beta(1, 1, 0.5) == pytest.approx(0.5, abs=1e-6)
    assert C2.reg_incomplete_beta(2, 3, 0.3) == pytest.approx(0.3483, abs=1e-3)   # vs scipy betainc


def test_clopper_pearson_matches_the_plan_headline():
    assert C2.clopper_pearson(0, 300)["hi"] == pytest.approx(0.0122, abs=1e-3)     # "300/300 => <=~1%"
    assert C2.clopper_pearson(0, 30)["hi"] > C2.clopper_pearson(0, 300)["hi"]      # fewer trials, looser bound
    cp = C2.clopper_pearson(3, 100)
    assert cp["lo"] < 0.03 < cp["hi"] and cp["p_fail"] == 0.03


# ---------------------------------------------------------------- model sanity + actuator level
def test_model_sanity_passes_a_valid_body():
    s = C2.model_sanity(_legged())
    assert s["ok"] and s["inertia_valid"] and s["mass_finite"]
    assert "inertia_provenance" in s                       # honest note that CAD inertia can be ~4x off


def test_actuator_level_L0_without_clamp_and_L1_with():
    assert C2.actuator_fidelity_level(_legged())["level"] == 0   # no torque clamp -> L0 ideal
    from virturoid.services.morphology_composer import compose_robot
    grounded = compose_robot("a simple four-legged walking robot", ensure_walkable=True)   # grounded -> clamps set
    assert C2.actuator_fidelity_level(grounded)["level"] == 1


# ---------------------------------------------------------------- the schema (cheap path)
def test_certificate_v2_schema_and_scope(monkeypatch):
    cert = C2.build_certificate_v2(_legged(), {"verdict": "CREDIBLE", "credible": True},
                                   run_dr=False, run_margins=False)
    assert cert["version"] == 2 and cert["tier"] == 1 and cert["valid"]
    # NASA-STD-7009 load-bearing blocks
    assert set(cert["scope"]) >= {"valid_iff", "not_modeled", "claim_ceiling"}
    assert "friction" in " ".join(cert["scope"]["not_modeled"]).lower()
    assert cert["use_history"]["predictivity_srcc"] is None      # honest "unmeasured" at N=0
    assert cert["actuator_fidelity_level"]["level"] in (0, 1)
    assert "optimistic by construction" in cert["honest_limits"]


def test_certificate_is_void_when_model_sanity_red(monkeypatch):
    monkeypatch.setattr(C2, "model_sanity", lambda g: {"ok": False, "issues": ["forced red"],
                                                       "inertia_valid": False, "mass_finite": True,
                                                       "joint_limits_present": True})
    cert = C2.build_certificate_v2(_legged(), {"verdict": "CREDIBLE", "credible": True})
    assert cert["valid"] is False and "voided_reason" in cert
    assert "robustness" not in cert and "margins" not in cert    # no downstream claims on a void certificate


# ---------------------------------------------------------------- full Tier-1 (real rollouts)
@pytest.mark.slow
def test_full_tier1_certificate_on_a_real_walker():
    from virturoid.services.morphology_composer import compose_robot
    walker = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    cert = C2.build_certificate_v2(walker, {"verdict": "CREDIBLE", "credible": True},
                                   run_dr=True, dr_draws=4, run_margins=True)
    assert cert["valid"]
    # robustness: pass-rate + a Clopper-Pearson bound + per-parameter failure boundaries (margins, not booleans)
    rob = cert["robustness"]
    assert 0.0 <= rob["pass_rate"] <= 1.0
    assert "hi" in rob["clopper_pearson_failure_bound"]
    assert "max_payload_kg" in rob["failure_boundaries"]
    assert "friction" in rob["scope_note"].lower()          # honestly scopes what it did NOT sweep
    assert cert["margins"]["available"] in (True, False)     # margins attempted from the verified trajectory
    assert "invariant" in cert["numerics"]
