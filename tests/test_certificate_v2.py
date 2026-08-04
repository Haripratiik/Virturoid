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


# ---------------------------------------------------------------- the honesty block is DERIVED, not declared
#
# The regression these guard: `scope.valid_iff`, `scope.not_modeled`, `scope.claim_ceiling` and `honest_limits`
# were four string CONSTANTS, byte-identical on every certificate for every robot, in a module whose docstring
# calls them "mandatory, not boilerplate". `claim_ceiling` in particular asserted "the failure modes we can
# model were swept and passed with the stated margins" on certificates where run_dr=False (nothing swept) and
# on bodies whose headline verdict was FELL.
def _canned_sweep(pass_rate=1.0, draws=12):
    k_fail = int(round((1.0 - pass_rate) * draws))
    return {"draws": draws, "pass_rate": pass_rate, "swept_ranges": C2.DR_RANGES,
            "clopper_pearson_failure_bound": C2.clopper_pearson(k_fail, draws),
            "failure_boundaries": {}, "scope_note": "friction is not swept"}


def test_the_four_honesty_fields_are_derived_from_the_run_not_hardcoded(monkeypatch):
    """Same robot, two runs that MEASURED different things -> all four fields must read differently."""
    body, v_ok = _legged(), {"verdict": "CREDIBLE", "credible": True}
    bare = C2.build_certificate_v2(body, v_ok, run_dr=False, run_margins=False)

    monkeypatch.setattr(C2, "dr_sweep", lambda *a, **k: _canned_sweep())
    monkeypatch.setattr(C2, "numerics_invariance", lambda *a, **k: {"probed": True, "invariant": True})
    monkeypatch.setattr(C2, "joint_margins", lambda *a, **k: {"available": True, "pass": True,
                                                              "weakest_joint_headroom": 1.9, "n_gates_pass": 6})
    full = C2.build_certificate_v2(body, v_ok, run_dr=True, run_margins=True)

    assert bare["scope"]["valid_iff"] != full["scope"]["valid_iff"]
    assert bare["scope"]["claim_ceiling"] != full["scope"]["claim_ceiling"]
    assert bare["scope"]["not_modeled"] != full["scope"]["not_modeled"]
    assert bare["honest_limits"] != full["honest_limits"]

    # ...and each one states the truth about ITS OWN run.
    assert "NO robustness claim" in bare["scope"]["claim_ceiling"]
    assert "no DR sweep ran" in bare["honest_limits"]
    assert any("no DR sweep ran" in s for s in bare["scope"]["not_modeled"])
    assert "12 draws" in full["scope"]["claim_ceiling"] and "ALL passed" in full["scope"]["claim_ceiling"]
    assert "SWEPT envelope" in full["scope"]["valid_iff"]
    assert "no perturbation envelope was swept" in bare["scope"]["valid_iff"]
    # the declared residue is still there, but LABELLED declared instead of passed off as a finding
    assert full["scope"]["not_modeled_provenance"]["declared"] == C2._DECLARED_NOT_MODELED
    assert full["honest_limits_declared"] == C2._DECLARED_LIMITS


def test_claim_ceiling_admits_failures_inside_the_swept_envelope(monkeypatch):
    monkeypatch.setattr(C2, "dr_sweep", lambda *a, **k: _canned_sweep(pass_rate=0.75))
    monkeypatch.setattr(C2, "numerics_invariance", lambda *a, **k: {"probed": True, "invariant": True})
    cert = C2.build_certificate_v2(_legged(), {"verdict": "CREDIBLE", "credible": True},
                                   run_dr=True, run_margins=False)
    ceiling = cert["scope"]["claim_ceiling"]
    assert "ALREADY CONTAINS FAILURES" in ceiling and "3/12" in ceiling
    assert "passed" not in ceiling.lower()          # it may never claim the sweep passed


def test_a_failed_verdict_cannot_carry_a_passing_claim_ceiling():
    cert = C2.build_certificate_v2(_legged(), {"verdict": "FELL", "credible": False},
                                   run_dr=False, run_margins=False)
    assert "does NOT claim the robot works" in cert["scope"]["claim_ceiling"]
    assert "FELL" in cert["scope"]["claim_ceiling"]
    assert "not credible" in cert["honest_limits"]


def test_numerics_probe_reports_unmeasured_rather_than_a_free_invariant_true(monkeypatch):
    monkeypatch.setattr(C2, "_credible_under", lambda g, p: (True, 1.0))
    n = C2.numerics_invariance(_legged(), gait_params=None)      # nothing to re-run -> it must not claim a pass
    assert n["probed"] is False and n["invariant"] is None
    assert "not probed" in n["note"]


def test_use_history_says_it_has_no_writer():
    cert = C2.build_certificate_v2(_legged(), {"verdict": "CREDIBLE", "credible": True},
                                   run_dr=False, run_margins=False)
    assert cert["use_history"]["measured"] is False
    assert "no writer" in cert["use_history"]["note"].lower()


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
