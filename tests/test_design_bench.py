"""WS-A.4 — the Design-Bench funnel: structural determinism + physics regression gate (master_plan_v6 §8.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import design_bench as DB

_BASELINE = json.loads((Path(__file__).resolve().parents[1] / "tests" / "fixtures"
                        / "design_bench_baseline_v1.json").read_text())


def _bad_gene() -> RobotGene:
    return RobotGene(id="bad", species="x", robot_class="quadruped",
                     segments=[GeneSegment(name="a", parent=None), GeneSegment(name="b", parent="ghost")])


# ---------------------------------------------------------------- credibility semantics
def test_is_credible_matches_the_honesty_engine_verbs():
    for v in ("CREDIBLE walk...", "DRIVES (0.5 m)", "ARTICULATES (reach 0.9 m)", "SWIMS (0.3 m)",
              "FLIES (1.2 m)", "CRAWLS (serpentine)", "PICKS UP (real grasp)"):
        assert DB._is_credible({"verdict": v}), v
    for v in ("SLIDE (feet barely lift)", "LURCHES", "TIPPED", "STUCK", "STANDS (static balance)",
              "SPRAY: no locomotion verdict", "FORWARD BUT SHORT"):
        assert not DB._is_credible({"verdict": v}), v


# ---------------------------------------------------------------- per-design funnel routing
def test_schema_failure_stops_at_schema_stage():
    r = DB.evaluate_design(_bad_gene(), verify=True)
    assert not r["schema_valid"] and r["error_class"] == "schema"
    assert not r["compiles"] and not r["credible"]


def test_none_design_is_an_honest_attempt():
    r = DB.evaluate_design(None)
    assert r["error_class"] == "schema" and not r["schema_valid"]


# ---------------------------------------------------------------- structural funnel (fast, deterministic)
def test_structural_funnel_over_cassette_is_deterministic():
    a = DB.bench_from_cassette(verify=False)
    b = DB.bench_from_cassette(verify=False)
    assert a["schema_valid@1"] == b["schema_valid@1"] == 1.0
    assert a["compile@1"] == b["compile@1"] == 1.0
    # single absolute denominator: n_attempts == the battery size
    assert a["n_attempts"] == _BASELINE_battery_size()
    # diversity + spec are computed and stable
    assert a["diversity"]["unique_ratio"] == b["diversity"]["unique_ratio"]
    assert a["spec_faithfulness"]["n_scored"] > 0


def _BASELINE_battery_size() -> int:
    from virturoid.services import design_battery as B
    return len(B.battery())


def test_denominator_is_absolute_not_survivorship():
    """Every headline rate must be over n_attempts, never over survivors (Text2CAD-Bench survivorship trap)."""
    out = DB.bench_from_cassette(verify=False)
    n = out["n_attempts"]
    # schema/compile counts reconstructable from the absolute rates land on whole designs
    assert abs(out["schema_valid@1"] * n - round(out["schema_valid@1"] * n)) < 1e-9
    assert "compile|schema" in out["conditional"] and "verdict|compile" in out["conditional"]


def test_diversity_flags_mode_collapse():
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    identical = DB.diversity_report([g, g, g])           # 3 copies of one body -> collapsed
    assert identical["unique_ratio"] == pytest.approx(1 / 3, abs=1e-3)   # reported rounded to 4 dp
    assert identical["mean_pairwise_similarity"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- the physics regression gate (headline)
@pytest.mark.slow
def test_verdict_at_1_does_not_regress():
    """The headline gate: a drop in verdict@1 (or the upstream stages) below the committed baseline blocks."""
    out = DB.bench_from_cassette(verify=True)
    tol = 0.05
    assert out["schema_valid@1"] >= _BASELINE["schema_valid@1"] - 1e-9
    assert out["compile@1"] >= _BASELINE["compile@1"] - 1e-9
    assert out["verdict@1"] >= _BASELINE["verdict@1"] - tol, \
        f"verdict@1 regressed: {out['verdict@1']} < {_BASELINE['verdict@1']} - {tol}"


@pytest.mark.slow
def test_no_individual_body_silently_stops_walking():
    """PER-CASE gate, because the aggregate above cannot resolve what it is asked to resolve.

    The battery is 20 prompts, so verdict@1 moves in steps of exactly 0.05 -- the same size as that test's
    tolerance -- and the cassette is deterministic and hermetic, so the ambiguity is lost information, not
    sampling noise. A change that fixes one body and breaks another reads as NO CHANGE, and the aggregate gate
    passes. That already happened here: measured 2026-07-30, the hybrid family (mobile manipulator) had gone
    1.0 -> 0.0 with BOTH cases lost while animal gained one, a net -0.05 the tolerance absorbed exactly.

    So gate on the bodies themselves. Any case credible in the recorded floor must stay credible, and the
    failure message names the body -- which is the whole point, since 'verdict@1 fell by 0.05' never did.
    """
    out = DB.bench_from_cassette(verify=True)
    got = out["per_case"]
    floor = _BASELINE.get("per_case") or {}
    assert floor, "baseline carries no per_case floor; re-record with scripts/design_bench.py --record"
    missing = sorted(set(floor) - set(got))
    assert not missing, f"battery no longer covers {missing}; re-record the baseline if that is intended"
    broke = sorted(k for k, was in floor.items() if was and not got.get(k))
    assert not broke, (
        "these bodies were credible and are not any more: " + ", ".join(broke)
        + f"  (verdict@1 {out['verdict@1']} vs baseline {_BASELINE['verdict@1']} — note the aggregate can "
          "absorb one flip entirely, which is why this per-case check exists)")


@pytest.mark.slow
def test_the_known_regressions_are_still_the_only_ones():
    """The cases the aggregate baseline says should pass, and which do not, are tracked debt -- not blessed.

    `known_regressions` exists so a real capability loss cannot be normalised by re-recording a lower number.
    If one of them starts passing, this fails and the list must shrink; if a NEW one appears, the per-case gate
    above catches it first. Either way the list stays honest. See task #246 for the hybrid investigation."""
    out = DB.bench_from_cassette(verify=True)
    known = set(_BASELINE.get("known_regressions") or [])
    fixed = sorted(k for k in known if out["per_case"].get(k))
    assert not fixed, (f"good news: {fixed} now pass. Remove them from known_regressions in "
                       "tests/fixtures/design_bench_baseline_v1.json so the list keeps meaning something.")
