"""WS-A.4 — the Design-Bench funnel: structural determinism + physics regression gate (master_plan_v6 §8.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import design_bench as DB

_FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
_BASELINE = json.loads((_FIXTURES / "design_bench_baseline_v1.json").read_text())
# The LIVE baseline (#280) — the same battery run through the production lane against a real model. Kept in a
# SEPARATE fixture from the offline one on purpose: they measure different generators, so neither may be graded
# against the other's floor.
_LIVE_BASELINE = json.loads((_FIXTURES / "design_bench_baseline_live_v1.json").read_text())
_LIVE_CASSETTE_PATH = _FIXTURES / "design_cassette_live_v1.json"


def _live_cassette():
    from virturoid.services.design_cassette import DesignCassette
    return DesignCassette(_LIVE_CASSETTE_PATH)


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
    assert r["outcome"] == DB.OUTCOME_FAILED and r["refused"] is False


# ---------------------------------------------------------------- the tri-state outcome (a refusal is not a bug)
def test_a_refusal_is_a_first_class_outcome_not_a_schema_failure():
    """THE representation defect. A design refusal and a malformed kinematic tree both arrive as ``gene=None``,
    so the funnel booked both as ``schema`` failures and both as ``per_case: false``. Under that encoding the
    recorded floor cannot say 'the product correctly declined' -- which is why the OFFLINE floor marked three
    REFUSED prompt ids ``true`` and nothing could tell.

    A refusal is still not credible (the prompt went unserved, so verdict@1 must not reward it) but it is a
    different fact about the product, and it now says so."""
    ref = DB.evaluate_design(None, refused=True, refusal_reason="center of mass falls outside the support polygon")
    bad = DB.evaluate_design(_bad_gene())
    assert ref["outcome"] == DB.OUTCOME_REFUSED and ref["error_class"] == "refused" and ref["refused"] is True
    assert "support polygon" in ref["refusal_reason"]
    assert not ref["credible"]                      # honesty: an unserved prompt is unserved
    assert bad["outcome"] == DB.OUTCOME_FAILED and bad["error_class"] == "schema"
    assert ref["outcome"] != bad["outcome"]         # ...and the two are no longer the same recorded fact


def test_the_outcome_gate_tells_a_refusal_apart_from_a_broken_body():
    """The transition policy, stated as a table. Only the moves that represent a real LOSS may fail, and the
    one the boolean floor structurally could not express -- refused -> failed, both sides False -- is the one
    that matters most: it means the grounding gate stopped firing and we now ship a body that does not work."""
    floor = {"walks": DB.OUTCOME_CREDIBLE, "declined": DB.OUTCOME_REFUSED, "debt": DB.OUTCOME_FAILED}
    # nothing moved
    assert DB.outcome_regressions(floor, dict(floor)) == {"broke": {}, "improved": {}, "missing": []}
    # a credible body stopped working, and a refusal turned into a body that does not work
    bad = DB.outcome_regressions(floor, {"walks": DB.OUTCOME_FAILED, "declined": DB.OUTCOME_FAILED,
                                         "debt": DB.OUTCOME_FAILED})
    assert bad["broke"] == {"walks": "credible -> failed", "declined": "refused -> failed"}
    # a credible body that starts being REFUSED is also a loss -- the customer stopped getting a robot
    assert DB.outcome_regressions(floor, dict(floor, walks=DB.OUTCOME_REFUSED))["broke"] \
        == {"walks": "credible -> refused"}
    # upward moves never fail, but they are reported so the floor is raised deliberately, not by drift
    up = DB.outcome_regressions(floor, {"walks": DB.OUTCOME_CREDIBLE, "declined": DB.OUTCOME_CREDIBLE,
                                        "debt": DB.OUTCOME_CREDIBLE})
    assert up["broke"] == {} and up["improved"] == {"debt": "failed -> credible",
                                                   "declined": "refused -> credible"}
    # tracked debt protects nothing: a failed case may do anything, including start being refused
    assert DB.outcome_regressions({"debt": DB.OUTCOME_FAILED}, {"debt": DB.OUTCOME_REFUSED})["broke"] == {}
    # a case that VANISHES is a gate condition, not a skip (see the reclassification test below)
    assert DB.outcome_regressions(floor, {"walks": DB.OUTCOME_CREDIBLE})["missing"] == ["debt", "declined"]


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

    Gated on the TRI-STATE map, not the boolean: this cassette holds no refusals, so the two agree today, but
    the floor must be written in a vocabulary that can express one. The boolean is checked alongside so the
    two representations cannot drift apart.
    """
    out = DB.bench_from_cassette(verify=True)
    floor = _BASELINE.get("per_case_outcome") or {}
    assert floor, "baseline carries no per_case_outcome floor; re-record with scripts/design_bench.py --record"
    reg = DB.outcome_regressions(floor, out["per_case_outcome"])
    assert not reg["missing"], \
        f"battery no longer covers {reg['missing']}; re-record the baseline if that is intended"
    assert not reg["broke"], (
        "these bodies changed for the worse: " + "; ".join(f"{k} ({v})" for k, v in reg["broke"].items())
        + f"  (verdict@1 {out['verdict@1']} vs baseline {_BASELINE['verdict@1']} — note the aggregate can "
          "absorb one flip entirely, which is why this per-case check exists)")
    # the boolean map is derived from the same rows; if they ever disagree one of them is lying
    assert out["per_case"] == {k: (v == DB.OUTCOME_CREDIBLE) for k, v in out["per_case_outcome"].items()}
    assert _BASELINE["per_case"] == {k: (v == DB.OUTCOME_CREDIBLE)
                                     for k, v in sorted(_BASELINE["per_case_outcome"].items())}


# ---------------------------------------------------------------- provenance gates (#208)
def test_the_run_label_is_a_function_of_the_data_not_of_a_flag():
    """The defect this closes: ``scripts/design_bench.py`` set ``model = 'live_llm_v1' if args.strict_llm else
    'offline_heuristic_v1'`` -- a label typed by a flag on the CURRENT run rather than read off the rows being
    scored -- so ``--strict-llm`` without ``--record`` printed a live label over a replay of the offline
    fixture. The label now comes from the cassette's measured mode, and nothing else."""
    out = DB.bench_from_cassette(verify=False)
    assert out["mode"] == "offline"
    assert out["model"] == DB.label_for_mode("offline") == "offline_heuristic_v1"
    assert out["provenance"]["n_authored_by_model"] == 0
    assert out["provenance"]["is_subset"] is False


def test_a_live_label_is_refused_on_replayed_offline_rows():
    """A number from a replay and a number from a live model must not be reportable as the same thing. The
    check runs in BOTH directions -- a live measurement filed under the offline label would corrupt the
    regression floor just as badly as the reverse."""
    with pytest.raises(DB.ProvenanceMismatch):
        DB.bench_from_cassette(verify=False, model="live_llm_v1")
    with pytest.raises(DB.ProvenanceMismatch):
        DB.check_label("gpt-5.5-live", "offline")
    with pytest.raises(DB.ProvenanceMismatch):
        DB.check_label("offline_heuristic_v1", "live")
    DB.check_label("offline_heuristic_v1", "offline")          # the honest pairings pass
    DB.check_label("live_llm_v1", "live")


def test_baseline_declares_the_generator_it_was_measured_on():
    """The committed floor and the committed cassette must agree about WHICH GENERATOR produced them. Without
    this, a live re-record of the cassette would be silently graded against a floor recorded from the offline
    builders -- two different products, one number, and no way to tell from the artifact."""
    from virturoid.services.design_cassette import DesignCassette
    assert _BASELINE["mode"] == DesignCassette().mode() == "offline"
    assert _BASELINE["model"] == DB.label_for_mode(_BASELINE["mode"])


def test_a_subset_cassette_is_marked_and_not_scored_as_phantom_failures(tmp_path):
    """A small live arm holds a handful of prompts. Scoring the absent 15 as schema failures would book a
    verdict@1 that looks like catastrophic regression, so the denominator narrows and the funnel says so."""
    from virturoid.services import design_battery as BB
    from virturoid.services.design_cassette import DesignCassette
    full = DesignCassette()
    part = DesignCassette(tmp_path / "part.json")
    keep = [BB.prompt_id(r) for r in BB.battery()][:3]
    for pid in keep:
        part._data["entries"][pid] = full.entry(pid)
    out = DB.bench_from_cassette(cassette=part, verify=False, only_recorded=True)
    assert out["n_attempts"] == 3 and out["provenance"]["is_subset"] is True
    assert "SUBSET" in out["subset_warning"]
    assert out["schema_valid@1"] == 1.0            # the three real rows, not 3/20

    # and WITHOUT only_recorded the absent rows are honest failures, never silently dropped
    wide = DB.bench_from_cassette(cassette=part, verify=False)
    assert wide["n_attempts"] == len(BB.battery())
    assert wide["schema_valid@1"] == round(3 / len(BB.battery()), 4)


def test_diff_compares_only_the_prompts_both_runs_scored():
    """The cassette-vs-live comparison is the point of the harness, and it is a per-case comparison: the two
    runs may cover different prompt sets, and even when they do not, verdict@1 moves in steps the size of the
    gate's own tolerance (see ``per_case``). So the diff names the bodies that changed."""
    base = {"mode": "offline", "model": "offline_heuristic_v1", "verdict@1": 0.5,
            "per_case": {"a": True, "b": False, "c": True},
            "per_case_outcome": {"a": "credible", "b": "failed", "c": "credible"}}
    cand = {"mode": "live", "model": "live_llm_v1", "verdict@1": 1.0,
            "per_case": {"a": False, "b": True, "d": True},
            "per_case_outcome": {"a": "refused", "b": "credible", "d": "credible"}}
    d = DB.diff_funnels(base, cand)
    assert d["n_shared"] == 2 and d["shared_prompt_ids"] == ["a", "b"]
    assert d["gained"] == ["b"] and d["lost"] == ["a"] and d["n_changed"] == 2
    assert d["only_in_baseline"] == ["c"] and d["only_in_candidate"] == ["d"]
    # shared-subset rates, NOT the full-battery ones, are what the delta is computed from
    assert d["verdict@1_baseline_shared"] == 0.5 and d["verdict@1_candidate_shared"] == 0.5
    assert d["delta_verdict@1_shared"] == 0.0
    assert d["baseline_mode"] == "offline" and d["candidate_mode"] == "live"
    # ...and the tri-state says WHICH KIND of change 'lost' was: 'a' is not broken, it is declined
    assert d["outcome_changes"] == {"a": "credible -> refused", "b": "failed -> credible"}
    assert d["declined_by_candidate"] == ["a"] and d["built_but_broken_in_candidate"] == []


# ---------------------------------------------------------------- the LIVE baseline (#280)
def test_the_live_baseline_declares_the_generator_it_was_measured_on():
    """The live floor and the live cassette must agree about WHICH GENERATOR produced them, exactly as the
    offline pair does. Without this, a re-record that fell back to the deterministic builders would be graded
    against a floor labelled 'live' and the artifact would stop describing itself."""
    cas = _live_cassette()
    assert _LIVE_BASELINE["mode"] == cas.mode() == "live"
    assert _LIVE_BASELINE["model"] == DB.label_for_mode("live") == "live_llm_v1"
    assert cas.provenance()["n_authored_by_builder"] == 0     # no offline body smuggled into the live floor


def test_the_two_baselines_cannot_be_graded_against_each_others_floor():
    """THE point of having two files. The offline number grades the deterministic builders; the live number
    grades the product. They are not interchangeable and the harness must refuse the swap in BOTH directions."""
    assert _BASELINE["mode"] == "offline" and _LIVE_BASELINE["mode"] == "live"
    assert _BASELINE["model"] != _LIVE_BASELINE["model"]
    with pytest.raises(DB.ProvenanceMismatch):
        DB.check_label(_LIVE_BASELINE["model"], _BASELINE["mode"])     # live label on offline rows
    with pytest.raises(DB.ProvenanceMismatch):
        DB.check_label(_BASELINE["model"], _LIVE_BASELINE["mode"])     # offline label on live rows
    # and the two disagree per case on prompts both measured -- so quoting one for the other is a real error,
    # not a pedantic one. MEASURED 2026-08-08: offline 0.5 vs live 0.0 on the six shared prompts.
    shared = set(_BASELINE["per_case"]) & set(_LIVE_BASELINE["per_case"])
    assert shared, "the two baselines share no prompt; the diff below would be vacuous"
    assert any(_BASELINE["per_case"][k] != _LIVE_BASELINE["per_case"][k] for k in shared)


def test_a_rate_limit_is_not_a_design_failure():
    """THE #280 defect, pinned. MEASURED 2026-08-08: a full live battery produced 16 'refusals' of which **14
    were HTTP 429** -- the org's 50-requests-per-day cap on the fast model. The funnel scored all 16 the same
    way, as schema failures in the absolute denominator, and printed ``verdict@1 = 0.0``. That number measured
    an exhausted API quota while carrying the product's label.

    Transport failures must leave the denominator (NAMED, never silently dropped); design refusals must stay in
    it, because a model declining to produce a sound body IS the product's measured behaviour."""
    from virturoid.services.design_cassette import is_infrastructure_failure
    assert is_infrastructure_failure("Error code: 429 - rate limit reached for gpt-4.1-mini")
    assert is_infrastructure_failure("APIConnectionError: connection reset")
    assert is_infrastructure_failure("Error code: 503 - service unavailable")
    # a real design refusal is NOT excused -- it is the product's honest answer and stays scored
    assert not is_infrastructure_failure(
        "LLMDesignUnavailable: center of mass falls outside the foot support polygon")
    assert not is_infrastructure_failure("LLMDesignUnavailable: 5 joint(s) near/over their static hold torque")
    assert not is_infrastructure_failure(None) and not is_infrastructure_failure("")

    out = DB.bench_from_cassette(cassette=_live_cassette(), verify=False)
    excluded = set(out["excluded_infrastructure_failures"])
    assert excluded == set(_LIVE_BASELINE["infrastructure_excluded"])
    assert out["n_attempts"] == _LIVE_BASELINE["n_attempts_scored"] == 20 - len(excluded)
    assert "429" in out["infrastructure_warning"] or "TRANSPORT" in out["infrastructure_warning"]
    # the excluded prompts are named in the funnel, so "unmeasured" can never read as "measured and failed"
    assert all(pid in out["infrastructure_warning"] for pid in excluded)
    # and they are absent from per_case rather than sitting there as False
    assert not (excluded & set(out["per_case"]))


def test_the_live_refusals_are_recorded_as_correct_behaviour_not_as_failures():
    """THE representation fix, at the fixture. Four prompts on the live battery were REFUSED by production in
    strict mode -- two for a rest pose whose centre of mass leaves the support polygon, two for joints at
    108-128% of their static hold torque. Declining to ship those bodies is the product working.

    Under the boolean floor they were recorded as ``false``, i.e. as failures, indistinguishable from a body
    that was built and fell over. So the floor could not protect the refusal, and the OFFLINE floor
    simultaneously marked three of the same prompt ids ``true``. Now each is ``refused``: still not credible
    (verdict@1 is a CAPABILITY rate and an unserved prompt is unserved), but a correct outcome that
    ``outcome_regressions`` will defend."""
    cas = _live_cassette()
    refused = {p for p in cas.prompt_ids() if cas.entry_provenance(p)["failure_kind"] == "design_refusal"}
    assert refused == set(_LIVE_BASELINE["refusals"]) and len(refused) == _LIVE_BASELINE["n_refused"] == 4
    # the two the task named, plus the two the same grounding rules caught on the other phrasings
    assert {"elephant__appearance", "palletizer__construction", "mobile_manip__appearance"} <= refused
    for pid, reason in _LIVE_BASELINE["refusals"].items():
        assert reason.strip(), f"{pid} refused with no recorded reason"
        assert _LIVE_BASELINE["per_case_outcome"][pid] == DB.OUTCOME_REFUSED      # correct, not failed
        assert _LIVE_BASELINE["per_case"][pid] is False                           # ...and still not credible


def test_correct_at_1_counts_an_honest_refusal_and_verdict_at_1_does_not():
    """Two rates, and neither may be quoted for the other. verdict@1 answers 'did the customer get a robot';
    correct@1 answers 'did the product do the right thing'. Collapsing them in either direction is a lie:
    counting a refusal as capability inflates the product, counting it as a defect punishes the honesty gate
    for firing. MEASURED on the live cassette: 0.1 and 0.5 -- the gap IS the four refusals."""
    out = DB.bench_from_cassette(cassette=_live_cassette(), verify=False)
    n = out["n_attempts"]
    assert out["refused@1"] == round(4 / n, 4) == _LIVE_BASELINE["refused@1"]
    # verify=False cannot produce a credible verdict, so correct@1 is exactly the refusal rate here
    assert out["correct@1"] == out["refused@1"] and out["verdict@1"] == 0.0
    # and the refusals are routed as refusals, NOT as schema failures, in the error histogram
    assert out["error_classes"].get("refused") == 4 and "schema" not in out["error_classes"]
    assert set(out["refusals"]) == set(_LIVE_BASELINE["refusals"])
    # the recorded floor keeps the two rates apart too
    assert _LIVE_BASELINE["verdict@1"] == 0.1 and _LIVE_BASELINE["correct@1"] == 0.5


def test_a_refusal_reclassified_as_transport_cannot_slip_out_of_the_gate():
    """The subtle way this gate could be silently emptied. A refusal row carries no gene, so its replayed
    outcome is a pure function of ``is_infrastructure_failure`` over the recorded error text. Add one word to
    INFRASTRUCTURE_ERROR_MARKERS -- 'capacity' appears verbatim in the palletizer refusal -- and that prompt
    stops being a design refusal, leaves the denominator as an infrastructure exclusion, and DISAPPEARS from
    per_case_outcome instead of failing anything. Absence must therefore be a gate condition."""
    from virturoid.services import design_cassette as DC
    out = DB.bench_from_cassette(cassette=_live_cassette(), verify=False)
    assert "palletizer__construction" in out["per_case_outcome"]

    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(DC, "INFRASTRUCTURE_ERROR_MARKERS", DC.INFRASTRUCTURE_ERROR_MARKERS + ("capacity",))
        widened = DB.bench_from_cassette(cassette=_live_cassette(), verify=False)
    finally:
        mp.undo()
    assert "palletizer__construction" not in widened["per_case_outcome"]      # vanished, not failed
    reg = DB.outcome_regressions(_LIVE_BASELINE["per_case_outcome"], widened["per_case_outcome"])
    assert "palletizer__construction" in reg["missing"]                      # ...and the gate says so


def test_the_offline_floor_cannot_claim_a_capability_the_product_lacks():
    """THE defect this task exists to close, machine-checked instead of written in a note.

    The offline per-case floor is a map of prompt ids to ``credible``, published beside ``verdict@1``, and it
    was being read as a statement about the product. It is not one. MEASURED 2026-08-08 across both live
    recording sessions: of the ELEVEN bodies the offline floor protects, **zero** have live corroboration --
    3 are prompts production correctly refuses, 2 are bodies the live model built that do not move, and 6 have
    never returned a real answer at all (every attempt died on HTTP 429).

    So the claim and the evidence now live in the same artifact, and this test keeps them from drifting: the
    offline fixture's ``production_outcome`` must AGREE with the live fixture wherever the live fixture has the
    prompt, and every uncorroborated ``credible`` must be named in ``offline_only_floor``. A reader of either
    file can no longer mistake the floor for a capability claim, and neither can a future re-record."""
    from virturoid.services import design_battery as BB
    ids = [BB.prompt_id(r) for r in BB.battery()]
    live = _LIVE_BASELINE["per_case_outcome"]
    prod = _BASELINE["production_outcome"]

    # 1) the offline fixture's picture of production is the LIVE fixture's, not an assertion of its own
    assert set(prod) == set(ids)
    for pid in ids:
        assert prod[pid] == live.get(pid, DB.PRODUCTION_UNMEASURED), pid
    # 'unmeasured' is not an outcome and must never be mistaken for one
    assert DB.PRODUCTION_UNMEASURED not in DB.OUTCOMES

    # 2) the stored reconciliation is exactly what the code computes from the two maps
    rec = DB.capability_reconciliation(_BASELINE["per_case_outcome"], live, prompt_ids=ids)
    stored = _BASELINE["capability_reconciliation"]
    for k in ("offline_credible", "corroborated", "unmeasured", "product_only", "offline_only_floor",
              "contradicted", "n_offline_credible", "capability_claim_rate", "n_prompts"):
        assert stored[k] == rec[k], f"{k}: fixture says {stored[k]}, measurement says {rec[k]}"

    # 3) every protected body without live corroboration is NAMED
    assert set(rec["offline_only_floor"]) == set(rec["contradicted"]) | set(rec["unmeasured"])
    assert set(rec["corroborated"]).isdisjoint(rec["offline_only_floor"])
    assert rec["capability_claim_rate"] == 0.0 and rec["n_offline_credible"] == 11, (
        "the reconciliation moved. That is allowed -- but update the fixture's floor_semantics prose with it, "
        "because that prose is what stops the map being read as capability.")

    # 4) the three the task named are contradicted by a REFUSAL specifically, not by a failure
    for pid in ("elephant__appearance", "palletizer__construction", "mobile_manip__appearance"):
        assert _BASELINE["per_case_outcome"][pid] == DB.OUTCOME_CREDIBLE       # the floor protects it...
        assert rec["contradicted"][pid] == DB.OUTCOME_REFUSED                  # ...and production declines it

    # 5) and the floor is wrong in the OTHER direction too, which is the part a boolean could never show:
    #    the one prompt production serves credibly is 'failed' in the offline floor.
    assert rec["product_only"] == ["welder__appearance"]
    assert _BASELINE["per_case_outcome"]["welder__appearance"] == DB.OUTCOME_FAILED


@pytest.mark.slow
def test_no_live_outcome_silently_changes_for_the_worse():
    """The per-case floor over the model's OWN outcomes, replayed token-free — the live twin of the offline
    per-case gate. Re-recording measures the generator and costs tokens; this replay measures the
    compiler+physics under those designs (and the classification of its refusals) and costs nothing.

    Tri-state, so it defends two different things at once: ``welder__appearance`` must keep articulating, and
    the four refusals must keep being refusals. The second is the one the boolean floor could not express --
    'refused' and 'built but broken' both read False, so a change that stopped the grounding gate firing and
    started shipping a topple-prone elephant would have moved nothing this test could see."""
    out = DB.bench_from_cassette(cassette=_live_cassette(), verify=True)
    floor = _LIVE_BASELINE["per_case_outcome"]
    reg = DB.outcome_regressions(floor, out["per_case_outcome"])
    assert not reg["missing"], f"live cassette no longer covers {reg['missing']}; re-record before moving floor"
    assert not reg["broke"], ("these live outcomes changed for the worse: "
                             + "; ".join(f"{k} ({v})" for k, v in reg["broke"].items()))
    assert not reg["improved"], (f"good news: {reg['improved']}. Re-record the live baseline so the floor "
                                 "rises deliberately rather than by drift.")
    # the headline rates the fixture publishes are the ones this replay produces
    assert (out["verdict@1"], out["correct@1"], out["refused@1"]) == \
        (_LIVE_BASELINE["verdict@1"], _LIVE_BASELINE["correct@1"], _LIVE_BASELINE["refused@1"])


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
