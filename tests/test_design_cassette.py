"""WS-A.2 — the design cassette records/replays deterministically (master_plan_v6 §8.1).

Plus #208: every recorded design carries the provenance of the generator that authored it, and the committed
fixture's own provenance is pinned here — because the number CI gates on is only meaningful if you can tell
which generator produced it.
"""
from __future__ import annotations

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import design_battery as B
from virturoid.services.design_cassette import (DesignCassette, authored_by_model, design_from_prompt)
from virturoid.services.heldout_set import body_key


def test_committed_fixture_covers_the_whole_battery():
    cas = DesignCassette()                                        # default = committed fixture
    ids = {B.prompt_id(r) for r in B.battery()}
    assert set(cas.prompt_ids()) == ids                          # every battery prompt has a recorded design
    assert len(cas) == len(B.battery())
    assert cas.summary()["n_failures"] == 0


def test_replay_is_deterministic_token_free():
    a = DesignCassette()
    b = DesignCassette()
    pid = B.prompt_id(B.battery()[0])
    ga, gb = a.get_gene(pid), b.get_gene(pid)
    assert ga is not None and gb is not None
    assert body_key(ga) == body_key(gb)                         # same cassette -> byte-identical design


def test_design_from_prompt_prefers_cassette_and_never_generates_on_replay():
    cas = DesignCassette()
    rec = B.battery()[0]
    gene, source = design_from_prompt(rec["prompt"], prompt_id=B.prompt_id(rec), cassette=cas,
                                      allow_generate=False)      # generation forbidden...
    assert gene is not None and source == "cassette"            # ...yet a design is returned from the cassette


def test_absent_entry_without_generation_is_honest():
    cas = DesignCassette()
    gene, source = design_from_prompt("a brand new never-seen prompt", prompt_id="nope__x",
                                      cassette=cas, allow_generate=False)
    assert gene is None and source == "absent"


def test_record_roundtrips_including_failures(tmp_path):
    p = tmp_path / "cass.json"
    cas = DesignCassette(p)
    rec = B.battery()[0]
    gene, source = design_from_prompt(rec["prompt"], prompt_id=B.prompt_id(rec), cassette=None,
                                      allow_generate=True)
    assert gene is not None
    cas.record(B.prompt_id(rec), prompt=rec["prompt"], gene=gene, source=source)
    cas.record("broken__x", prompt="an impossible design", gene=None, source="error", error="boom")
    cas.save(battery_version=B.BATTERY_VERSION)

    reloaded = DesignCassette(p)
    assert reloaded.has(B.prompt_id(rec)) and reloaded.get_gene(B.prompt_id(rec)) is not None
    assert reloaded.has("broken__x") and reloaded.get_gene("broken__x") is None    # a recorded failure
    assert reloaded.summary()["n_failures"] == 1


# ---------------------------------------------------------------- provenance (#208)
def test_committed_fixture_was_produced_offline_and_the_artifact_says_so():
    """THE FINDING, pinned as a test. Every number Design-Bench publishes today is scored on these 20 rows,
    and NOT ONE of them was authored by a model: 14 ``heuristic`` + 6 ``anatomy_generic``, both deterministic
    builders. The bench's headline was therefore a measurement of the offline fallback, not of the LLM-first
    product a customer gets.

    This test is not a wish. If someone re-records the fixture live, it FAILS -- and it should, because the
    committed regression floor and the baseline's declared mode must move together (see the baseline gate in
    test_design_bench.py). Promote deliberately; never let the floor drift generators silently."""
    prov = DesignCassette().provenance()
    assert prov["mode"] == "offline"
    assert prov["n_authored_by_model"] == 0
    assert prov["n_authored_by_builder"] == prov["n_rows"] == len(B.battery())
    assert set(prov["design_sources"]) <= {"heuristic", "anatomy_generic"}


def test_authored_by_model_reads_the_design_source_not_a_flag():
    """The classification has to be decidable from a recorded row alone -- that is what makes the LEGACY
    fixture classifiable without re-recording it. Sources assigned only inside ``llm is not None`` branches
    count as model-authored; the deterministic builders do not, even though a live model may have been CALLED
    before its proposal failed grounding and landed there."""
    for src in ("llm", "llm_anatomy_graph", "anatomy"):
        assert authored_by_model(src), src
    for src in ("heuristic", "anatomy_generic", "archetype", "anthropometric", None, "", "unknown"):
        assert not authored_by_model(src), src


def _stub_gene(design_source: str) -> RobotGene:
    g = RobotGene(id="g", species="x", robot_class="quadruped",
                  segments=[GeneSegment(name="torso", parent=None)])
    g.design_source = design_source
    return g


def test_recorded_provenance_round_trips_and_drives_the_mode(tmp_path):
    p = tmp_path / "cass.json"
    cas = DesignCassette(p)
    cas.record("a__x", prompt="p", gene=_stub_gene("llm_anatomy_graph"), source="llm_anatomy_graph",
               llm_calls=3, backend="openai", model="gpt-5.5", strict_llm=True)
    cas.save(battery_version=B.BATTERY_VERSION)
    prov = DesignCassette(p).provenance()
    assert prov["mode"] == "live"
    assert prov["llm_calls_total"] == 3 and prov["n_rows_with_measured_calls"] == 1
    assert prov["backends"] == ["openai"] and prov["models"] == ["gpt-5.5"]


def test_a_refused_live_design_does_not_relabel_the_run_as_offline(tmp_path):
    """MEASURED 2026-08-07: the 5-prompt live arm produced 3 designs and 2 hard refusals
    (``LLMDesignUnavailable`` on mobile_manip and starfish). Refusal is the CORRECT strict-mode behaviour --
    production would rather return nothing than substitute a template -- so counting those rows as evidence of
    the offline generator would have mislabelled a live measurement. Failures are counted, not voted."""
    p = tmp_path / "refused.json"
    cas = DesignCassette(p)
    cas.record("a__x", prompt="p", gene=_stub_gene("llm_anatomy_graph"), source="llm_anatomy_graph", llm_calls=4)
    cas.record("b__x", prompt="q", gene=None, source="error", error="LLMDesignUnavailable: ...", llm_calls=6)
    assert cas.mode() == "live"
    prov = cas.provenance()
    assert (prov["n_rows"], prov["n_built"], prov["n_failed"]) == (2, 1, 1)
    # #280 split the old undifferentiated "(design failed)" bucket: a refusal is the product's honest answer,
    # a 429 is not an answer at all, and the artifact now distinguishes them.
    assert prov["design_sources"]["(design refused)"] == 1
    assert prov["n_design_refused"] == 1 and prov["n_infrastructure_failed"] == 0
    assert prov["llm_calls_total"] == 10          # the refusal still cost tokens, and the artifact says so


def test_a_cassette_holding_both_generators_reports_mixed_not_live(tmp_path):
    """The dangerous middle case: a live re-record where some prompts fell back to a deterministic builder.
    Calling that 'live' would credit the model with bodies it did not design, so it reads ``mixed`` and the
    bench's label check (design_bench.check_label) refuses a live label for it."""
    p = tmp_path / "mixed.json"
    cas = DesignCassette(p)
    cas.record("a__x", prompt="p", gene=_stub_gene("llm_anatomy_graph"), source="llm_anatomy_graph", llm_calls=2)
    cas.record("b__x", prompt="q", gene=_stub_gene("heuristic"), source="heuristic", llm_calls=2)
    assert cas.mode() == "mixed"
    prov = cas.provenance()
    # tokens WERE spent on the fallback row -- spend does not equal authorship, and the artifact keeps both.
    assert prov["llm_calls_total"] == 4 and prov["n_authored_by_model"] == 1


def test_recording_measures_llm_calls_rather_than_trusting_the_caller(tmp_path, monkeypatch):
    """``design_from_prompt`` reads the spend ledger around the compose. With no backend the honest answer is
    zero -- which is exactly how the committed fixture came to be offline while nobody noticed."""
    monkeypatch.setenv("VIRTUROID_NO_INTERNAL_LLM", "1")
    p = tmp_path / "c.json"
    cas = DesignCassette(p)
    rec = B.battery()[0]
    gene, _ = design_from_prompt(rec["prompt"], prompt_id=B.prompt_id(rec), cassette=cas,
                                 allow_generate=True, record=True)
    assert gene is not None
    row = cas.entry_provenance(B.prompt_id(rec))
    assert row["llm_calls"] == 0
    assert row["authored_by_model"] is False
    assert cas.mode() == "offline"


# ---------------------------------------------------------------- infrastructure vs design failure (#280)
def test_a_transport_failure_is_recorded_as_infrastructure_not_as_a_refusal(tmp_path):
    """MEASURED 2026-08-08: a full 20-prompt live battery produced 16 failures, and 14 of them were HTTP 429 --
    the org's 50-requests-per-day cap on the fast model. Every one was stored exactly like a design refusal, so
    Design-Bench scored them in the absolute denominator and printed ``verdict@1 = 0.0``.

    A failure now carries WHY, decided from the recorded error text so legacy rows classify without a re-record
    (the same property that let #208 classify the 2026-07 fixture)."""
    p = tmp_path / "mixed_failures.json"
    cas = DesignCassette(p)
    cas.record("ok__x", prompt="p", gene=_stub_gene("llm_anatomy_graph"), source="llm_anatomy_graph", llm_calls=2)
    cas.record("quota__x", prompt="q", gene=None, source="error", llm_calls=0,
               error="LLMDesignUnavailable: the anatomy graph did not compile: Error code: 429 - rate limit")
    cas.record("refused__x", prompt="r", gene=None, source="error", llm_calls=3,
               error="LLMDesignUnavailable: center of mass falls outside the foot support polygon")
    cas.save(battery_version=B.BATTERY_VERSION)

    prov = DesignCassette(p).provenance()
    assert prov["n_failed"] == 2
    assert prov["n_infrastructure_failed"] == 1 and prov["infrastructure_prompt_ids"] == ["quota__x"]
    assert prov["n_design_refused"] == 1 and prov["design_refused_prompt_ids"] == ["refused__x"]
    # usable = everything that is a real observation of the product: the build AND the honest refusal
    assert prov["n_usable"] == 2
    # a transport failure must not vote the run offline either -- it is not evidence about a generator at all
    assert prov["mode"] == "live"


def test_a_transport_failed_row_can_be_dropped_so_a_re_record_retries_it(tmp_path):
    """Recording REPLAYS anything the cassette already holds, so without an explicit drop a single rate-limit
    window would poison those prompts forever -- every later re-record would skip them and keep serving the
    outage. ``scripts/design_bench.py`` drops transport-failed rows before recording; this is that primitive."""
    p = tmp_path / "drop.json"
    cas = DesignCassette(p)
    cas.record("a__x", prompt="p", gene=None, source="error",
               error="LLMDesignUnavailable: Error code: 429 - rate limit reached")
    assert cas.has("a__x") and cas.entry_provenance("a__x")["infrastructure_failure"]
    assert cas.drop("a__x") is True
    assert not cas.has("a__x")
    assert cas.drop("a__x") is False              # idempotent, and honest about having done nothing

    # and with the row gone, the generate path is reachable again instead of replaying the outage
    rec = B.battery()[0]
    gene, source = design_from_prompt(rec["prompt"], prompt_id=B.prompt_id(rec), cassette=cas,
                                      allow_generate=True, record=True)
    assert gene is not None and source != "cassette"


def test_the_committed_live_cassette_carries_its_outage_openly():
    """The committed live fixture was recorded through a rate-limit window. That is not hidden: the artifact
    reports which prompts died on transport, so nobody can read '6 built of 20' as a capability statement.

    It MERGES two sessions of the identical production lane (2026-08-07 and 2026-08-08). The 08-08 full-battery
    run lost 14 rows to the org's daily cap; where the earlier arm had a real answer for one of those prompts
    it is used instead, which is why coverage is 10/20 and not 6/20. A row is never merged over a real answer,
    only over a transport failure, and every entry carries its own ``recorded_at``."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "design_cassette_live_v1.json"
    cas = DesignCassette(path)
    prov = cas.provenance()
    assert prov["mode"] == "live" and prov["n_rows"] == len(B.battery())
    assert prov["n_infrastructure_failed"] == 10 and prov["n_usable"] == 10
    assert prov["n_built"] == 6 and prov["n_design_refused"] == 4
    assert prov["models"] == ["gpt-5.5"]
    # the merge is self-describing: both sessions declared, and every row dates itself
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw["recording_sessions"]) == {"2026-08-07", "2026-08-08"}
    assert all(e.get("recorded_at", "")[:10] in raw["recording_sessions"] for e in raw["entries"].values())
    # ...and every row ran the same lane, so the two sessions are commensurable
    assert all(cas.entry_provenance(p)["strict_llm"] is True for p in cas.prompt_ids())
    # and the call counter is documented as a FLOOR -- the ledger records a call only after complete_json
    # returns, so every completion that raised (all the 429s) billed latency and counted zero.
    assert "FLOOR" in prov["llm_calls_caveat"]


def test_battery_is_disjoint_from_heldout():
    """Design-Bench (battery) and the compounding eval (held-out) must not share prompts, or the funnel would
    leak into the transfer curve."""
    from virturoid.services import heldout_set as H
    battery_prompts = {H.normalize_prompt(r["prompt"]) for r in B.battery()}
    held = {H.normalize_prompt(p) for p in H.held_out_prompts()}
    assert battery_prompts.isdisjoint(held)
