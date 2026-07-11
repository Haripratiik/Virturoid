"""WS-A.2 — the design cassette records/replays deterministically (master_plan_v6 §8.1)."""
from __future__ import annotations

from virturoid.services import design_battery as B
from virturoid.services.design_cassette import DesignCassette, design_from_prompt
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


def test_battery_is_disjoint_from_heldout():
    """Design-Bench (battery) and the compounding eval (held-out) must not share prompts, or the funnel would
    leak into the transfer curve."""
    from virturoid.services import heldout_set as H
    battery_prompts = {H.normalize_prompt(r["prompt"]) for r in B.battery()}
    held = {H.normalize_prompt(p) for p in H.held_out_prompts()}
    assert battery_prompts.isdisjoint(held)
