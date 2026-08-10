"""The deploy-select A/B has to compare against the DEFAULT, and the ledger has to say which number shipped.

Two defects, both found by re-deriving ``/api/moat``'s headline from the provenance table (2026-08-07, live bank):

1. ``crawl_gait_rollout`` falls back to ``gene.metadata['gait_params']`` for every gait kwarg it is not handed
   (morph_policy.py:1285-1299). The deploy-select called it bare for the "shipped default" arm, so on a body
   carrying a tuned cache BOTH ARMS RAN THE SAME GAIT. All 1598 ``tuned_for_this_body`` rows in the live bank
   carry delta EXACTLY 0.000 — a full wasted rollout each, and a safety net that could not fire.

2. ``delta`` is the COUNTERFACTUAL (recalled - default), not what shipped. The bank's flywheel-hint arm reads
   -0.1288 m counterfactual and +0.0949 m shipped: opposite signs. With only the counterfactual recorded, the
   panel reads as "the flywheel makes robots worse" when the measured shipped effect is positive.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from virturoid.services import ai_native_tools as AIT
from virturoid.services import morph_policy as MP


@pytest.fixture(scope="module")
def quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four-legged walking robot", ensure_walkable=False)


def _rollout(forward: float):
    """A CREDIBLE walk at a chosen distance (level body -> roll/pitch 0, so classify() passes its full bar)."""
    return {"survived": True, "upright_frac": 0.9, "height_ratio": 0.9, "cadence": 2.1, "support_frac": 0.8,
            "forward": forward, "speed": 0.31, "alive": 900,
            "qpos_frames": [np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])] * 4}


# --------------------------------------------------------------- 1: the default arm is actually the default
def test_default_arm_does_not_inherit_the_bodys_tuned_cache(quad, monkeypatch):
    """The baseline rollout must name the shipped constants, or a tuned body is compared against ITSELF."""
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT

    tuned = {"freq": 2.6189, "hip_amp": 1.08, "knee_amp": 0.939, "kp": 59.57, "kd": 14.0}
    gene = SimpleNamespace(**{k: getattr(quad, k) for k in dir(quad) if not k.startswith("_")})
    gene.metadata = {**(getattr(quad, "metadata", None) or {}), "gait_params": dict(tuned)}

    calls: list[dict] = []

    def fake_crawl(g, **kw):
        calls.append(kw)
        return _rollout(0.80 if len(calls) == 1 else 0.50)

    monkeypatch.setattr(MP, "crawl_gait_rollout", fake_crawl)
    monkeypatch.setattr(AIT, "_record_gait_hint_outcome", lambda *a, **k: None)
    # NO MINED HINT for this body. Deploy-select runs a THIRD arm when a controller is landed on a body the
    # flywheel has a hint region for (see ``_honest_gait``), and whether this session's scratch bank has banked
    # two walks near a composed quadruped depends on which other tests ran first. Pinning it empty keeps this
    # test measuring the thing it names — that arm 2 is the SHIPPED DEFAULT and not the metadata fallback.
    monkeypatch.setattr(AIT, "_mined_hint_params", lambda _g: {})

    AIT._honest_gait(gene, steps=600)

    assert len(calls) == 2, "the deploy-select must run both arms"
    deployed, baseline = calls[0], calls[1]
    # arm 1 deploys the body's own tuned point...
    assert deployed["freq"] == pytest.approx(tuned["freq"])
    # ...and arm 2 must PIN the shipped default, not leave it to the metadata fallback.
    for key, value in _DEFAULT_GAIT.items():
        assert key in baseline, f"the default arm left {key!r} to gene.metadata -> it re-ran the tuned gait"
        assert baseline[key] == pytest.approx(value)
    assert baseline["freq"] != pytest.approx(tuned["freq"])


# --------------------------------------------------- 2: the ledger records the shipped number, not only the CF
def _recorded(monkeypatch, source: str, hint_m: float, default_m: float, selected_default: bool) -> dict:
    seen: dict = {}

    class _VM:
        def __init__(self, _db):
            pass

        def record_provenance(self, *a, **kw):
            seen.update(kw)
            seen["child"] = a

    import virturoid.services.robotics_vector_memory as RVM
    monkeypatch.setattr(RVM, "RoboticsVectorMemory", _VM)
    AIT._record_gait_hint_outcome(SimpleNamespace(id="b", metadata={}), _rollout(hint_m), _rollout(default_m),
                                  selected_default=selected_default, source=source)
    return seen


def test_shipped_delta_is_recorded_beside_the_counterfactual(quad, monkeypatch):
    """Both arms stay in the row: delta is what a blind deploy WOULD have done, shipped_delta is what happened."""
    seen = _recorded(monkeypatch, "flywheel_hint", hint_m=0.30, default_m=0.80, selected_default=True)
    meta = seen["meta"]
    # the counterfactual is preserved exactly as before (this is the series we must never lose)
    assert seen["delta"] == pytest.approx(-0.50)
    assert meta["delta_is_counterfactual"] is True
    # ...and the number the robot actually walked away with says the guard caught the bad recall
    assert meta["shipped_forward_m"] == pytest.approx(0.80)
    assert meta["shipped_delta"] == pytest.approx(0.0)
    json.dumps(meta)  # the row must survive the round-trip into the provenance table


def test_a_winning_hint_ships_its_gain(quad, monkeypatch):
    seen = _recorded(monkeypatch, "flywheel_hint", hint_m=0.90, default_m=0.60, selected_default=False)
    assert seen["delta"] == pytest.approx(0.30)
    assert seen["meta"]["shipped_delta"] == pytest.approx(0.30)


# ------------------------------------------------------------- 3: one kind per mechanism, both still recorded
def test_own_tuned_point_is_not_banked_as_a_flywheel_hint(quad, monkeypatch):
    """A body's OWN fitted op-point is not a mined cross-body hint; pooling them diluted the hint arm ~4x."""
    hint = _recorded(monkeypatch, "flywheel_hint", 0.9, 0.6, False)
    own = _recorded(monkeypatch, "tuned_for_this_body", 0.9, 0.6, False)
    assert hint["kind"] == "gait_hint_deploy"
    assert own["kind"] == "gait_own_point_deploy"
    # neither is dropped -- separated, not deleted
    assert own["meta"]["source"] == "tuned_for_this_body"
    assert own["delta"] == pytest.approx(0.30)
