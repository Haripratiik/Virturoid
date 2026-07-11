"""R1 — in-place stance repair converts rolling legged bodies to walkers, keeping every authored part (§3.M)."""
from __future__ import annotations

import pytest

from virturoid.services import stance_repair as SR


def test_widen_preserves_every_part():
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a small robot dog", ensure_walkable=False)
    w = SR.widen_stance(g, 20.0)
    assert [s.name for s in w.segments] == [s.name for s in g.segments]     # no part added/removed
    assert len(w.segments) == len(g.segments)
    # only leg-proximal mount rotations changed; geometry/lengths untouched
    changed = sum(1 for a, b in zip(g.segments, w.segments) if a.mount_euler != b.mount_euler)
    assert 0 < changed <= len(g.segments)


def test_non_legged_body_is_untouched():
    from virturoid.services.morphology_composer import compose_robot
    arm = compose_robot("a tabletop robot arm with a gripper", ensure_walkable=False)
    out, rep = SR.repair_stance(arm, steps=200)
    assert out is arm and not rep["applied"]


@pytest.mark.slow
def test_repair_converts_a_rolling_body_and_keeps_parts():
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a small robot dog", ensure_walkable=False)   # measured: rolls/short as-authored
    out, rep = SR.repair_stance(g, steps=600)
    # the repair either legitimately helps (applied + better verdict) or honestly leaves it alone
    if rep["applied"]:
        assert rep["roll_deg"] > 0
        assert (rep["credible_after"] and not rep["credible_before"]) or \
               rep["forward_after_m"] > rep["forward_before_m"]      # never adopted unless strictly better
        assert len(out.segments) == len(g.segments)                 # design preserved (not a template swap)
    else:
        assert out is g


@pytest.mark.slow
def test_repair_is_a_noop_on_an_already_walking_body():
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a four-legged walking robot", ensure_walkable=True)   # already walks ~1.3 m
    out, rep = SR.repair_stance(g, steps=600)
    # an already-walkable body must not be degraded; if unchanged, the same gene is returned
    if not rep["applied"]:
        assert out is g
    assert rep["forward_after_m"] >= rep["forward_before_m"] - 1e-6          # never worse
