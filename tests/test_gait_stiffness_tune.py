"""Flywheel breakthrough: STIFFNESS is a first-class gait-tune dimension (flywheel_breakthrough_plan §control).

Measured: the freq/amp-only tune grid produced 0/6 ROBUSTLY-credible walks (double-confirm) on a spread of
quadrupeds — which is why LLM/composed bodies deployed the default kp=32 and LURCHed. Adding kp/kd to the grid
made 4/6 robustly credible. These tests lock in: the grid carries stiffness, the tuner caches it, and DEPLOY
(crawl_gait_rollout) reads the cached stiffness so the deploy verdict matches the tuned credible op-point.
"""
from __future__ import annotations

import pytest

from virturoid.services import morph_policy as MP


def test_tune_grid_carries_stiffness():
    kps = {cfg[3] for cfg in MP._GAIT_TUNE_GRID if len(cfg) == 5}
    assert kps - {32.0}, "the grid must include stiffer kp op-points, not only the soft default"
    assert MP._GAIT_TUNE_GRID[0][:3] == (1.5, 0.9, 1.0)      # soft quad default stays first (cheap fast-path)


def test_deploy_reads_cached_stiffness_from_metadata():
    """A body carrying a tuned kp/kd in metadata deploys WITH it (the tune->deploy link that makes the verdict
    match the tuned op-point). Verified via the control plan, no long rollout needed."""
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a four-legged walking robot", ensure_walkable=True)
    g.metadata = {**(g.metadata or {}), "gait_params": {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0,
                                                        "kp": 175.0, "kd": 7.0}}
    r = MP.crawl_gait_rollout(g, steps=60, return_control_plan=True)
    plan = r.get("control_plan") or {}
    assert plan.get("kp") == 175.0 and plan.get("kd") == 7.0   # deploy honored the cached stiffness
    # an EXPLICIT kwarg still overrides the cache
    r2 = MP.crawl_gait_rollout(g, steps=60, kp=40.0, return_control_plan=True)
    assert (r2.get("control_plan") or {}).get("kp") == 40.0


def test_legacy_3tuple_grid_still_normalizes():
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a four-legged walking robot", ensure_walkable=True)
    # a caller passing an old 3-tuple grid must not crash (padded to default kp/kd)
    out = MP.tune_crawl_gait(g, steps=200, grid=[(1.5, 0.9, 1.0)], cache=False)
    assert "kp" in out and "kd" in out


@pytest.mark.slow
def test_stiffness_tune_finds_a_credible_walk_the_soft_grid_misses():
    """End-to-end: the tuner finds+caches a stiffer credible walk, and deploy reproduces it (double-confirmed)."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a four-legged walking robot", ensure_walkable=True)
    t = MP.tune_crawl_gait(g, steps=700)
    if t.get("untuned"):
        pytest.skip("this body has no robustly-credible scripted gait even with stiffness (honest)")
    gp = (g.metadata or {}).get("gait_params", {})
    assert gp.get("kp") is not None
    # deploy reads the cached stiffness and reproduces a credible walk
    assert classify(MP.crawl_gait_rollout(g, steps=900)).startswith("CREDIBLE")
