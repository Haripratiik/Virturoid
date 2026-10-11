"""A landed controller is not swapped for a mined hint that walked backwards.

When a body carries a controller of its own, verify also tries the flywheel's mined hint for it and keeps the
better walk. That comparison used ``abs(forward)``, so between two rollouts that were not credible walks, a hint
going BACKWARD out-ranked the robot's own controller going forward. Measured 2026-10-11 on the composed hexapod
(its own fitted gait walks 7.9 m at the 6000-step judging horizon): an 800-step verify swapped in the mined hint
at -0.132 m and reported it as the robot's verdict. The comparison is signed now, like the one against the
shipped default beside it.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs MuJoCo")

_OWN = {"freq": 1.3, "hip_amp": 0.6, "knee_amp": 0.9, "kp": 80.0, "kd": 6.0}
_HINT = {"freq": 2.2, "hip_amp": 1.1, "knee_amp": 1.2, "kp": 120.0, "kd": 9.0}


def _rollout(forward: float) -> dict:
    """A rollout that is upright and stepping but too short to be a credible walk."""
    return {"finite": True, "forward": forward, "height_ratio": 0.95, "alive": 800, "survived": True,
            "speed": forward / 1.6, "cadence": 6.0, "upright_frac": 1.0, "support_frac": 0.8, "n_feet": 6,
            "yaw_change": 0.0, "lateral": 0.0, "path_m": abs(forward), "net_m": abs(forward), "straightness": 1.0,
            "heading_dev_p95_deg": 1.0, "heading_dev_max_deg": 1.0, "gait": "crawl", "n_tokens": 18,
            "steps": 800, "frame_every": 0, "qpos_frames": []}


def test_a_backward_hint_never_replaces_the_controller_the_robot_carries(monkeypatch):
    from virturoid.services import ai_native_tools as AN
    from virturoid.services import morph_policy as MP
    from virturoid.services.gait_flywheel import _DEFAULT_GAIT
    from virturoid.services.gait_quality import classify
    from virturoid.services.morphology_composer import compose_robot

    gene = compose_robot("a hexapod robot", ensure_walkable=False)
    gene.metadata = {**(gene.metadata or {}), "gait_params": dict(_OWN),
                     "gait_provenance": {"door": "train_held"}}

    def fake(_gene, **kw):
        point = {k: kw.get(k) for k in ("freq", "hip_amp", "knee_amp", "kp", "kd")}
        if point == _HINT:
            return _rollout(-0.50)                    # the hint walks backwards, further
        if point == {k: float(_DEFAULT_GAIT[k]) for k in point}:
            return _rollout(0.02)
        return _rollout(0.10)                         # the robot's own controller, forwards
    monkeypatch.setattr(MP, "crawl_gait_rollout", fake)
    monkeypatch.setattr(AN, "_mined_hint_params", lambda _g: dict(_HINT))
    monkeypatch.setattr(AN, "_record_gait_hint_outcome", lambda *a, **k: None)
    monkeypatch.setattr(AN, "_record_gait_lesson", lambda *a, **k: None)
    assert not classify(_rollout(0.10)).startswith("CREDIBLE")
    out = AN._honest_gait(gene, steps=800)
    assert out["gait_source"] == "tuned_for_this_body::train_held", out["gait_source"]
    assert out["forward_m"] == pytest.approx(0.10)
