"""An argument this tool does not honour verbatim has to say so.

Measured through ``call_tool`` on a real Go2: ``simulate_bench_log {delay_ms: 6.0}`` returned
``injected.delay_ms: 10.0`` and nothing else. Snapping to the control tick is CORRECT -- delay is injected by
holding commands for whole control periods, so at 100 Hz only multiples of 10 ms exist. Saying nothing about
it is not. A caller sweeping 5/6/7 ms would have got three identical runs, read them as three measurements,
and concluded the fit was insensitive to delay; the quantisation was invisible in every artifact, including
the filename, which carries ticks rather than milliseconds.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="a bench log needs MuJoCo")


@pytest.fixture(scope="module")
def held():
    from virturoid.services import session_state as S
    from virturoid.services.morphology_composer import compose_robot
    return S.put_robot(compose_robot("a four legged robot dog", llm=None), label="sysid-coercion")


def _run(held, tmp_path, **kw):
    from virturoid.services.sysid.tools import simulate_bench_log
    args = {"robot_id": held, "hold_only": True, "budget_s": 6.0, "out_dir": str(tmp_path)}
    args.update(kw)
    out = simulate_bench_log(args)
    assert out.get("ok"), out
    return out


def test_a_snapped_delay_is_reported_as_snapped(held, tmp_path):
    out = _run(held, tmp_path, delay_ms=6.0)
    assert out["injected"]["delay_ms_requested"] == 6.0
    assert out["injected"]["delay_ms"] == 10.0
    assert out["injected"]["delay_control_ticks"] == 1

    coerced = out["coerced_arguments"]
    assert len(coerced) == 1
    c = coerced[0]
    assert c["argument"] == "delay_ms" and c["requested"] == 6.0 and c["used"] == 10.0
    assert "whole control ticks" in c["why"] and "100 Hz" in c["why"]
    assert "multiple of 10" in c["to_get_what_you_asked_for"]
    assert out["coercion_summary"] == "delay_ms 6 -> 10"


def test_a_delay_on_the_grid_reports_no_coercion(held, tmp_path):
    """The disclosure must be a signal, not decoration -- a caller who trips nothing should read nothing."""
    out = _run(held, tmp_path, delay_ms=20.0)
    assert "coerced_arguments" not in out and "coercion_summary" not in out
    assert out["injected"]["delay_ms"] == 20.0
    assert out["injected"]["delay_ms_requested"] == 20.0


def test_the_default_delay_is_on_the_grid(held, tmp_path):
    """The 20 ms default must not report itself as a coercion on every single call."""
    out = _run(held, tmp_path)
    assert "coerced_arguments" not in out
    assert out["injected"]["delay_ms"] == 20.0


def test_a_sub_tick_delay_says_it_became_zero(held, tmp_path):
    """The worst case: 2 ms rounds to NO delay at all. A caller who believed they had injected one would
    read the fit's recovered delay of ~0 as a success."""
    out = _run(held, tmp_path, delay_ms=2.0)
    assert out["injected"]["delay_control_ticks"] == 0
    assert out["injected"]["delay_ms"] == 0.0
    assert out["coerced_arguments"][0]["requested"] == 2.0
    assert out["coerced_arguments"][0]["used"] == 0.0


def test_the_quantisation_rule_is_stated_whether_or_not_it_bit(held, tmp_path):
    """`delay_quantised_to` is unconditional: the grid a caller has to aim at should not appear only after
    they have already missed it."""
    for ms in (20.0, 6.0):
        out = _run(held, tmp_path, delay_ms=ms)
        assert "control ticks at 100 Hz" in out["injected"]["delay_quantised_to"]
