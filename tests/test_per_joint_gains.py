"""``crawl_gait_rollout`` can run PER-JOINT PD gains, and can make ``duty`` live — both OPT-IN, both DEFAULT-OFF.

WHY THESE EXIST (measured, task #265):

* ``recipe_gains`` is the literature's zero-rollout PD rule (``Kp = J_eff * wn^2`` off the ``mj_fullM`` diagonal).
  It is finished and correct, but its ONLY callers were the recipe/CPG path — while ``crawl_gait_rollout``, the
  controller behind every credible walk this product has, applied ONE SCALAR ``kp`` to every joint. Measured on
  the grounded authored bodies, the law wants a 5.8x (dog) / 8.6x (cheetah) / 9.4x (horse) spread between the
  stiffest and softest joint: at kp=32 the hip carrying a whole leg runs at 0.49x (dog) to 0.20x (horse) of the
  stiffness it needs, while the knee runs 1.8x-2.8x TOO stiff — the "neck out-torques knee" defect, in numbers.
* ``duty`` was DEAD. ``crawl_gait_rollout`` never read it (the live swing fraction, ``lift_duty``, is derived
  structurally from ``beta``), yet ``gait_search.PARAM_NAMES`` searched it over [0.12, 0.42] and ``bank_gait``
  stored it — a sixth of the CEM's dimensionality spent on a no-op, and every banked ``duty`` describing nothing.

Both are REAL CONTROLLER CHANGES on a body/gait pair that is co-tuned ([[body-and-gait-are-co-tuned]]), so the
contract these tests lock in is: the wiring exists and demonstrably bites when asked for, and is EXACTLY INERT
otherwise. Turning either on by default needs its own measured before/after, which is not what this is.
"""
from __future__ import annotations

import numpy as np
import pytest

from virturoid.services import morph_policy as MP


def _template():
    """The canonical taught template — the one body the whole shipped operating point is co-tuned with."""
    from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
    return build_from_anatomy(_generic_legged_graph(n_pairs=2, girth=0.22, fan=True))


def test_default_rollout_is_untouched_in_shape():
    """OFF by default: no disclosure keys appear, so every existing consumer sees the same dict it always saw."""
    r = MP.crawl_gait_rollout(_template(), steps=120)
    assert "gain_mode" not in r and "kp_span" not in r and "lift_duty" not in r


def test_the_scalar_kp_collapses_a_real_spread():
    """The defect itself: the per-joint law wants materially different stiffness per joint on ONE body."""
    from virturoid.services.morph_graph import encode_robot
    g = _template()
    model = MP.compiled_model(MP.robot_mjcf(g), solver_iterations=20)
    kp, _kd, _ = MP.recipe_gains(model, encode_robot(model))
    assert float(kp.max()) / float(kp.min()) > 2.0, "if the spread were small this wiring would not be worth it"
    r = MP.crawl_gait_rollout(g, steps=120, per_joint_gains="recipe")
    assert r["gain_mode"] == "recipe"
    assert r["kp_span"] == [round(float(kp.min()), 3), round(float(kp.max()), 3)]   # the law reached the loop


def test_scaled_mode_preserves_the_bodys_operating_point():
    """``scaled`` composes with ``gait_flywheel.fit_gait_for_body``: it redistributes the FITTED scalar across
    joints instead of discarding it, so the typical joint keeps exactly the stiffness that was fitted."""
    r = MP.crawl_gait_rollout(_template(), steps=120, kp=90.0, kd=6.0, per_joint_gains="scaled",
                              return_control_plan=True)
    plan = r["control_plan"]
    assert np.isclose(float(np.median(plan["kp_per_joint"])), 90.0, rtol=1e-3)
    assert np.isclose(float(np.median(plan["kd_per_joint"])), 6.0, rtol=1e-3)
    assert min(plan["kp_per_joint"]) < 90.0 < max(plan["kp_per_joint"])   # genuinely spread, not a flat rescale


def test_per_joint_plan_refuses_to_call_itself_exportable():
    """The frozen target program carries a scalar kp/kd. A plan built under per-joint gains cannot reproduce the
    rollout it describes, so it must not claim it can — the vectors ride along for an exporter that can."""
    plan = MP.crawl_gait_rollout(_template(), steps=120, per_joint_gains="recipe",
                                 return_control_plan=True)["control_plan"]
    assert plan["exportable"] is False and "per-joint" in plan["reason"]
    assert len(plan["kp_per_joint"]) == len(plan["joint_names"])
    # the scalar path is still exportable, unchanged
    assert MP.crawl_gait_rollout(_template(), steps=120, return_control_plan=True)["control_plan"]["exportable"]


def test_unknown_gain_mode_is_refused():
    with pytest.raises(ValueError):
        MP.crawl_gait_rollout(_template(), steps=60, per_joint_gains="stiff")


def test_duty_is_inert_by_default_and_live_only_on_request():
    """The regression test for the dead knob. OFF, duty cannot move the gait AT ALL (that is the bug being
    documented, not a feature). ON, it must actually move it — otherwise the flag is decorative."""
    g = _template()
    off_lo = MP.crawl_gait_rollout(g, steps=500, duty=0.12)
    off_hi = MP.crawl_gait_rollout(g, steps=500, duty=0.42)
    assert off_lo["forward"] == off_hi["forward"], "duty is INERT unless live_duty=True; if this fails, the "\
                                                  "search space and every banked duty just became load-bearing"
    on_lo = MP.crawl_gait_rollout(g, steps=500, duty=0.12, live_duty=True)
    on_hi = MP.crawl_gait_rollout(g, steps=500, duty=0.42, live_duty=True)
    assert on_lo["lift_duty"] == 0.12 and on_hi["lift_duty"] == 0.42
    assert on_lo["forward"] != on_hi["forward"]


def test_live_duty_at_the_structural_value_is_a_no_op_on_a_quad():
    """Anchors WHERE the wiring lands: a quad's structural swing fraction is 1-beta = 0.25, so asking for 0.25
    explicitly must reproduce the shipped gait exactly — same number, not merely a similar one."""
    g = _template()
    assert (MP.crawl_gait_rollout(g, steps=500, duty=0.25, live_duty=True)["forward"]
            == MP.crawl_gait_rollout(g, steps=500)["forward"])
