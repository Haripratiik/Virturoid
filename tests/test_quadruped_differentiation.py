"""B1 (2026-07-24 audit, the #1 share-it blocker): different SIZE prompts must build DIFFERENT quadrupeds that
STILL walk. Before the fix, "a robot dog" / "a small robot dog" / "a large robot dog" / "a quadruped horse"
returned ONE byte-identical robot (the walkability fallback swapped a fixed template, erasing the composer's
size scaling). The fix scales the fanned WALKABLE template to the authored body's size (dynamic similarity), so
the outputs differentiate AND keep a credible gait.
"""
from __future__ import annotations

import importlib.util

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composing + evaluating needs MuJoCo")

_SIZE_PROMPTS = ["a robot dog", "a small robot dog", "a large robot dog", "a quadruped horse robot"]


def _body_sig(gene):
    legs = sum(float(getattr(s, "length_m", 0.0) or 0.0) for s in gene.segments if "leg" in (s.name or "").lower())
    mass = sum(float(getattr(s, "mass_kg", 0.0) or 0.0) for s in gene.segments)
    return (round(legs, 2), round(mass, 1))


def test_size_prompts_build_distinct_bodies():
    from virturoid.services.agent_tools import call_tool
    from virturoid.services import session_state as S
    sigs = set()
    for p in _SIZE_PROMPTS:
        rid = call_tool("create_robot", {"prompt": p})["result"]["robot_id"]
        sigs.add(_body_sig(S.get_robot(rid)))
    # at least 3 of the 4 prompts must be structurally distinct (was 1 byte-identical robot before the fix)
    assert len(sigs) >= 3, f"quadruped family collapsed to {len(sigs)} distinct bodies: {sigs}"


def test_scaled_walkable_templates_still_walk():
    """A scaled fanned template must keep a credible gait at its own scale (dynamic similarity), so a
    differentiated body isn't a body that fell over."""
    from virturoid.services.anatomy_compiler import _leg_len, ensure_walkable_quad
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.task_matched_eval import evaluate_robot
    walked = 0
    for p in ("a small robot dog", "a large robot dog", "a quadruped horse robot"):
        g = ensure_walkable_quad(compose_robot(p, llm=None), p, force=True)
        md = getattr(g, "metadata", None) or {}
        assert md.get("scaled_to_body"), f"{p}: template was not scaled to the body"
        if float(evaluate_robot(g).get("value", 0.0)) >= 0.5:
            walked += 1
    assert walked >= 3, f"only {walked}/3 scaled templates walked credibly"


def test_scale_is_clamped_to_the_walkable_band():
    """An extreme size never scales the template past the measured walkable band (~[0.6, 1.4])."""
    from virturoid.services.anatomy_compiler import _leg_len, ensure_walkable_quad
    from virturoid.services.morphology_composer import compose_robot
    g = ensure_walkable_quad(compose_robot("a gigantic robot dog", llm=None), "a gigantic robot dog", force=True)
    sc = (getattr(g, "metadata", None) or {}).get("scaled_to_body")
    if sc:                                                    # if it adopted a scaled template, the scale is bounded
        assert 0.6 <= sc["scale"] <= 1.4
