"""Different prompts must produce DIFFERENT robots — the "is this a generator or a template?" test.

MEASURED 2026-07-21, the single most damaging thing a skeptical reviewer would find in 30 seconds:

    a robot dog                                -> total link length 2.155 m
    a cheetah robot that runs fast             -> total link length 2.155 m   (BYTE-IDENTICAL geometry)
    a large quadruped robot                    -> total link length 2.241 m
    a small lightweight quadruped              -> total link length 2.241 m   (same body as "large"!)
    a heavy-duty quadruped for carrying loads  -> total link length 2.241 m

Two independent causes, both fixed:
  1. `animal_proportions` had no `cheetah`, and there was NO size axis at all — plain-English size words were
     dropped on the floor because the composer's uniform scale was only ever fed by an explicit scale_m /
     nominal_dims from a spec.
  2. `ensure_walkable_quad` rebuilt every non-walking quadruped from HARDCODED dims, so each one collapsed onto
     one canonical body. It now tries the gene's own dims first and falls back to canonical RATIOS re-scaled to
     the gene's own SIZE, adopting whichever actually walks.

These assertions are deliberately about PROPERTIES (relative size, distinctness), not pinned constants, so they
keep their meaning when the bodies improve. The point is un-gameable: you cannot satisfy them by returning one
template, and you cannot satisfy them by returning junk that does not walk.
"""
from __future__ import annotations

import importlib.util

import pytest

_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="composition + walk verdicts need MuJoCo")


def _compose(prompt: str, walkable: bool = False):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt, llm=None, ensure_walkable=walkable)


def _total_len(gene) -> float:
    return sum(s.length_m for s in gene.segments)


def test_a_neutral_prompt_is_unchanged_by_the_size_axis():
    """GUARD: prompts that say nothing about size must stay on the gait-tuned baseline, so adding the size
    axis cannot silently re-tune every robot the locomotion tests pin."""
    from virturoid.services.animal_proportions import size_scale
    for neutral in ("a quadruped robot", "a four-legged walking robot", "a robot that walks"):
        assert size_scale(neutral) == (1.0, 1.0), neutral


def test_size_words_change_the_body_size():
    small, large = _compose("a small lightweight quadruped"), _compose("a large quadruped robot")
    assert _total_len(small) < _total_len(large), (
        f"'small' must build a SMALLER robot than 'large' "
        f"(got {_total_len(small):.3f} m vs {_total_len(large):.3f} m)")


def test_size_words_are_ordered_not_merely_different():
    """tiny < small < (neutral) < large < giant — a real axis, not three arbitrary bodies."""
    from virturoid.services.animal_proportions import size_scale
    order = [size_scale(f"a {w} quadruped robot")[0]
             for w in ("tiny", "small", "plain", "large", "giant")]
    assert order == sorted(order), f"size words must be monotonically ordered, got {order}"
    assert order[0] < 1.0 < order[-1]


def test_a_named_animal_differs_from_the_generic_dog():
    """A cheetah is a long-spined, long-legged runner; it must not compile to the dog body."""
    dog, cheetah = _compose("a robot dog"), _compose("a cheetah robot that runs fast")
    d, c = _total_len(dog), _total_len(cheetah)
    assert abs(c - d) / max(d, 1e-6) > 0.10, (
        f"cheetah must differ from dog by >10% total link length (dog {d:.3f} m, cheetah {c:.3f} m)")
    assert dog.id != cheetah.id


def test_distinct_prompts_stay_distinct_through_the_walkable_path_and_still_walk():
    """The regression that actually mattered: the walkability fallback used to REPLACE differentiated bodies
    with one canonical walker, so the product 'walked' only because every robot was the same robot. Bodies must
    survive the walkable path distinct AND still earn a real (un-gameable) walk verdict."""
    from virturoid.services.gait_quality import classify
    from virturoid.services.morph_policy import crawl_gait_rollout

    prompts = ["a robot dog", "a cheetah robot that runs fast",
               "a horse-like quadruped with long legs", "a bear-like quadruped"]
    genes = {p: _compose(p, walkable=True) for p in prompts}
    ids = {p: g.id for p, g in genes.items()}
    assert len(set(ids.values())) == len(prompts), f"walkable path collapsed distinct bodies: {ids}"

    for p, g in genes.items():
        verdict = classify(crawl_gait_rollout(g, steps=800, record_qpos=True))
        assert verdict.startswith("CREDIBLE"), f"{p!r} must still walk credibly, got {verdict}"
