"""Animal PROPORTION priors (P4) — the OFFLINE-ONLY heuristic for the no-LLM fallback path.

IMPORTANT (the honest layering): in strict PRODUCTION the LLM authors proportions — a giraffe's long legs, a
gecko's splayed short ones come from the MODEL (which knows biology, open-world), grounded by physics + the
robotics embedding. This deterministic table is NOT that; it is a finite class-template and is reached ONLY on
the no-LLM heuristic fallback path (VIRTUROID_ALLOW_HEURISTIC_FALLBACK / offline tests), so offline/demo bodies
aren't all identical. It is subordinate to the LLM, never the diversity solution — the LLM + grounding is.

Both legged fallback builders (the anatomy `_generic_legged_graph` and the `morphology_composer` composed path)
scaled bodies uniformly by size but kept FIXED ratios, so "horse"/"gecko"/"turtle" composed to identical
proportions offline (embedding cosine 1.0). These per-animal multipliers on leg length / stance width / torso /
mass make the OFFLINE bodies distinct. Generic/dog/wolf stay 1.0 -> byte-identical to the gait-pinned baseline.
Word-boundary match so 'ox' never hits 'box'.
"""
from __future__ import annotations

import re

_UNIT = {"leg": 1.0, "stance": 1.0, "torso": 1.0, "mass": 1.0}

_ANIMAL_PROPORTIONS: dict[str, dict] = {
    #                        leg   stance  torso  mass   (multipliers; 1.0 = the baseline quadruped)
    "giraffe":   {"leg": 1.9, "stance": 0.80, "torso": 1.10, "mass": 1.2},
    "horse":     {"leg": 1.5, "stance": 0.85, "torso": 1.20, "mass": 1.3},
    "deer":      {"leg": 1.5, "stance": 0.82, "torso": 1.10, "mass": 0.9},
    "moose":     {"leg": 1.6, "stance": 0.88, "torso": 1.20, "mass": 1.6},
    "ostrich":   {"leg": 1.7, "stance": 0.80, "torso": 0.90, "mass": 0.9},
    "cat":       {"leg": 0.9, "stance": 0.95, "torso": 1.00, "mass": 0.7},
    "gecko":     {"leg": 0.62, "stance": 1.55, "torso": 0.90, "mass": 0.55},
    "lizard":    {"leg": 0.68, "stance": 1.50, "torso": 1.00, "mass": 0.7},
    "salamander": {"leg": 0.60, "stance": 1.55, "torso": 1.05, "mass": 0.6},
    "crocodile": {"leg": 0.62, "stance": 1.60, "torso": 1.30, "mass": 1.6},
    "turtle":    {"leg": 0.55, "stance": 1.35, "torso": 1.15, "mass": 1.4},
    "tortoise":  {"leg": 0.52, "stance": 1.35, "torso": 1.15, "mass": 1.5},
    "bull":      {"leg": 0.92, "stance": 1.10, "torso": 1.18, "mass": 1.8},
    "hippo":     {"leg": 0.70, "stance": 1.25, "torso": 1.35, "mass": 2.2},
    "rhino":     {"leg": 0.85, "stance": 1.15, "torso": 1.30, "mass": 2.1},
    "elephant":  {"leg": 1.35, "stance": 1.10, "torso": 1.35, "mass": 2.4},
    "mouse":     {"leg": 0.70, "stance": 1.05, "torso": 0.90, "mass": 0.4},
    "corgi":     {"leg": 0.60, "stance": 1.05, "torso": 1.20, "mass": 0.8},
    "dachshund": {"leg": 0.55, "stance": 1.00, "torso": 1.45, "mass": 0.8},
    # Sight-hunters / cursorial runners: long thin legs, long flexible spine, narrow track, light frame. MEASURED
    # GAP 2026-07-21: "a cheetah robot that runs fast" composed BYTE-IDENTICAL geometry to "a robot dog" (the
    # table simply had no cheetah), which is the single most obvious "this is a template, not a generator" tell.
    "cheetah":   {"leg": 1.35, "stance": 0.82, "torso": 1.25, "mass": 0.75},
    "greyhound": {"leg": 1.30, "stance": 0.82, "torso": 1.20, "mass": 0.70},
    "antelope":  {"leg": 1.45, "stance": 0.82, "torso": 1.00, "mass": 0.8},
    "gazelle":   {"leg": 1.45, "stance": 0.80, "torso": 0.95, "mass": 0.7},
    "zebra":     {"leg": 1.45, "stance": 0.88, "torso": 1.18, "mass": 1.25},
    "camel":     {"leg": 1.55, "stance": 0.90, "torso": 1.15, "mass": 1.4},
    "leopard":   {"leg": 1.10, "stance": 0.95, "torso": 1.15, "mass": 1.0},
    "jaguar":    {"leg": 1.05, "stance": 1.00, "torso": 1.15, "mass": 1.1},
    "panther":   {"leg": 1.10, "stance": 0.95, "torso": 1.15, "mass": 1.0},
    "tiger":     {"leg": 1.00, "stance": 1.05, "torso": 1.15, "mass": 1.5},
    "lion":      {"leg": 1.00, "stance": 1.05, "torso": 1.10, "mass": 1.5},
    "wolf":      {"leg": 1.05, "stance": 0.95, "torso": 1.05, "mass": 0.95},
    "fox":       {"leg": 0.95, "stance": 0.95, "torso": 1.00, "mass": 0.6},
    "goat":      {"leg": 1.05, "stance": 0.90, "torso": 0.95, "mass": 0.7},
    "sheep":     {"leg": 0.95, "stance": 1.00, "torso": 1.05, "mass": 0.9},
    "pig":       {"leg": 0.60, "stance": 1.15, "torso": 1.25, "mass": 1.3},
    "cow":       {"leg": 1.00, "stance": 1.08, "torso": 1.25, "mass": 2.0},
    "bear":      {"leg": 0.85, "stance": 1.20, "torso": 1.20, "mass": 2.0},
    "badger":    {"leg": 0.58, "stance": 1.25, "torso": 1.15, "mass": 0.8},
    "otter":     {"leg": 0.55, "stance": 1.20, "torso": 1.20, "mass": 0.6},
    "rabbit":    {"leg": 0.85, "stance": 0.95, "torso": 0.85, "mass": 0.4},
    "hare":      {"leg": 0.95, "stance": 0.92, "torso": 0.85, "mass": 0.4},
}

# SIZE words -> a uniform body-scale multiplier (and a mass nudge on top of the s^3 volume law).
#
# MEASURED GAP 2026-07-21: "a large quadruped robot" and "a small lightweight quadruped" composed to the SAME
# body (identical total link length 2.241 m) — the composer's uniform scale `s` was only ever fed by an explicit
# scale_m / nominal_dims from a spec, so plain-English size words were dropped on the floor. A user typing
# "small" and getting the same robot as "large" is the fastest way to conclude the thing is a template.
#
# NOT included on purpose: "heavy-duty"/"rugged"/"lightweight-for-flight" already drive a REAL, separate
# mechanism (bom_builder._SKELETON_THICKNESS thickens or slims the load path, and the task material choice
# changes density), so adding them here would double-count. This table is body SIZE only.
_SIZE_WORDS: tuple[tuple[str, float, float], ...] = (   # (pattern, size_scale, extra mass multiplier)
    (r"\b(?:miniature|tiny|micro|palm[- ]?sized|insect[- ]?sized)\b", 0.55, 1.0),
    (r"\b(?:small|compact|little|mini)\b",                            0.72, 1.0),
    (r"\b(?:giant|huge|massive|enormous)\b",                          1.70, 1.0),
    (r"\b(?:large|big|full[- ]?size[d]?)\b",                          1.35, 1.0),
    (r"\b(?:lightweight|featherweight)\b",                            1.00, 0.70),
)


def size_scale(prompt: str) -> tuple[float, float]:
    """Return ``(size_scale, mass_multiplier)`` implied by plain-English SIZE words in the prompt.

    ``(1.0, 1.0)`` when the prompt says nothing about size — so a neutral prompt stays BYTE-IDENTICAL to the
    gait-tuned baseline that the locomotion tests pin. Deterministic, first-hit-wins, word-boundary matched
    (so 'little' never fires on 'littoral'). Subordinate to an explicit scale_m / nominal_dims, which are a
    real measurement and always win over a word.
    """
    p = (prompt or "").lower()
    scale, mass = 1.0, 1.0
    for pat, s, m in _SIZE_WORDS:
        if re.search(pat, p):
            scale, mass = s, m
            break
    # a size word AND a weight word can both apply ("a small lightweight quadruped")
    if scale != 1.0 and re.search(r"\b(?:lightweight|featherweight)\b", p):
        mass *= 0.70
    return scale, mass


def animal_proportions(prompt: str) -> dict:
    """Per-animal ratio multipliers from the prompt (leg/stance/torso/mass). Default all-1.0 (generic/dog/wolf ->
    the pinned baseline). First keyword hit wins; deterministic; word-boundary so 'ox' won't match 'box'."""
    p = (prompt or "").lower()
    for animal, prop in _ANIMAL_PROPORTIONS.items():
        if re.search(r"\b" + re.escape(animal) + r"\b", p):
            return {**_UNIT, **prop}
    return dict(_UNIT)
