"""Animal PROPORTION priors (P4) — the shared table that makes distinct animals STRUCTURALLY distinct.

Both legged builders (the anatomy path `_generic_legged_graph` and the `morphology_composer` composed fallback)
scaled bodies uniformly by size but kept FIXED ratios, so "horse", "gecko", "turtle" composed to identical
proportions (measured: identical 0.677 gait, embedding cosine 1.0). These per-animal multipliers on leg length /
stance width / torso length / mass make them distinct — and because the fine-variants experiment showed leg
length drives the optimal gait (freq 0.8-3.1), the diversity is FUNCTIONAL, not cosmetic. Generic/dog/wolf stay
1.0 -> byte-identical to the gait-pinned Go1-class baseline. Word-boundary match so 'ox' never hits 'box'.
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
}


def animal_proportions(prompt: str) -> dict:
    """Per-animal ratio multipliers from the prompt (leg/stance/torso/mass). Default all-1.0 (generic/dog/wolf ->
    the pinned baseline). First keyword hit wins; deterministic; word-boundary so 'ox' won't match 'box'."""
    p = (prompt or "").lower()
    for animal, prop in _ANIMAL_PROPORTIONS.items():
        if re.search(r"\b" + re.escape(animal) + r"\b", p):
            return {**_UNIT, **prop}
    return dict(_UNIT)
