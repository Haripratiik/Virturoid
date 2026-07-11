"""Fidelity keystone — extremity proportion coherence (the measured drum-foot bottleneck).

A LIVE gpt-5.5 build authored an otherwise-sensible quadruped whose four feet were 0.19 m-radius boxes on
0.02 m-radius legs: visually four drums that swallowed the legs, the walk CROUCHed (fitness -0.22), and EVERY
existing gate (validity, hygiene, part-balance, stance, M16) passed it. Two composed causes, both fixed:
  (1) no gate compared a limb TIP to the limb carrying it -> anatomy_critic gains ``extremity_proportion``
      (high severity, so it drives the LLM repair loop with a teaching detail);
  (2) the compiler multiplied an EXPLICITLY-authored foot's dims by the auto-foot widening (1.6x/1.9x —
      calibrated for feet inheriting slender leg girth) -> authored foot dims are now respected (floors only).
The live drum-foot gene is a committed fixture so this exact body class can never silently pass again.
"""
from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.gene import RobotGene
from virturoid.services.anatomy_compiler import build_from_anatomy
from virturoid.services.anatomy_critic import critique_gene

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "live_llm_drumfoot_gene.json"


def _drumfoot_gene() -> RobotGene:
    return RobotGene.from_dict(json.loads(_FIXTURE.read_text(encoding="utf-8")))


def _graph(foot_girth=None):
    parts = [
        {"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
        {"name": "leg_f", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out",
         "size": 0.4, "girth": 0.02, "segments": 3, "symmetry": "left_right", "joint": "revolute"},
        {"name": "leg_r", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down_out",
         "size": 0.4, "girth": 0.02, "segments": 3, "symmetry": "left_right", "joint": "revolute"},
    ]
    if foot_girth is not None:
        parts.append({"name": "foot_f", "role": "foot", "parent": "leg_f", "attach": "tip",
                      "size": 0.06, "girth": foot_girth, "symmetry": "left_right"})
    return {"robot_class": "quadruped", "name": "t", "parts": parts}


# ---------------------------------------------------------------- the critic check
def test_critic_flags_the_live_drumfoot_body():
    c = critique_gene(_drumfoot_gene())
    ext = [i for i in c["issues"] if i["check"] == "extremity_proportion"]
    assert len(ext) == 4, "all four drum feet must flag"
    assert all(i["severity"] == "high" for i in ext)       # high -> enters the LLM repair loop
    assert not c["ok"]
    # the detail TEACHES (names the part, the ratio, and a CONCRETE numeric fix) — repair context, not just a flag
    assert "girth" in ext[0]["detail"] and "radius" in ext[0]["detail"] and "0.0" in ext[0]["detail"]


def test_critic_passes_a_proportionate_body():
    from virturoid.services.morphology_composer import compose_robot
    q = compose_robot("a sturdy four-legged walking robot", ensure_walkable=True)
    c = critique_gene(q)
    assert not [i for i in c["issues"] if i["check"] == "extremity_proportion"]


def test_critic_check_is_positional_not_name_based():
    """A big HEAD on a long neck (top of the body) must NOT flag — only ground extremities do."""
    g = _drumfoot_gene()
    # sanity: the flagged parts are all feet (bottom), never the sensor head
    c = critique_gene(g)
    flagged = {i["detail"].split("'")[1] for i in c["issues"] if i["check"] == "extremity_proportion"}
    assert all("foot" in n for n in flagged)
    assert not any("head" in n or "antenna" in n for n in flagged)


# ---------------------------------------------------------------- the compiler fix
def test_authored_foot_dims_are_respected_not_multiplied():
    gene = build_from_anatomy(_graph(foot_girth=0.03))
    feet = [s for s in gene.segments if "foot_f" in s.name]
    assert feet and all(abs(s.radius_m - 0.03) < 1e-6 for s in feet), \
        f"authored girth 0.03 must realize at 0.03 (was 1.9x -> 0.057): {[s.radius_m for s in feet]}"


def test_auto_feet_keep_the_leg_calibrated_widening():
    """A leg WITHOUT an authored foot still gets its widened pad (the walkable-stance behavior, unchanged)."""
    bare = build_from_anatomy(_graph(foot_girth=None))
    authored = build_from_anatomy(_graph(foot_girth=0.03))
    # same leg spec in both graphs -> identical auto-realization for leg_r (no authored foot attached)
    r_bare = sorted(round(s.radius_m, 4) for s in bare.segments if "leg_r" in s.name)
    r_auth = sorted(round(s.radius_m, 4) for s in authored.segments if "leg_r" in s.name)
    assert r_bare == r_auth


def test_fixed_authored_feet_no_longer_flag():
    """End-to-end: the same drum-foot body with feet shrunk to a sane girth clears the critic check."""
    g = _drumfoot_gene()
    for s in g.segments:
        if "traction_foot" in s.name:
            s.radius_m = 0.04                              # the repair the teaching detail asks for
    c = critique_gene(g)
    assert not [i for i in c["issues"] if i["check"] == "extremity_proportion"]
