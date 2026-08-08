"""#285 — the composer must READ the proportion words, and reading them must CHANGE THE STRUCTURE.

MEASURED 2026-08-08, before this landed: the only phrase ``compose_robot`` read off a functional walker prompt
was the SIZE word. "long slender legs", "short thick legs", "a wide stance", "a long body", "a slender body"
and leg counts of 3 or 5 all composed a BYTE-IDENTICAL robot, so eighteen hand-written corpus-factory prompts
collapsed to three bodies and the honest distinct-body ceiling of the whole offline composer was about ten.
A 324-prompt grid (leg count x size x proportion) produced 20 distinct structures before and 192 after.

These assertions are about PROPERTIES, not pinned constants:
  * a neutral prompt is BYTE-IDENTICAL to the pinned baseline (adding an axis may not re-tune every robot);
  * each axis moves the structure in the DIRECTION asked for, and in the right dimension;
  * the axes COMPOSE with the species prior rather than replacing it;
  * the leg count the customer typed is the leg count they get.
You cannot satisfy them by returning one template, and you cannot satisfy them by scaling everything at once.
"""
from __future__ import annotations

import importlib.util

import pytest

from virturoid.services.morphology_composer import _leg_count, compose_from_spec, morphology_from_requirements
from virturoid.services.morphology_priors import PROPORTION_UNIT, body_proportions

_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _spec(prompt: str) -> dict:
    return morphology_from_requirements(0.65, 0.25, prompt=prompt, robot_class="quadruped")


def _legs(spec: dict) -> list[dict]:
    return [limb for limb in spec["limbs"] if limb["prefix"].startswith("leg")]


def _stride_len(spec: dict) -> float:
    """Thigh + calf of the first leg — the gait-determining lever."""
    links = _legs(spec)[0]["links"]
    return round(float(links[1]["length"]) + float(links[2]["length"]), 6)


def _stride_radius(spec: dict) -> float:
    return round(float(_legs(spec)[0]["links"][1]["radius"]), 6)


def _stance(spec: dict) -> float:
    return max(abs(float(limb["mount_offset"][1])) for limb in _legs(spec))


def _station_span(spec: dict) -> float:
    xs = [float(limb["mount_offset"][0]) for limb in _legs(spec)]
    return round(max(xs) - min(xs), 6)


# ---------------------------------------------------------------- the parser
def test_a_neutral_prompt_reads_as_no_proportion_ask():
    """GUARD: prompts that describe no proportion must return all-1.0, so the axis cannot silently re-tune
    every robot the locomotion suite pins."""
    for neutral in ("a four-legged walking robot", "a quadruped robot", "a robot dog",
                    "a small four-legged walking robot", "a cheetah robot that runs fast"):
        assert body_proportions(neutral) == PROPORTION_UNIT, neutral


def test_an_adjective_only_counts_next_to_the_noun_it_modifies():
    """A bare adjective must NOT fire: "a compact 3-joint pick arm" is not a request for a short torso, and
    "a small four-legged robot" must not double-count the size word as a body-length change."""
    for unrelated in ("a compact 3-joint pick arm", "a long slender inspection arm",
                      "a small four-legged walking robot", "a wide six-wheeled hauler"):
        assert body_proportions(unrelated) == PROPORTION_UNIT, unrelated


@pytest.mark.parametrize("prompt,axis,direction", [
    ("a four-legged walking robot with long legs", "leg", +1),
    ("a four-legged walking robot with short legs", "leg", -1),
    ("a long-legged walking robot", "leg", +1),
    ("a four-legged walking robot with slender legs", "thick", -1),
    ("a four-legged walking robot with thick legs", "thick", +1),
    ("a four-legged walking robot with a wide stance", "stance", +1),
    ("a four-legged walking robot with a narrow stance", "stance", -1),
    ("a four-legged walking robot with a long body", "torso", +1),
    ("a four-legged walking robot with a compact body", "torso", -1),
    ("a four-legged walking robot with a slender body", "girth", -1),
    ("a four-legged walking robot with a broad body", "girth", +1),
])
def test_each_axis_is_read_in_the_direction_asked_for(prompt, axis, direction):
    got = body_proportions(prompt)[axis]
    assert (got > 1.0) if direction > 0 else (got < 1.0), f"{prompt!r} -> {axis}={got}"


def test_two_adjectives_on_one_noun_both_land():
    """"long slender legs" is a request on TWO axes at once, and reading only the nearer word is what made
    the whole phrase inert."""
    r = body_proportions("a four-legged walking robot with long slender legs")
    assert r["leg"] > 1.0 and r["thick"] < 1.0, r


# ---------------------------------------------------------------- the composer honours them
def test_a_neutral_prompt_composes_the_byte_identical_baseline():
    """The pinned gait-tuned body must not move because an axis was added."""
    baseline = _spec("a four-legged walking robot")
    assert _stride_len(baseline) == pytest.approx(0.24)
    assert baseline["base"]["cross_section"][0] == pytest.approx(baseline["base"]["cross_section"][1])


def test_leg_length_and_thickness_are_separate_structural_axes():
    base = _spec("a four-legged walking robot")
    longer = _spec("a four-legged walking robot with long legs")
    thinner = _spec("a four-legged walking robot with slender legs")
    assert _stride_len(longer) > _stride_len(base), "'long legs' must lengthen the stride lever"
    assert _stride_len(thinner) == pytest.approx(_stride_len(base)), "thickness must not change LENGTH"
    assert _stride_radius(thinner) < _stride_radius(base), "'slender legs' must thin the limb"
    assert _stride_radius(longer) == pytest.approx(_stride_radius(base)), "length must not change DIAMETER"


def _splay(spec: dict) -> float:
    import math
    return max(abs(float(limb["mount_euler"][0]) - math.pi) for limb in _legs(spec))


def test_a_wide_stance_is_abduction_not_a_hip_hanging_in_the_air():
    """The hip must stay ON the trunk; the extra width comes from ROLLING the leg out, which is the hip
    abduction the crawl gait is built around. Moving the mount outboard instead detaches the leg."""
    base, wide = _spec("a four-legged walking robot"), _spec("a four-legged walking robot with a wide stance")
    shell_y = max(abs(float(sec[1])) for sec in wide["base"]["geometry"]["sections"])
    assert _stance(wide) <= shell_y + 1e-9, "a wide stance must not root the hip off the body"
    assert _splay(wide) > _splay(base) == pytest.approx(0.0), "the extra width must come from a splay"
    assert _stride_len(wide) == pytest.approx(_stride_len(base)), "stance must not change leg LENGTH"


def test_a_narrow_stance_pulls_the_legs_in():
    base = _spec("a four-legged walking robot")
    narrow = _spec("a four-legged walking robot with a narrow stance")
    assert _stance(narrow) < _stance(base)
    assert _splay(narrow) == pytest.approx(0.0), "narrowing needs no abduction"


def test_body_length_moves_the_leg_stations_AND_the_trunk_that_carries_them():
    """A "long body" whose shell stayed the baseline length would render with its legs hanging off the nose."""
    base, long_b = _spec("a four-legged walking robot"), _spec("a four-legged walking robot with a long body")
    assert _station_span(long_b) > _station_span(base), "leg stations must spread along a longer body"
    hx_base, hx_long = base["base"]["cross_section"][0], long_b["base"]["cross_section"][0]
    assert hx_long > hx_base, "the trunk itself must lengthen with its legs"
    assert long_b["base"]["cross_section"][1] == pytest.approx(base["base"]["cross_section"][1]), \
        "body LENGTH must not silently change body WIDTH"


def test_body_width_is_its_own_axis():
    base, slim = _spec("a four-legged walking robot"), _spec("a four-legged walking robot with a slender body")
    assert slim["base"]["cross_section"][1] < base["base"]["cross_section"][1]
    assert slim["base"]["cross_section"][0] == pytest.approx(base["base"]["cross_section"][0])


def test_the_muzzle_is_outside_the_trunk_it_mounts_on():
    """The head used to sit at x=0.12 inside a shell reaching 0.21 — a head in the model and none in the
    picture. It must clear the trunk's own forward extent."""
    spec = _spec("a four-legged walking robot")
    head = next(limb for limb in spec["limbs"] if limb["prefix"] == "head")
    nose = float(head["mount_offset"][0]) + float(head["links"][0]["length"])
    shell = max(abs(float(sec[2])) for sec in spec["base"]["geometry"]["sections"])
    assert nose > shell, f"the muzzle tip ({nose:.3f} m) is inside the trunk shell ({shell:.3f} m)"


# ---------------------------------------------------------------- leg count
@pytest.mark.parametrize("prompt,n", [
    ("a three-legged walking robot", 3), ("a five-legged walking robot", 5),
    ("a seven-legged walking robot", 7), ("a four-legged walking robot", 4),
    ("a six-legged walking robot", 6), ("a 10-legged walking robot", 10),
])
def test_the_leg_count_asked_for_is_the_leg_count_built(prompt, n):
    """3 and 5 used to round to the nearest PAIR (3 -> 2 -> 4, 5 -> 4) and compose a body byte-identical to
    a plain quadruped, so two whole cells of the corpus grid were inert."""
    assert _leg_count(prompt) == n
    assert len(_legs(_spec(prompt))) == n


def test_an_odd_leg_sits_on_the_centreline_not_in_a_phantom_pair():
    legs = _legs(_spec("a three-legged walking robot"))
    ys = sorted(round(float(limb["mount_offset"][1]), 4) for limb in legs)
    assert ys.count(0.0) == 1, f"exactly one centreline leg expected, got {ys}"
    assert ys[0] == pytest.approx(-ys[-1]), "the remaining legs must still be a mirrored pair"


# ---------------------------------------------------------------- composition with the species prior
def test_the_ask_multiplies_the_species_prior_rather_than_replacing_it():
    """"a horse-like quadruped with long legs" must be longer-legged than a horse, not merely long-legged."""
    horse = _spec("a horse-like quadruped robot")
    horse_long = _spec("a horse-like quadruped robot with long legs")
    plain_long = _spec("a four-legged walking robot with long legs")
    assert _stride_len(horse_long) > _stride_len(horse) > _stride_len(_spec("a four-legged walking robot"))
    assert _stride_len(horse_long) > _stride_len(plain_long)


# ---------------------------------------------------------------- it still builds
@pytest.mark.skipif(not _MUJOCO, reason="building + validating a gene needs MuJoCo")
@pytest.mark.parametrize("prompt", [
    "a four-legged walking robot with long slender legs",
    "a four-legged walking robot with short thick legs",
    "a four-legged walking robot with a wide stance",
    "a four-legged walking robot with a long body",
    "a three-legged walking robot",
    "a five-legged walking robot",
])
def test_every_axis_still_produces_a_buildable_connected_body(prompt):
    from virturoid.services.morphology_composer import _morphology_quality_issues
    gene = compose_from_spec(_spec(prompt))
    assert not gene.validate(), gene.validate()
    assert not _morphology_quality_issues(gene), _morphology_quality_issues(gene)
