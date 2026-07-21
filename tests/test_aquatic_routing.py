"""Aquatic ROUTING vs aquatic ENVIRONMENT — one vocabulary was doing two jobs (MVP audit 2026-07-15).

Measured bug: an "octopus-like robot with eight tentacles" was not recognised as aquatic at all, so it received a
LAND-WALK verdict it can never pass (SLIDE, forward -0.00 m) on a showcased prompt. Two lists had drifted apart —
``aquatic._AQUATIC_WORDS`` (which gates the undulator RESHAPE) knew manta/salmon/koi but not octopus, while
``ai_native_tools._AQUATIC_WORDS`` (which gates the honest ENVELOPE annotation) knew octopus but not manta.

The naive fix (add "octopus" to the reshape list) would have been WORSE: that branch REPLACES the composed body
with a generic snake, discarding the octopus's authored radial morphology (33 segments, sphere mantle, 8 radial
limbs) — the silent template substitution the architecture forbids. So the two questions are now separate:
  * is_undulator_prompt -> compose a serial spine + compress it   (fish, eel, shark…)
  * is_aquatic_prompt   -> environment/verdict tier               (…plus octopus, squid, jellyfish)
and ai_native_tools imports the ONE canonical vocabulary so the lists cannot drift again.
"""
from __future__ import annotations

import pytest

from virturoid.services import aquatic as AQ


def test_cephalopods_are_aquatic_but_never_undulators():
    """The load-bearing distinction: aquatic for the VERDICT, radial for the MORPHOLOGY."""
    for p in ("an octopus-like robot with eight tentacles", "a squid robot", "a jellyfish robot",
              "an octopus robot that swims"):                    # 'swims' must NOT drag it into the reshape
        assert AQ.is_aquatic_prompt(p), p
        assert not AQ.is_undulator_prompt(p), p


def test_true_undulators_still_route_to_the_spine_reshape():
    for p in ("a swimming eel robot", "a robot fish", "a shark robot", "a lamprey robot"):
        assert AQ.is_aquatic_prompt(p) and AQ.is_undulator_prompt(p), p


def test_land_prompts_are_neither():
    for p in ("a four-legged walking robot", "a wheeled rover", "a robotic arm"):
        assert not AQ.is_aquatic_prompt(p) and not AQ.is_undulator_prompt(p), p


def test_one_canonical_vocabulary_no_second_copy():
    """ai_native_tools must IMPORT the vocabulary, not keep its own (that duplication caused the drift)."""
    from virturoid.services import ai_native_tools as AIT
    assert AIT._AQUATIC_WORDS is AQ.AQUATIC_ENV_WORDS
    for w in ("octopus", "squid", "jellyfish", "manta", "salmon", "koi", "lamprey"):
        assert w in AQ.AQUATIC_ENV_WORDS, w          # the union of what the two drifted lists each knew


def test_envelope_note_never_claims_an_unrun_swim():
    """Reaching the annotation means the body was NOT simulated in water, so the note must say LAND-BASED PROXY
    (it used to assert 'it was simulated in water (see swim_m)' — false on this branch)."""
    from virturoid.services.ai_native_tools import _flag_physics_envelope
    res = {"kind": "legged", "verdict": "SLIDE (feet barely lift / no real stepping)"}
    _flag_physics_envelope(res, "an octopus-like robot with eight tentacles", "legged")
    assert res["physics_envelope"] == "aquatic" and res["credible_walk"] is False
    note = res["envelope_note"].lower()
    assert "land-based proxy" in note and "simulated in water" not in note
    # a body ACTUALLY simulated in its medium keeps its verdict, unannotated
    swum = {"kind": "aquatic", "verdict": "SWIMS"}
    _flag_physics_envelope(swum, "a swimming eel robot", "aquatic")
    assert "physics_envelope" not in swum


@pytest.mark.slow
def test_octopus_keeps_its_radial_body_and_the_eel_still_becomes_a_spine():
    """End-to-end: the fix must not cost the octopus its authored morphology, nor change the eel."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.aquatic import _is_serial_spine
    oct_g = compose_robot("an octopus-like robot with eight tentacles", llm=None)
    assert len(oct_g.children_of(oct_g.root().name)) >= 6      # radial limbs survive (measured 8)
    assert not _is_serial_spine(oct_g) and len(oct_g.segments) >= 20
    eel = compose_robot("a swimming eel robot", llm=None)
    assert _is_serial_spine(eel) and eel.robot_class == "aquatic"
