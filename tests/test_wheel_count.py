"""Wheel count comes from the PROMPT, and the layout stays physically honest (MVP audit MAJOR).

Measured before: "a two-wheeled delivery robot" built the identical 4-wheel rover as "a four-wheeled rover" --
the drivetrain ignored the request, which any reader of the prompt spots immediately.

The fix is not merely cosmetic. A differential-drive body with only two ground contacts tips over, so a
2-wheeled base also gets a passive CASTER (as every real diff-drive robot does); otherwise honouring the number
would have traded one wrong answer for a robot that cannot drive.
"""
from __future__ import annotations

import pytest

from virturoid.services.morphology_composer import _wheel_count_from_prompt, compose_robot


def test_prompt_wheel_count_is_parsed_but_only_next_to_a_wheel_noun():
    assert _wheel_count_from_prompt("a two-wheeled delivery robot") == 2
    assert _wheel_count_from_prompt("a four-wheeled flat-deck rover") == 4
    assert _wheel_count_from_prompt("a wide six-wheeled hauler") == 6
    assert _wheel_count_from_prompt("a 4 wheel cart") == 4
    # the number must qualify a WHEEL, or unrelated counts would resize the drivetrain
    assert _wheel_count_from_prompt("a robot that carries two boxes") is None
    assert _wheel_count_from_prompt("a wheeled rover") is None          # unspecified -> builder default


@pytest.mark.parametrize("prompt,driven", [("a two-wheeled delivery robot", 2),
                                           ("a four-wheeled flat-deck rover", 4),
                                           ("a wide six-wheeled hauler", 6)])
def test_built_body_has_the_requested_driven_wheels(prompt, driven):
    g = compose_robot(prompt, llm=None)
    names = [s.name for s in g.segments]
    assert sum(1 for n in names if n.startswith("wheel_")) == driven
    assert ("caster" in names) is (driven == 2)             # caster ONLY where it is needed for stability


@pytest.mark.slow
def test_a_two_wheeled_base_actually_drives():
    """The honest bar: honouring the count must not cost the robot its DRIVE verdict."""
    from virturoid.services.ai_native_tools import _honest_drive
    out = _honest_drive(compose_robot("a two-wheeled delivery robot", llm=None), steps=700)
    assert str(out.get("verdict", "")).startswith("DRIVES"), out.get("verdict")
