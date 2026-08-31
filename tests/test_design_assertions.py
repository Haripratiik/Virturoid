"""A designer must be able to state what it MEANT, and be checked against it.

`probe_robot` answers questions -- but only the ones someone thinks to ask. The other half of the Articraft
harness is the agent declaring intent ("the foot should touch the ground", "these two are supposed to overlap,
that is a joint housing") so the checker can find the gap between intent and geometry.

That is aimed at one specific failure mode Articraft reports: "poor global shape quality despite passing local
structural checks". Every part can be individually fine while the assembly means nothing, and an assertion is
the only artefact that says what the assembly was FOR.

The tests that matter here are the NEGATIVE ones: an assertion framework that cannot fail is decoration.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("VIRTUROID_DISABLE_GAIT_HINTS", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
pytestmark = pytest.mark.skipif(not _MUJOCO, reason="checking assertions needs MuJoCo")


@pytest.fixture(scope="module")
def dog():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four legged robot dog", llm=None)


def _foot(gene):
    return next(s.name for s in gene.segments if s.name.endswith("_3"))


def test_true_claims_about_a_real_body_pass(dog):
    from virturoid.services.design_assertions import check
    r = check(dog, [{"kind": "expect_contact", "a": _foot(dog), "b": "floor"},
                    {"kind": "expect_above", "a": "head", "b": "torso"},
                    {"kind": "expect_clearance", "a": "torso", "min_m": 0.10}])
    assert r["ok"] and r["failed"] == 0, r["results"]


def test_a_false_claim_FAILS_and_says_by_how_much(dog):
    """The load-bearing test. A framework that cannot fail is decoration, and a failure that does not carry the
    measurement cannot be acted on."""
    from virturoid.services.design_assertions import check
    r = check(dog, [{"kind": "expect_within", "a": "head", "b": "torso", "max_m": 0.05}])
    assert r["failed"] == 1
    row = r["results"][0]
    assert row["passed"] is False
    assert "apart" in row["detail"] and "limit" in row["detail"], row


def test_an_assertion_about_a_part_that_does_not_exist_fails_by_name(dog):
    """A rename that leaves the intent behind is exactly the drift worth surfacing -- so this is a failure that
    names the part, not a crash and not a silent skip."""
    from virturoid.services.design_assertions import check
    r = check(dog, [{"kind": "expect_contact", "a": "nonexistent_part", "b": "floor"}])
    assert r["failed"] == 1 and "nonexistent_part" in r["results"][0]["detail"]


def test_a_malformed_assertion_teaches_instead_of_raising():
    from virturoid.services.design_assertions import validate
    errs = validate([{"kind": "expect_within", "a": "x"}])
    assert any("'b'" in e for e in errs) and any("max_m" in e for e in errs), errs
    assert validate([{"kind": "nope", "a": "x"}])[0].startswith("assertion 0: unknown kind")


def test_allow_forms_record_a_reason_rather_than_loosening_the_check(dog):
    """An `allow_` dismisses ONE false positive, in writing, with the reason kept -- instead of relaxing a
    threshold for every design that follows."""
    from virturoid.services.design_assertions import check
    r = check(dog, [{"kind": "allow_overlap", "a": "torso", "b": "neck", "reason": "shoulder housing"}])
    assert r["failed"] == 0
    assert "shoulder housing" in r["results"][0]["detail"]
    assert r["allowed_overlaps"] == [["torso", "neck"]]


def test_assertions_stored_on_the_gene_are_re_checked_without_being_passed_again(dog):
    """Intent has to survive into an AMEND: a later change that quietly breaks what the design was for should be
    caught by the design's own words, with nobody re-supplying them."""
    import copy

    from virturoid.services.design_assertions import check
    g = copy.deepcopy(dog)
    g.metadata = {**(g.metadata or {}), "assertions": [
        {"kind": "expect_contact", "a": _foot(g), "b": "floor"}]}
    r = check(g)
    assert r["ok"] and r["n"] == 1 and r["failed"] == 0, r
