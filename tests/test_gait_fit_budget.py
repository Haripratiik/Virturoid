"""Making the per-body gait fit AFFORDABLE must not make it OPTIONAL.

``create_robot`` grounds the customer's body and fits an operating point to it rather than substituting a
template — the correctness win of docs/breaking_the_cotuning_wall.md — and that costs real seconds. MEASURED
2026-08-05 on this checkout, per call: grounded authored hexapod 24.2 s / 3 evals, cat 56.0 s / 87 evals,
dog 124.7 s / 360 evals (the whole budget spent establishing an honest "no open-loop walk for this body").
Through ``create_robot`` end to end, one "a quadruped robot dog" is 143.6 s.

The suite calls ``create_robot`` about fifty times and ``tests/test_ai_native.py`` alone rebuilds that same dog
NINE times, so two switches exist — both DEFAULT OFF, so a product run is unchanged:

    VIRTUROID_GAIT_FIT_CACHE=1   memoize on the body's STRUCTURAL key (143.6 s -> 3.4 s on the repeat)
    VIRTUROID_SKIP_GAIT_FIT=1    the fit becomes a DISCLOSED no-op

This file is the guard rail on both. The cache must be indistinguishable from a real fit, the skip must be
impossible to mistake for a finding, and the tests whose SUBJECT is the fit must never be allowed to buy the
speed-up.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from virturoid.services import gait_flywheel as GF

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_TESTS = Path(__file__).resolve().parent


# ------------------------------------------------------------------ the skip must announce itself
def test_a_skipped_fit_says_it_was_skipped_and_never_reads_as_a_finding(monkeypatch):
    """``searched: False, adopted: False`` off a REAL fit means "measured, and it kept its default". Off a
    skipped one it means "never measured". Those two must not be the same dict — this module's whole reason for
    reporting ``ok=False`` on a crash is that a non-answer shaped like an answer is the defect."""
    monkeypatch.setenv("VIRTUROID_SKIP_GAIT_FIT", "1")

    class _G:
        def __init__(self):
            self.metadata = {}
            self.id = "skip-gene"

    g = _G()
    out = GF.fit_gait_for_body(g)
    assert out["skipped"] is True and out["searched"] is False and out["adopted"] is False
    assert "VIRTUROID_SKIP_GAIT_FIT" in out["reason"], out["reason"]
    assert "never searched" in out["reason"], out["reason"]
    # and the disclosure is stashed on the body, so a downstream reader sees it too
    assert g.metadata["gait_fit"]["skipped"] is True
    # ...and no operating point was invented for a body nothing measured
    assert "gait_params" not in g.metadata


def test_the_skip_is_off_by_default(monkeypatch):
    """A product run must never silently stop fitting. Absent the variable the stub gene reaches the real
    fitter, which reports a genuine crash (no segments) rather than a skip."""
    monkeypatch.delenv("VIRTUROID_SKIP_GAIT_FIT", raising=False)

    class _G:
        def __init__(self):
            self.metadata = {}
            self.id = "nodefault-gene"

    out = GF.fit_gait_for_body(_G())
    assert out.get("skipped") is not True, out


# ------------------------------------------------------------------ the cache must be invisible
@pytest.mark.skipif(not _MUJOCO, reason="fitting a real body needs MuJoCo")
def test_a_cached_fit_returns_the_same_answer_AND_the_same_body_state(monkeypatch):
    """A cache that returned the disclosure dict but not the adopted operating point would hand back a robot
    with no controller — passing every assertion about the dict and shipping a different robot."""
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morphology_composer import compose_robot

    monkeypatch.setenv("VIRTUROID_GAIT_FIT_CACHE", "1")
    monkeypatch.delenv("VIRTUROID_SKIP_GAIT_FIT", raising=False)
    GF.clear_fit_cache()

    def _body():
        g = compose_robot("a six-legged hexapod walker", llm=None)
        ground_and_repair(g)
        return g

    a, b = _body(), _body()
    # A TINY BUDGET ON PURPOSE. The subject here is the cache's identity contract, not the quality of the
    # search -- the real search is exercised at full budget by every unmarked test that calls create_robot.
    # At the shipped budget these two fits alone cost ~48 s to prove a bookkeeping property.
    kw = dict(cache=True, deploy_steps=600, search_steps=600, max_evals=2, warm_evals=1, seed_restarts=1,
              bank=False)
    first = GF.fit_gait_for_body(a, **kw)
    assert len(GF._FIT_CACHE) == 1, "the first fit must be remembered"
    second = GF.fit_gait_for_body(b, **kw)

    assert second["adopted"] == first["adopted"]
    assert second.get("params") == first.get("params")

    # THE REASON IS COMPARED ON ITS SUBSTANCE, AND THE REPLAY MARKER IS ASSERTED SEPARATELY. This read
    # ``second["reason"] == first["reason"]`` until 2026-08-13, when the replay started saying so: a memoized
    # fit was returning ``{searched: True, n_evals: 6}`` for a call that ran ZERO rollouts, which is a claim
    # about work done and the same defect as every other number this repo corrected that day. Verbatim equality
    # cannot express "same answer, different provenance", so it is split into the two things it was standing in
    # for -- and that is strictly MORE constraint, not less: the substance must match exactly, the first call
    # must NOT be labelled a replay, the second MUST be, and both surfaces must agree.
    strip = lambda s: str(s or "").split("] ", 1)[-1] if str(s or "").startswith(GF._REPLAY_TAG) else str(s or "")
    assert strip(second["reason"]) == strip(first["reason"]), "a replay must not change the ANSWER"
    assert not first.get("replayed_from_cache"), "the first call really did search; labelling it a replay would "\
                                                 "make the flag meaningless"
    assert second.get("replayed_from_cache") is True, second
    assert second.get("evals_spent_by_this_call") == 0, second
    assert second["reason"].startswith(GF._REPLAY_TAG), second["reason"]

    # the BODY, not just the dict: the adopted operating point and the disclosure both land on the twin
    assert (b.metadata or {}).get("gait_params") == (a.metadata or {}).get("gait_params")
    # ...and the gene's own copy must tell the SAME story as the returned dict. The first version of the replay
    # marked only the dict, so one fit read two ways depending where you looked -- the four-surfaces-disagree
    # defect (#215/#218) reintroduced in miniature. This assertion is what caught it.
    _md_fit = (b.metadata or {}).get("gait_fit", {})
    assert strip(_md_fit.get("reason")) == strip(first["reason"])
    assert _md_fit.get("replayed_from_cache") is True, _md_fit
    assert _md_fit.get("reason", "").startswith(GF._REPLAY_TAG), _md_fit.get("reason")


@pytest.mark.skipif(not _MUJOCO, reason="fitting a real body needs MuJoCo")
def test_the_cache_does_nothing_unless_it_is_asked_twice_over(monkeypatch):
    """Two independent gates: the caller opts in (``cache=True``) AND the environment enables it. The build path
    opts in permanently, so the environment variable is what keeps a PRODUCT run searching every time."""
    from virturoid.services.gene_build import ground_and_repair
    from virturoid.services.morphology_composer import compose_robot

    g = compose_robot("a six-legged hexapod walker", llm=None)
    ground_and_repair(g)

    kw = dict(deploy_steps=600, search_steps=600, max_evals=2, warm_evals=1, seed_restarts=1, bank=False)

    monkeypatch.setenv("VIRTUROID_GAIT_FIT_CACHE", "1")
    GF.clear_fit_cache()
    GF.fit_gait_for_body(g, **kw)                            # caller did NOT opt in
    assert GF._FIT_CACHE == {}, "cache=False must never populate the cache"

    monkeypatch.delenv("VIRTUROID_GAIT_FIT_CACHE", raising=False)
    GF.fit_gait_for_body(g, cache=True, **kw)                # environment does not enable it
    assert GF._FIT_CACHE == {}, "the product default must not memoize"


def test_a_stub_body_is_never_cached(monkeypatch):
    """``structural_gait_key`` answers for a gene with NO SEGMENTS — every stub in
    tests/test_gait_fit_robustness.py hashes to the same value — so keying on it alone would let one stubbed
    fit answer for another. It did: the first version of this cache handed ``test_restarts_are_bounded``
    another test's adopted walk. A body with no segments is never cached."""
    monkeypatch.setenv("VIRTUROID_GAIT_FIT_CACHE", "1")
    GF.clear_fit_cache()

    class _G:
        def __init__(self):
            self.metadata = {}
            self.id = "stub"
            self.robot_class = "quadruped"

    assert GF.structural_gait_key(_G()), "precondition: a segment-less gene still hashes"
    assert GF._fit_cache_key(_G(), {}) is None, "...and must still be refused a cache slot"
    GF.fit_gait_for_body(_G(), cache=True)
    assert GF._FIT_CACHE == {}


def test_a_crashed_fit_is_not_cached(monkeypatch):
    """``ok=False`` is not an answer about the body. Memoizing it would turn one transient failure into a
    permanent one for every structurally identical body in the process."""
    from virturoid.services import gait_search as GS

    monkeypatch.setenv("VIRTUROID_GAIT_FIT_CACHE", "1")
    monkeypatch.delenv("VIRTUROID_SKIP_GAIT_FIT", raising=False)
    GF.clear_fit_cache()

    class _Seg:
        name = "s"
        shape = "capsule"
        length_m = radius_m = mass_kg = 0.1
        joint_type = "revolute"
        joint_axis = (0.0, 0.0, 1.0)
        joint_lower = joint_upper = 0.0
        mount_euler = (0.0, 0.0, 0.0)
        parent = None

    class _G:
        def __init__(self):
            self.metadata = {}
            self.id = "boom"
            self.segments = [_Seg()]

    def _boom(*a, **k):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr(GS, "evaluate_gait", _boom)
    out = GF.fit_gait_for_body(_G(), cache=True)
    assert out["ok"] is False, out
    assert GF._FIT_CACHE == {}, "a crash must not be remembered as this body's answer"


# ------------------------------------------------------------------ who is NOT allowed to buy the speed-up
#: Files whose SUBJECT is the grounded-and-fitted body: they must pay for the real search every run. A fast
#: suite that stopped testing the thing it was made fast around is worse than a slow one, and the cheapest way
#: to lose that is for someone to add ``no_gait_fit`` here while chasing a number.
_MUST_RUN_THE_REAL_FIT = (
    "test_gait_tuning.py",                  # the per-body tuning IS the subject
    "test_gait_fit_robustness.py",          # the fitter's own contract
    "test_gait_flywheel.py",                # recall / bank / adoption gate
    "test_gait_search.py",                  # the search underneath it
    "test_held_body_is_buildable_body.py",  # grounding + fit is what the file exists to prove
    "test_same_robot_throughout.py",        # the identity of the body create_robot hands back
    "test_quadruped_differentiation.py",    # different prompts -> different bodies that STILL walk
    "test_phase2_majors.py",                # create_robot and autobuild agree on a CREDIBLE WALK
    "test_walk_settling.py",                # the horizon the fit is judged at
    "test_walkable_default.py",             # the shipped example walks out of the box
)


@pytest.mark.parametrize("name", _MUST_RUN_THE_REAL_FIT)
def test_the_tests_that_are_about_the_gait_still_pay_for_it(name):
    path = _TESTS / name
    if not path.is_file():
        pytest.skip(f"{name} not present on this checkout")
    src = path.read_text(encoding="utf-8", errors="replace")
    assert "no_gait_fit" not in src, (
        f"{name} is about the per-body gait fit; it must not opt out of running it. "
        "If this file genuinely stopped depending on locomotion, remove it from _MUST_RUN_THE_REAL_FIT "
        "in the same change, so the decision is visible in review."
    )
    assert "VIRTUROID_SKIP_GAIT_FIT" not in src, f"{name} must not disable the fit by hand either"


@pytest.mark.no_gait_fit
def test_the_marker_actually_reaches_the_environment():
    """A marker nothing acts on is decoration. This is the end-to-end check of the conftest wiring."""
    assert os.environ.get("VIRTUROID_SKIP_GAIT_FIT") == "1"


def test_an_unmarked_test_gets_the_real_fitter():
    """...and the marker is scoped to the test that carries it. The pattern this replaces -- module-level
    ``os.environ.setdefault`` -- would have leaked the line above to every test in the process, because pytest
    imports every module before running anything."""
    assert os.environ.get("VIRTUROID_SKIP_GAIT_FIT") is None


def test_the_marker_is_declared_so_a_typo_cannot_silently_do_nothing():
    """``no_gait_fit`` is applied per FILE. A misspelled marker is not an error by default, and it would fail
    open in the safe direction (a slow test) — but ``--strict-markers`` users deserve the registration, and the
    declaration is where the rule about what may carry it is written down."""
    cfg = (_TESTS.parent / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    conf = (_TESTS / "conftest.py").read_text(encoding="utf-8", errors="replace")
    assert "no_gait_fit" in cfg or "no_gait_fit" in conf
    assert "VIRTUROID_GAIT_FIT_CACHE" in conf, "the suite must turn the cache on, or nothing gets faster"


def test_the_suite_really_has_the_cache_on():
    """The saving is worth nothing if conftest stops setting it, and reading the file is not the same as
    observing the process. ``setdefault`` means a developer can still run the whole suite the slow way with
    ``VIRTUROID_GAIT_FIT_CACHE=0``, which is a legitimate thing to want before trusting a timing claim — that
    case is honoured, everything else is a regression."""
    val = os.environ.get("VIRTUROID_GAIT_FIT_CACHE")
    if val is not None and val != "1":
        pytest.skip(f"cache deliberately disabled for this run (VIRTUROID_GAIT_FIT_CACHE={val!r})")
    assert val == "1", "tests/conftest.py must enable the gait-fit cache for the suite"
