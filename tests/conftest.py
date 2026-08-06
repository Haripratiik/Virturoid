"""Test session setup.

Keep the suite hermetic: never read the developer's project-local ``.env`` during
tests, so a configured OpenAI/Claude/local backend can't leak into the offline,
deterministic test expectations. Set before any virturoid import resolves a backend.
"""

import os

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
# Production builds are LLM-first and fail closed when no design model is
# configured.  Tests exercise the explicitly supported offline compatibility
# lane instead of silently depending on that production fallback.
os.environ.setdefault("VIRTUROID_ALLOW_HEURISTIC_FALLBACK", "1")

# ---------------------------------------------------------------- affording the per-body gait fit
#
# ``create_robot`` grounds the body and then fits an operating point TO IT, which is the correctness win that
# stopped the product throwing the customer's design away -- and it is expensive on purpose. MEASURED
# 2026-08-05 on this checkout, per call: hexapod 24 s, cat 56 s, grounded authored dog 125 s (360 evaluations
# spent establishing an honest "this body has no open-loop walk"). The suite calls ``create_robot`` ~50 times
# and rebuilds the SAME "a quadruped robot dog" nine times inside ``test_ai_native.py`` alone.
#
# So the suite memoizes the fit on the body's STRUCTURAL key for the life of the process. This is not a
# weakening: every structurally distinct body still runs the real bounded CEM search over 6000-step rollouts,
# including the expensive negative; only the Nth identical rebuild is free, and it is handed back exactly the
# operating point and the disclosure the first one earned. Measured end-to-end through ``create_robot``:
# 144.3 s -> 3.4 s on the repeat, same adopted parameters, same reason string.
os.environ.setdefault("VIRTUROID_GAIT_FIT_CACHE", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_gait_fit: this file's subject is NOT locomotion -- skip the per-body gait fit inside create_robot. "
        "Never put this on a test that reads a walk verdict, a gait parameter, or the identity of the body "
        "create_robot hands back (without a fitted controller a quadruped is substituted for a template).",
    )


@pytest.fixture(autouse=True)
def _gait_fit_budget(request, monkeypatch):
    """Turn the gait fit into a DISCLOSED no-op for files marked ``no_gait_fit``.

    OPT-IN, not opt-out, and set through ``monkeypatch`` rather than at module import. Both choices are load
    bearing:

    * Opt-in means a NEW test gets the real fit by default. The failure mode of the other direction is a test
      that silently stops exercising the thing it was written to protect, which is strictly worse than a slow
      suite.
    * Module-level ``os.environ.setdefault`` -- the pattern ``VIRTUROID_DISABLE_GAIT_HINTS`` uses in eleven
      files -- LEAKS, because pytest imports every test module into one process before running anything. A
      variable that disables a load-bearing behaviour must not be settable by import order, so this one is
      scoped to the marked test and unset again afterwards.

    A consequence worth stating: exporting ``VIRTUROID_SKIP_GAIT_FIT=1`` in a shell will NOT make this suite
    faster, because the ``delenv`` below takes it back for every unmarked test. That is the intent. A whole
    suite silently running without the fit is the outcome this fixture exists to make impossible; the way to
    make the suite fast is the cache above, which costs no coverage at all.
    """
    if request.node.get_closest_marker("no_gait_fit"):
        monkeypatch.setenv("VIRTUROID_SKIP_GAIT_FIT", "1")
    else:
        monkeypatch.delenv("VIRTUROID_SKIP_GAIT_FIT", raising=False)
