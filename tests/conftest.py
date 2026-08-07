"""Test session setup.

Keep the suite hermetic: never read the developer's project-local ``.env`` during
tests, so a configured OpenAI/Claude/local backend can't leak into the offline,
deterministic test expectations. Set before any virturoid import resolves a backend.
"""

import atexit
import os
import shutil
import tempfile

import pytest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

# ---------------------------------------------------------------- the suite is not a writer to the real bank
#
# MEASURED 2026-08-07: running pytest grew ``build/memory/virturoid_memory.db`` from 97 to 101 locomotion
# rows, and four rows in that bank carry the body class ``totally_made_up_xyz`` -- a fixture name from
# ``tests/test_structural_dispatch.py``. Nothing was misbehaving: the suite calls ``verify_robot``, which
# calls ``_auto_bank_gait``, which banks. The path was simply a constant, so every run banked into the
# developer's own memory.
#
# The reason that matters more than tidiness: THE BANK IS THE THING WE MEASURE. The evidence gates, the
# fragility re-measurement and the hint-mining runs all read it, so a suite that writes to it is a suite
# editing its own evidence -- and fixture rows were being counted as observations in an analysis of whether
# the flywheel has any signal at all. It is also the shared state that makes two concurrent suites corrupt
# each other.
#
# Set here, at conftest import, because pytest imports conftest before any test module and therefore before
# any virturoid module binds ``DEFAULT_DB_PATH`` / ``DEFAULT_MEMORY_DIR``. An autouse fixture would be too
# late for those import-time bindings.
#
# The session bank starts EMPTY rather than as a copy of the developer's. A test that needs banked rows
# should bank them itself: depending on whatever a particular machine has accumulated is not reproducible,
# and a green suite that silently requires 101 local rows is not evidence of anything.
_SESSION_MEMORY_DIR = tempfile.mkdtemp(prefix="virturoid-test-memory-")
os.environ.setdefault("VIRTUROID_MEMORY_DIR", _SESSION_MEMORY_DIR)


@atexit.register
def _drop_session_memory_dir() -> None:
    # Only ever the directory this process created, and only if it is still the one in force.
    if os.environ.get("VIRTUROID_MEMORY_DIR") == _SESSION_MEMORY_DIR:
        shutil.rmtree(_SESSION_MEMORY_DIR, ignore_errors=True)
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

# ---------------------------------------------------------------- affording REAL robots instead of fixtures
#
# The same argument, one layer up. Fixtures have lied in this repo, so the import tests are pointed at the real
# MuJoCo Menagerie -- and a real robot description is expensive to parse: MEASURED per ``import_robot`` call,
# Unitree G1 13.88 s, Go2 11.29 s, Talos 7.45 s, UR5e 3.49 s. The suite names ~115 Menagerie model references
# across nine files (go2.xml 26 times, g1.xml 13, ur5e.xml 11), each one re-parsing a file that has not changed.
#
# Memoizing on the SOURCE's identity -- content digest of the model file, plus a fingerprint of the directory
# its meshes and <include>s live in, plus the options that alter the result -- makes the repeats 2-4 ms and
# costs no coverage: every distinct model still runs the real importer. Measured saving on the eight most-cited
# files alone: 563 s. Default-off in the product (see robot_import._IMPORT_CACHE), on here.
os.environ.setdefault("VIRTUROID_IMPORT_CACHE", "1")


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
