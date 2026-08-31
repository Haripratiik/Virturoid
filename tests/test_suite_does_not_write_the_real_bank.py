"""The suite must not bank into the developer's own memory.

This is a ratchet, not a feature test. It exists because the defect it guards was invisible for
months: ``verify_robot`` -> ``_auto_bank_gait`` -> ``bank_gait`` is the ORDINARY product path, so
every test that verified a robot banked a gait, and ``DEFAULT_DB_PATH`` was a constant pointing at
``build/memory/virturoid_memory.db``. Nothing looked wrong at any single call site.

MEASURED 2026-08-07 before the fix: one session took the live bank from 97 to 101 locomotion rows,
and four rows in it carry the body class ``totally_made_up_xyz`` -- a fixture name from
``tests/test_structural_dispatch.py``.

Why this is worth a standing test rather than a comment: the bank is the substrate we MEASURE
against. The evidence gates, the fragility re-measurement and the hint-mining runs all read it. A
suite that writes to it is a suite editing its own evidence, and fixture rows were being counted as
observations in an analysis of whether the flywheel has any signal at all. It is also the shared
state behind "two concurrent suites corrupt each other".
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB, default_memory_dir
from virturoid.services.memory_store import DEFAULT_MEMORY_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MEMORY_DIR = REPO_ROOT / "build" / "memory"


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def test_the_db_this_session_writes_to_is_not_the_repos_own_bank():
    assert not _is_under(Path(DEFAULT_DB_PATH), REAL_MEMORY_DIR), (
        f"the test session would bank into {DEFAULT_DB_PATH}, which is the developer's real memory. "
        "conftest sets VIRTUROID_MEMORY_DIR before any virturoid import for exactly this reason."
    )


def test_the_json_memory_dir_follows_the_same_rule():
    # memory_store keeps its per-(class, task) JSON records beside the sqlite file, so redirecting
    # one and not the other would leave half the write path pointed at the real directory.
    assert not _is_under(Path(DEFAULT_MEMORY_DIR), REAL_MEMORY_DIR)
    assert Path(DEFAULT_MEMORY_DIR).resolve() == Path(DEFAULT_DB_PATH).parent.resolve()


def test_banking_through_the_real_path_lands_in_the_session_dir():
    """Not a mock: open the DB the product would open and write a row through it."""
    before = REAL_MEMORY_DIR / "virturoid_memory.db"
    before_rows = _locomotion_rows(before)

    Path(DEFAULT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with MemoryDB(Path(DEFAULT_DB_PATH)) as db:
        db.record_skill(
            "ratchet-probe",
            "quadruped",
            "locomotion",
            success_rate=1.0,
            species="ratchet",
            notes="written by test_suite_does_not_write_the_real_bank",
        )

    assert _locomotion_rows(Path(DEFAULT_DB_PATH)) >= 1
    assert _locomotion_rows(before) == before_rows, (
        "writing through the product's own banking path changed the REAL bank's row count"
    )


def _locomotion_rows(db_path: Path) -> int:
    import sqlite3

    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as con:
        try:
            return con.execute(
                "select count(*) from skills where task_type='locomotion'"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            return 0


def test_the_override_is_read_per_call_not_frozen_at_import():
    """A corpus-factory night and a test session must be able to differ in one process.

    ``default_memory_dir()`` is a function precisely so that the corpus factory can point a run at
    its own database -- a "clean rebuild" that warm-starts from the corpus it is replacing is not a
    clean rebuild.
    """
    original = os.environ.get("VIRTUROID_MEMORY_DIR")
    try:
        os.environ["VIRTUROID_MEMORY_DIR"] = str(Path("nonexistent") / "elsewhere")
        assert default_memory_dir() == Path("nonexistent") / "elsewhere"
    finally:
        if original is None:
            os.environ.pop("VIRTUROID_MEMORY_DIR", None)
        else:
            os.environ["VIRTUROID_MEMORY_DIR"] = original
    assert default_memory_dir() == Path(DEFAULT_MEMORY_DIR)


def test_without_the_override_the_product_still_defaults_to_build_memory():
    """The redirect is a TEST affordance. The product's default must be unchanged.

    Checked in a subprocess with the variable cleared, because this process has it set and the
    module-level ``DEFAULT_DB_PATH`` was bound under it.
    """
    env = {k: v for k, v in os.environ.items() if k != "VIRTUROID_MEMORY_DIR"}
    # The package is not installed; pytest puts ``src`` on the path for the suite, so the
    # subprocess has to be told the same thing.
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    out = subprocess.run(
        [sys.executable, "-c",
         "from virturoid.services.memory_db import DEFAULT_DB_PATH; print(DEFAULT_DB_PATH)"],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), check=True,
    )
    assert out.stdout.strip() == str(Path("build") / "memory" / "virturoid_memory.db")
