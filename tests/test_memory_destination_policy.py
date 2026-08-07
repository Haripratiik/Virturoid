"""WHERE A WRITE TO THE MEMORY BANK IS ALLOWED TO LAND — the classification, made executable.

``tests/conftest.py`` fixed PYTEST. It did not fix the bank. Two things were still true on 2026-08-07, both
measured, both reproduced by the tests below:

  1. THERE ARE TWO DEFAULT RULES AND ONLY ONE READS THE ENV VAR. ``agent_tools.safe_build_path(None, "memory")``
     computes ``<cwd>/build/memory`` from scratch, and it is the destination behind ``fit_gait_for_body(db=None)``
     (so ``ensure_walkable_quad``, so ``create_robot``), behind ``submit_design``'s ``record_run``, and behind
     ``adapt_gait``. With ``VIRTUROID_MEMORY_DIR`` pointed at an empty directory, ``fit_gait_for_body(db=None)``
     created a 122 KB database in ``<cwd>/build/memory`` and left the redirect target empty. The live bank's
     newest ``runs`` row was written today and one of them carries the fixture class ``totally_made_up_xyz``.
  2. NOTHING LOADS CONFTEST EXCEPT PYTEST. ``python -c``, ``python -m unittest`` and the REPL import the product
     with no redirect at all.

The classification these tests pin down:

  DELIBERATE WRITER (a corpus night)  -> must NAME its destination; ``--memory`` is required and is exported so
                                         the nested defaults three levels down land there too.
  PRODUCT RUNTIME (workspace/session) -> already passes an explicit path; the policy never touches it.
  AD-HOC PROBE (-c, -, -i, unittest)  -> may READ the real bank, may not WRITE it without saying so.

The bias, stated: polluting the corpus is irreversible (gitignored, no backup, rows are never deleted, and every
evidence gate reads it), while sandboxing a run is recoverable. So the guard refuses the write. It is built so the
other direction cannot happen silently either: a probe still reads the REAL bank byte for byte (an empty-looking
corpus would be the worse lie), the only substituted destination is one the process itself exported, and both
branches announce themselves.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from virturoid.services.memory_db import (MemoryDB, adhoc_entry_point, conventional_memory_dir,
                                          default_memory_dir, resolve_memory_destination)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

_SEED = textwrap.dedent("""
    from virturoid.services.memory_db import MemoryDB
    import sys
    with MemoryDB(sys.argv[1]) as db:
        db.record_skill("seed__row", "quadruped", "locomotion", success_rate=0.9)
""")

#: import the product and bank WITHOUT naming a destination -- exactly the shape of an ad-hoc probe.
_BANK_AT_DEFAULT = textwrap.dedent("""
    import json
    from virturoid.services.memory_db import MemoryDB, MemoryBankIsReadOnly
    db = MemoryDB()                                   # no argument: the product default
    out = {"path": str(db.path), "read_only": db.read_only, "rows_before": db.stats()["skills"]}
    try:
        db.record_skill("probe__wrote_this", "quadruped", "locomotion", success_rate=1.0)
        out["write"] = "SUCCEEDED"
    except MemoryBankIsReadOnly as exc:
        out["write"] = "REFUSED"; out["msg"] = str(exc)
    out["rows_after"] = db.stats()["skills"]
    db.close()
    print("RESULT" + json.dumps(out))
""")


def _run(args, *, cwd: Path, memory_dir: str | None = None) -> subprocess.CompletedProcess:
    """A child process that has NEVER heard of conftest — which is the entire point."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUROID_MEMORY_DIR"}
    env["PYTHONPATH"] = str(SRC)
    env["VIRTUROID_NO_LOCAL_ENV"] = "1"
    if memory_dir is not None:
        env["VIRTUROID_MEMORY_DIR"] = memory_dir
    return subprocess.run([sys.executable, *args], cwd=str(cwd), env=env, capture_output=True, text=True)


def _payload(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def _seed_bank(cwd: Path) -> Path:
    """Put a real bank where the env-blind rule looks, so the child has a corpus that could be polluted."""
    bank = cwd / "build" / "memory" / "virturoid_memory.db"
    bank.parent.mkdir(parents=True, exist_ok=True)
    assert _run(["-c", _SEED, str(bank)], cwd=cwd).returncode == 0
    return bank


def _skills(bank: Path) -> int:
    import sqlite3
    conn = sqlite3.connect(f"file:{bank.as_posix()}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0])
    finally:
        conn.close()


# ------------------------------------------------------------------ 1. an explicit destination is never touched
def test_an_explicitly_named_destination_is_never_rewritten(tmp_path):
    """The product runtime lane. ``webapp``/``desktop``/``compose``/``import_robot`` all pass
    ``workspace/memory/virturoid_memory.db``, and every test in this suite passes a ``tmp_path``. A policy that
    moved those would be a policy that breaks the product to protect the bank."""
    asked = tmp_path / "workspace" / "memory" / "virturoid_memory.db"
    used, read_only, why = resolve_memory_destination(asked)
    assert (used, read_only, why) == (asked, False, "")
    with MemoryDB(asked) as db:
        db.record_skill("explicit__row", "quadruped", "locomotion", success_rate=0.5)
        assert db.path == asked and not db.read_only
    assert asked.exists()


# ------------------------------------------------------------------ 2. the redirect becomes TOTAL
def test_the_env_blind_default_is_rewritten_onto_the_redirect(tmp_path, monkeypatch):
    """THE LEAK THAT DEFEATED THE CONFTEST FIX. ``safe_build_path(None, "memory")`` never read
    ``VIRTUROID_MEMORY_DIR``, so ``fit_gait_for_body(db=None)`` and ``submit_design`` banked into
    ``<cwd>/build/memory`` even under pytest. Fixing it at those call sites is not possible from here and would not
    hold anyway — they are three levels below any caller that takes a destination. So the chokepoint decides."""
    monkeypatch.setattr("virturoid.services.memory_db._INITIAL_CWD", tmp_path)
    monkeypatch.setenv("VIRTUROID_MEMORY_DIR", str(tmp_path / "night"))
    env_blind = conventional_memory_dir() / "virturoid_memory.db"     # what safe_build_path would hand us

    used, read_only, why = resolve_memory_destination(env_blind)
    assert used == tmp_path / "night" / "virturoid_memory.db"
    assert not read_only and "safe_build_path" in why

    with MemoryDB(env_blind) as db:
        db.record_skill("night__row", "quadruped", "locomotion", success_rate=0.7)
        assert db.requested == env_blind and db.path == used
    assert used.exists(), "the night's own directory got the row"
    assert not env_blind.exists(), "nothing was written to the env-blind default"


def test_under_this_very_suite_the_real_bank_resolves_to_the_session_dir():
    """The load-bearing assertion, made against the REAL conventional path and opening nothing.

    ``conventional_memory_dir()`` here is the repo's own ``build/memory`` — the gitignored, unbacked corpus. If the
    policy regressed, this is the path a stray ``safe_build_path`` write would land in."""
    real = conventional_memory_dir() / "virturoid_memory.db"
    used, read_only, _ = resolve_memory_destination(real)
    assert used.parent == Path(default_memory_dir()), used
    assert used != real and not read_only


# ------------------------------------------------------------------ 3. the ad-hoc lane
@pytest.mark.parametrize("argv0,expected", [("-c", "python -c"), ("-", "python - (stdin)"),
                                            ("", "the interactive interpreter"),
                                            ("python.exe -m unittest", "python -m unittest"),
                                            ("scripts/corpus_factory_night.py", None),
                                            ("/usr/bin/pytest", None)])
def test_only_processes_with_no_entry_point_count_as_ad_hoc(argv0, expected, monkeypatch):
    """A NAMED script is deliberately not ad-hoc: it is a place where a destination can be stated. That is why
    ``corpus_factory_night.py`` states one and why pytest, Studio and the MCP server are untouched."""
    monkeypatch.setattr(sys, "argv", [argv0])
    assert adhoc_entry_point() == expected


def test_a_python_c_probe_reads_the_real_bank_but_cannot_write_it(tmp_path):
    """TASK ITEM 4: import the product and bank with no explicit destination, from the exact entry point that was
    polluting the corpus. Both halves are asserted, because getting only one of them right is a different bug:
    the READ must still return the real corpus (a probe told the bank is empty is the worse failure), and the
    WRITE must not land."""
    bank = _seed_bank(tmp_path)
    before = _skills(bank)

    proc = _run(["-c", _BANK_AT_DEFAULT], cwd=tmp_path)
    got = _payload(proc)

    assert got["read_only"] is True
    assert got["rows_before"] == before == 1, "the probe read the REAL bank, not an empty sandbox"
    assert got["write"] == "REFUSED"
    assert "VIRTUROID_MEMORY_DIR" in got["msg"], got["msg"]      # the message says how to opt in
    assert "VIRTUROID_MEMORY_DIR" in proc.stderr                 # ...and so does the open-time notice
    assert _skills(bank) == before, "an ad-hoc probe added a row to the corpus"


def test_python_m_unittest_cannot_write_the_bank_either(tmp_path):
    """``python -m unittest`` runs this repo's tests without ever importing ``tests/conftest.py``, so the redirect
    that protects pytest does not exist for it. Same corpus, same fixture rows, no guard — until now."""
    bank = _seed_bank(tmp_path)
    case = tmp_path / "t_bank_case.py"
    case.write_text(textwrap.dedent("""
        import unittest
        from virturoid.services.memory_db import MemoryDB
        class T(unittest.TestCase):
            def test_banks(self):
                try:
                    with MemoryDB() as db:
                        db.record_skill("unittest__row", "quadruped", "locomotion", success_rate=1.0)
                except Exception:
                    pass                      # mirrors the product's best-effort banking sites
    """), encoding="utf-8")

    proc = _run(["-m", "unittest", "discover", "-s", str(tmp_path), "-p", "t_bank_case.py"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert _skills(bank) == 1, "python -m unittest banked a fixture row into the corpus"


def test_naming_the_real_bank_makes_it_writable_from_a_probe(tmp_path):
    """The escape hatch, and there is deliberately only one. A developer who MEANS to write the corpus from
    ``python -c`` says so with the same variable everything else uses — no second 'allow writes' flag, because a
    destination you have to name is the entire mechanism."""
    bank = _seed_bank(tmp_path)
    got = _payload(_run(["-c", _BANK_AT_DEFAULT], cwd=tmp_path, memory_dir=str(bank.parent)))
    assert got["read_only"] is False and got["write"] == "SUCCEEDED"
    assert _skills(bank) == 2


# ------------------------------------------------------------------ 4. the other failure direction
def test_a_named_run_is_not_silently_sandboxed(tmp_path):
    """BOTH directions matter. A real run that states its destination must land THERE — not in a scratch dir of
    our invention, and not in the conventional bank either."""
    bank = _seed_bank(tmp_path)
    night = tmp_path / "night"
    script = tmp_path / "a_real_run.py"
    script.write_text(_BANK_AT_DEFAULT, encoding="utf-8")

    got = _payload(_run([str(script)], cwd=tmp_path, memory_dir=str(night)))
    assert got["read_only"] is False and got["write"] == "SUCCEEDED"
    assert Path(got["path"]).parent == night
    assert _skills(night / "virturoid_memory.db") == 1
    assert _skills(bank) == 1, "the named run leaked into the conventional bank"


def test_a_named_script_with_no_redirect_still_writes_the_default(tmp_path):
    """The honest limit of the guard, asserted rather than hidden: a script under ``scripts/`` is classified as a
    DELIBERATE WRITER, and deliberate writers are made safe by stating a destination (see
    ``corpus_factory_night.claim_destination``), not by having the library refuse them. Silently sandboxing every
    named script would break the product's own banking, which is the failure direction we refused to take."""
    bank = _seed_bank(tmp_path)
    script = tmp_path / "some_script.py"
    script.write_text(_BANK_AT_DEFAULT, encoding="utf-8")
    got = _payload(_run([str(script)], cwd=tmp_path))
    assert got["read_only"] is False and got["write"] == "SUCCEEDED"
    assert _skills(bank) == 2


# ------------------------------------------------------------------ 5. the deliberate writer states it
def test_corpus_factory_night_requires_a_destination():
    proc = _run([str(REPO / "scripts" / "corpus_factory_night.py"), "--bodies", "1"], cwd=REPO)
    assert proc.returncode != 0
    assert "--memory" in proc.stderr and "required" in proc.stderr.lower()


def test_corpus_factory_night_exports_its_destination_before_importing(tmp_path, monkeypatch):
    """``claim_destination`` is the whole fix: passing ``memory_dir=`` down cannot reach the leaking call sites,
    which take no destination argument at all. Exporting it does, via the rewrite asserted above."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from corpus_factory_night import claim_destination
    finally:
        sys.path.pop(0)
    monkeypatch.delenv("VIRTUROID_MEMORY_DIR", raising=False)
    target = claim_destination(str(tmp_path / "corpus"))
    assert os.environ["VIRTUROID_MEMORY_DIR"] == str(target) == str((tmp_path / "corpus").resolve())
    assert target.is_dir()

    monkeypatch.setenv("VIRTUROID_MEMORY_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(SystemExit, match="Pick one"):
        claim_destination(str(tmp_path / "corpus"))
