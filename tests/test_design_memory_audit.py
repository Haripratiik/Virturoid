"""The re-measurement harness has to be trustworthy in two specific ways, or its answer is worthless.

``scripts/audit_design_memory.py`` asks the only question the fixture census exists to answer: with the
suite's rows removed, does any design-side number we quote change? It answers by DELETING rows -- which is
exactly the operation the quarantine was designed to avoid on the real bank. Two properties keep that safe
and keep the answer honest:

  1. THE DELETES HAPPEN ON COPIES. The source bank is opened read-only and comes back byte-identical. If this
     ever stopped holding, an audit run would destroy a gitignored, unbacked-up database.
  2. THE STRICT ARM IS ACTUALLY STRICTER. ``ex_strict`` drops only rows a NAME or a REPRODUCTION attributes to
     the suite; ``ex_full`` also drops rows attributed by a shared prompt phrase. If the two arms silently
     became the same filter, "this number only moves under the weakest channel" -- the caveat that decides
     whether a finding is quotable -- could no longer be stated.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import audit_design_memory as A  # noqa: E402
import bank_provenance as BP  # noqa: E402


def _bank(tmp_path: Path) -> Path:
    """Three runs: one named after a fixture, one attributed only by a shared phrase, one operator request."""
    db = tmp_path / "virturoid_memory.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, prompt TEXT, robot_class TEXT, species TEXT, "
                "task_type TEXT, converged_design TEXT, success_rate REAL, design_source TEXT, "
                "created_at TEXT)")
    con.execute("CREATE TABLE designs (id INTEGER PRIMARY KEY, run_id INTEGER, prompt TEXT, robot_class TEXT,"
                " task_type TEXT, design TEXT, success_rate REAL, source TEXT, created_at TEXT)")
    con.execute("CREATE TABLE provenance (id INTEGER PRIMARY KEY, child_type TEXT, child_id TEXT, "
                "parent_type TEXT, parent_id TEXT, kind TEXT, delta REAL, meta TEXT, created_at TEXT)")
    con.execute("CREATE TABLE species (name TEXT PRIMARY KEY, robot_class TEXT, best_success_rate REAL, "
                "best_design TEXT, runs INTEGER, updated_at TEXT)")
    con.execute("CREATE TABLE vectors (obj_type TEXT, obj_id TEXT)")
    rows = [(1, "[agent] agent_lynx", "quadruped", "sp.fixture", 0.10),
            (2, "a phrase only a test uses today", "quadruped", "sp.mixed", 0.90),
            (3, "a cobalt origami crane that folds parcels", "quadruped", "sp.mixed", 0.50)]
    for rid, prompt, cls, sp, rate in rows:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, prompt, cls, sp, "locomotion", json.dumps({"id": f"g{rid}"}), rate, "agent",
                     "2026-08-01"))
        con.execute("INSERT INTO designs VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, rid, prompt, cls, "locomotion", "{}", rate, "agent", "2026-08-01"))
        con.execute("INSERT INTO provenance VALUES (?,?,?,?,?,?,?,?,?)",
                    (rid, "design", f"g{rid}", "design", "prior", "design_search_gain", 0.01,
                     json.dumps({"prompt": prompt}), "2026-08-01"))
        con.execute("INSERT INTO vectors VALUES ('run', ?)", (str(rid),))
    con.execute("INSERT INTO species VALUES ('sp.fixture','quadruped',0.10,NULL,1,'2026-08-01')")
    con.execute("INSERT INTO species VALUES ('sp.mixed','quadruped',0.90,NULL,2,'2026-08-01')")
    con.commit()
    con.close()
    return db


def _verdicts(tmp_path: Path, db: Path) -> dict:
    """Classify against a synthetic tree where run 2's prompt is a fixture and run 3's is nobody's."""
    root = tmp_path / "repo"
    for name, body in (("tests", 'P = "a phrase only a test uses today"'), ("scripts", ""), ("src", "")):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "f.py").write_text(body, encoding="utf-8")
    return BP.classify_design_side(db, root)


def test_the_arms_are_built_from_copies_and_never_touch_the_source(tmp_path):
    """THE PROPERTY THAT MAKES AN AUDIT RUN SAFE ON A DATABASE WITH NO BACKUP."""
    db = _bank(tmp_path)
    before = db.read_bytes()
    A.build_arms(db, tmp_path / "arms", _verdicts(tmp_path, db))
    assert db.read_bytes() == before


def test_the_strict_arm_keeps_what_only_a_shared_phrase_accuses(tmp_path):
    """Run 1 is named after a fixture design (proof-grade); run 2 is attributed only because its prompt
    appears in tests/. ``ex_strict`` must drop the first and KEEP the second -- that difference is what lets
    the report say whether a moved number rests on the weakest channel."""
    db = _bank(tmp_path)
    v = _verdicts(tmp_path, db)
    assert [r["bucket"] for r in v["runs"]] == [BP.SUITE, BP.SUITE, BP.REAL]
    assert A._is_proof_grade(v["runs"][0]["evidence"]) is True
    assert A._is_proof_grade(v["runs"][1]["evidence"]) is False, "a shared phrase is not proof"

    arms = A.build_arms(db, tmp_path / "arms", v)
    kept = {arm: sorted(r[0] for r in sqlite3.connect(p).execute("SELECT id FROM runs"))
            for arm, p in arms.items()}
    assert kept == {"full": [1, 2, 3], "ex_strict": [2, 3], "ex_full": [3]}
    # the same filter applies to every table the metrics read, not just runs
    for table in ("designs", "provenance"):
        ids = sorted(r[0] for r in sqlite3.connect(arms["ex_full"]).execute(f"SELECT id FROM {table}"))
        assert ids == [3], table
    # a deleted run must not keep occupying a slot in the kNN index it can no longer fill
    assert [r[0] for r in sqlite3.connect(arms["ex_full"]).execute(
        "SELECT obj_id FROM vectors WHERE obj_type='run'")] == ["3"]


def test_the_species_accumulator_is_rederived_not_filtered(tmp_path):
    """``species.runs`` is a counter bumped per ``record_run`` and ``best_success_rate`` a running max, so
    neither can be filtered -- both have to be recomputed from the surviving runs. A species left with no
    runs is dropped: it existed only because the suite built it."""
    db = _bank(tmp_path)
    arms = A.build_arms(db, tmp_path / "arms", _verdicts(tmp_path, db))

    def species(arm):
        return {n: (r, b) for n, r, b in sqlite3.connect(arms[arm]).execute(
            "SELECT name, runs, best_success_rate FROM species")}

    assert species("full") == {"sp.fixture": (1, 0.10), "sp.mixed": (2, 0.90)}
    assert species("ex_strict") == {"sp.mixed": (2, 0.90)}, "the pure-fixture species is gone entirely"
    # run 2 carried sp.mixed's 0.90 best; with it excluded the species' own best drops to run 3's 0.50
    assert species("ex_full") == {"sp.mixed": (1, 0.50)}
