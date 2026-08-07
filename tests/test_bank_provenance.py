"""The fixture census must be a measurement, not a vibe -- and quarantine must never destroy a row.

``scripts/bank_provenance.py`` decides who wrote each banked skill and stamps the answer in place. Two
properties are worth a standing test:

  1. THE STAMP IS ADDITIVE. ``build/memory/virturoid_memory.db`` is gitignored and has no backup, so a
     quarantine pass that dropped or rewrote a row would be unrecoverable. The stamper may only add keys.
  2. THE THIRD BUCKET IS REAL. A row no channel speaks to must read ``unattributed``, never ``real``.
     Collapsing to two buckets is how "we removed the fixture rows" would come to mean "we kept everything we
     could not prove was a fixture" -- which is the same pollution with a cleaner label.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import bank_provenance as BP  # noqa: E402


def _bank(tmp_path: Path, rows: list[dict]) -> Path:
    """A minimal bank with the three tables the attribution reads."""
    db = tmp_path / "virturoid_memory.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE skills (skill_id TEXT, robot_class TEXT, species TEXT, task_type TEXT, "
                "gene_id TEXT, base_config TEXT, created_at TEXT, notes TEXT)")
    con.execute("CREATE TABLE vectors (obj_type TEXT, obj_id TEXT, meta TEXT)")
    con.execute("CREATE TABLE designs (prompt TEXT, design TEXT)")
    con.execute("CREATE TABLE runs (prompt TEXT, robot_class TEXT, created_at TEXT)")
    for r in rows:
        con.execute("INSERT INTO skills VALUES (?,?,?,?,?,?,?,?)",
                    (r["skill_id"], r.get("robot_class", "quadruped"), r.get("species", "quadruped.anatomy"),
                     r.get("task_type", "locomotion"), r.get("gene_id"),
                     json.dumps(r.get("base_config", {"gait_params": {"freq": 1.5}})),
                     r.get("created_at", "2026-08-01T00:00:00"), r.get("notes", "")))
        con.execute("INSERT INTO vectors VALUES ('skill', ?, ?)",
                    (r["skill_id"], json.dumps({"gene": {"id": r.get("gene_id")}})))
    con.commit()
    con.close()
    return db


def _repo(tmp_path: Path, *, tests: str = "", scripts: str = "", src: str = "") -> Path:
    """A synthetic repo tree to grep against.

    Deliberately NOT the real one: this test file has to spell out the fake robot names it is classifying,
    so grepping the live ``tests/`` would find them and the classifier would correctly report that the
    phrase is a fixture -- proving nothing except that the test wrote it.
    """
    root = tmp_path / "repo"
    for name, body in (("tests", tests), ("scripts", scripts), ("src", src)):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "f.py").write_text(body, encoding="utf-8")
    return root


def _classify(db: Path, repo_root: Path, *, reproduced=None):
    rows = BP._skills(db)
    return {v["skill_id"]: v for v in BP.classify(
        rows, reproduced=reproduced or {}, prompts=BP._designs_prompts(db),
        repo_root=repo_root, fixture_times=[])}


def test_a_fixture_body_class_is_attributed_to_the_suite(tmp_path):
    """``totally_made_up_xyz`` is a literal from tests/test_structural_dispatch.py and nowhere else."""
    db = _bank(tmp_path, [{"skill_id": "gait::totally_made_up_xyz::aa", "robot_class": "totally_made_up_xyz",
                           "gene_id": "anatomy_creature_deadbeef"}])
    v = _classify(db, _repo(tmp_path))["gait::totally_made_up_xyz::aa"]
    assert v["bucket"] == BP.SUITE
    assert any("fixture-class" in e for e in v["evidence"])


def test_a_reproduced_skill_id_is_proof_not_inference(tmp_path):
    """``skill_id`` is ``gait::<class>::<structural key>`` -- a hash of the BODY. If a controlled pytest run
    into an empty database produced the same id, the suite builds that exact body."""
    db = _bank(tmp_path, [{"skill_id": "gait::quadruped::beef", "gene_id": "anatomy_creature_1234abcd"}])
    plain = _classify(db, _repo(tmp_path))["gait::quadruped::beef"]
    assert plain["bucket"] == BP.UNATTRIBUTED, "a generic body with no repro evidence must not be guessed at"

    v = _classify(db, _repo(tmp_path),
                  reproduced={"gait::quadruped::beef": ["probe-run"]})["gait::quadruped::beef"]
    assert v["bucket"] == BP.SUITE
    assert "repro:probe-run" in v["evidence"]


def test_a_body_named_after_a_request_nobody_in_the_repo_makes_reads_as_real(tmp_path):
    db = _bank(tmp_path, [{"skill_id": "gait::quadruped::cc",
                           "gene_id": "anatomy_lilac_hovering_teapot_robot_0badcafe"}])
    v = _classify(db, _repo(tmp_path))["gait::quadruped::cc"]
    assert v["bucket"] == BP.REAL
    assert any("nowhere-in-repo" in e for e in v["evidence"])

    # ...and the same body reads as UNATTRIBUTED the moment a test in the tree asks for that phrase, even
    # spelled with hyphens. This is the channel that keeps the census from over-claiming "real".
    named = _repo(tmp_path / "b", tests='PROMPT = "a lilac hovering-teapot robot"')
    assert _classify(db, named)["gait::quadruped::cc"]["bucket"] == BP.UNATTRIBUTED


def test_a_generic_compiler_body_stays_in_the_third_bucket(tmp_path):
    """``anatomy_creature`` is what the compiler names a body when the request named none, so its id carries
    no evidence either way. It must NOT be rounded into ``real`` just because it is not provably a fixture."""
    db = _bank(tmp_path, [{"skill_id": "gait::quadruped::dd", "gene_id": "anatomy_creature_91b931bf"}])
    v = _classify(db, _repo(tmp_path))["gait::quadruped::dd"]
    assert v["bucket"] == BP.UNATTRIBUTED
    assert any("generic-body-name" in e for e in v["evidence"])


def test_the_phrase_match_survives_hyphens_and_underscores():
    """A gene id spells the request ``sturdy_four_legged_walking_robot``; the source spells it "a sturdy
    four-legged walking robot". Comparing raw forms answers "is this in the tree?" with a confident NO for
    every multiword phrase, which silently promotes fixture rows to ``real``."""
    assert BP._norm("sturdy_four_legged_walking_robot") == "sturdy four legged walking robot"
    assert BP._norm("A sturdy four-legged walking robot.").startswith("a sturdy four legged walking robot")
    assert BP._slug_of("anatomy_medium_warehouse_dog_robot_86ba6779") == "medium warehouse dog robot"
    assert BP._slug_of("built_quadruped_18seg") is None, "composed bodies carry no prompt trace"


def test_the_stamp_is_purely_additive_and_deletes_nothing(tmp_path):
    """THE PROPERTY THAT MAKES THIS SAFE TO RUN ON THE REAL BANK."""
    db = _bank(tmp_path, [{"skill_id": "gait::totally_made_up_xyz::aa", "robot_class": "totally_made_up_xyz",
                           "gene_id": "anatomy_creature_deadbeef",
                           "base_config": {"gait_params": {"freq": 1.25}, "forward_m": 0.9,
                                           "bank_gate": "fragility_v1"}},
                          {"skill_id": "gait::quadruped::bb", "gene_id": "anatomy_creature_91b931bf"}])
    before = sqlite3.connect(db).execute("SELECT COUNT(*) FROM skills").fetchone()[0]
    verdicts = list(_classify(db, _repo(tmp_path)).values())

    assert BP.stamp(db, verdicts, apply=False) == 0
    assert not json.loads(sqlite3.connect(db).execute(
        "SELECT base_config FROM skills WHERE skill_id='gait::quadruped::bb'").fetchone()[0]).get("row_source")

    assert BP.stamp(db, verdicts, apply=True) == 2
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM skills").fetchone()[0] == before, "quarantine must not drop rows"
    bc = json.loads(con.execute(
        "SELECT base_config FROM skills WHERE skill_id='gait::totally_made_up_xyz::aa'").fetchone()[0])
    assert bc["row_source"] == BP.SUITE
    assert bc["row_source_stamp"] == BP.ROW_SOURCE_STAMP
    # every pre-existing key survives untouched, including another pass's stamp
    assert bc["gait_params"] == {"freq": 1.25} and bc["forward_m"] == 0.9 and bc["bank_gate"] == "fragility_v1"


def test_source_of_treats_an_unstamped_row_as_an_open_question():
    assert BP.source_of(None) == BP.UNATTRIBUTED
    assert BP.source_of({}) == BP.UNATTRIBUTED
    assert BP.source_of({"row_source": BP.SUITE}) == BP.SUITE


def test_the_census_reports_three_buckets(tmp_path):
    """``audit_gait_bank.census`` is where the next person reads provenance without redoing this."""
    pytest.importorskip("virturoid.services.gait_flywheel")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import audit_gait_bank as A

    rows = [{"base_config": {"row_source": BP.SUITE}}, {"base_config": {}},
            {"base_config": {"row_source": BP.REAL}}]
    assert dict(A.row_sources(rows)) == {BP.SUITE: 1, BP.REAL: 1, BP.UNATTRIBUTED: 1}


# ---------------------------------------------------------------- the stamp has to SURVIVE a re-bank
#
# OBSERVED LIVE 2026-08-07, and it is the failure that makes a quarantine worthless rather than merely
# incomplete: ``MemoryDB.record_skill`` REPLACES ``base_config``, so a row stamped by the audit pass was
# re-banked minutes later and came back with no stamp at all. An absent ``row_source`` reads as
# ``unattributed`` -- which is precisely the bucket a suite-authored row would launder itself into.
#
# ``bank_gait``'s gate guard cannot cover this. That guard refuses a write which knows LESS about robustness;
# a re-bank that knows strictly MORE is allowed through, and it still carries no provenance, because
# provenance is not something a banking call can know. So it must be COPIED FORWARD, not defended.

def test_a_rebank_does_not_erase_the_quarantine_stamp():
    from virturoid.services.gait_flywheel import _carry_provenance

    prior = {"gait_params": {"freq": 1.0}, "row_source": "suite", "row_source_stamp": "provenance_v1",
             "row_source_evidence": ["repro:mem2"], "row_source_stamped_at": "2026-08-07T09:00:00"}
    fresh = {"gait_params": {"freq": 2.0}, "forward_m": 6.4, "bank_gate": "fragility_v1"}

    out = _carry_provenance(prior, dict(fresh))

    assert out["row_source"] == "suite", "a re-bank laundered a suite row into 'unattributed'"
    assert out["row_source_evidence"] == ["repro:mem2"]
    assert out["row_source_stamped_at"] == "2026-08-07T09:00:00"
    # ...while everything the NEW measurement established still wins.
    assert out["gait_params"] == {"freq": 2.0}
    assert out["forward_m"] == 6.4
    assert out["bank_gate"] == "fragility_v1"


def test_carrying_provenance_never_invents_it():
    """No prior row, or a prior with no stamp, must not manufacture one."""
    from virturoid.services.gait_flywheel import _carry_provenance

    assert "row_source" not in _carry_provenance(None, {"forward_m": 1.0})
    assert "row_source" not in _carry_provenance({"gait_params": {}}, {"forward_m": 1.0})


def test_an_explicit_new_source_is_not_overwritten_by_the_old_one():
    """If a write genuinely knows its own provenance, it wins -- carry-forward only fills a GAP."""
    from virturoid.services.gait_flywheel import _carry_provenance

    out = _carry_provenance({"row_source": "suite"}, {"row_source": "real"})
    assert out["row_source"] == "real"
