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


# ======================================================== the design side: runs / designs / species (#278)
#
# ``skills`` is the gait flywheel's store. The DESIGN flywheel warm-starts from ``runs`` (``best_design``),
# retrieves over them (``similar_runs``), distils ``designs`` (``training_dataset``) and aggregates both into
# ``species`` -- and 97% of those rows are the suite's. The same three-bucket contract has to hold there, with
# one extra property those tables force: they have no JSON column to add a key to, so the stamp lives in a
# SIDE TABLE and the audited rows must come back BYTE-IDENTICAL.


def _design_bank(tmp_path: Path, *, runs: list[tuple], designs: list[tuple] = (),
                 species: list[tuple] = ()) -> Path:
    """A bank with the design-side tables. ``runs`` rows are (prompt, robot_class, species, design_json)."""
    db = tmp_path / "virturoid_memory.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE skills (skill_id TEXT, robot_class TEXT, species TEXT, task_type TEXT, "
                "gene_id TEXT, base_config TEXT, created_at TEXT, notes TEXT)")
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, prompt TEXT, robot_class TEXT, species TEXT, "
                "task_type TEXT, converged_design TEXT, success_rate REAL, design_source TEXT, created_at TEXT)")
    con.execute("CREATE TABLE designs (id INTEGER PRIMARY KEY, run_id INTEGER, prompt TEXT, robot_class TEXT, "
                "task_type TEXT, design TEXT, success_rate REAL, source TEXT, created_at TEXT)")
    con.execute("CREATE TABLE species (name TEXT PRIMARY KEY, robot_class TEXT, best_success_rate REAL, "
                "best_design TEXT, runs INTEGER, updated_at TEXT)")
    con.execute("CREATE TABLE provenance (id INTEGER PRIMARY KEY, child_type TEXT, child_id TEXT, "
                "parent_type TEXT, parent_id TEXT, kind TEXT, delta REAL, meta TEXT, created_at TEXT)")
    for i, (prompt, cls, sp, gene) in enumerate(runs, start=1):
        payload = json.dumps({"id": gene}) if gene else None
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                    (i, prompt, cls, sp, "locomotion", payload, 1.0, "agent", "2026-08-01T00:00:00"))
        con.execute("INSERT INTO designs VALUES (?,?,?,?,?,?,?,?,?)",
                    (i, i, prompt, cls, "locomotion", payload or "{}", 1.0, "agent", "2026-08-01T00:00:00"))
    for name, cls, best_gene, n in species:
        con.execute("INSERT INTO species VALUES (?,?,?,?,?,?)",
                    (name, cls, 1.0, json.dumps({"id": best_gene}) if best_gene else None, n,
                     "2026-08-01T00:00:00"))
    con.commit()
    con.close()
    return db


def _design_verdicts(db: Path, repo_root: Path) -> dict:
    return BP.classify_design_side(db, repo_root)


def test_the_agent_submission_prompt_names_the_fixture_outright(tmp_path):
    """``submit_design`` records ``[agent] <design name>``. ``agent_lynx`` is a shipped worked example that
    only tests/ submits, so the prompt column alone attributes the row -- no slug recovery needed."""
    db = _design_bank(tmp_path, runs=[("[agent] agent_lynx", "quadruped", "quadruped", "anatomy_agent_lynx_1")])
    v = _design_verdicts(db, _repo(tmp_path))
    assert v["runs"][0]["bucket"] == BP.SUITE
    assert any("fixture-design-name:agent_lynx" in e for e in v["runs"][0]["evidence"])
    assert v["designs"][0]["bucket"] == BP.SUITE, "the design row is the same submission, one table over"


def test_a_prompt_shared_by_a_test_and_a_script_decides_nothing(tmp_path):
    """THE FLOOR PROPERTY. A phrase both a fixture and a demo script use cannot say which one ran, so the
    row is unjudged -- NOT rounded to 'real' (which would launder it) and NOT to 'suite' (which would
    manufacture pollution). This is why every ``suite=N`` is a lower bound."""
    prompt = "a lilac hovering teapot robot that patrols"
    repo = _repo(tmp_path, tests=f'P = "{prompt}"', scripts=f'P = "{prompt}"')
    db = _design_bank(tmp_path, runs=[(prompt, "quadruped", "quadruped", "anatomy_creature_1")])
    assert _design_verdicts(db, repo)["runs"][0]["bucket"] == BP.UNATTRIBUTED

    # ...and the moment the script stops using it, the same row is provably the suite's.
    only_tests = _repo(tmp_path / "b", tests=f'P = "{prompt}"')
    assert _design_verdicts(db, only_tests)["runs"][0]["bucket"] == BP.SUITE


def test_a_prompt_nobody_in_the_repo_makes_reads_as_real(tmp_path):
    db = _design_bank(tmp_path, runs=[("a cobalt origami crane that folds parcels", "mobile_base",
                                       "mobile_base", "anatomy_creature_2")])
    v = _design_verdicts(db, _repo(tmp_path))["runs"][0]
    assert v["bucket"] == BP.REAL
    assert any("nowhere-in-repo" in e for e in v["evidence"])


def test_a_generic_submission_name_is_not_evidence(tmp_path):
    """``submit_design`` falls back to ``graph.get("name", "design")`` and ``train_held`` writes the robot
    CLASS, so ``[agent] design`` / ``[agent-trained] quadruped`` name nobody. They must stay unattributed
    unless another channel speaks -- and when the body id IS a fixture, that channel decides."""
    db = _design_bank(tmp_path, runs=[("[agent-trained] quadruped", "quadruped", "quadruped", None),
                                      ("[agent-trained] quadruped", "quadruped", "quadruped",
                                       "anatomy_agent_lynx_9debd7b5")])
    v = _design_verdicts(db, _repo(tmp_path))["runs"]
    assert v[0]["bucket"] == BP.UNATTRIBUTED
    assert v[1]["bucket"] == BP.SUITE, "the trained body is a fixture even though the prompt says 'quadruped'"


def test_a_two_word_prompt_is_too_generic_to_decide(tmp_path):
    """"a robot dog" appears in 25 test files, 1 script and 4 source files. A phrase that common is not
    evidence of a writer, and treating it as such is how a census acquires false precision."""
    db = _design_bank(tmp_path, runs=[("a robot dog", "quadruped", "quadruped", "anatomy_creature_3")])
    v = _design_verdicts(db, _repo(tmp_path))["runs"][0]
    assert v["bucket"] == BP.UNATTRIBUTED
    assert any("too-generic" in e for e in v["evidence"])


def test_a_species_is_judged_by_the_runs_that_reference_it(tmp_path):
    """``species`` has no prompt: its ``runs`` counter and ``best_design`` ARE its runs. Decided only when
    they agree; a mixed species stays unattributed and carries the split."""
    db = _design_bank(
        tmp_path,
        runs=[("[agent] agent_lynx", "quadruped", "pure.suite", "anatomy_agent_lynx_1"),
              ("[agent] agent_lynx", "quadruped", "mixed.sp", "anatomy_agent_lynx_1"),
              ("a cobalt origami crane that folds parcels", "quadruped", "mixed.sp", "anatomy_creature_9")],
        species=[("pure.suite", "quadruped", "anatomy_agent_lynx_1", 1),
                 ("mixed.sp", "quadruped", None, 2)])
    by_name = {s["name"]: s for s in _design_verdicts(db, _repo(tmp_path))["species"]}
    assert by_name["pure.suite"]["bucket"] == BP.SUITE
    assert by_name["mixed.sp"]["bucket"] == BP.UNATTRIBUTED
    assert by_name["mixed.sp"]["suite_run_fraction"] == 0.5
    assert any("suite=1 real=1" in e for e in by_name["mixed.sp"]["evidence"])


def test_the_design_side_stamp_never_writes_the_audited_rows(tmp_path):
    """THE PROPERTY THAT MAKES THIS SAFE TO RUN ON THE REAL BANK -- and it is stronger here than for
    ``skills``: those tables are opened read-only by the classifier and never updated at all. The verdict
    goes to a side table, so ``runs``/``designs``/``species`` come back byte-identical."""
    db = _design_bank(tmp_path,
                      runs=[("[agent] agent_lynx", "quadruped", "quadruped", "anatomy_agent_lynx_1"),
                            ("a cobalt origami crane that folds parcels", "mobile_base", "mb", None)],
                      species=[("quadruped", "quadruped", "anatomy_agent_lynx_1", 1)])
    before = db.read_bytes()
    verdicts = _design_verdicts(db, _repo(tmp_path))
    assert BP.stamp_table(db, "runs", verdicts["runs"], apply=False)["written"] == 0
    assert db.read_bytes() == before, "a dry run must not touch the file at all"

    for table in ("runs", "designs", "species"):
        BP.stamp_table(db, table, verdicts[table], apply=True)

    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2, "quarantine must not drop rows"
    # every audited row is unchanged -- no new column, no rewritten payload
    assert [tuple(r) for r in con.execute("SELECT prompt, converged_design, success_rate FROM runs")] == [
        ("[agent] agent_lynx", '{"id": "anatomy_agent_lynx_1"}', 1.0),
        ("a cobalt origami crane that folds parcels", None, 1.0)]
    assert con.execute("SELECT COUNT(*) FROM row_provenance").fetchone()[0] == 5, "2 runs + 2 designs + 1 sp"


def test_the_audited_tables_keep_their_exact_schema(tmp_path):
    """Widening ``runs`` would put audit metadata in a table the product SELECTs *-from; stuffing it into
    ``converged_design`` would hand it to the co-design search as part of the warm start."""
    db = _design_bank(tmp_path, runs=[("[agent] agent_lynx", "quadruped", "quadruped", "anatomy_agent_lynx_1")])
    cols_before = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)")]
    v = _design_verdicts(db, _repo(tmp_path))
    BP.stamp_table(db, "runs", v["runs"], apply=True)
    assert [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)")] == cols_before


def test_an_unstamped_row_reads_unattributed_and_a_suite_stamp_is_never_downgraded(tmp_path):
    """Two halves of the same rule. An absent stamp is an open question, never innocence; and a later pass
    that runs WITHOUT the evidence an earlier one had must not talk a proven fixture row back out."""
    db = _design_bank(tmp_path, runs=[("[agent] agent_lynx", "quadruped", "quadruped", "anatomy_agent_lynx_1")])
    assert BP.row_sources(db, "runs") == {}, "an unstamped table yields no keys at all"
    assert BP.source_of_key({}, "1") == BP.UNATTRIBUTED

    v = _design_verdicts(db, _repo(tmp_path))["runs"]
    BP.stamp_table(db, "runs", v, apply=True)
    assert BP.row_sources(db, "runs") == {"1": BP.SUITE}

    res = BP.stamp_table(db, "runs", [{**v[0], "bucket": BP.UNATTRIBUTED, "evidence": []}], apply=True)
    assert res["retained_suite"] == 1
    assert BP.row_sources(db, "runs") == {"1": BP.SUITE}
    # ...but a positive re-verdict still lands: the guard only blocks the slide back to "unknown".
    BP.stamp_table(db, "runs", [{**v[0], "bucket": BP.REAL, "evidence": ["operator-probe-name:trex"]}],
                   apply=True)
    assert BP.row_sources(db, "runs") == {"1": BP.REAL}


def test_the_census_counts_three_buckets_per_table(tmp_path):
    db = _design_bank(tmp_path,
                      runs=[("[agent] agent_lynx", "quadruped", "quadruped", "anatomy_agent_lynx_1"),
                            ("a cobalt origami crane that folds parcels", "mobile_base", "mb", None),
                            ("a robot dog", "quadruped", "quadruped", None)])
    c = BP.census(_design_verdicts(db, _repo(tmp_path))["runs"])
    assert c == {BP.SUITE: 1, BP.REAL: 1, BP.UNATTRIBUTED: 1}
