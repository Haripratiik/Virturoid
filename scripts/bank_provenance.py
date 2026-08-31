"""Decide, for every banked row, whether the SUITE wrote it or a real run did -- and stamp the answer in place.

    PYTHONPATH=src python scripts/bank_provenance.py [--memory build/memory]
        [--repro-db DIR_OR_DB ...] [--tables skills,runs,designs,species,provenance] [--apply] [--json OUT]

WHY THIS EXISTS. Until 2026-08-07 the test suite banked into the developer's own database: ``verify_robot``
-> ``_auto_bank_gait`` -> ``bank_gait`` is the ORDINARY product path and ``DEFAULT_DB_PATH`` was a constant
(``tests/conftest.py`` now redirects it; ``tests/test_suite_does_not_write_the_real_bank.py`` is the ratchet).
So for months every pytest run added fixture bodies to the corpus that the evidence gates, the fragility
re-measurement and the hint miners read as observations. Four rows carry the body class
``totally_made_up_xyz`` outright. The rest do not announce themselves.

This script does NOT delete anything, and it must not: ``build/memory/virturoid_memory.db`` is gitignored and
has no backup, so a wrong delete is unrecoverable. It QUARANTINES -- it writes a provenance stamp into
``base_config``, the same shape as ``bank_gate``, so a reader can exclude fixture rows and every row survives
to be re-judged when better evidence arrives.

THE FOUR EVIDENCE CHANNELS, strongest first. Each verdict records which ones fired.

  1. REPRODUCTION (``repro:<db>``). Run pytest with ``VIRTUROID_MEMORY_DIR`` pointed at an empty directory and
     see which skills it banks. ``skill_id`` is ``gait::<class>::<structural key>`` -- a hash of the BODY -- so
     an id that comes back identical is the same body, built by the suite, on this checkout. This is the only
     channel that is proof rather than inference. Regenerate it with:

         VIRTUROID_MEMORY_DIR=/tmp/probe VIRTUROID_IMPORT_CACHE=1 VIRTUROID_GAIT_FIT_CACHE=1 \
             python -m pytest <modules that call create_robot/verify_robot> --basetemp=/tmp/pt -q

     then pass ``--repro-db /tmp/probe``. Coverage is whatever you ran: this channel can prove a row IS the
     suite's and can never prove one is not.

  2. FIXTURE IDENTITY (``fixture-class``, ``fixture-body``). The row carries a name that exists nowhere but
     ``tests/`` -- ``totally_made_up_xyz`` as a body class, or a gene id built from a design name that only a
     test submits (the shipped ``get_design_schema()["examples"]`` are submitted from ``tests/`` and nowhere
     else in the repo; ``octo1``/``domed``/``t`` are literals in specific test modules).

  3. PROMPT (``fixture-prompt``, ``real-prompt``). ``designs`` stores the prompt beside the compiled body, and
     an anatomy body's id is a slug of its prompt, so a gene id can often be read back to the request that
     made it. A slug whose words appear in no test and no script is evidence of a real request; one that
     appears in tests is evidence of a fixture -- weaker than (1)-(2) because a phrase can be shared.

  4. TIMESTAMP. MEASURED and REJECTED as a decision channel. Suite sessions are sparse writers (a full run
     leaves ~32 ``runs`` rows over ~35 min), so clustering them into "session windows" is unstable: at a
     5-minute gap threshold 4 of 101 locomotion rows land inside a window, at 10 minutes 12, at 15 minutes 31.
     The same rows change bucket with an arbitrary parameter, so this script reports the nearest fixture-run
     distance as CONTEXT and never lets it decide. Reporting it as though it decided is how a fixture census
     would acquire false precision.

THE SAME POLLUTION, ONE TABLE OVER (task #278). ``skills`` is the gait flywheel's store; the DESIGN flywheel,
species memory and warm-start read ``runs``, ``designs`` and ``species``, and the compounding ledger reads
``provenance``. Those tables carry the same fixture rows, in bulk -- the suite submits the shipped worked
examples thousands of times, so ``[agent] agent_lynx`` alone is over half of ``runs``. All four are stamped
here, by the SAME channels, with one addition ``skills`` could not use: ``runs``/``designs`` store the PROMPT
VERBATIM, so the phrase channel reads a literal request instead of a slug recovered from a gene id. That is
strictly stronger evidence, and it is what lets a free-text prompt reach ``suite`` at all.

WHERE THE STAMP GOES. ``skills`` has a JSON ``base_config`` to add a key to. The other four do not, and
widening their schema -- or stuffing provenance into ``converged_design``, which ``best_design`` hands to the
co-design search as a warm start -- would put audit metadata inside a payload the product consumes. So those
rows are stamped in a SIDE TABLE, ``row_provenance``, keyed by (table, row key). Additive in the strongest
available sense: the audited rows are never written at all, only read. It also survives writes that would
erase an in-row stamp -- ``species`` is UPSERTed on every run, which would strip a stamp kept in the row the
way ``record_skill`` strips one from ``base_config`` (see ``gait_flywheel._carry_provenance``).

SUITE IS A FLOOR, NEVER A TOTAL. Every count printed here is the number of rows some channel could PROVE the
suite wrote. A row lands in ``unattributed`` when the channels say nothing or disagree -- a prompt that
appears in ``tests/`` AND ``scripts/`` genuinely is unknowable from the text. Those rows are not "probably
real", they are unjudged, and some are certainly the suite's. Read every ``suite=N`` as "at least N".
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sqlite3
from pathlib import Path

#: The stamp, mirroring ``gait_flywheel.BANK_GATE``: one flat string key plus its evidence.
ROW_SOURCE_STAMP = "provenance_v1"
SUITE, REAL, UNATTRIBUTED = "suite", "real", "unattributed"


def source_of(base_config: dict | None) -> str:
    """The bucket a banked row was stamped with. Unstamped rows read ``unattributed`` -- never ``real``.

    Same contract as ``gait_flywheel.gate_of``: absence is not innocence, it is an unanswered question.
    """
    return str((base_config or {}).get("row_source") or UNATTRIBUTED)


# --------------------------------------------------------------------------------------- reading the bank
def _skills(db_path: Path, task_type: str | None = None) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    genes = {}
    for r in conn.execute("SELECT obj_id, meta FROM vectors WHERE obj_type='skill'"):
        meta = json.loads(r["meta"]) if r["meta"] else {}
        if isinstance(meta.get("gene"), dict):
            genes[r["obj_id"]] = meta["gene"]
    q = "SELECT * FROM skills"
    args: tuple = ()
    if task_type:
        q += " WHERE task_type=?"
        args = (task_type,)
    out = []
    for r in conn.execute(q + " ORDER BY created_at", args):
        bc = json.loads(r["base_config"]) if r["base_config"] else {}
        gene = genes.get(r["skill_id"]) or {}
        out.append({"skill_id": r["skill_id"], "robot_class": r["robot_class"], "species": r["species"],
                    "task_type": r["task_type"], "created_at": r["created_at"], "notes": r["notes"] or "",
                    "base_config": bc, "gene_id": gene.get("id") or r["gene_id"]})
    conn.close()
    return out


def _designs_prompts(db_path: Path) -> dict[str, collections.Counter]:
    """gene id -> the prompts that produced a design with that id. The bank's own audit trail."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for prompt, design in conn.execute("SELECT prompt, design FROM designs"):
        try:
            gid = (json.loads(design) or {}).get("id")
        except Exception:  # noqa: BLE001 - a malformed design row is not worth aborting a census over
            continue
        if gid:
            out[gid][prompt or ""] += 1
    conn.close()
    return out


def _fixture_run_times(db_path: Path) -> list[dt.datetime]:
    """When a design that only a test submits was last built. CONTEXT ONLY -- see channel 4 in the header."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    for prompt, cls, ts in conn.execute("SELECT prompt, robot_class, created_at FROM runs ORDER BY created_at"):
        name = (prompt or "")[8:] if (prompt or "").startswith("[agent] ") else None
        if cls == "totally_made_up_xyz" or (name and name in TEST_ONLY_DESIGN_NAMES):
            try:
                out.append(dt.datetime.fromisoformat(ts))
            except Exception:  # noqa: BLE001
                pass
    conn.close()
    return out


# ------------------------------------------------------------------------------ what only the suite builds
#: Design names that reach ``submit_design`` from ``tests/`` and from nowhere else in the repo. The first
#: five are the shipped worked examples in ``agent_design_tools``; the ONLY caller that submits them is
#: ``tests/`` (``get_design_schema()["examples"][...]`` in test_agent_first / test_shipped_examples...).
#: The rest are literals in one or two named test modules. Verified by grep on 2026-08-07, and re-verified on
#: every run by ``_recheck_name_lists`` below, which prints a warning for any name that has stopped appearing
#: where this comment claims it does -- a hardcoded list of "test-only" names that quietly stops being true is
#: exactly how a census acquires false confidence.
TEST_ONLY_DESIGN_NAMES = {
    "agent_lynx", "agent_hexapod", "agent_rover", "agent_scara", "agent_excavator",
    "domed", "octo1", "octo2", "octopus", "t", "x", "bad", "good", "rover", "loader", "test_quad",
}
#: Body classes / species tokens that exist only in ``tests/``.
TEST_ONLY_CLASS_TOKENS = {"totally_made_up_xyz"}
#: Design names seen ONCE each in ``runs`` and present in no test -- an operator poking the product by hand.
#: Evidence of a real (if exploratory) run, not of a customer.
AD_HOC_PROBE_NAMES = {
    "trex", "audit_biped", "fatwheel", "okbot", "portal_gantry", "portal_gantry_audit",
    "delta120", "radial120", "surveyor", "cycler",
}

_SLUG = re.compile(r"^anatomy_(?P<slug>.+?)(?:_[0-9a-f]{8})?$")
#: Names the anatomy compiler invents for itself when the request named no body. ``anatomy_creature`` is the
#: literal "no recognizable robot body plan was found" fallback, so its id says nothing about who asked.
GENERIC_BODY_SLUGS = {"creature", "quadruped", "robot", "walking robot"}


def _slug_of(gene_id: str | None) -> str | None:
    """``anatomy_medium_warehouse_dog_robot_86ba6779`` -> ``medium warehouse dog robot``.

    Anatomy bodies are named from the request that asked for them, which is why a gene id can be read back
    to a prompt at all. Composed/imported bodies (``built_quadruped_18seg``, ``go2``) carry no such trace.
    """
    m = _SLUG.match(gene_id or "")
    if not m:
        return None
    slug = _norm(m.group("slug"))
    return slug or None


def _norm(text: str) -> str:
    """Lowercase, and collapse every run of non-alphanumerics to one space.

    Load-bearing, not cosmetic: a gene id spells the request with underscores
    (``sturdy_four_legged_walking_robot``) and the source spells it with hyphens and spaces
    ("a sturdy four-legged walking robot"). Comparing the raw forms answers "is this prompt in the tree?"
    with a confident NO for every multiword phrase -- which would move rows into ``real`` on a formatting
    artefact. MEASURED: normalising moved 3 of 101 rows out of ``real``.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class _Corpus:
    """The repo's own text, normalised once, so a phrase lookup is a substring test and not a tree walk."""

    def __init__(self, roots: dict[str, Path], *, exclude: set[Path]) -> None:
        self.blobs: dict[str, list[tuple[str, str]]] = {}
        for label, root in roots.items():
            entries = []
            if root.exists():
                for p in sorted(root.rglob("*")):
                    if p.suffix not in {".py", ".json", ".jsonl", ".md"} or "__pycache__" in p.parts:
                        continue
                    if p.resolve() in exclude:
                        continue            # never let THIS script's own docstring count as repo evidence
                    try:
                        entries.append((str(p), _norm(p.read_text(encoding="utf-8", errors="ignore"))))
                    except OSError:
                        continue
            self.blobs[label] = entries

    def where(self, phrase: str) -> dict[str, list[str]]:
        needle = _norm(phrase)
        return {label: [name for name, text in entries if needle in text]
                for label, entries in self.blobs.items()}


def _recheck_name_lists(corpus: "_Corpus") -> list[str]:
    """Re-derive the hardcoded name lists against the working tree, and say so when one has stopped holding.

    ``TEST_ONLY_DESIGN_NAMES`` claims a name reaches ``submit_design`` only from ``tests/``. Two ways that
    rots: the fixture is deleted (the name now proves nothing) or a script/product path starts using it (the
    name now proves the WRONG thing). Only the second is dangerous, so only the second is a warning.
    """
    out = []
    for name in sorted(TEST_ONLY_DESIGN_NAMES):
        hits = corpus.where(f'"{name}"')
        if not hits["tests"] and not hits["src"]:
            out.append(f"'{name}' is listed as a test-only design name but appears in neither tests/ nor src/")
        elif hits["scripts"] and not hits["tests"]:
            out.append(f"'{name}' now appears only in scripts/ -- it is no longer evidence of the suite")
    return out


def _gene_id_bucket(gene_id: str | None) -> tuple[str | None, list[str]]:
    """The verdict a gene id carries on its own, if any. ``(bucket|None, evidence)``.

    ``anatomy_agent_lynx_9debd7b5`` names a shipped worked example that only ``tests/`` submits;
    ``anatomy_trex_...`` names a body an operator typed once. The two name sets are disjoint and the match is
    anchored, so at most one fires. Precedence against the OTHER channels stays with the caller -- a repro hit
    must not be demoted by an operator-probe name.
    """
    gid = gene_id or ""
    for name in sorted(TEST_ONLY_DESIGN_NAMES, key=len, reverse=True):
        if re.match(rf"^anatomy_{re.escape(name)}(_[0-9a-f]{{8}})?$", gid):
            return SUITE, [f"fixture-body:{name}"]
    for name in sorted(AD_HOC_PROBE_NAMES, key=len, reverse=True):
        if re.match(rf"^anatomy_{re.escape(name)}(_[0-9a-f]{{8}})?$", gid):
            return REAL, [f"operator-probe:{name}"]
    return None, []


# ------------------------------------------------------------------------------------------- the verdict
def classify(rows: list[dict], *, reproduced: dict[str, list[str]], prompts: dict[str, collections.Counter],
             repo_root: Path, fixture_times: list[dt.datetime]) -> list[dict]:
    """One verdict per row, with the evidence that produced it. Buckets, never guesses: a row that no channel
    speaks to stays ``unattributed`` rather than being rounded to whichever bucket is convenient."""
    # The provenance tooling names the bodies it is judging, so grepping it finds them. Excluded, or every
    # row this audit discusses would end up "phrase found in tests" -- evidence of nothing but this audit.
    corpus = _Corpus({"tests": repo_root / "tests", "scripts": repo_root / "scripts",
                      "src": repo_root / "src"},
                     exclude={Path(__file__).resolve(),
                              (repo_root / "tests" / "test_bank_provenance.py").resolve()})
    for warning in _recheck_name_lists(corpus):
        print(f"  WARNING {warning}")
    slug_cache: dict[str, dict[str, list[str]]] = {}
    out = []
    for row in rows:
        ev: list[str] = []
        bucket = UNATTRIBUTED

        if row["skill_id"] in reproduced:
            ev.append("repro:" + ",".join(reproduced[row["skill_id"]]))
            bucket = SUITE

        blob = f"{row['robot_class']} {row['species']} {row['gene_id']} {row['skill_id']}"
        for tok in TEST_ONLY_CLASS_TOKENS:
            if tok in blob:
                ev.append(f"fixture-class:{tok}")
                bucket = SUITE

        gid = row["gene_id"] or ""
        gene_bucket, gene_ev = _gene_id_bucket(gid)
        ev.extend(gene_ev)
        if gene_bucket == SUITE:
            bucket = SUITE
        elif gene_bucket == REAL and bucket == UNATTRIBUTED:
            bucket = REAL

        # The prompts that produced a design with this body id. Only decisive when EVERY one of them is a
        # fixture submission -- a body the suite and a real run both reach says nothing on its own.
        pr = prompts.get(gid)
        if pr and bucket == UNATTRIBUTED:
            names = [p[8:] for p in pr if p.startswith("[agent] ")]
            if names and all(n in TEST_ONLY_DESIGN_NAMES for n in names) and len(names) == len(pr):
                ev.append("fixture-prompt:" + ",".join(sorted(set(names))))
                bucket = SUITE

        slug = _slug_of(gid)
        if slug in GENERIC_BODY_SLUGS:
            # ``anatomy_creature`` is what the compiler names a body when the request named none. Grepping
            # the repo for the word "creature" answers a question nobody asked.
            ev.append(f"generic-body-name:{slug!r} (carries no prompt)")
        elif slug:
            if slug not in slug_cache:
                slug_cache[slug] = corpus.where(slug)
            hits = slug_cache[slug]
            if hits["tests"]:
                ev.append(f"prompt-phrase-in-tests:{slug!r}->{len(hits['tests'])} files")
            elif hits["scripts"]:
                ev.append(f"prompt-phrase-only-in-scripts:{slug!r}->{Path(hits['scripts'][0]).name}")
                if bucket == UNATTRIBUTED:
                    bucket = REAL          # a corpus-factory / demo script run: real physics, real body
            elif hits["src"]:
                ev.append(f"prompt-phrase-in-src:{slug!r} (a name the product generates, not a request)")
            else:
                ev.append(f"prompt-phrase-nowhere-in-repo:{slug!r}")
                if bucket == UNATTRIBUTED:
                    bucket = REAL          # an operator typed this request; nothing in the tree asks for it

        # Context only. Never decides -- see channel 4.
        try:
            ts = dt.datetime.fromisoformat(row["created_at"])
            gaps = [abs((t - ts).total_seconds()) for t in fixture_times]
            near = round(min(gaps)) if gaps else None
        except Exception:  # noqa: BLE001
            near = None

        out.append({"skill_id": row["skill_id"], "task_type": row["task_type"], "created_at": row["created_at"],
                    "robot_class": row["robot_class"], "gene_id": row["gene_id"], "bucket": bucket,
                    "evidence": ev, "seconds_to_nearest_fixture_run": near})
    return out


# ----------------------------------------------------------------------------------------- the quarantine
def stamp(db_path: Path, verdicts: list[dict], *, apply: bool) -> int:
    """Write the verdict into ``base_config``. NOTHING IS DELETED and no other field is touched.

    Not idempotent against re-banking: ``MemoryDB.record_skill`` REPLACES ``base_config`` when a run re-banks
    the same ``skill_id`` with an equal-or-better success rate, so a stamped row that is banked again loses
    its stamp. That is the right behaviour -- the re-bank is a new observation whose provenance is whatever
    wrote it -- but it means this script is a pass to re-run, not a one-time migration.
    """
    if not apply:
        return 0
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    now = dt.datetime.now().isoformat(timespec="seconds")
    n = 0
    for v in verdicts:
        row = conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (v["skill_id"],)).fetchone()
        if row is None:
            continue
        bc = json.loads(row["base_config"]) if row["base_config"] else {}
        bc["row_source"] = v["bucket"]
        bc["row_source_stamp"] = ROW_SOURCE_STAMP
        bc["row_source_evidence"] = v["evidence"]
        bc["row_source_stamped_at"] = now
        conn.execute("UPDATE skills SET base_config=? WHERE skill_id=?",
                     (json.dumps(bc), v["skill_id"]))
        n += 1
    conn.commit()
    conn.close()
    return n


# ====================================================================== the design side: runs/designs/species
#
# ``runs`` and ``designs`` are what the DESIGN flywheel reads (``best_design`` -> the co-design warm start,
# ``similar_runs`` -> retrieval, ``training_dataset`` -> the distillation corpus); ``species`` aggregates them
# into per-species bests; ``provenance`` is the compounding ledger. Same three buckets, same additive
# contract, one extra channel: these rows carry the PROMPT, so the phrase lookup reads a literal request.

#: The prompt shape ``submit_design`` / ``train_held`` write (``agent_design_tools`` L558/L973).
_AGENT_PROMPT = re.compile(r"^\[agent(?:-trained)?\]\s+(?P<name>.+)$")
#: Submission names that are a fallback or a robot CLASS rather than a design anybody named -- ``submit_design``
#: writes ``graph.get("name", "design")`` and ``train_held`` writes the robot class. They identify nobody, so
#: like ``anatomy_creature`` they carry no evidence and must not be rounded into either decided bucket.
GENERIC_AGENT_NAMES = {"design", "robot", "quadruped", "biped", "humanoid", "legged", "manipulator",
                       "mobile base", "mobile manipulator", "fixed arm", "creature"}
#: Below this many words a free-text prompt is too generic for the corpus lookup to decide. "a robot dog"
#: appears in 25 test files, 1 script and 4 source files; a phrase that common proves nothing about a writer.
MIN_DECISIVE_PROMPT_WORDS = 4

_TEST_ONLY_NORM = {_norm(n) for n in TEST_ONLY_DESIGN_NAMES}
_AD_HOC_NORM = {_norm(n) for n in AD_HOC_PROBE_NAMES}


def _prompt_bucket(prompt: str | None, corpus: "_Corpus", cache: dict) -> tuple[str | None, list[str]]:
    """The verdict a recorded prompt carries. ``(bucket|None, evidence)``.

    Two shapes. ``[agent] <name>`` is an agent submission and ``<name>`` is the design's own name, so the
    test-only / ad-hoc name lists apply directly -- no slug recovery, no guessing. Anything else is the
    request as the operator (or the test) typed it, and is looked up verbatim in the tree.

    The free-text rule is deliberately asymmetric. tests-only decides ``suite``; tests AND scripts decides
    NOTHING, because a phrase both a fixture and a demo script use cannot say which one ran. That asymmetry is
    why ``suite`` is a floor: a prompt a script builds by concatenation would read as tests-only here, and a
    prompt shared with a script reads as unjudged even when the suite wrote every row of it.
    """
    text = (prompt or "").strip()
    if not text:
        return None, ["no-prompt-recorded"]
    m = _AGENT_PROMPT.match(text)
    if m:
        raw = m.group("name").strip()
        # Matched on the NORMALISED form (``agent_lynx`` and "agent lynx" are the same submission) but
        # reported with the name as recorded, so the evidence string greps back to the row that produced it.
        name = _norm(raw)
        if name in _TEST_ONLY_NORM:
            return SUITE, [f"fixture-design-name:{raw}"]
        if name in _AD_HOC_NORM:
            return REAL, [f"operator-probe-name:{raw}"]
        if name in GENERIC_AGENT_NAMES:
            return None, [f"generic-submission-name:{raw!r} (names no design)"]
        return None, [f"unlisted-submission-name:{raw!r}"]

    words = _norm(text).split()
    if len(words) < MIN_DECISIVE_PROMPT_WORDS:
        return None, [f"prompt-too-generic-to-decide:{text[:40]!r} ({len(words)} words)"]
    if text not in cache:
        cache[text] = corpus.where(text)
    hits = cache[text]
    in_tests, in_scripts, in_src = bool(hits["tests"]), bool(hits["scripts"]), bool(hits["src"])
    if in_tests and in_scripts:
        return None, [f"prompt-verbatim-in-tests-AND-scripts:{len(hits['tests'])}+{len(hits['scripts'])} files"]
    if in_tests:
        return SUITE, [f"prompt-verbatim-only-in-tests:{len(hits['tests'])} files"]
    if in_scripts:
        return REAL, [f"prompt-verbatim-only-in-scripts:{Path(hits['scripts'][0]).name}"]
    if in_src:
        return None, ["prompt-verbatim-in-src (a phrase the product ships, not a request)"]
    return REAL, ["prompt-verbatim-nowhere-in-repo"]


def _combine(bucket: str, new: str | None, ev: list[str], new_ev: list[str], *, strong: bool = False) -> str:
    """Fold one channel's answer into the running verdict. ``suite`` is sticky; ``real`` only fills a gap.

    Sticky ``suite`` is the floor rule in code: once a channel has PROVEN the suite wrote a row, a weaker
    channel that merely fails to see the suite must not talk it back out again.
    """
    ev.extend(new_ev)
    if new == SUITE or (strong and new):
        return new
    if new == REAL and bucket == UNATTRIBUTED:
        return REAL
    return bucket


# ------------------------------------------------------------------------------------------------- readers
def _rows(db_path: Path, sql: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out = [dict(r) for r in conn.execute(sql)]
    conn.close()
    return out


def _gene_id_of(payload) -> str | None:
    """The body id inside a stored ``converged_design`` / ``design`` JSON blob."""
    if not payload:
        return None
    try:
        d = json.loads(payload) if isinstance(payload, str) else payload
    except Exception:  # noqa: BLE001 - a malformed payload is not worth aborting a census over
        return None
    return (d or {}).get("id") if isinstance(d, dict) else None


def read_runs(db_path: Path) -> list[dict]:
    out = []
    for r in _rows(db_path, "SELECT id, prompt, robot_class, species, task_type, converged_design, "
                            "success_rate, design_source, created_at FROM runs ORDER BY id"):
        out.append({"key": str(r["id"]), "prompt": r["prompt"], "robot_class": r["robot_class"],
                    "species": r["species"], "task_type": r["task_type"],
                    "gene_id": _gene_id_of(r["converged_design"]), "success_rate": r["success_rate"],
                    "design_source": r["design_source"], "created_at": r["created_at"]})
    return out


def read_designs(db_path: Path) -> list[dict]:
    out = []
    for r in _rows(db_path, "SELECT id, run_id, prompt, robot_class, task_type, design, success_rate, "
                            "source, created_at FROM designs ORDER BY id"):
        out.append({"key": str(r["id"]), "run_id": r["run_id"], "prompt": r["prompt"],
                    "robot_class": r["robot_class"], "species": None, "task_type": r["task_type"],
                    "gene_id": _gene_id_of(r["design"]), "success_rate": r["success_rate"],
                    "source": r["source"], "created_at": r["created_at"]})
    return out


def read_species(db_path: Path) -> list[dict]:
    out = []
    for r in _rows(db_path, "SELECT name, robot_class, best_success_rate, best_design, runs, updated_at "
                            "FROM species ORDER BY runs DESC"):
        out.append({"key": r["name"], "name": r["name"], "robot_class": r["robot_class"],
                    "best_success_rate": r["best_success_rate"], "best_gene_id": _gene_id_of(r["best_design"]),
                    "runs": r["runs"], "created_at": r["updated_at"]})
    return out


def read_provenance(db_path: Path) -> list[dict]:
    out = []
    for r in _rows(db_path, "SELECT id, child_type, child_id, parent_type, parent_id, kind, delta, meta, "
                            "created_at FROM provenance ORDER BY id"):
        try:
            meta = json.loads(r["meta"]) if r["meta"] else {}
        except Exception:  # noqa: BLE001
            meta = {}
        out.append({"key": str(r["id"]), "child_type": r["child_type"], "child_id": r["child_id"],
                    "parent_type": r["parent_type"], "parent_id": r["parent_id"], "kind": r["kind"],
                    "delta": r["delta"], "prompt": (meta or {}).get("prompt"),
                    "robot_class": (meta or {}).get("robot_class"), "species": None,
                    "gene_id": r["child_id"] if r["child_type"] == "gene" else None,
                    "created_at": r["created_at"]})
    return out


def skill_sources(db_path: Path) -> dict[str, str]:
    """``skill_id -> stamped bucket``, for rows that already carry the ``skills`` stamp. Unstamped reads
    ``unattributed``, so a skill this pass has not judged never lends ``real`` to anything downstream."""
    out = {}
    try:
        rows = _rows(db_path, "SELECT skill_id, base_config FROM skills")
    except sqlite3.OperationalError:
        return {}                    # no skills table: no parent evidence, which lends nothing either way
    for r in rows:
        try:
            bc = json.loads(r["base_config"]) if r["base_config"] else {}
        except Exception:  # noqa: BLE001
            bc = {}
        out[r["skill_id"]] = source_of(bc)
    return out


# --------------------------------------------------------------------------------------------- classifiers
def classify_prompt_rows(rows: list[dict], *, corpus: "_Corpus", parent_sources: dict[str, str] | None = None,
                         reproduced_prompts: set[str] | None = None) -> list[dict]:
    """One verdict per prompt-bearing row (``runs``, ``designs``, ``provenance``).

    Channel order mirrors the skills pass: reproduction, then fixture class token, then the body id, then the
    prompt, then (``provenance`` only) the stamped provenance of the parent skill the edge points at.
    """
    cache: dict[str, dict] = {}
    out = []
    for row in rows:
        ev: list[str] = []
        bucket = UNATTRIBUTED

        if reproduced_prompts and (row.get("prompt") or "") in reproduced_prompts:
            bucket = _combine(bucket, SUITE, ev, [f"repro-prompt:{row['prompt']}"])

        blob = f"{row.get('robot_class')} {row.get('species')} {row.get('gene_id')} {row.get('child_id')}"
        for tok in TEST_ONLY_CLASS_TOKENS:
            if tok in blob:
                bucket = _combine(bucket, SUITE, ev, [f"fixture-class:{tok}"])

        gene_bucket, gene_ev = _gene_id_bucket(row.get("gene_id"))
        bucket = _combine(bucket, gene_bucket, ev, gene_ev)

        p_bucket, p_ev = _prompt_bucket(row.get("prompt"), corpus, cache)
        bucket = _combine(bucket, p_bucket, ev, p_ev)

        if parent_sources and row.get("parent_id") in parent_sources:
            parent = parent_sources[row["parent_id"]]
            if parent != UNATTRIBUTED:
                bucket = _combine(bucket, parent, ev, [f"parent-{row.get('parent_type')}-is-{parent}"])

        out.append({**row, "bucket": bucket, "evidence": ev})
    return out


def classify_species_rows(rows: list[dict], run_verdicts: list[dict]) -> list[dict]:
    """A species is judged by the runs that reference it -- it has no prompt of its own.

    ``species.runs`` is a counter bumped once per ``record_run``, and ``best_design``/``best_success_rate`` are
    whichever run scored highest, so a species IS its runs. Decided only when the referencing runs agree:
    every judged run ``suite`` and none ``real`` -> ``suite``, and vice versa. Anything mixed stays
    ``unattributed`` and carries the split in its evidence, because "62% of this species' runs are the
    suite's" is the honest statement and "this species is a fixture" is not.
    """
    by_species: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in run_verdicts:
        by_species[str(v.get("species"))][v["bucket"]] += 1
    out = []
    for row in rows:
        ev: list[str] = []
        bucket = UNATTRIBUTED
        for tok in TEST_ONLY_CLASS_TOKENS:
            if tok in f"{row['name']} {row['robot_class']}":
                bucket = _combine(bucket, SUITE, ev, [f"fixture-class:{tok}"])
        gene_bucket, gene_ev = _gene_id_bucket(row.get("best_gene_id"))
        if gene_bucket:
            ev.append(f"best_design-{gene_ev[0]}")
            if gene_bucket == SUITE:
                bucket = SUITE                       # the species' own headline design is a fixture body
        c = by_species.get(row["name"], collections.Counter())
        total = sum(c.values())
        ev.append(f"runs-referencing: suite={c[SUITE]} real={c[REAL]} unattributed={c[UNATTRIBUTED]}")
        if total and bucket == UNATTRIBUTED:
            if c[SUITE] and not c[REAL]:
                bucket = SUITE
            elif c[REAL] and not c[SUITE]:
                bucket = REAL
        if not total:
            ev.append("no runs reference this species")
        out.append({**row, "bucket": bucket, "evidence": ev,
                    "suite_run_fraction": round(c[SUITE] / total, 4) if total else None})
    return out


# ------------------------------------------------------------------------------------ the side-table stamp
ROW_PROVENANCE_TABLE = "row_provenance"
DESIGN_SIDE_TABLES = ("runs", "designs", "species", "provenance")


def ensure_row_provenance(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {ROW_PROVENANCE_TABLE} (
        table_name TEXT NOT NULL,
        row_key    TEXT NOT NULL,
        row_source TEXT NOT NULL,
        row_source_stamp TEXT NOT NULL,
        row_source_evidence TEXT NOT NULL,
        row_source_stamped_at TEXT NOT NULL,
        row_source_first_stamped_at TEXT NOT NULL,
        PRIMARY KEY (table_name, row_key))""")


def stamp_table(db_path: Path, table: str, verdicts: list[dict], *, apply: bool) -> dict:
    """Write the verdicts for one table into ``row_provenance``. The audited table is never written.

    Re-running is expected (new evidence, e.g. a fresh repro bank). A re-stamp refreshes the verdict, with one
    guard: a row already stamped ``suite`` is never downgraded to ``unattributed``. Losing a proof because a
    later pass ran without the repro db that produced it is exactly how a quarantine would leak.
    """
    counts = collections.Counter(v["bucket"] for v in verdicts)
    if not apply:
        return {"table": table, "written": 0, "buckets": dict(counts)}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_row_provenance(conn)
    now = dt.datetime.now().isoformat(timespec="seconds")
    prior = {r["row_key"]: r for r in conn.execute(
        f"SELECT * FROM {ROW_PROVENANCE_TABLE} WHERE table_name=?", (table,))}
    written = retained = 0
    for v in verdicts:
        key, bucket, ev = str(v["key"]), v["bucket"], list(v["evidence"])
        was = prior.get(key)
        if was is not None and was["row_source"] == SUITE and bucket == UNATTRIBUTED:
            bucket, retained = SUITE, retained + 1
            ev.append("retained-prior-suite-verdict (a stamp is never downgraded to unattributed)")
        first = was["row_source_first_stamped_at"] if was is not None else now
        conn.execute(f"INSERT OR REPLACE INTO {ROW_PROVENANCE_TABLE} VALUES (?,?,?,?,?,?,?)",
                     (table, key, bucket, ROW_SOURCE_STAMP, json.dumps(ev), now, first))
        written += 1
    conn.commit()
    conn.close()
    return {"table": table, "written": written, "retained_suite": retained, "buckets": dict(counts)}


def row_sources(db_path: Path, table: str) -> dict[str, str]:
    """``row key -> bucket`` for one stamped table. A key with no stamp is simply absent; callers must read
    an absent key as ``unattributed`` (``source_of_key`` does), never as ``real``."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"SELECT row_key, row_source FROM {ROW_PROVENANCE_TABLE} WHERE table_name=?",
                            (table,)).fetchall()
    except sqlite3.OperationalError:
        rows = []                                     # never stamped: every row is an open question
    conn.close()
    return {str(k): str(v) for k, v in rows}


def source_of_key(stamps: dict[str, str], key) -> str:
    """Same contract as ``source_of``: absence is an unanswered question, not innocence."""
    return stamps.get(str(key), UNATTRIBUTED)


def census(verdicts: list[dict], by: str | None = None) -> dict:
    """Bucket counts, optionally split by a field. ``suite`` is a FLOOR in every cell."""
    if by is None:
        return dict(collections.Counter(v["bucket"] for v in verdicts))
    grouped: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in verdicts:
        grouped[str(v.get(by))][v["bucket"]] += 1
    return {k: dict(c) for k, c in sorted(grouped.items(), key=lambda kv: -sum(kv[1].values()))}


def classify_design_side(db_path: Path, repo_root: Path) -> dict[str, list[dict]]:
    """Every design-side table, classified in dependency order (species needs the run verdicts)."""
    corpus = _Corpus({"tests": repo_root / "tests", "scripts": repo_root / "scripts", "src": repo_root / "src"},
                     exclude={Path(__file__).resolve(),
                              (repo_root / "tests" / "test_bank_provenance.py").resolve(),
                              (repo_root / "scripts" / "audit_design_memory.py").resolve()})
    runs = classify_prompt_rows(read_runs(db_path), corpus=corpus)
    designs = classify_prompt_rows(read_designs(db_path), corpus=corpus)
    species = classify_species_rows(read_species(db_path), runs)
    prov = classify_prompt_rows(read_provenance(db_path), corpus=corpus,
                                parent_sources=skill_sources(db_path))
    return {"runs": runs, "designs": designs, "species": species, "provenance": prov}


def _reproduced(paths: list[str]) -> dict[str, list[str]]:
    """skill_id -> which controlled pytest bank(s) produced it."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for raw in paths:
        p = Path(raw)
        db = p / "virturoid_memory.db" if p.is_dir() else p
        if not db.exists():
            print(f"  (repro db missing, skipped: {db})")
            continue
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for (sid,) in conn.execute("SELECT skill_id FROM skills"):
            out[sid].append(p.name)
        conn.close()
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="build/memory")
    ap.add_argument("--repro-db", action="append", default=[],
                    help="a bank produced by pytest under VIRTUROID_MEMORY_DIR (dir or .db). Repeatable.")
    ap.add_argument("--task-type", default=None, help="restrict to one task_type (default: every skill)")
    ap.add_argument("--tables", default="skills",
                    help="comma-separated: skills,runs,designs,species,provenance (default: skills)")
    ap.add_argument("--apply", action="store_true", help="write the stamp (default: dry run, report only)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    db = Path(args.memory) / "virturoid_memory.db"
    repo_root = Path(__file__).resolve().parent.parent
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    unknown = [t for t in tables if t not in ("skills", *DESIGN_SIDE_TABLES)]
    if unknown:
        raise SystemExit(f"unknown table(s): {unknown}")

    design_side = {}
    if any(t in DESIGN_SIDE_TABLES for t in tables):
        print("=== design side (runs / designs / species / provenance) — 'suite' is a FLOOR in every cell\n")
        design_side = classify_design_side(db, repo_root)
        for name in DESIGN_SIDE_TABLES:
            if name not in tables:
                continue
            v = design_side[name]
            print(f"  {name}: {len(v)} rows | " + " ".join(f"{b}={census(v).get(b, 0)}"
                                                           for b in (SUITE, REAL, UNATTRIBUTED)))
            res = stamp_table(db, name, v, apply=args.apply)
            if args.apply:
                print(f"    stamped {res['written']} rows into {ROW_PROVENANCE_TABLE} "
                      f"(retained prior suite: {res.get('retained_suite', 0)}); nothing in {name} was written")
        print()

    if "skills" not in tables:
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps({k: v for k, v in design_side.items() if k in tables},
                                                  indent=2, default=str), encoding="utf-8")
            print(f"wrote {args.json}")
        if not args.apply:
            print("DRY RUN - no write. Pass --apply to stamp.")
        return

    rows = _skills(db, args.task_type)
    verdicts = classify(rows, reproduced=_reproduced(args.repro_db), prompts=_designs_prompts(db),
                        repo_root=repo_root, fixture_times=_fixture_run_times(db))

    by_task: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for v in verdicts:
        by_task[v["task_type"]][v["bucket"]] += 1
    print(json.dumps({t: dict(c) for t, c in by_task.items()}, indent=2))
    print(f"\n{len(verdicts)} rows | " + " ".join(f"{b}={sum(c[b] for c in by_task.values())}"
                                                  for b in (SUITE, REAL, UNATTRIBUTED)))
    for v in verdicts:
        print(f"  {v['bucket']:<13} {v['created_at'][:19]} {str(v['gene_id'])[:38]:<38} "
              f"{'; '.join(v['evidence'])[:110]}")

    n = stamp(db, verdicts, apply=args.apply)
    print(f"\n{'STAMPED ' + str(n) + ' rows in place (nothing deleted)' if args.apply else 'DRY RUN - no write. Pass --apply to stamp.'}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
