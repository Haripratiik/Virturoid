"""Decide, for every banked skill, whether the SUITE wrote it or a real run did -- and stamp the answer in place.

    PYTHONPATH=src python scripts/bank_provenance.py [--memory build/memory]
        [--repro-db DIR_OR_DB ...] [--apply] [--json OUT]

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
        for name in sorted(TEST_ONLY_DESIGN_NAMES, key=len, reverse=True):
            if re.match(rf"^anatomy_{re.escape(name)}(_[0-9a-f]{{8}})?$", gid):
                ev.append(f"fixture-body:{name}")
                bucket = SUITE
                break
        for name in sorted(AD_HOC_PROBE_NAMES, key=len, reverse=True):
            if re.match(rf"^anatomy_{re.escape(name)}(_[0-9a-f]{{8}})?$", gid):
                ev.append(f"operator-probe:{name}")
                if bucket == UNATTRIBUTED:
                    bucket = REAL
                break

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
    ap.add_argument("--apply", action="store_true", help="write the stamp (default: dry run, report only)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    db = Path(args.memory) / "virturoid_memory.db"
    repo_root = Path(__file__).resolve().parent.parent
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
