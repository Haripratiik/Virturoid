"""Re-measure every design-side memory/flywheel metric with the fixture rows in, and with them out.

    PYTHONPATH=src python scripts/audit_design_memory.py [--memory build/memory] [--work DIR] [--json OUT]

WHY. ``scripts/bank_provenance.py`` decides who wrote each banked row. That census is only worth having if
somebody then asks the question it exists for: DOES IT CHANGE ANY NUMBER WE QUOTE? On the gait side it did --
the fixture rows were the bank's best performers, and three parameters that appeared to pass the evidence
gates were carried entirely by them. ``runs``, ``designs`` and ``species`` feed the DESIGN flywheel, species
memory and warm-start, so every design-side memory number inherits the same anchoring risk. This script
answers it by measurement rather than by argument.

HOW. Three arms, each a full COPY of the bank (the real one is opened read-only and never written):

  * ``full``       -- every row, i.e. what the product reads today;
  * ``ex_strict``  -- drop only rows a PROOF-GRADE channel attributes to the suite (a reproduced id, the
                      ``totally_made_up_xyz`` class, or a body/design named after a fixture the tests submit
                      and nothing else does). The prompt-phrase channel is excluded here on purpose;
  * ``ex_full``    -- drop every row stamped ``suite``, phrase channel included.

Reporting both exclusions is the point of the second arm: if a number only moves under ``ex_full`` then the
move rests on the weakest channel, and that is worth knowing before anyone re-quotes it.

The metrics are then computed through the PRODUCTION code (``MemoryDB.best_design`` / ``similar_runs`` /
``training_dataset``, ``flywheel_status.moat_status``, ``RoboticsVectorMemory.compounding_summary``) rather
than re-implemented here, so a metric that changes shape later cannot silently stop being audited.

SUITE IS A FLOOR, IN EVERY ROW OF EVERY TABLE BELOW. An excluded-arm number is what the metric reads with
the PROVABLE fixture rows removed. Rows the channels could not judge stay in, and some of them are certainly
the suite's -- so each ``ex_`` column is the smallest correction the evidence supports, not the true one.

``--work`` holds the three copies and is rewritten on every run; give two concurrent runs different ones.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bank_provenance as BP  # noqa: E402

#: Evidence prefixes that identify a row by NAME or by reproduction rather than by a shared phrase. These are
#: the channels that cannot be defeated by an operator happening to type what a test types.
PROOF_GRADE = ("repro:", "repro-prompt:", "fixture-class:", "fixture-body:", "fixture-design-name:")

#: Paraphrases, not the banked prompts -- retrieval's real job is the request nobody has typed before. Asking
#: with the verbatim stored prompt would measure exact match and call it retrieval.
QUERY_BATTERY = [
    ("a four-legged robot that walks over rough terrain", "quadruped"),
    ("a warehouse patrol robot with four legs", "quadruped"),
    ("a robot dog for inspection rounds", "quadruped"),
    ("a tabletop arm that picks up blocks", "manipulator"),
    ("a wheeled delivery robot for an office", "mobile_base"),
    ("a humanoid that stacks boxes onto a pallet", "humanoid"),
    ("a six-legged walking robot", "legged"),
    ("a robot that carries parcels around a building", None),
]


# ------------------------------------------------------------------------------------------- the three arms
def _is_proof_grade(evidence) -> bool:
    return any(str(e).startswith(PROOF_GRADE) for e in (evidence or []))


def build_arms(src_db: Path, work: Path, verdicts: dict) -> dict[str, Path]:
    """Copy the bank three times and delete the fixture rows from two of the copies.

    Deleting is safe HERE and only here: these are throwaway copies in a scratch directory. The real bank is
    never opened for writing by this script -- the whole point of the additive stamp is that the destructive
    experiment can be run somewhere else.
    """
    arms = {}
    for arm in ("full", "ex_strict", "ex_full"):
        # one directory per arm, with the bank under its canonical name: ``moat_status`` takes a memory DIR
        # and looks for ``virturoid_memory.db`` inside it, so the arm has to look like a workspace.
        (work / arm).mkdir(parents=True, exist_ok=True)
        dst = work / arm / "virturoid_memory.db"
        dst.unlink(missing_ok=True)
        con = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
        out = sqlite3.connect(dst)
        con.backup(out)
        con.close()
        if arm != "full":
            strict = arm == "ex_strict"
            for table, key in (("runs", "id"), ("designs", "id"), ("provenance", "id")):
                drop = [v["key"] for v in verdicts[table]
                        if v["bucket"] == BP.SUITE and (_is_proof_grade(v["evidence"]) or not strict)]
                out.executemany(f"DELETE FROM {table} WHERE {key}=?", [(int(k),) for k in drop])
                if table == "runs":
                    # the run sub-space of the vector index points at run ids; leaving the vectors behind
                    # would let a deleted fixture keep occupying a kNN slot it can no longer fill
                    out.executemany("DELETE FROM vectors WHERE obj_type='run' AND obj_id=?",
                                    [(str(k),) for k in drop])
        # EVERY arm gets the same rebuild, including ``full``. ``species`` is an accumulator, so an excluded
        # arm has to be re-derived; comparing a re-derived column against the stored one would report the
        # rebuild as if it were the exclusion. The stored-vs-rebuilt drift is reported separately instead.
        _rebuild_species(out)
        out.commit()
        out.close()
        arms[arm] = dst
    return arms


def _rebuild_species(con: sqlite3.Connection) -> None:
    """Recompute the ``species`` aggregate from the runs that survived.

    ``species`` stores a counter and a best-so-far, both accumulated one ``record_run`` at a time, so it
    cannot be filtered -- it has to be re-derived. A species with no surviving run is dropped: it existed
    only because the suite built it.
    """
    rows = con.execute("SELECT species, robot_class, success_rate, converged_design, created_at FROM runs "
                       "WHERE species IS NOT NULL ORDER BY id").fetchall()
    best: dict[str, dict] = {}
    for sp, cls, sr, design, ts in rows:
        e = best.setdefault(sp, {"cls": cls, "rate": -1.0, "design": None, "n": 0, "ts": ts})
        e["n"] += 1
        e["ts"] = ts
        if (sr if sr is not None else -1.0) >= e["rate"]:
            e["rate"], e["design"] = (sr if sr is not None else -1.0), design
    con.execute("DELETE FROM species")
    for sp, e in best.items():
        con.execute("INSERT INTO species (name, robot_class, best_success_rate, best_design, runs, updated_at) "
                    "VALUES (?,?,?,?,?,?)", (sp, e["cls"], max(e["rate"], 0.0) if e["rate"] >= 0 else None,
                                             e["design"], e["n"], e["ts"]))


# ----------------------------------------------------------------------------------------------- the metrics
def _db(path: Path):
    from virturoid.services.memory_db import MemoryDB
    return MemoryDB(path)


def metric_counts(path: Path) -> dict:
    with _db(path) as db:
        s = db.stats()
    return {k: s[k] for k in ("runs", "designs", "species", "skills", "lessons")}


def metric_best_design(path: Path, pairs) -> dict:
    """``best_design(class, task)`` IS the design flywheel's warm start (``design_flywheel`` L34/L114): the
    body the next co-design search starts from. If this row changes, the next build starts somewhere else."""
    out = {}
    with _db(path) as db:
        for cls, task in pairs:
            r = db.best_design(cls, task)
            if r is None:
                out[f"{cls}/{task}"] = None
                continue
            gid = None
            cd = r.get("converged_design")
            if isinstance(cd, dict):
                gid = cd.get("id")
            out[f"{cls}/{task}"] = {"run_id": r.get("id"), "prompt": r.get("prompt"),
                                    "success_rate": r.get("success_rate"), "gene_id": gid}
    return out


def metric_transfer_candidates(path: Path, classes) -> dict:
    out = {}
    with _db(path) as db:
        for cls in classes:
            out[cls] = [{"key": c["key"], "prompt": c["prompt"], "success_rate": c["success_rate"]}
                        for c in db.transfer_candidates(cls)]
    return out


def metric_similar_runs(path: Path, stamps: dict[str, str]) -> dict:
    """The retrieval seam (``agent_tools.recall_similar`` / ``memory_cli``): what prior work a NEW request is
    shown. Reported as the top hit plus the provenance mix of the top 5."""
    out = {}
    with _db(path) as db:
        for prompt, cls in QUERY_BATTERY:
            hits = db.similar_runs(prompt, robot_class=cls, limit=5)
            mix = collections.Counter(BP.source_of_key(stamps, h["id"]) for h in hits)
            top = hits[0] if hits else None
            out[prompt] = {"n_hits": len(hits),
                           "top1_run_id": (top or {}).get("id"),
                           "top1_prompt": (top or {}).get("prompt"),
                           "top1_similarity": (top or {}).get("similarity"),
                           "top1_source": BP.source_of_key(stamps, (top or {}).get("id")) if top else None,
                           "top5_mix": dict(mix)}
    return out


def metric_training_dataset(path: Path) -> dict:
    """The distillation corpus (``proposer_distill`` / ``export_training_jsonl``) -- the rows a fine-tune of
    the design proposer would see."""
    out = {}
    with _db(path) as db:
        for floor in (0.0, 0.8):
            rows = db.training_dataset(min_success=floor)
            rates = [r["success_rate"] for r in rows if r["success_rate"] is not None]
            out[f"min_success>={floor}"] = {
                "rows": len(rows),
                "mean_success": round(sum(rates) / len(rates), 4) if rates else None,
                "distinct_prompts": len({r["prompt"] for r in rows}),
                "distinct_classes": len({r["robot_class"] for r in rows})}
    return out


def metric_species(path: Path) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("SELECT name, runs, best_success_rate FROM species ORDER BY runs DESC").fetchall()
    con.close()
    return {"n_species": len(rows), "total_species_runs": sum(r[1] or 0 for r in rows),
            "per_species": {r[0]: {"runs": r[1], "best_success_rate": r[2]} for r in rows}}


def metric_compounding(path: Path) -> dict:
    """The compounding ledger -- ``moat.json`` quotes its edge count and positive fraction as the moat's
    headline, and ``flywheel_status.warm_start.provenance_reuse_edges`` re-exports it."""
    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
    with _db(path) as db:
        summary = RoboticsVectorMemory(db).compounding_summary()
        by_kind = {}
        for r in db.conn.execute(
                "SELECT kind, COUNT(*) n, AVG(delta) d, SUM(CASE WHEN delta>0 THEN 1 ELSE 0 END) w "
                "FROM provenance GROUP BY kind").fetchall():
            by_kind[r["kind"]] = {"edges": int(r["n"]),
                                  "mean_delta": round(float(r["d"]), 6) if r["d"] is not None else None,
                                  "positive_fraction": round(int(r["w"] or 0) / int(r["n"]), 4)}
    return {"summary": summary, "by_kind": by_kind}


def metric_moat_status(path: Path) -> dict:
    from virturoid.services.flywheel_status import moat_status
    st = moat_status(path.parent)
    return {"counts": st["counts"], "compounding": st["compounding"], "accumulating": st["accumulating"],
            "warm_start": {k: st["warm_start"][k] for k in ("trainable_builds", "warm_started_builds",
                                                            "utilization", "provenance_reuse_edges")},
            "summary": st["summary"]}


# ------------------------------------------------------- the file-backed memory, which has no stamp at all
def audit_file_memory(memory_dir: Path, repo_root: Path, pairs) -> dict:
    """``memory_store.find_similar_design`` -- ``autonomous_build`` L343's warm start -- is NOT in the
    database. It is one JSON file per (class, task) in the same directory, so the stamp cannot reach it and
    the only available attribution is the prompt it recorded."""
    corpus = BP._Corpus({"tests": repo_root / "tests", "scripts": repo_root / "scripts",
                         "src": repo_root / "src"},
                        exclude={Path(BP.__file__).resolve(), Path(__file__).resolve(),
                                 (repo_root / "tests" / "test_bank_provenance.py").resolve()})
    cache: dict = {}
    out = {}
    for path in sorted(Path(memory_dir).glob("*__*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        bucket, ev = BP._prompt_bucket(rec.get("prompt"), corpus, cache)
        out[path.name] = {"prompt": rec.get("prompt"), "success_rate": rec.get("success_rate"),
                          "bucket": bucket or BP.UNATTRIBUTED, "evidence": ev}
    # HIT RATE. ``find_similar_design(class, task)`` is what ``autonomous_build`` L343 warm-starts from. It is
    # a per-(class, task) file with a same-class fallback, so "hit" means the builder found ANY prior design.
    # Excluded = the suite-authored records treated as absent, which is what removing them would do.
    from virturoid.services import memory_store
    hits_full = hits_ex = 0
    per_pair = {}
    suite_files = {n for n, v in out.items() if isinstance(v, dict) and v.get("bucket") == BP.SUITE}
    for cls, task in pairs:
        rec = memory_store.find_similar_design(cls, task, Path(memory_dir))
        name = f"{cls}__{task}.json".replace("/", "_")
        # the fallback can return a record from a DIFFERENT file, so identify the record by its prompt
        served = next((n for n, v in out.items()
                       if isinstance(v, dict) and rec is not None and v.get("prompt") == rec.get("prompt")),
                      None)
        hit = rec is not None
        clean = hit and served not in suite_files
        hits_full += int(hit)
        hits_ex += int(clean)
        per_pair[f"{cls}/{task}"] = {"hit": hit, "served_by": served or name,
                                     "hit_after_excluding_suite": clean}
    out["_hit_rate"] = {"pairs": len(pairs), "hits_full": hits_full,
                        "hits_excluding_suite_records": hits_ex,
                        "rate_full": round(hits_full / len(pairs), 4) if pairs else None,
                        "rate_excluded": round(hits_ex / len(pairs), 4) if pairs else None,
                        "per_pair": per_pair}
    arch = Path(memory_dir) / "design_archive.json"
    if arch.exists():
        try:
            cells = (json.loads(arch.read_text(encoding="utf-8")) or {}).get("cells", [])
        except Exception:  # noqa: BLE001
            cells = []
        mix = collections.Counter()
        for c in cells:
            b, _ = BP._gene_id_bucket(((c.get("gene") or {}).get("id")))
            mix[b or BP.UNATTRIBUTED] += 1
        out["design_archive.json"] = {"niches": len(cells),
                                      "qd_score": round(sum(float(c.get("score") or 0) for c in cells), 4),
                                      "bucket_mix": dict(mix)}
    return out


# ------------------------------------------------------------------------------------------------- reporting
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def print_table(title: str, rows: dict[str, dict], arms) -> None:
    print(f"\n--- {title}   (suite is a FLOOR: each ex_ column is the SMALLEST correction the evidence supports)")
    width = max((len(k) for k in rows), default=10)
    print(f"  {'metric'.ljust(width)} | " + " | ".join(a.ljust(22) for a in arms) + " | moves?")
    for k, per_arm in rows.items():
        cells = [_fmt(per_arm.get(a)) for a in arms]
        moved = "MOVES" if len({str(c) for c in cells}) > 1 else "."
        print(f"  {k.ljust(width)} | " + " | ".join(c[:22].ljust(22) for c in cells) + f" | {moved}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="build/memory")
    ap.add_argument("--work", default="build/design_memory_audit")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    memory_dir = Path(args.memory)
    src = memory_dir / "virturoid_memory.db"
    repo_root = Path(__file__).resolve().parent.parent
    work = Path(args.work)

    print("classifying the design-side tables ...")
    verdicts = BP.classify_design_side(src, repo_root)
    for table in ("runs", "designs", "species", "provenance"):
        c = BP.census(verdicts[table])
        strict = sum(1 for v in verdicts[table] if v["bucket"] == BP.SUITE and _is_proof_grade(v["evidence"]))
        print(f"  {table:<11} {len(verdicts[table]):>5} rows | suite={c.get('suite', 0)} "
              f"(proof-grade {strict}) real={c.get('real', 0)} unattributed={c.get('unattributed', 0)}")

    arms = build_arms(src, work, verdicts)
    names = list(arms)
    stamps = {a: BP.row_sources(src, "runs") for a in names}     # provenance of a run id is arm-independent

    pairs, classes = set(), set()
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    for cls, task in con.execute("SELECT DISTINCT robot_class, task_type FROM runs"):
        pairs.add((cls, task))
        classes.add(cls)
    con.close()
    pairs, classes = sorted(pairs), sorted(classes)

    results = {a: {} for a in names}
    for arm, path in arms.items():
        results[arm]["counts"] = metric_counts(path)
        results[arm]["best_design"] = metric_best_design(path, pairs)
        results[arm]["transfer_candidates"] = metric_transfer_candidates(path, classes)
        results[arm]["similar_runs"] = metric_similar_runs(path, stamps[arm])
        results[arm]["training_dataset"] = metric_training_dataset(path)
        results[arm]["species"] = metric_species(path)
        results[arm]["compounding"] = metric_compounding(path)
        results[arm]["moat_status"] = metric_moat_status(path)

    flat: dict[str, dict] = collections.defaultdict(dict)
    for arm in names:
        r = results[arm]
        for k, v in r["counts"].items():
            flat[f"counts.{k}"][arm] = v
        for k, v in r["training_dataset"].items():
            flat[f"training_dataset[{k}].rows"][arm] = v["rows"]
            flat[f"training_dataset[{k}].mean_success"][arm] = v["mean_success"]
            flat[f"training_dataset[{k}].distinct_prompts"][arm] = v["distinct_prompts"]
        flat["species.n_species"][arm] = r["species"]["n_species"]
        flat["species.total_runs"][arm] = r["species"]["total_species_runs"]
        cs = r["compounding"]["summary"]
        for k in ("edges", "seeded_builds", "measured_deltas", "mean_delta", "positive_fraction"):
            flat[f"compounding.{k}"][arm] = cs[k]
        for kind, kv in sorted(r["compounding"]["by_kind"].items()):
            flat[f"provenance[{kind}].edges"][arm] = kv["edges"]
            flat[f"provenance[{kind}].mean_delta"][arm] = kv["mean_delta"]
            flat[f"provenance[{kind}].positive_fraction"][arm] = kv["positive_fraction"]
        ws = r["moat_status"]["warm_start"]
        for k, v in ws.items():
            flat[f"moat.warm_start.{k}"][arm] = v
        flat["moat.compounding"][arm] = r["moat_status"]["compounding"]
    print_table("headline counts, corpora and ledgers", dict(flat), names)

    bd: dict[str, dict] = collections.defaultdict(dict)
    for arm in names:
        for key, v in results[arm]["best_design"].items():
            bd[f"{key} sr"][arm] = None if v is None else f"run{v['run_id']} sr={v['success_rate']}"
            bd[f"{key} body"][arm] = None if v is None else (v["gene_id"] or (v["prompt"] or "")[:22])
    print_table("THE WARM START: best_design(class, task) -- what the next co-design search starts from",
                {k: bd[k] for k in sorted(bd)}, names)

    sr: dict[str, dict] = collections.defaultdict(dict)
    for arm in names:
        for q, v in results[arm]["similar_runs"].items():
            sr[f"{q[:42]} | top1"][arm] = f"run{v['top1_run_id']}/{v['top1_source']}"
            sr[f"{q[:42]} | suite in topK"][arm] = f"{v['top5_mix'].get('suite', 0)}/{v['n_hits']}"
    print_table("RETRIEVAL: similar_runs(prompt) -- the prior work a new request is shown",
                {k: sr[k] for k in sorted(sr)}, names)

    print("\n--- WHO SCORES BETTER? the design-side analogue of the gait finding (fixture rows were the BEST)")
    print("    success_rate of runs by stamped provenance, in the FULL bank. NULL rates are the agent")
    print("    submissions that never ran a task and are excluded from training_dataset by the >= filter.")
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    by_src: dict[str, list] = collections.defaultdict(list)
    nulls: collections.Counter = collections.Counter()
    for rid, rate in con.execute("SELECT id, success_rate FROM runs"):
        b = BP.source_of_key(stamps["full"], rid)
        (by_src[b].append(float(rate)) if rate is not None else nulls.update([b]))
    con.close()
    print(f"    {'bucket':<14} {'n_scored':>9} {'mean':>8} {'median':>8} {'>=0.8':>8} {'null rate':>10}")
    scored_rows = {}
    for b in (BP.SUITE, BP.REAL, BP.UNATTRIBUTED):
        v = sorted(by_src.get(b, []))
        if not v:
            print(f"    {b:<14} {'0':>9}")
            continue
        med = v[len(v) // 2]
        hi = sum(1 for x in v if x >= 0.8) / len(v)
        scored_rows[b] = {"n": len(v), "mean": round(sum(v) / len(v), 4), "median": round(med, 4),
                          "frac_ge_0.8": round(hi, 4), "null_rate_rows": nulls[b]}
        print(f"    {b:<14} {len(v):>9} {sum(v) / len(v):>8.4f} {med:>8.4f} {hi:>8.1%} {nulls[b]:>10}")

    sp: dict[str, dict] = collections.defaultdict(dict)
    for arm in names:
        per = results[arm]["species"]["per_species"]
        for name in sorted(set().union(*(set(results[a]["species"]["per_species"]) for a in names))):
            e = per.get(name)
            sp[f"{name} runs"][arm] = "-" if e is None else e["runs"]
            sp[f"{name} best"][arm] = "-" if e is None else e["best_success_rate"]
    print_table("SPECIES MEMORY: species_usage() is the trim guard -- a node with builds is never trimmed "
                "(species_maintenance L89/L101)", {k: sp[k] for k in sorted(sp)}, names)

    print("\n--- species: what the STORED accumulator says vs what its own runs say (all three arms rebuilt)")
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    stored = {n: r for n, r in con.execute("SELECT name, runs FROM species")}
    con.close()
    rebuilt = results["full"]["species"]["per_species"]
    drift = {n: (stored.get(n), (rebuilt.get(n) or {}).get("runs"))
             for n in sorted(set(stored) | set(rebuilt))
             if stored.get(n) != (rebuilt.get(n) or {}).get("runs")}
    print(f"    stored total={sum(stored.values())}  rebuilt-from-runs total="
          f"{results['full']['species']['total_species_runs']}  species with a drifting counter: {len(drift)}")
    for n, (a, b) in drift.items():
        print(f"      {n:<32} stored runs={a}  runs table says={b}")

    print("\n--- file-backed memory (memory_store.find_similar_design + the MAP-Elites archive)")
    print("    NOT in the database, so the row stamp cannot reach it; attributed by recorded prompt / gene id")
    fm = audit_file_memory(memory_dir, repo_root, pairs)
    for name, v in fm.items():
        print(f"  {name:<42} {json.dumps(v, default=str)[:150]}")

    payload = {"census": {t: BP.census(verdicts[t]) for t in verdicts}, "arms": results, "file_memory": fm}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
