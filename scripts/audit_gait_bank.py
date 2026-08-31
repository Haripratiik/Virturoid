"""Audit a gait bank: what is in it, which rows still walk, which survive the fragility gate, and whether any
parameter shows minable structure once the fragile rows are removed.

    PYTHONPATH=src python scripts/audit_gait_bank.py [--memory build/memory] [--remeasure] [--gates] [--json OUT]

WHY THIS EXISTS. The bank is the flywheel's evidence base, and until 2026-08-04 nothing screened what went into
it. A row banked before that date, or through one of the three ``bank_gait`` call sites that measure no
robustness margin (``ai_native_tools._auto_bank_gait``, ``r2prime.seed_corpus``, ``reward_loop``), records an
operating point about which the fragility question was never asked. Those rows are not interchangeable with
gated ones, and pooling them is how a coordinate the controller never read (``duty``) came to be reported to
operators as the bank's tightest-clustered parameter (#265/#266). ``bank_gate`` now stamps the distinction; this
script reads it, and ``--remeasure`` re-derives it from physics for the rows that predate the stamp.

WHO WROTE THESE ROWS. Until 2026-08-07 the test suite banked into this same database — ``verify_robot`` ->
``_auto_bank_gait`` -> ``bank_gait`` is the ordinary product path and the DB path was a constant — so the
corpus contains fixture bodies that every previous measurement counted as observations. Four rows carry the
body class ``totally_made_up_xyz`` outright. ``scripts/bank_provenance.py`` attributes each row and stamps
``base_config['row_source']``; the census below reports the split so the next reader gets provenance without
redoing the attribution. ``unattributed`` is a real third answer, not a synonym for "real": it means no
channel spoke to that row, and on the live bank it is the LARGEST bucket.

    (default)     STATIC census — rows, provenance, gate split, junk, duplicate operating points. Seconds.
    --remeasure   re-run every banked operating point at the settling horizon and re-run the joint robustness
                  ladder on the ones still walking. THIS IS THE HONEST NUMBER and it costs physics: ~1 rollout
                  per row plus up to 12 for the ladder (measured ~3 s/row on a quadruped bank, 5 min for 97).
    --gates       run gait_hints' three evidence gates over the whole bank AND over the surviving subset, each
                  arm counted THREE ways (rows as observations / one row per distinct body / one row per
                  distinct operating point) — the falsifiable test of whether a cleaner corpus reveals structure
                  the pooled one hides, and of how much of any apparent structure is one well-sampled body
                  counted many times (#274) or one repeated controller counted many times (``_op_point``).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

_DEFAULT = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "kp": 32.0, "kd": 1.5}


def _rows(db_path: Path) -> list[dict]:
    """Every banked locomotion skill, with the body that earned it (carried inline in its skill vector)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    genes = {}
    for r in conn.execute("SELECT obj_id, meta FROM vectors WHERE obj_type='skill'"):
        meta = json.loads(r["meta"]) if r["meta"] else {}
        if isinstance(meta.get("gene"), dict):
            genes[r["obj_id"]] = meta["gene"]
    out = []
    for r in conn.execute("SELECT * FROM skills WHERE task_type='locomotion' ORDER BY created_at"):
        bc = json.loads(r["base_config"]) if r["base_config"] else {}
        out.append({"skill_id": r["skill_id"], "robot_class": r["robot_class"], "species": r["species"],
                    "created_at": r["created_at"], "success_rate": r["success_rate"], "base_config": bc,
                    "params": bc.get("gait_params") or {}, "gene": genes.get(r["skill_id"])})
    conn.close()
    return out


def _is_default(p: dict) -> bool:
    return all(abs(float(p.get(k, -9)) - v) < 1e-9 for k, v in _DEFAULT.items())


def body_families(rows: list[dict]) -> collections.Counter:
    """How many DISTINCT BODIES the rows actually cover, by the held-out coarse structural key.

    THE NUMBER THAT DECIDES WHETHER A GATE RESULT MEANS ANYTHING, and the one the bank's row count hides.
    MEASURED 2026-08-07 on the live bank: the 55 rows that survive the fragility gate cover 21 body families,
    and ONE family supplies 18 of them. Run the evidence gates on those 55 rows counting ROWS and ``freq`` clears
    the association gate (rank-corr -0.39, p=0.0075); take one row per family and the association disappears
    (rank-corr +0.12, wrong sign). That is why ``gait_hints`` now counts bodies — this Counter is the census that
    made it visible, and it uses the SAME identity function the gate does, so the two cannot disagree.
    """
    return collections.Counter(_body_of(r) for r in rows)


def row_sources(rows: list[dict]) -> collections.Counter:
    """WHO WROTE EACH ROW — suite fixture, real run, or nobody can say. THREE buckets, and the third is real.

    Reads the stamp ``scripts/bank_provenance.py`` writes into ``base_config['row_source']``. It is deliberately
    a read of a recorded verdict rather than a re-derivation: the attribution needs the ``designs`` table, the
    repo's own text and (for the only channel that is proof rather than inference) a controlled pytest run
    banking into an empty database, none of which belong in a census that is supposed to take seconds.

    ``unattributed`` is NOT a polite word for ``real``. An unstamped or unattributable row is one where the
    fixture question was asked and came back "no evidence either way", and on the live bank that is the
    majority: the generic bodies ``anatomy_creature_*`` / ``built_quadruped_18seg`` carry no trace of who
    asked for them, because the compiler names them after itself and not after the request.
    """
    return collections.Counter(_source_of()(r["base_config"]) for r in rows)


def _source_of():
    """``bank_provenance.source_of``, imported however this script was invoked. One definition of the stamp key
    lives in ``bank_provenance``; copying the literal here is how the writer and the reader drift apart."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bank_provenance import source_of  # noqa: PLC0415 - sibling script, resolved at call time
    return source_of


def census(rows: list[dict]) -> dict:
    from virturoid.services.gait_flywheel import BANK_GATE, gate_of
    gates = collections.Counter(gate_of(r["base_config"]) for r in rows)
    uniq = {tuple(sorted((k, round(float(v), 6)) for k, v in r["params"].items())) for r in rows}
    fam = body_families(rows)
    src = row_sources(rows)
    return {"rows": len(rows),
            "by_row_source": dict(src),
            "suite_authored_fraction": round(src.get("suite", 0) / max(1, len(rows)), 3),
            "distinct_body_families": len(fam),
            "rows_in_largest_family": (fam.most_common(1)[0][1] if fam else 0),
            "by_class": dict(collections.Counter(r["robot_class"] for r in rows)),
            "by_gate": dict(gates),
            "gated_fraction": round(gates.get(BANK_GATE, 0) / max(1, len(rows)), 3),
            "params_are_the_shipped_default": sum(1 for r in rows if _is_default(r["params"])),
            "carry_the_dead_duty_coordinate": sum(1 for r in rows if "duty" in r["params"]),
            "distinct_operating_points": len(uniq),
            "no_inline_body": sum(1 for r in rows if r["gene"] is None),
            "first": rows[0]["created_at"] if rows else None,
            "last": rows[-1]["created_at"] if rows else None}


def remeasure(rows: list[dict], *, verbose: bool = True) -> list[dict]:
    """Re-run each banked operating point at the settling horizon, then the JOINT robustness ladder on the ones
    still walking. ``per_param`` is off: it does not enter the bank decision and costs 3-5x."""
    from virturoid.schemas.gene import RobotGene
    from virturoid.services.gait_flywheel import _FIT_PARAMS, _SETTLE_STEPS, robustness_margin
    from virturoid.services.gait_search import evaluate_gait
    out = []
    for row in rows:
        rec = {k: row[k] for k in ("skill_id", "robot_class", "species", "created_at")}
        rec["params"] = {k: float(row["params"][k]) for k in _FIT_PARAMS if k in row["params"]}
        t0 = time.perf_counter()
        if row["gene"] is None:
            rec["error"] = "no inline body — this row cannot be re-measured at all"
        else:
            try:
                gene = RobotGene.from_dict(row["gene"])
                r = evaluate_gait(gene, rec["params"], steps=_SETTLE_STEPS)
                rec["settle_forward_m"] = round(float(r["forward"]), 4)
                rec["settle_verdict"] = r.get("verdict")
                rec["still_walks"] = bool(r.get("credible")) and bool(r["survived"])
                if rec["still_walks"]:
                    rob = robustness_margin(gene, rec["params"], steps=_SETTLE_STEPS, per_param=False)
                    rec.update(robustness_rel=rob["robustness_rel"], probes=rob["probes"])
                rec["survives_gate"] = bool(rec.get("still_walks") and rec.get("robustness_rel") is not None)
            except Exception as exc:  # noqa: BLE001 - one odd row is a diagnostic, never the end of the audit
                rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        out.append(rec)
        if verbose:
            print(f"  {rec['skill_id'][:44]:46s} walks={rec.get('still_walks')} "
                  f"rel={rec.get('robustness_rel')} {rec['wall_s']}s {rec.get('error', '')}", flush=True)
    return out


def _body_of(row: dict) -> str:
    """The row's DISTINCT-BODY identity, exactly as ``gait_hints`` computes it (one implementation, no drift)."""
    from virturoid.services.gait_hints import _body_of
    return _body_of(row["gene"], row["skill_id"])


def _op_point(params: dict) -> str:
    """The row's OPERATING-POINT identity — the five fitted coordinates, rounded to what a fit can resolve.

    Body dedup removes replication of the ROBOT and leaves replication of the CONTROLLER: two different bodies
    that both shipped the byte-identical default (1.5/0.9/1.0/32/1.5), or two warm-starts that converged on one
    clone point, are one observation of that point counted twice, and they carry that replication into every
    parameter at once. ``gait_hints._DEDUP_RULE`` records this as the third check that dissolved knee_amp on the
    live bank (band 0.098 -> 0.438 of the prior, correlation flipping sign); it was done by hand there. Here it
    is a first-class dedup arm, run through the SAME ``representative_rows`` rule as the body arm.

    THIS ARM IS ORTHOGONAL TO THE BODY ARM, NOT STRONGER THAN IT, and reading it as stronger inverts the
    conclusion. MEASURED on the live bank's 51 fragility-surviving rows (2026-08-08): 21 distinct bodies, 34
    distinct operating points — and after op-point dedup the surviving 34 rows still come from only 18 bodies,
    with ONE body supplying 9 of them. So ``freq`` PASSES this arm 5/5 and FAILS the body arm at 1/5, and the
    body arm is the one telling the truth: the association is one well-sampled robot counted nine times. A claim
    has to survive BOTH arms. Neither alone is the gate.
    """
    from virturoid.services.gait_hints import _PARAM_KEYS
    return "|".join(f"{k}={float(params[k]):.4f}" if isinstance(params.get(k), (int, float)) else f"{k}=?"
                    for k in _PARAM_KEYS)


def gates(rows: list[dict], measured: list[dict] | None) -> dict:
    """gait_hints' three evidence gates, per parameter, over the whole bank and over the surviving subset —
    each arm run THREE ways: counting rows as independent observations (the pre-#274 behaviour, kept only as the
    comparison), counting DISTINCT BODIES (what the shipped gate now does), and counting DISTINCT OPERATING
    POINTS (the pseudo-replication body dedup cannot see — see ``_op_point``).

    The dedup itself is not reimplemented here: ``_region_evidence(bodies=...)`` collapses to one row per group
    through ``gait_hints.representative_rows``, so this script and the product can never drift apart on the rule.

    ``selections_ok`` is reported because it is the answer to a DIFFERENT question than ``ok``: it is the
    selection-stability panel's tally, and a parameter that passes 5/5 is a different claim from one that passes
    because every group happened to hold exactly one row (panel not exercised, reported as null).

    The outcome is the RE-MEASURED distance when ``--remeasure`` ran, otherwise the distance recorded at bank
    time. Prefer the re-measured one: a banked ``forward_m`` from a 1500-step horizon is a claim about a horizon
    that ends before some of these bodies fall.
    """
    from virturoid.services.gait_hints import _PARAM_KEYS, _region_evidence, _search_range
    by_id = {m["skill_id"]: m for m in (measured or [])}
    arms = {"whole bank": [(r["params"], (by_id.get(r["skill_id"], {}).get("settle_forward_m")
                                          if measured else r["base_config"].get("forward_m")),
                            _body_of(r)) for r in rows]}
    if measured:
        ok = {m["skill_id"] for m in measured if m.get("survives_gate")}
        body = {r["skill_id"]: _body_of(r) for r in rows}
        arms["survives the fragility gate"] = [(m["params"], m.get("settle_forward_m"),
                                                body.get(m["skill_id"], "row:" + m["skill_id"]))
                                               for m in measured if m["skill_id"] in ok]
    out = {}
    for arm, triples in arms.items():
        for dedup in ("rows as observations", "ONE ROW PER DISTINCT BODY", "ONE ROW PER DISTINCT OP-POINT"):
            res = {}
            for key in _PARAM_KEYS:
                vals = [(float(p[key]), abs(float(f)),
                         b if dedup == "ONE ROW PER DISTINCT BODY" else _op_point(p)) for p, f, b in triples
                        if isinstance(p.get(key), (int, float)) and isinstance(f, (int, float))]
                lo, hi = _search_range(key)
                ev = _region_evidence([v for v, _, _ in vals], [f for _, f, _ in vals], lo, hi,
                                      bodies=(None if dedup == "rows as observations"
                                              else [b for _, _, b in vals]))
                res[key] = {k: ev[k] for k in ("ok", "support", "rows", "selections_ok", "center", "band",
                                               "spread_vs_prior", "rho_center_distance", "p_association",
                                               "p_vs_winner_null", "why")}
            out[f"{arm}, {dedup}"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="build/memory")
    ap.add_argument("--remeasure", action="store_true", help="re-run every operating point (costs physics)")
    ap.add_argument("--gates", action="store_true", help="run the three evidence gates per parameter")
    ap.add_argument("--exclude-suite", action="store_true",
                    help="drop rows stamped row_source=suite before measuring anything — the honest re-run of "
                         "any number that was computed while the test suite was writing into this bank")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    # THIS SCRIPT IS A READER, AND IT SHOULD BE INCAPABLE OF BEING ANYTHING ELSE. ``_rows`` already opens
    # ``--memory`` with ``mode=ro``, but the census/remeasure/gate code below imports ``gait_flywheel`` and
    # ``gait_hints``, and several product paths under those open a bank from a DEFAULT they compute themselves
    # (measured: ``fit_gait_for_body(db=None)`` -> ``safe_build_path(None, "memory")`` -> ``<cwd>/build/memory``,
    # with ``bank=True``). An audit that banked a row into the corpus it is auditing would be the same
    # measuring-your-own-evidence problem as the test suite's. So point every DEFAULT at a throwaway directory:
    # the bank we READ is still exactly the one named on the command line, and nothing here can write anything.
    os.environ["VIRTUROID_MEMORY_DIR"] = tempfile.mkdtemp(prefix="virturoid-audit-scratch-")

    rows = _rows(Path(args.memory) / "virturoid_memory.db")
    if args.exclude_suite:
        src = _source_of()
        keep = [r for r in rows if src(r["base_config"]) != "suite"]
        print(f"excluding {len(rows) - len(keep)} suite-authored rows; {len(keep)} remain\n")
        rows = keep
    report = {"memory": args.memory, "excluded_suite_rows": bool(args.exclude_suite), "census": census(rows)}
    print(json.dumps(report["census"], indent=2))
    measured = None
    if args.remeasure:
        print(f"\nre-measuring {len(rows)} operating points at the settling horizon:")
        measured = remeasure(rows)
        ok = [m for m in measured if m.get("survives_gate")]
        walk = [m for m in measured if m.get("still_walks")]
        report["remeasured"] = measured
        report["survivorship"] = {"rows": len(measured), "still_walk": len(walk), "survive_gate": len(ok),
                                 "rel": dict(collections.Counter(m.get("robustness_rel") for m in measured))}
        print(f"\nstill a credible walk at the settling horizon: {len(walk)}/{len(measured)}")
        print(f"survives the fragility gate:                   {len(ok)}/{len(measured)}")
    if args.gates:
        report["gates"] = gates(rows, measured)
        for arm, res in report["gates"].items():
            print(f"\n=== evidence gates — {arm} ===")
            for key, ev in res.items():
                print(f"  {key:9s} {'PASS' if ev['ok'] else 'fail'}  n={ev['support']}/{ev['rows']} rows "
                      f"spread/prior={ev['spread_vs_prior']} rho={ev['rho_center_distance']} "
                      f"p_assoc={ev['p_association']} p_winner_null={ev['p_vs_winner_null']} "
                      f"panel={ev['selections_ok'] or 'n/a'}")
                if not ev["ok"]:
                    print(f"            {ev['why']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
