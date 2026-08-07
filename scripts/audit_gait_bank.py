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

    (default)     STATIC census — rows, provenance, gate split, junk, duplicate operating points. Seconds.
    --remeasure   re-run every banked operating point at the settling horizon and re-run the joint robustness
                  ladder on the ones still walking. THIS IS THE HONEST NUMBER and it costs physics: ~1 rollout
                  per row plus up to 12 for the ladder (measured ~3 s/row on a quadruped bank, 5 min for 97).
    --gates       run gait_hints' three evidence gates over the whole bank AND over the surviving subset — the
                  falsifiable test of whether a cleaner corpus reveals structure the pooled one hides.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
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
    and ONE family supplies 18 of them. Run the evidence gates on those 55 rows and ``freq`` looks like it has a
    real association with distance (rank-corr -0.39, p=0.008); take one row per family and the association
    disappears (rank-corr +0.12, wrong sign). The gates treat each row as an independent observation, so a bank
    that re-fits a handful of near-identical bodies can pass gate 2 on pseudo-replication alone.
    """
    from virturoid.schemas.gene import RobotGene
    from virturoid.services.heldout_set import body_key
    out = collections.Counter()
    for r in rows:
        if r["gene"] is None:
            continue
        try:
            out[body_key(RobotGene.from_dict(r["gene"]))] += 1
        except Exception:  # noqa: BLE001 - an unreadable body just does not count toward coverage
            continue
    return out


def census(rows: list[dict]) -> dict:
    from virturoid.services.gait_flywheel import BANK_GATE, gate_of
    gates = collections.Counter(gate_of(r["base_config"]) for r in rows)
    uniq = {tuple(sorted((k, round(float(v), 6)) for k, v in r["params"].items())) for r in rows}
    fam = body_families(rows)
    return {"rows": len(rows),
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


def gates(rows: list[dict], measured: list[dict] | None) -> dict:
    """gait_hints' three evidence gates, per parameter, over the whole bank and over the surviving subset.

    The outcome is the RE-MEASURED distance when ``--remeasure`` ran, otherwise the distance recorded at bank
    time. Prefer the re-measured one: a banked ``forward_m`` from a 1500-step horizon is a claim about a horizon
    that ends before some of these bodies fall.
    """
    from virturoid.services.gait_hints import _PARAM_KEYS, _region_evidence, _search_range
    by_id = {m["skill_id"]: m for m in (measured or [])}
    arms = {"whole bank": [(r["params"], (by_id.get(r["skill_id"], {}).get("settle_forward_m")
                                          if measured else r["base_config"].get("forward_m")))
                           for r in rows]}
    if measured:
        ok = {m["skill_id"] for m in measured if m.get("survives_gate")}
        arms["survives the fragility gate"] = [(m["params"], m.get("settle_forward_m"))
                                               for m in measured if m["skill_id"] in ok]
        # ONE ROW PER DISTINCT BODY — the same gates with pseudo-replication removed. See ``body_families``:
        # on the live bank this arm is where the surviving arm's apparent ``freq`` association evaporates.
        by_id = {m["skill_id"]: m for m in measured}
        fam: dict = {}
        from virturoid.schemas.gene import RobotGene
        from virturoid.services.heldout_set import body_key
        for r in rows:
            m = by_id.get(r["skill_id"])
            if r["gene"] is None or m is None or r["skill_id"] not in ok:
                continue
            try:
                k = body_key(RobotGene.from_dict(r["gene"]))
            except Exception:  # noqa: BLE001
                continue
            if k not in fam or abs(m["settle_forward_m"]) > abs(fam[k]["settle_forward_m"]):
                fam[k] = m                                    # the family's furthest-travelling walk represents it
        arms["survives the gate, ONE ROW PER DISTINCT BODY"] = [(m["params"], m["settle_forward_m"])
                                                                for m in fam.values()]
    out = {}
    for arm, pairs in arms.items():
        res = {}
        for key in _PARAM_KEYS:
            vals = [(float(p[key]), abs(float(f))) for p, f in pairs
                    if isinstance(p.get(key), (int, float)) and isinstance(f, (int, float))]
            lo, hi = _search_range(key)
            ev = _region_evidence([v for v, _ in vals], [f for _, f in vals], lo, hi)
            res[key] = {k: ev[k] for k in ("ok", "support", "spread_vs_prior", "rho_center_distance",
                                           "p_association", "p_vs_winner_null", "why")}
        out[arm] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default="build/memory")
    ap.add_argument("--remeasure", action="store_true", help="re-run every operating point (costs physics)")
    ap.add_argument("--gates", action="store_true", help="run the three evidence gates per parameter")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    rows = _rows(Path(args.memory) / "virturoid_memory.db")
    report = {"memory": args.memory, "census": census(rows)}
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
                print(f"  {key:9s} {'PASS' if ev['ok'] else 'fail'}  n={ev['support']} "
                      f"spread/prior={ev['spread_vs_prior']} p_assoc={ev['p_association']} "
                      f"p_winner_null={ev['p_vs_winner_null']}")
                if not ev["ok"]:
                    print(f"            {ev['why']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
