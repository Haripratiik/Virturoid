"""What the verified-morphology memory actually holds, and what recalling it actually did.

This is the read model behind Studio's **Memory** tab. The bank is the differentiator, and until now the
product never showed it: the Library tab rendered three cards (robot count, flywheel cycles, design-brain
coverage) and none of them answered the only questions a buyer or a reviewer asks about a memory --

  1. *What is in it?*            rows, and what kind of body each row came from
  2. *How sure are you?*         the fragility gate each row carries, and which call site wrote it
  3. *Where did it come from?*   suite fixture, real build, or unattributed (``scripts/bank_provenance.py``)
  4. *Did it help?*              the measured delta of every deploy that consulted it -- wins AND losses
  5. *Did it help HERE?*         the recall events recorded against the robot currently open

TWO DELIBERATE CHOICES, both of which exist because the honest answer is currently unflattering.

**It reports losses at the same weight as wins.** The live bank's dominant recall kind, ``gait_hint_deploy``,
has a mean delta of roughly -0.034 m over 2163 recorded deploys: 246 wins, 283 losses, 1634 ties. A panel that
rendered only ``gait_warm_start`` (+0.375 m, 122 edges) would be true in every individual number and a lie in
aggregate. So every kind is returned, each carries a ``direction`` computed from the sign of its own mean, and
the headline is built from the LARGEST kind by edge count -- not the best-looking one.

**It reads the database read-only.** ``sqlite3`` is opened with ``mode=ro`` rather than going through
``MemoryDB``, which would run its schema migrations against the developer's real bank every time a UI panel
polled. A status surface must not be able to modify the substrate it reports on; that is the same class of
mistake as a test suite banking into the bank it measures.

Everything degrades to an honest empty: a missing directory, a missing database or a schema that predates a
column returns zeros and a ``notes`` entry saying so, never an exception and never a plausible-looking zero.
"""

from __future__ import annotations

import collections
import json
import sqlite3
from pathlib import Path

#: The task whose rows are the moat. Locomotion is the only one with a gate stack behind it today.
LOCOMOTION = "locomotion"

#: Recall kinds, in the order the panel should read them, with what each one MEANS. A kind absent from the
#: database is simply absent from the output -- the panel never invents a row to fill a slot.
_KIND_MEANING = {
    "gait_hint_deploy": "a mined hint was deployed on a real body and measured against that body's default gait",
    "gait_warm_start": "a new search started from a banked operating point instead of cold",
    "gait_warm_start_no_bank": "the same search with the bank withheld -- the control arm for the line above",
    "design_search_gain": "a design search seeded from a prior converged body",
    "warm_start": "a policy trained from a prior policy's weights",
}


def _connect_ro(db_path: Path) -> sqlite3.Connection | None:
    """A read-only handle, or ``None``. Never creates the file, never migrates it."""
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    con.row_factory = sqlite3.Row
    return con


def _base_config(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rows(con: sqlite3.Connection, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    try:
        return con.execute(sql, args).fetchall()
    except sqlite3.Error:                       # a table or column this build predates
        return []


def bank_census(con: sqlite3.Connection, *, task: str = LOCOMOTION) -> dict:
    """Rows, gates, doors, provenance sources and BODY CONCENTRATION for one task.

    The body concentration is not decoration. Every evidence gate over this bank treats rows as independent
    observations, and they are not: on the live bank one gene supplies 34 of 101 locomotion rows, so a gate
    that "passes on 101 samples" may be passing on far fewer bodies than it thinks. The number the panel shows
    is the largest single body's share, because that is the number that tells you how bad the pooling is.
    """
    rows = _rows(con, "SELECT robot_class, species, gene_id, base_config FROM skills WHERE task_type=?", (task,))
    gates: collections.Counter = collections.Counter()
    doors: collections.Counter = collections.Counter()
    sources: collections.Counter = collections.Counter()
    classes: collections.Counter = collections.Counter()
    bodies: collections.Counter = collections.Counter()
    for r in rows:
        bc = _base_config(r["base_config"])
        gates[str(bc.get("bank_gate") or "ungated")] += 1
        doors[str(bc.get("bank_door") or "unnamed")] += 1
        # An ABSENT row_source is "unstamped", which is NOT the same word as "unattributed". Unattributed is a
        # verdict the audit pass reached; unstamped means the audit pass never looked at this row.
        sources[str(bc.get("row_source") or "unstamped")] += 1
        classes[str(r["robot_class"] or "unknown")] += 1
        bodies[str(r["gene_id"] or r["species"] or "unknown")] += 1
    n = len(rows)
    top_body, top_rows = bodies.most_common(1)[0] if bodies else (None, 0)
    by_task = {str(r["task_type"] or "unknown"): int(r["n"])
               for r in _rows(con, "SELECT task_type, COUNT(*) n FROM skills GROUP BY task_type")}
    return {
        "task": task,
        "rows": n,
        "by_task": dict(sorted(by_task.items(), key=lambda kv: -kv[1])),
        "by_gate": dict(gates),
        "gated_rows": int(gates.get("fragility_v1", 0)),
        "gated_fraction": round(gates.get("fragility_v1", 0) / n, 4) if n else 0.0,
        "by_door": dict(doors),
        "by_source": dict(sources),
        "by_class": dict(classes),
        "bodies": {
            "distinct": len(bodies),
            "largest_share_body": top_body,
            "largest_share_rows": int(top_rows),
            "largest_share_fraction": round(top_rows / n, 4) if n else 0.0,
        },
    }


def recall_ledger(con: sqlite3.Connection) -> dict:
    """Every recorded reuse edge, grouped by kind, with wins, losses, ties and the MEAN DELTA.

    ``ties`` is reported separately and is never folded into either side: a tie usually means recall handed the
    body the same operating point it would have found anyway, which is neither evidence for the moat nor
    against it, and burying 1634 of them inside a hit rate is how a flat memory comes to look like a winning one.
    """
    kinds = []
    for r in _rows(con,
                   "SELECT kind, COUNT(*) n, AVG(delta) d, "
                   "SUM(CASE WHEN delta > 0 THEN 1 ELSE 0 END) wins, "
                   "SUM(CASE WHEN delta < 0 THEN 1 ELSE 0 END) losses, "
                   "SUM(CASE WHEN delta = 0 THEN 1 ELSE 0 END) ties "
                   "FROM provenance GROUP BY kind ORDER BY n DESC"):
        mean = round(float(r["d"]), 4) if r["d"] is not None else None
        wins, losses, ties = int(r["wins"] or 0), int(r["losses"] or 0), int(r["ties"] or 0)
        decided = wins + losses
        kinds.append({
            "kind": str(r["kind"]),
            "means": _KIND_MEANING.get(str(r["kind"]), "a recorded reuse edge"),
            "edges": int(r["n"]),
            "mean_delta_m": mean,
            "wins": wins, "losses": losses, "ties": ties,
            # Of the deploys that CHANGED anything, how many changed it for the better. Ties excluded from the
            # denominator on purpose -- see the docstring.
            "decided_win_rate": round(wins / decided, 4) if decided else None,
            "direction": ("helps" if (mean or 0) > 1e-6 else "hurts" if (mean or 0) < -1e-6 else "neutral"),
        })
    dominant = max(kinds, key=lambda k: k["edges"], default=None)
    if dominant is None:
        headline = "Nothing has been recalled yet: no reuse edges are recorded."
    else:
        verb = {"helps": "AHEAD of", "hurts": "BEHIND", "neutral": "level with"}[dominant["direction"]]
        headline = (f"{dominant['edges']} recorded deploys of {dominant['kind']} put recall {verb} "
                    f"the no-recall default by {dominant['mean_delta_m']:+.4f} m on average "
                    f"({dominant['wins']} better, {dominant['losses']} worse, {dominant['ties']} identical).")
    return {"kinds": kinds, "dominant_kind": dominant["kind"] if dominant else None, "headline": headline}


def _gene_id_candidates(genome_id: str) -> list[str]:
    """The exported genome id, peeled back toward the id the provenance ledger actually recorded.

    The export wraps the gene id: ``built_quadruped_18seg`` is banked, and the package ships
    ``genome_built_quadruped_18seg_v_v``. So the panel peels one known prefix (``genome_``) and any number of
    trailing ``_v`` variant markers, and matches the results EXACTLY.

    Exact, not prefix. The ledger contains both ``anatomy_creature`` and ``anatomy_creature_91b931bf`` as
    distinct child ids, so a prefix match would silently credit one body with another body's recall events --
    the panel would be inventing the very evidence it exists to audit. When nothing matches exactly the honest
    answer is "no recorded event", which ``build_recall`` says out loud.
    """
    out: list[str] = []
    cur = genome_id
    if cur.startswith("genome_"):
        out.append(cur)
        cur = cur[len("genome_"):]
    while True:
        if cur and cur not in out:
            out.append(cur)
        if not cur.endswith("_v"):
            break
        cur = cur[: -len("_v")]
    return out


def _gene_ids_for_package(package_dir: Path) -> list[str]:
    """Every id this package could be known by in the provenance ledger, most specific first."""
    genome = Path(package_dir) / "robot" / "robot_genome.json"
    try:
        data = json.loads(genome.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    out: list[str] = []
    for key in ("id", "name"):
        val = data.get(key)
        if isinstance(val, str) and val:
            for cand in _gene_id_candidates(val):
                if cand not in out:
                    out.append(cand)
    return out


def build_recall(con: sqlite3.Connection, gene_ids: list[str]) -> dict:
    """What recall did FOR THIS ROBOT: the deploy events recorded against its gene id.

    Returns an explicit ``matched: False`` when nothing links, because "we have no record of consulting memory
    for this robot" and "memory was consulted and did nothing" are different facts and the UI must not blur
    them into an empty list that reads as the second.
    """
    if not gene_ids:
        return {"matched": False, "gene_ids": [], "events": [], "summary":
                "This robot's package does not record a gene id, so no recall event can be attributed to it."}
    wanted = {g for g in gene_ids if g}
    events = []
    for r in _rows(con, "SELECT child_id, parent_id, kind, delta, meta, created_at FROM provenance "
                        "WHERE kind LIKE 'gait_%' ORDER BY id DESC"):
        child = str(r["child_id"] or "")
        if child not in wanted:
            continue
        try:
            meta = json.loads(r["meta"]) if r["meta"] else {}
        except (TypeError, ValueError):
            meta = {}
        events.append({
            "when": r["created_at"],
            "kind": str(r["kind"]),
            "gene_id": child,
            "region": r["parent_id"],
            "delta_m": round(float(r["delta"]), 4) if r["delta"] is not None else None,
            "source": meta.get("source"),
            "selected": meta.get("selected"),
            "hint_forward_m": meta.get("hint_forward_m"),
            "default_forward_m": meta.get("default_forward_m"),
            "hint_credible": meta.get("hint_credible"),
            "default_credible": meta.get("default_credible"),
        })
    if not events:
        return {"matched": False, "gene_ids": gene_ids, "events": [], "summary":
                "No recall event is recorded against this robot's gene id. Either its build predates the "
                "provenance ledger, or memory was never consulted for it."}
    kept = [e for e in events if e.get("selected") == "hint"]
    deltas = [e["delta_m"] for e in events if e["delta_m"] is not None]
    mean = round(sum(deltas) / len(deltas), 4) if deltas else None
    summary = (f"{len(events)} recall event(s) recorded for this robot; the recalled gait was kept in "
               f"{len(kept)} of them"
               + (f", for a mean of {mean:+.4f} m against this body's own default gait." if mean is not None
                  else "; no delta was measured."))
    return {"matched": True, "gene_ids": gene_ids, "events": events[:25], "event_count": len(events),
            "kept": len(kept), "mean_delta_m": mean, "summary": summary}


def moat_panel(memory_dir, *, package_dir=None, task: str = LOCOMOTION) -> dict:
    """The whole Memory tab in one read. Never raises; an unreadable bank reads as an empty, labelled one."""
    memory_dir = Path(memory_dir)
    db_path = memory_dir / "virturoid_memory.db"
    notes: list[str] = []
    con = _connect_ro(db_path)
    if con is None:
        return {
            "memory_dir": str(memory_dir), "db_present": False,
            "bank": {"task": task, "rows": 0, "by_task": {}, "by_gate": {}, "gated_rows": 0, "gated_fraction": 0.0,
                     "by_door": {}, "by_source": {}, "by_class": {},
                     "bodies": {"distinct": 0, "largest_share_body": None, "largest_share_rows": 0,
                                "largest_share_fraction": 0.0}},
            "recall": {"kinds": [], "dominant_kind": None,
                       "headline": "No memory database at this build root — nothing has been banked here yet."},
            "this_build": {"matched": False, "gene_ids": [], "events": [],
                           "summary": "No memory database to attribute recall against."},
            "notes": [f"no readable database at {db_path}"],
        }
    try:
        bank = bank_census(con, task=task)
        recall = recall_ledger(con)
        this_build = build_recall(con, _gene_ids_for_package(package_dir) if package_dir else [])
    finally:
        con.close()

    # The caveats are part of the payload, not the prose around it: whoever reads this number in the UI reads
    # its limits in the same glance. Each one is emitted only when the data actually earns it.
    if bank["rows"] and bank["gated_fraction"] < 0.5:
        notes.append(f"{bank['gated_rows']} of {bank['rows']} rows carry a measured fragility margin "
                     f"({bank['gated_fraction']:.1%}). The rest record an operating point with no error bar.")
    share = bank["bodies"]["largest_share_fraction"]
    if bank["rows"] and share > 0.15:
        notes.append(f"{bank['rows']} rows come from only {bank['bodies']['distinct']} distinct bodies, and one "
                     f"supplies {bank['bodies']['largest_share_rows']} of them ({share:.1%}) — treat row counts "
                     f"as pseudo-replicated, not as independent observations.")
    unattributed = bank["by_source"].get("unattributed", 0) + bank["by_source"].get("unstamped", 0)
    if unattributed:
        notes.append(f"{unattributed} rows are not attributed to a real build (run scripts/bank_provenance.py "
                     f"to re-audit); {bank['by_source'].get('suite', 0)} are known to be suite-authored.")
    return {"memory_dir": str(memory_dir), "db_present": True,
            "bank": bank, "recall": recall, "this_build": this_build, "notes": notes}
