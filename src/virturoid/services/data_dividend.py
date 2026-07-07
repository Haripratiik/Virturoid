"""Data-dividend service — compute + bank + summarize the flywheel ledger (dossier "Data Dividend Ledger").

Every run should answer "which reusable prior did I improve, and by how much?". This service turns before/after
metrics into a :class:`DataDividendRecord` (computing the measured delta and whether the prior becomes reusable by
default), appends it to ``data_dividend.jsonl`` beside the existing ``flywheel_manifest.jsonl``, and aggregates the
ledger into a moat view. Pure/offline/best-effort — a missing ledger reads as empty rather than erroring.
"""

from __future__ import annotations

import json
from pathlib import Path

from virturoid.schemas.data_dividend import DataDividendRecord
from virturoid.services.memory_store import DEFAULT_MEMORY_DIR

LEDGER_NAME = "data_dividend.jsonl"


def _num(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_dividend(
    *,
    run_id: str,
    improved_prior_type: str,
    improved_prior_ref: str,
    before_metrics: dict,
    after_metrics: dict,
    key_metric: str | None = None,
    higher_is_better: bool = True,
    min_delta: float = 0.0,
    evidence_refs: list[str] | None = None,
    permission_scope: str = "workspace_private",
    record_id: str | None = None,
) -> DataDividendRecord:
    """Build a :class:`DataDividendRecord` from before/after metrics.

    ``measured_delta`` is ``after - before`` for every shared numeric metric. ``reusable_by_default`` is True only
    when the ``key_metric`` improved past ``min_delta`` in the correct direction AND the permission scope allows
    reuse — so a run banks a prior as reusable only on *measured* improvement, never on hope.
    """
    delta: dict[str, float] = {}
    for name, after in after_metrics.items():
        before = before_metrics.get(name)
        if _num(after) and _num(before):
            delta[name] = round(float(after) - float(before), 6)

    if key_metric is None:                                     # default to the first shared numeric metric
        key_metric = next((k for k in after_metrics if k in delta), None)

    improved = False
    if key_metric in delta:
        d = delta[key_metric]
        improved = d > min_delta if higher_is_better else d < -min_delta
    reusable = bool(improved and permission_scope != "no_reuse")

    return DataDividendRecord(
        id=record_id or f"dd_{run_id}_{improved_prior_type}",
        run_id=run_id,
        improved_prior_type=improved_prior_type,
        improved_prior_ref=improved_prior_ref,
        evidence_refs=list(evidence_refs or []),
        before_metrics=dict(before_metrics),
        after_metrics=dict(after_metrics),
        measured_delta=delta,
        key_metric=key_metric,
        permission_scope=permission_scope,
        reusable_by_default=reusable,
    )


def record_dividend(record: DataDividendRecord, *, memory_dir=DEFAULT_MEMORY_DIR) -> str:
    """Append a dividend record to ``<memory_dir>/data_dividend.jsonl`` (creating it). Returns the ledger path."""
    memory_dir = Path(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / LEDGER_NAME
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return str(path)


def read_dividends(*, memory_dir=DEFAULT_MEMORY_DIR) -> list[dict]:
    """Every dividend row ever appended (best-effort; empty if the ledger is missing)."""
    path = Path(memory_dir) / LEDGER_NAME
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def dividend_summary(*, memory_dir=DEFAULT_MEMORY_DIR) -> dict:
    """Aggregate the ledger into a moat view: totals, per-prior-type counts, and reuse conversion."""
    rows = read_dividends(memory_dir=memory_dir)
    by_type: dict[str, int] = {}
    reusable = 0
    improved = 0
    for row in rows:
        by_type[row.get("improved_prior_type", "unknown")] = by_type.get(row.get("improved_prior_type", "unknown"), 0) + 1
        if row.get("reusable_by_default"):
            reusable += 1
        km = row.get("key_metric")
        if km and isinstance(row.get("measured_delta"), dict) and km in row["measured_delta"]:
            if row["measured_delta"][km] > 0:
                improved += 1
    total = len(rows)
    return {
        "total_dividends": total,
        "by_prior_type": by_type,
        "reusable_by_default": reusable,
        "improved_key_metric": improved,
        "reuse_conversion": round(reusable / total, 4) if total else 0.0,
        "summary": (
            f"{total} data dividends banked across {len(by_type)} prior types; "
            f"{reusable} became reusable by default."
            if total else "No data dividends banked yet."
        ),
    }
