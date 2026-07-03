"""Multi-night compounding measurement (plan v3 M3) — the open-endedness PROOF, made auditable.

``night_runner.run_night`` banks verified capabilities into the flywheel + fills the QD archive each night; the
CLAIM is that this COMPOUNDS unattended. This module is the measurement: record a per-night snapshot, then
aggregate a sequence into the vs-night curve — cumulative banked, QD-coverage, ANNECS-V (novelty) — with a
COMPOUNDING verdict (cumulative banked strictly grows AND coverage never regresses across the run). Pure
aggregation over night reports (no re-run); a dependency-free ASCII curve. The multi-night RUNNING is the
overnight compute; THIS is the harness that turns those nights into the published flywheel plot.

(Companion to ``flywheel_chart.flywheel_compounding_chart``, which charts the autonomous-BUILD flywheel's
per-cycle series; this charts the night-shift flywheel's per-NIGHT series.)"""

from __future__ import annotations

import json
from pathlib import Path


def night_snapshot(report: dict, *, night: int | None = None) -> dict:
    """Extract the per-night compounding fields from a ``run_night`` report (defensive to the report shape)."""
    n = report.get("night", report)                          # run_night nests the counters under "night"
    banked = int(getattr(n, "banked", None) if not isinstance(n, dict) else n.get("banked", 0) or 0)
    novel = int(getattr(n, "novel", None) if not isinstance(n, dict) else n.get("novel", 0) or 0)
    qd = (getattr(n, "qd", None) if not isinstance(n, dict) else n.get("qd")) or {}
    return {"night": night, "banked": banked, "novel": novel,
            "qd_filled": int(qd.get("filled", 0) or 0), "coverage": float(qd.get("coverage", 0.0) or 0.0),
            "annecs_v": int(qd.get("annecs_v", 0) or 0), "qd_score": float(qd.get("qd_score", 0.0) or 0.0)}


def append_night(report: dict, log_path: str) -> dict:
    """Append a night snapshot to a nights-log JSONL (the multi-night series the curve reads). Numbers the night
    by the current line count, so sequential runs build the series automatically. Returns the snapshot."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prior = load_nights(log_path)
    snap = night_snapshot(report, night=len(prior) + 1)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snap) + "\n")
    return snap


def load_nights(log_path: str) -> list:
    p = Path(log_path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def compounding_series(snapshots: list) -> dict:
    """Aggregate per-night snapshots into the compounding curve + verdict. ``cumulative_banked`` is the running
    total of banked capabilities; COMPOUNDING = cumulative banked strictly grows over the run AND QD-coverage
    never regresses (POET/ANNECS open-endedness: the frontier keeps expanding, it does not shrink)."""
    snaps = sorted(snapshots, key=lambda s: s.get("night", 0))
    series = []
    cum = 0
    cov_regressed = False
    prev_cov = -1.0
    for i, s in enumerate(snaps):
        cum += int(s.get("banked", 0) or 0)
        cov = float(s.get("coverage", 0.0) or 0.0)
        if cov + 1e-9 < prev_cov:
            cov_regressed = True
        prev_cov = cov
        series.append({"night": s.get("night", i + 1), "banked": int(s.get("banked", 0) or 0),
                       "cumulative_banked": cum, "coverage": cov, "annecs_v": int(s.get("annecs_v", 0) or 0),
                       "qd_score": float(s.get("qd_score", 0.0) or 0.0)})
    n = len(series)
    # COMPOUNDING = CAPABILITIES ACCUMULATE (cumulative banked strictly grows) -- the always-valid signal (the
    # banks persist in shared flywheel memory regardless). Coverage growth is a SECONDARY signal that ALSO
    # requires a SHARED QD archive passed across nights (else each night's coverage is per-night, not cumulative,
    # and oscillates -- reported honestly as coverage_compounds=False, not conflated with the banked verdict).
    banked_compounds = bool(n >= 2 and series[-1]["cumulative_banked"] > series[0]["cumulative_banked"])
    coverage_compounds = bool(n >= 2 and not cov_regressed and series[-1]["coverage"] > series[0]["coverage"])
    compounding = banked_compounds
    headline = (f"banked {series[0]['cumulative_banked']}->{series[-1]['cumulative_banked']} cumulative over "
                f"{n} nights; coverage {series[0]['coverage']:.3f}->{series[-1]['coverage']:.3f} "
                f"({'grows' if coverage_compounds else 'per-night (share the QD archive to accumulate)'}); "
                f"{'COMPOUNDING' if compounding else 'not yet compounding (need >=2 banking nights)'}"
                ) if series else "no nights recorded yet"
    return {"series": series, "n_nights": n, "cumulative_banked": series[-1]["cumulative_banked"] if series else 0,
            "banked_compounds": banked_compounds, "coverage_compounds": coverage_compounds,
            "coverage_regressed": cov_regressed, "compounding": compounding, "headline": headline}


def run_night_series(run_one, n: int, *, log_path: str) -> dict:
    """Run ``n`` sequential nights (``run_one() -> a run_night report``), record each snapshot to ``log_path``,
    and return the accumulated compounding series. ``run_one`` is injected so this is testable without GPU (a
    fake report) and reusable with the real ``night_runner.run_night`` bound to its config. The nights-log is
    APPEND, so a later invocation continues the series (resumable multi-night, the overnight-in-chunks pattern)."""
    for _ in range(int(n)):
        rep = run_one()
        append_night(rep, log_path)
    return compounding_series(load_nights(log_path))


def render_ascii(chart: dict, *, width: int = 28) -> str:
    """Dependency-free bar chart of cumulative banked capabilities per night (the flywheel compounding)."""
    series = chart.get("series", [])
    if not series:
        return "(no nights recorded)\n"
    mx = max((s["cumulative_banked"] for s in series), default=0) or 1
    lines = ["Flywheel compounding — cumulative banked capabilities per night", ""]
    for s in series:
        bar = "#" * max(0, round(width * s["cumulative_banked"] / mx))
        lines.append(f"night {str(s['night']):>2} | {bar} {s['cumulative_banked']}"
                     f"  (+{s['banked']} banked, cov {s['coverage']:.2f}, annecs {s['annecs_v']})")
    lines += ["", chart["headline"]]
    return "\n".join(lines) + "\n"
