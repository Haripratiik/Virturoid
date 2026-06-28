"""§4.4 — the flywheel-compounding chart: the moat KPI, made viewable.

``flywheel_runner.run_flywheel`` already runs N build cycles against one shared memory and reports whether banked
assets compound. This turns that report into a chart — banked assets per cycle, warm-start availability, and
success rate — so the moat ("the Nth warm-started build starts from more than the 1st") is auditable at a glance.
Pure aggregation over the report (no re-run); a dependency-free ASCII chart plus a best-effort matplotlib PNG.
"""

from __future__ import annotations


def flywheel_compounding_chart(report: dict) -> dict:
    """Extract the per-cycle compounding series + KPIs from a ``run_flywheel`` report."""
    cycles = report.get("cycles", [])
    if not isinstance(cycles, list):
        cycles = []
    series = []
    for c in cycles:
        after = c.get("banked_after", {}) or {}
        banked = sum(int(v) for v in after.values() if isinstance(v, (int, float)))
        series.append({"cycle": c.get("cycle"), "banked": banked,
                       "warm_start": bool(c.get("warm_start_available")),
                       "success_rate": float(c.get("success_rate", 0.0) or 0.0)})
    banked_start = series[0]["banked"] if series else 0
    banked_end = series[-1]["banked"] if series else 0
    warm_frac = round(sum(1 for s in series if s["warm_start"]) / max(1, len(series)), 3)
    compounding = bool(report.get("compounding", banked_end > banked_start and len(series) >= 2))
    headline = (f"banked assets {banked_start}->{banked_end} over {len(series)} cycles; "
                f"warm-started {int(warm_frac * 100)}% of cycles; "
                f"{'COMPOUNDING' if compounding else 'flat (no compounding)'}") if series else "no cycles to chart"
    return {"series": series, "n_cycles": len(series), "banked_start": banked_start, "banked_end": banked_end,
            "banked_growth": banked_end - banked_start, "warm_start_fraction": warm_frac,
            "compounding": compounding, "headline": headline}


def render_ascii_chart(chart: dict, *, width: int = 24) -> str:
    """A dependency-free bar chart of banked assets per cycle (the moat growing)."""
    series = chart.get("series", [])
    if not series:
        return "(no flywheel cycles)\n"
    mx = max((s["banked"] for s in series), default=0) or 1
    lines = ["Flywheel compounding — banked assets per cycle", ""]
    for s in series:
        bar = "#" * max(0, round(width * s["banked"] / mx))
        ws = "warm" if s["warm_start"] else "cold"
        lines.append(f"cycle {str(s['cycle']):>2} [{ws}] | {bar} {s['banked']}  ({int(s['success_rate'] * 100)}%)")
    lines += ["", chart["headline"]]
    return "\n".join(lines) + "\n"


def maybe_save_png(chart: dict, path) -> str | None:
    """Best-effort matplotlib PNG of the compounding curve (banked assets + success rate vs cycle). Returns the
    path, or None if matplotlib is unavailable (the ASCII chart is always available)."""
    series = chart.get("series", [])
    if not series:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        xs = [s["cycle"] for s in series]
        fig, ax1 = plt.subplots(figsize=(6, 3.2))
        ax1.plot(xs, [s["banked"] for s in series], "-o", color="#2a7", label="banked assets")
        ax1.set_xlabel("cycle"); ax1.set_ylabel("banked assets", color="#2a7")
        ax2 = ax1.twinx()
        ax2.plot(xs, [s["success_rate"] for s in series], "--s", color="#36c", label="success rate")
        ax2.set_ylabel("success rate", color="#36c"); ax2.set_ylim(0, 1)
        ax1.set_title(chart["headline"], fontsize=8)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
        return str(path)
    except Exception:  # noqa: BLE001 - the chart is value-add; ASCII always works
        return None
