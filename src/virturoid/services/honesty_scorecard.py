"""§4.1 — the honesty scorecard: pair every CLAIM with the gate's VERDICT in one view.

The dossier's wedge: "a weak result shown WITH the gate's verdict reads as the product working; the same result
shown without it reads as broken." This unifies the three honest signals a build already produces into one
scorecard so the weak results are surfaced next to their verdict, never hidden:
  - the Product Readiness Ledger  — is each stage provably real? (CAD/physics/controller/...)
  - the spec-compliance report    — did the build honor the detailed prompt? (height/weight/payload/pinned)
  - the sim2sim transfer verdict  — does nominal performance predict perturbed?

Pure aggregation over dicts the build writes — no new evaluation, so it can never inflate a claim.
"""

from __future__ import annotations

from virturoid.schemas.readiness_ledger import ATTAINED, NOT_REQUIRED

_REAL_STATUSES = {ATTAINED, NOT_REQUIRED}


def honesty_scorecard(*, readiness: dict | None = None, spec_compliance: dict | None = None,
                      sim2sim: dict | None = None) -> dict:
    """Combine the honest signals into one scorecard. Each input is the corresponding report dict (or None)."""
    rows: list[dict] = []
    for st in (readiness or {}).get("stages", []):
        rows.append({"claim": st.get("stage", ""), "verdict": st.get("status", ""),
                     "honest": st.get("status") in _REAL_STATUSES, "evidence": st.get("detail", "")})
    for c in (spec_compliance or {}).get("constraints", []):
        honored = c.get("honored")
        rows.append({"claim": "spec:" + str(c.get("constraint", "")),
                     "verdict": ("honored" if honored else "NOT honored" if honored is False else "reported"),
                     "honest": honored is not False, "evidence": c})
    if sim2sim:
        rows.append({"claim": "sim2sim transfer", "verdict": sim2sim.get("verdict", ""), "honest": True,
                     "evidence": {"pearson_r": sim2sim.get("pearson_r"), "mmrv": sim2sim.get("mmrv")}})
    n_honest = sum(1 for r in rows if r["honest"])
    n_flagged = len(rows) - n_honest
    if rows:
        headline = f"{n_honest}/{len(rows)} claims pass the honest gate"
        if n_flagged:
            headline += f"; {n_flagged} flagged openly (the product working, not broken)"
    else:
        headline = "no claims to score"
    return {"rows": rows, "n_claims": len(rows), "n_honest": n_honest, "n_flagged": n_flagged,
            "headline": headline}


def format_scorecard_md(scorecard: dict) -> str:
    lines = ["# Honesty Scorecard", "", f"**{scorecard['headline']}**", "",
             "| Claim | Verdict | Honest? |", "|---|---|---|"]
    for r in scorecard["rows"]:
        lines.append(f"| {r['claim']} | {r['verdict']} | {'OK' if r['honest'] else 'FLAGGED'} |")
    return "\n".join(lines) + "\n"


def scorecard_from_package(package_dir) -> dict:
    """Build the scorecard from a built package's on-disk reports (best-effort; missing reports are skipped)."""
    import json
    from pathlib import Path
    pkg = Path(package_dir)

    def _read(name: str):
        for p in pkg.rglob(name):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    return honesty_scorecard(readiness=_read("product_readiness_ledger.json"),
                             spec_compliance=_read("spec_compliance.json"),
                             sim2sim=_read("sim2sim_report.json"))
