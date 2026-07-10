"""R2' with a LIVE LLM designer — the shape-DESIGN arm (complements r2prime.py's gait arm).

The decisive Thesis A question, run with a real LLM (Claude) as the designer and no paid API: does retrieving a
PHYSICS-VERIFIED exemplar from the corpus raise an LLM designer's verified-solve rate vs authoring from scratch?
Each task is a functional part role + a real-world shape constraint (a fin is a flat panel, a mantle is bulbous,
a tentacle is an elongated taper...). Two arms:
  * OFF    — the LLM's genuine first-attempt shape program authored from scratch (``off`` below).
  * CORPUS — the verified exemplar RETRIEVED for that role (``recall_shape``).
The verdict is UN-GAMEABLE: shape_verdict (realizes a valid solid) AND the constraint measured on the REALIZED
solid's bounding box + volume — never the designer's say-so. Deterministic (the authored programs are fixed), so
it is reproducible in the test suite. Transparency: one designer authored both arms and is not blinded; the OFF
programs are honest first attempts (e.g. a "tapered fin" that comes out round, a full-length mantle loft that
comes out tubular), and the per-task dims are reported so every verdict is auditable.
"""
from __future__ import annotations


def _dims_vol(program: dict):
    from virturoid.services.cad_geometry import realize_shape
    s = realize_shape(program)
    bb = s.bounding_box().size
    d = sorted([bb.X / 1000.0, bb.Y / 1000.0, bb.Z / 1000.0], reverse=True)     # meters, longest..shortest
    return d, float(s.volume) / 1000.0                                          # cm^3


def _elongated(p):  # a rod: longest >> the next
    d, _ = _dims_vol(p); return d[0] >= 3.0 * d[1]


def _flat_panel(p):  # a plate: thinnest is a small fraction of the longest
    d, _ = _dims_vol(p); return d[2] <= 0.2 * d[0]


def _bulbous(p):  # roundish AND bulky
    d, v = _dims_vol(p); return d[0] <= 1.8 * d[2] and v >= 150.0


def _wide_flat(p):  # a footprint: two comparable large dims + one thin
    d, _ = _dims_vol(p); return d[2] <= 0.5 * d[1] and d[1] >= 0.6 * d[0]


_CONSTRAINTS = {"elongated": _elongated, "flat_panel": _flat_panel, "bulbous": _bulbous, "wide_flat": _wide_flat}

# (role, constraint, OFF = the LLM's honest first-attempt, CORPUS_CANDIDATE = a verified exemplar for the role)
LLM_DESIGN_BATTERY = [
    ("tentacle", "elongated",
     {"family": "tapered", "length": 0.5, "r0": 0.04, "r1": 0.015},
     {"family": "tapered", "length": 0.5, "r0": 0.045, "r1": 0.007}),
    ("leg", "elongated",
     {"family": "tapered", "length": 0.42, "r0": 0.03, "r1": 0.022},
     {"family": "tapered", "length": 0.4, "r0": 0.035, "r1": 0.028}),
    ("wing", "flat_panel",
     {"family": "extrude", "profile": [[0, 0], [0.28, 0], [0.28, 0.14], [0, 0.14]], "height": 0.02},
     {"family": "extrude", "profile": [[0, 0], [0.3, 0], [0.3, 0.12], [0, 0.12]], "height": 0.012}),
    ("fin", "flat_panel",
     {"family": "tapered", "length": 0.2, "r0": 0.05, "r1": 0.02},              # first instinct: a "tapered fin" (round!)
     {"family": "extrude", "profile": [[0, 0], [0.22, 0], [0.11, 0.16]], "height": 0.01}),
    ("mantle", "bulbous",
     {"family": "loft", "sections": [[0.0, 0.10, 0.10], [0.5, 0.17, 0.17], [1.0, 0.08, 0.08]]},  # full-length (tubular)
     {"family": "loft", "sections": [[0.0, 0.10, 0.10], [0.18, 0.17, 0.17], [0.35, 0.09, 0.09]]}),
    ("head", "bulbous",
     {"family": "revolve", "profile": [[0, 0], [0.06, 0], [0.055, 0.05], [0, 0.08]]},   # a smallish dome
     {"family": "revolve", "profile": [[0, 0], [0.11, 0], [0.10, 0.09], [0, 0.15]]}),
    ("foot", "wide_flat",
     {"family": "extrude", "profile": [[0, 0], [0.1, 0], [0.1, 0.06], [0, 0.06]], "height": 0.03},
     {"family": "extrude", "profile": [[0, 0], [0.12, 0], [0.12, 0.08], [0, 0.08]], "height": 0.02}),
    ("deck", "wide_flat",
     {"family": "extrude", "profile": [[0, 0], [0.6, 0], [0.6, 0.4], [0, 0.4]], "height": 0.05},
     {"family": "extrude", "profile": [[0, 0], [0.7, 0], [0.7, 0.45], [0, 0.45]], "height": 0.04}),
]


def design_solved(program: dict, constraint: str) -> bool:
    """UN-GAMEABLE per-task verdict: the program realizes a VALID solid AND satisfies the role's shape constraint,
    both measured on the realized geometry (never the designer's claim)."""
    from virturoid.services.shape_flywheel import shape_verdict
    try:
        return bool(shape_verdict(program).get("credible")) and bool(_CONSTRAINTS[constraint](program))
    except Exception:  # noqa: BLE001 - a program that won't realize is not a solve
        return False


def run_llm_designer_battery(db) -> dict:
    """Seed the corpus with each CORPUS exemplar that genuinely verifies+satisfies its role, then score both arms.
    Returns ``{off, corpus, n, rows}`` where rows carry per-task PASS/FAIL + the OFF program's realized dims."""
    from virturoid.services.shape_flywheel import bank_shape
    for role, cname, _off, cand in LLM_DESIGN_BATTERY:
        if design_solved(cand, cname):                         # only a verified, role-correct exemplar enters the corpus
            bank_shape(db, role, cand)

    from virturoid.services.shape_flywheel import recall_shape
    off_n = corpus_n = 0
    rows = []
    for role, cname, off, _cand in LLM_DESIGN_BATTERY:
        o = design_solved(off, cname)
        recalled = recall_shape(db, role)                      # the CORPUS arm RETRIEVES (not the authored var)
        c = bool(recalled) and design_solved(recalled, cname)
        off_n += int(o); corpus_n += int(c)
        try:
            d, v = _dims_vol(off)
            det = {"off_dims_m": [round(x, 3) for x in d], "off_vol_cm3": round(v, 0)}
        except Exception:  # noqa: BLE001
            det = {}
        rows.append({"role": role, "constraint": cname, "off": o, "corpus": c, **det})
    n = len(LLM_DESIGN_BATTERY)
    return {"off": off_n, "corpus": corpus_n, "n": n, "off_rate": round(off_n / n, 3),
            "corpus_rate": round(corpus_n / n, 3), "lift": round((corpus_n - off_n) / n, 3), "rows": rows}
