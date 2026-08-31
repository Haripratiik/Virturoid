"""Let the designer state its INTENT as claims the harness checks.

`probe_robot` answers questions, but only if someone thinks to ask them. The other half of the Articraft harness
is the agent declaring what it MEANT -- "the foot should touch the ground", "the gripper should be within 40 cm
of the mount", "these two parts are supposed to overlap, that is a joint housing not a defect" -- so the checker
can find the gap between intent and geometry rather than the agent having to notice it.

That matters because of the specific failure mode Articraft reports: "poor global shape quality despite passing
local structural checks". Every part can be individually fine while the assembly means nothing. An assertion is
the agent saying what the assembly was FOR, which is the only thing that makes global wrongness detectable.

Assertions are stored on the gene, so a later AMEND re-checks the original intent -- a change that quietly breaks
what the design was for gets caught by the design's own words.

Nothing here calls an LLM. `allow_*` forms exist so an agent can dismiss a false positive ONCE, in writing, with
the reason recorded -- rather than the checker being loosened for everyone.
"""
from __future__ import annotations

_KINDS = ("expect_contact", "expect_within", "expect_clearance", "expect_above",
          "allow_overlap", "allow_isolated_part")


def describe() -> dict:
    """The assertion vocabulary, for the design schema the agent reads."""
    return {
        "expect_contact": {"args": "a, b", "means": "these two parts should touch (feet on the floor: use "
                                                    "b='floor')"},
        "expect_within": {"args": "a, b, max_m", "means": "a's centre is within max_m of b's centre"},
        "expect_clearance": {"args": "a, min_m", "means": "a sits at least min_m above the lowest point"},
        "expect_above": {"args": "a, b", "means": "a's centre is higher than b's"},
        "allow_overlap": {"args": "a, b, reason", "means": "these two are MEANT to intersect (a joint housing, a "
                                                           "shrouded actuator); records the reason rather than "
                                                           "loosening the check for everyone"},
        "allow_isolated_part": {"args": "a, reason", "means": "a is deliberately unattached"},
    }


def validate(assertions) -> list[str]:
    """Structural problems with the assertion list itself — returned as teaching errors, never raised."""
    errs = []
    for i, a in enumerate(assertions or []):
        if not isinstance(a, dict):
            errs.append(f"assertion {i} is not an object")
            continue
        k = str(a.get("kind") or "")
        if k not in _KINDS:
            errs.append(f"assertion {i}: unknown kind {k!r}; use one of {list(_KINDS)}")
            continue
        if not a.get("a"):
            errs.append(f"assertion {i} ({k}): needs 'a' (a part name)")
        if k in ("expect_within", "expect_contact", "expect_above", "allow_overlap") and not a.get("b"):
            errs.append(f"assertion {i} ({k}): needs 'b'")
        if k == "expect_within" and a.get("max_m") is None:
            errs.append(f"assertion {i} (expect_within): needs 'max_m' in metres")
        if k == "expect_clearance" and a.get("min_m") is None:
            errs.append(f"assertion {i} (expect_clearance): needs 'min_m' in metres")
    return errs


def check(gene, assertions=None) -> dict:
    """Check each assertion against the compiled body. Returns per-assertion results, never raises.

    A missing part is a FAILURE naming the part, not a crash and not a silent skip: an assertion about a part
    that does not exist is exactly the kind of drift worth surfacing (a rename that left the intent behind)."""
    import numpy as np

    from virturoid.services.robot_probe import _body_extents, _model, _posed_data

    src = assertions if assertions is not None else ((getattr(gene, "metadata", None) or {}).get("assertions") or [])
    bad = validate(src)
    if bad:
        return {"ok": False, "errors": bad}
    model = _model(gene)
    data = _posed_data(model, gene)
    ext = _body_extents(model, data)
    floor = min(float(e["lo"][2]) for e in ext.values()) if ext else 0.0

    results, failed = [], 0
    for a in src:
        k, an, bn = a["kind"], str(a["a"]), str(a.get("b") or "")
        row = {"kind": k, "a": an, **({"b": bn} if bn else {})}
        missing = [n for n in ((an,) + ((bn,) if bn and bn != "floor" else ())) if n not in ext]
        if missing:
            row.update(passed=False, detail=f"no such part(s): {missing}")
            results.append(row); failed += 1
            continue
        ea = ext[an]
        if k == "expect_contact":
            if bn == "floor":
                gap = float(ea["lo"][2]) - floor
                row.update(passed=gap <= 0.02, detail=f"{gap:.4f} m above the lowest point (<=0.02 counts as down)")
            else:
                eb = ext[bn]
                touch = all(ea["lo"][i] <= eb["hi"][i] + 5e-3 and eb["lo"][i] <= ea["hi"][i] + 5e-3
                            for i in range(3))
                row.update(passed=touch, detail="AABBs meet" if touch else "AABBs are apart")
        elif k == "expect_within":
            d = float(np.linalg.norm(ea["centre"] - ext[bn]["centre"]))
            row.update(passed=d <= float(a["max_m"]), detail=f"{d:.4f} m apart, limit {float(a['max_m']):.4f}")
        elif k == "expect_clearance":
            c = float(ea["lo"][2]) - floor
            row.update(passed=c >= float(a["min_m"]), detail=f"{c:.4f} m clear, need {float(a['min_m']):.4f}")
        elif k == "expect_above":
            dz = float(ea["centre"][2] - ext[bn]["centre"][2])
            row.update(passed=dz > 0.0, detail=f"{dz:+.4f} m higher")
        else:                                                # allow_* record intent; they never fail
            row.update(passed=True, detail=f"allowed: {a.get('reason') or 'no reason given'}")
        results.append(row)
        failed += 0 if row["passed"] else 1

    return {"ok": True, "n": len(results), "failed": failed, "results": results,
            "allowed_overlaps": [[a["a"], a.get("b")] for a in src if a["kind"] == "allow_overlap"]}
