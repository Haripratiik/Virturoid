"""Coercion audit — the determinism sweep (master_plan_v6 §8.2.4 / WS-B.4).

The north star says deterministic code lives ONLY in the grounding layer and *nothing user-facing is a template*.
The anatomy compiler, though, quietly mutates an LLM-authored graph: it clamps sub-buildable sizes, fills omitted
girths/joints/aims with role defaults, and caps an over-fat torso. Each such mutation is either

  (a) a **physics/buildability-required clamp** — legitimate, but it must be *reported back as a teaching note*
      (grounding), so the model learns the constraint instead of being silently overruled; or
  (b) a **silent default/override** — the antipattern; the model's stated intent was changed without it knowing.

This module makes every one of those visible. ``detect_coercions(graph)`` inspects an authored graph and returns
the coercions the compiler WILL apply (so ``submit_design`` can surface them); ``audit_table()`` is the complete
static enumeration of the compiler's coercion sites with each classified. The rules here are kept in lockstep
with ``anatomy_compiler._role_geometry`` / ``_ARTICULATED`` / ``_DOWN_ROLES`` — a drift test asserts they match.
"""
from __future__ import annotations

# Rules mirrored from anatomy_compiler (kept in sync by test_coercion_audit::test_rules_track_the_compiler).
SIZE_FLOOR_M = 0.02            # _role_geometry: L = max(0.02, size)
GIRTH_FLOOR_M = 0.006         # g = max(0.006, girth) if girth else ...
GIRTH_DEFAULT_FRAC = 0.18     # ...else max(0.01, 0.18 * L)
GIRTH_DEFAULT_FLOOR_M = 0.01
BODY_LONG_WIDTH_CAP_FRAC = 0.32   # default 'long' torso caps half-width footprint at 0.32*L

# graph-level roles (what the LLM authors) that articulate / aim-down by default once the compiler splits them.
_ARTICULABLE_ROLES = frozenset({"leg", "arm", "tail", "wing", "fin", "flipper", "claw", "neck", "tentacle", "trunk"})
_DOWN_ROLES = frozenset({"leg", "foot", "paw"})


def _coercion(part: str, field: str, kind: str, authored, applied, physics_required: bool, note: str) -> dict:
    """One coercion record. ``kind``: 'clamp' (sub-buildable value bounded) | 'default' (omitted field filled) |
    'cap' (an explicit value reduced by a proportion rule — the class that OVERRIDES stated intent)."""
    return {"part": part, "field": field, "kind": kind, "authored": authored, "applied": applied,
            "physics_required": physics_required, "note": note}


def detect_coercions(graph: dict) -> list[dict]:
    """Return the coercions the compiler will apply to this authored graph — the per-build grounding note."""
    parts = graph.get("parts") or []
    body = next((p for p in parts if p.get("role") == "body" and not p.get("parent")), None)
    out: list[dict] = []
    for p in parts:
        role = str(p.get("role") or "").lower()
        name = str(p.get("name") or "?")
        size = p.get("size")
        girth = p.get("girth")
        # --- size clamp (buildability) ---
        if size is not None and float(size) < SIZE_FLOOR_M:
            out.append(_coercion(name, "size", "clamp", size, SIZE_FLOOR_M, True,
                                 f"size {size} m is below the buildable floor; clamped to {SIZE_FLOOR_M} m"))
        # --- girth: default (omitted) or clamp (sub-buildable) ---
        if girth is None and size is not None:
            d = round(max(GIRTH_DEFAULT_FLOOR_M, GIRTH_DEFAULT_FRAC * max(SIZE_FLOOR_M, float(size))), 4)
            out.append(_coercion(name, "girth", "default", None, d, False,
                                 f"girth omitted → compiler uses ~0.18×size = {d} m; set girth to control thickness"))
        elif girth is not None and float(girth) < GIRTH_FLOOR_M:
            out.append(_coercion(name, "girth", "clamp", girth, GIRTH_FLOOR_M, True,
                                 f"girth {girth} m is below the buildable floor; clamped to {GIRTH_FLOOR_M} m"))
        # --- default 'long' torso width cap (OVERRIDE of an explicit girth — the dangerous class) ---
        if body is not None and p is body and girth is not None and size:
            aspect = str(p.get("aspect") or "").lower()
            cap = BODY_LONG_WIDTH_CAP_FRAC * float(size)
            if not aspect and float(girth) > cap:
                out.append(_coercion(name, "girth", "cap", girth, round(cap, 4), True,
                                     f"a default 'long' torso caps its width at 0.32×length = {cap:.3f} m, so your "
                                     f"girth {girth} m is reduced — set aspect='wide'/'round'/'flat' to opt out"))
        # --- joint default (an articulable appendage authored without a joint) ---
        if role in _ARTICULABLE_ROLES and "joint" not in p:
            out.append(_coercion(name, "joint", "default", None, "revolute", False,
                                 f"a '{role}' articulates (revolute) by default; author joint explicitly to be sure "
                                 f"(a WALKING leg needs joint='revolute')"))
        # --- aim default (a down-hanging role authored without an aim) ---
        if role in _DOWN_ROLES and "aim" not in p:
            out.append(_coercion(name, "aim", "default", None, "down_out", False,
                                 f"a '{role}' aims 'down_out' by default; author aim explicitly to control stance"))
    return out


def summarize(coercions: list[dict]) -> dict:
    """Split coercions into the two plan classes: physics-required clamps (legitimate, reported) vs conveniences
    (defaults the model should make explicit) — and count overrides (the class we most want to see, or eliminate)."""
    return {
        "n": len(coercions),
        "physics_required": [c for c in coercions if c["physics_required"]],
        "conveniences": [c for c in coercions if not c["physics_required"]],
        "overrides": [c for c in coercions if c["kind"] == "cap"],
        "kinds": {k: sum(1 for c in coercions if c["kind"] == k) for k in ("clamp", "default", "cap")},
    }


# ------------------------------------------------------------------ the complete static table (the gate artifact)
def audit_table() -> list[dict]:
    """Every coercion site the compiler applies to an authored graph, classified. The 'complete table' WS-B.4's
    gate requires — each site is either a reported physics clamp or a reported default; none is a silent rewrite."""
    return [
        {"site": "_role_geometry: L = max(0.02, size)", "field": "size", "kind": "clamp",
         "physics_required": True, "surfaced": True,
         "rationale": "a segment below ~2 cm is not buildable/simulable; reported via detect_coercions"},
        {"site": "_role_geometry: g = max(0.006, girth)", "field": "girth", "kind": "clamp",
         "physics_required": True, "surfaced": True,
         "rationale": "a radius below ~6 mm is not buildable; reported"},
        {"site": "_role_geometry: girth omitted → 0.18×L", "field": "girth", "kind": "default",
         "physics_required": False, "surfaced": True,
         "rationale": "a convenience default; surfaced so the model can be explicit (not a silent choice)"},
        {"site": "_role_geometry body 'long': half-width ≤ 0.32×L", "field": "girth", "kind": "cap",
         "physics_required": True, "surfaced": True,
         "rationale": "prevents an over-fat torso (a stability/geometry rule); OVERRIDE — surfaced prominently, "
                      "and the model can opt out with aspect='wide'/'round'/'flat'"},
        {"site": "_ARTICULATED roles: joint defaults revolute", "field": "joint", "kind": "default",
         "physics_required": False, "surfaced": True,
         "rationale": "articulable appendages articulate by default; surfaced so a welded intent is explicit"},
        {"site": "_DOWN_ROLES: aim defaults down_out", "field": "aim", "kind": "default",
         "physics_required": False, "surfaced": True,
         "rationale": "legs/feet hang down by default; surfaced so a different stance is explicit"},
        {"site": "per-role half-width min/max (head/wheel/limb …)", "field": "girth", "kind": "clamp",
         "physics_required": True, "surfaced": False,
         "rationale": "role-appropriate proportion bounds; buildability — the realized dims are visible in the "
                      "returned summary/render, and the aggregate clamp is documented here (not a per-field note)"},
    ]
