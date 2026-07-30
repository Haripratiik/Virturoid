"""What will this amend actually touch, and what stops being true once it lands?

An edit currently just happens. The customer asks to make a robot taller and gets a new robot back, with no
statement of what was meant to change, what was meant to stay, or which previously-established facts the change
invalidates. That is the difference between a tool that edits a file and one that amends an engineering design:
the second tells you the blast radius BEFORE it commits, and afterwards tells you what has to be re-checked.

Two questions, answered separately because they fail differently:

  scope(gene, ops)   -> {editable, preserved, invalidates}   BEFORE the edit, so it can be shown or refused
  verify_preserved() -> what actually moved                  AFTER, so a claim of "untouched" is checkable

The second exists because the first is a promise. A promise that nothing else moved is worth exactly as much as
the check that confirms it -- and mount offsets, mass and inertia all propagate along a chain, so "I only changed
the calf" is very easy to believe and often false.
"""
from __future__ import annotations

# What each recheck MEANS, so a report names consequences rather than emitting tags.
_RECHECK = {
    "joint_range": "joint travel and limits on the changed chain",
    "self_collision": "parts passing through each other, including through their joint ranges",
    "reach": "the reachable envelope / end-effector workspace",
    "torque": "actuator demand vs rating on every joint below the change",
    "mass": "mass, centre of mass and inertia",
    "stability": "standing/driving stability — support polygon and tipping margin",
    "gait": "the locomotion verdict; a body that walked may not any more",
    "ground_contact": "which parts touch down, and the standing height",
}

# Per operator: what it edits, and what stops being trustworthy once it has. Deliberately CONSERVATIVE -- a
# missed recheck is a silent wrong answer, an extra one costs a few seconds of simulation.
_OP_IMPACT = {
    "scale_group":            ("the named group", ["mass", "torque", "self_collision", "stability", "gait",
                                                   "reach", "ground_contact"]),
    "set_height":             ("every ground-reaching chain", ["stability", "gait", "ground_contact", "torque",
                                                               "self_collision"]),
    "scale_robot":            ("every part", ["mass", "torque", "stability", "gait", "reach", "ground_contact",
                                              "self_collision"]),
    "set_material":           ("materials only", ["mass", "torque", "stability"]),
    "set_leg_count":          ("the whole leg topology", ["gait", "stability", "mass", "torque", "reach",
                                                          "ground_contact", "self_collision", "joint_range"]),
    "set_payload":            ("carried mass", ["mass", "torque", "stability", "gait"]),
    "add_limb":               ("a new chain plus its mount", ["mass", "torque", "stability", "reach",
                                                              "self_collision", "gait"]),
    "adopt_walkable_template": ("the entire body", ["gait", "stability", "mass", "torque", "reach",
                                                    "ground_contact", "self_collision", "joint_range"]),
}
_DEFAULT_IMPACT = ("unclassified", sorted(_RECHECK))          # unknown op -> assume it touches everything


def _names(gene) -> list[str]:
    return [s.name for s in getattr(gene, "segments", []) or []]


def _descendants(gene, root: str) -> set[str]:
    kids: dict[str, list[str]] = {}
    for s in gene.segments:
        if s.parent:
            kids.setdefault(s.parent, []).append(s.name)
    out, stack = set(), list(kids.get(root, []))
    while stack:
        n = stack.pop()
        if n in out:
            continue
        out.add(n)
        stack += kids.get(n, [])
    return out


def scope(gene, ops) -> dict:
    """What an amend intends to touch, and what it invalidates — computed BEFORE anything is edited."""
    all_names = set(_names(gene))
    editable: set[str] = set()
    rechecks: set[str] = set()
    unknown: list[str] = []
    per_op = []
    for spec in ops or []:
        op = str((spec or {}).get("op") or "")
        args = (spec or {}).get("args") or {}
        touched, checks = _OP_IMPACT.get(op, _DEFAULT_IMPACT if op else ("", []))
        if op and op not in _OP_IMPACT:
            unknown.append(op)
        # Name the parts when the operator says which; a chain edit takes everything BELOW it, because a
        # descendant's placement is defined relative to its parent and therefore moves whether or not it is named.
        named = set()
        for key in ("part", "name", "target", "group", "parent"):
            v = args.get(key)
            if isinstance(v, str) and v in all_names:
                named |= {v} | _descendants(gene, v)
        editable |= named
        rechecks |= set(checks)
        per_op.append({"op": op or "(missing)", "touches": touched,
                       "named_parts": sorted(named) or None, "invalidates": sorted(checks)})
    # An op that names no part is NOT an op that touches nothing -- set_height reshapes every ground-reaching
    # chain and scale_robot touches all of it. Returning an empty `editable` for those would read as "nothing
    # changes", which is the most dangerous possible answer, so say plainly that the scope is unbounded.
    unbounded = bool(ops) and not editable
    return {
        "ok": True,
        "editable": sorted(editable),
        "scope_is_bounded": not unbounded,
        "scope_note": ("this edit names no specific part, so it may touch ANY of them — nothing is promised "
                       "preserved. Re-check everything in `invalidates`."
                       if unbounded else "scope is limited to the parts listed in `editable`"),
        # "preserved" is a PROMISE, not a finding. verify_preserved() is what turns it into a fact.
        "preserved": sorted(all_names - editable) if editable else [],
        "preserved_is_a_claim": ("nothing here has been checked yet — call verify_preserved(before, after) after "
                                 "the edit to find what actually moved"),
        "invalidates": sorted(rechecks),
        "invalidates_meaning": {k: _RECHECK[k] for k in sorted(rechecks) if k in _RECHECK},
        "unknown_ops": unknown,
        "per_op": per_op,
    }


def verify_preserved(before, after, preserved=None) -> dict:
    """What ACTUALLY moved between two genes. The check behind the promise.

    Compares the fields an edit can propagate through — geometry, placement, mass, joint spec — so a claim that a
    part was untouched is verified rather than trusted. Mount offsets and mass travel along a chain, which is why
    "I only changed the calf" is easy to believe and often wrong."""
    fields = ("length_m", "radius_m", "mass_kg", "shape", "joint_type", "joint_axis",
              "joint_lower", "joint_upper", "mount_offset", "mount_euler", "actuator_torque_nm")

    def snap(g):
        return {s.name: {f: getattr(s, f, None) for f in fields} for s in getattr(g, "segments", []) or []}

    a, b = snap(before), snap(after)
    watch = set(preserved) if preserved is not None else set(a)
    moved, gone, added = {}, [], sorted(set(b) - set(a))
    for n in sorted(watch):
        if n not in b:
            gone.append(n)
            continue
        if n not in a:
            continue
        d = {f: [a[n][f], b[n][f]] for f in fields if a[n][f] != b[n][f]}
        if d:
            moved[n] = d
    return {"ok": True, "held": not moved and not gone, "changed": moved,
            "removed": gone, "added": added,
            "n_watched": len(watch), "n_changed": len(moved)}
