"""Does this body carry a gripper? One STRUCTURAL answer, read by every place that asks.

Three places used to answer the question three different ways, and on real robots they disagreed:

* ``robot_import`` declared ``end_effector_type = "gripper"`` only when some segment's NAME contained "grip".
  A Franka Panda (links ``hand`` / ``left_finger`` / ``right_finger``) therefore imported as ``"none"``, and so
  did an i2rt YAM, an AgileX Piper and an ARX L5, each of which carries two sliding fingers.
* ``gene_compiler`` emits the ``grasp_site`` our grasp scripts aim at exactly when the end-effector segment has
  a prismatic child. That is the detector ``verify_robot`` actually runs: no site, every attempt reports
  ``no_grasp_site``, and the verdict reads "NO GRIPPER in your model". It attempted a grasp on the Panda.
* ``capability_registry._run_pick_place`` tried the real contact grasp only on a DECLARED gripper, so the same
  Panda never got one in ``run_task`` and went straight to the idealized pin, in the same session in which
  ``verify_robot`` had tried to grasp with it.

THE RULE. A gripper is a pair of jaws on a PALM at the end of the arm, found in one of two forms:

* SLIDING jaws: prismatic children of the palm. :func:`jaws_on` is the compiler's grasp-site rule character
  for character, so whenever the palm is the flagged end effector the importer's answer and the grasp site
  cannot drift apart (``tests/test_gripper_detection.py`` pins the agreement). These are the only jaws our grasp
  scripts can close (``closable``).
* COUPLED revolute jaws: two revolute children of a rigidly mounted palm that the source model couples 1:1
  (an equality joint), on a palm an arm leads to. That is a Robotiq 2F-85's linkage (``left_driver`` /
  ``right_driver``), measured on the Menagerie Stanford Tidybot. The extra conditions are measured, not
  cautious: a ToddlerBot couples its elbow and shoulder joints to sibling GEAR links 1:1 exactly like a pair
  of jaws, but those palms are themselves revolute; its neck pair sits on the ROOT torso, which no arm leads
  to. Without both conditions a humanoid would have imported carrying a gripper.

WHICH SEGMENT IS THE PALM is the part the compiler never had to know. An importer that flags the last leaf it
finds lands on a FINGER whenever the fingers are the last links in the file, and a finger has nothing under
it. Measured on the Menagerie: the Piper and the ARX L5 flag ``link8``, the PAL Tiago flags
``gripper_right_finger_link``, and the Tidybot flags ``left_silicone_pad``, four links below its gripper's
base. :func:`palm_and_jaws` finds the palm from wherever the flag landed.

Deliberately NOT a gripper by this rule: uncoupled revolute fingers. A dexterous hand's five finger chains, a
soft foot's five phalanges (Menagerie ``iit_softfoot``) and an arm's last wrist joint are all revolute joints at
the end of a chain, and nothing structural separates one of them from a single revolute jaw. A body whose
source NAMES its gripper still declares one (the importer keeps that evidence), and :func:`gripper_status`
reports it as declared-but-not-closable rather than as absent.
"""
from __future__ import annotations

#: End-effector types that hold an object: jaws close on it, or a cup holds it by vacuum. ``"hand"`` is not in
#: the schema's enum but several consumers already test for it, so it is honoured here as well.
HOLDING_TOOLS = frozenset({"gripper", "hand", "suction"})

#: How far above the flagged end effector a palm may sit. The Tidybot's flag lands on a finger pad four links
#: below its gripper's base (pad <- pad mount <- follower <- spring link <- base); ToddlerBot's root torso,
#: which also carries a coupled pair, is seven above its flagged hand.
_PALM_SEARCH_DEPTH = 4


def _children(segments, name: str) -> list:
    return [s for s in segments if getattr(s, "parent", None) == name]


def jaws_on(segments, name: str) -> list:
    """The SLIDING jaws carried by segment ``name``: its direct prismatic children.

    This is ``gene_compiler``'s grasp-site rule (``fingers = [c for c in gene.children_of(seg.name) if
    c.joint_type == "prismatic"]``): when this list is non-empty for the flagged end effector, the compiled body
    carries ``grasp_site`` and the grasp scripts can close the jaws."""
    return [c for c in _children(segments, name) if getattr(c, "joint_type", None) == "prismatic"]


def _coupled_revolute_pairs(segments, coupled) -> dict:
    """``{palm: [jaw_a, jaw_b]}`` for every revolute sibling pair the source couples 1:1 on a rigid palm that an
    actuated joint leads to (see the module docstring for why each condition is there)."""
    by = {s.name: s for s in segments}

    def arm_leads_to(name: str) -> bool:
        seen = set()
        cur = by.get(name)
        while cur is not None and cur.parent is not None and cur.name not in seen:
            seen.add(cur.name)
            parent = by.get(cur.parent)
            if parent is None:
                return False
            if getattr(parent, "joint_type", None) in ("revolute", "prismatic"):
                return True
            cur = parent
        return False

    out: dict = {}
    for c in coupled or []:
        if not isinstance(c, dict):
            continue
        a, b = by.get(str(c.get("a"))), by.get(str(c.get("b")))
        if a is None or b is None or a.parent is None or a.parent != b.parent:
            continue
        if a.joint_type != "revolute" or b.joint_type != "revolute":
            continue
        try:
            ratio, offset = float(c.get("ratio", 1.0)), float(c.get("offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if abs(abs(ratio) - 1.0) > 1e-3 or abs(offset) > 1e-6:
            continue
        palm = by.get(a.parent)
        if palm is None or getattr(palm, "joint_type", None) not in (None, "fixed") or not arm_leads_to(palm.name):
            continue
        out.setdefault(palm.name, [a, b])
    return out


def palm_and_jaws(segments, ee_name: str | None, coupled=None) -> tuple[str | None, list]:
    """``(palm, jaws)`` for a body whose end effector is flagged on ``ee_name``.

    ``palm`` is the segment that carries the jaws and ``jaws`` the jaws on it, searched in this order:

    1. sliding jaws on the flagged segment itself: the compiler's own case, the only one with a grasp site;
    2. sliding jaws on its parent, when the flagged segment is ITSELF a sliding jaw with at least one sliding
       sibling (the flag landed on one finger of a pair);
    3. sliding jaws on a rigid child of the flagged segment (no joint, or a fixed one) carrying at least two of
       them (the fingers hang under an adaptor plate);
    4. a coupled revolute pair (``coupled`` is the gene's ``coupled_joints``) whose palm is the flagged segment,
       one of its ancestors up to ``_PALM_SEARCH_DEPTH`` links up, or a rigid child of it.

    Cases 2 and 3 ask for a PAIR, because a single prismatic leaf one level away from the end effector is as
    likely to be a linear axis as a jaw. ``(ee_name, [])`` when none of the four finds jaws.
    """
    by = {s.name: s for s in segments}
    ee = by.get(ee_name) if ee_name else None
    if ee is None:
        return None, []
    jaws = jaws_on(segments, ee.name)
    if jaws:
        return ee.name, jaws
    if getattr(ee, "joint_type", None) == "prismatic" and ee.parent in by:
        pair = jaws_on(segments, ee.parent)
        if len(pair) >= 2:
            return ee.parent, pair
    rigid_children = [c for c in _children(segments, ee.name) if getattr(c, "joint_type", None) in (None, "fixed")]
    for child in rigid_children:
        pair = jaws_on(segments, child.name)
        if len(pair) >= 2:
            return child.name, pair
    pairs = _coupled_revolute_pairs(segments, coupled)
    if pairs:
        near = [ee.name]
        cur = ee
        for _ in range(_PALM_SEARCH_DEPTH):
            cur = by.get(cur.parent) if cur.parent else None
            if cur is None:
                break
            near.append(cur.name)
        near += [c.name for c in rigid_children]
        for name in near:
            if name in pairs:
                return name, pairs[name]
    return ee.name, []


def gripper_status(gene) -> dict:
    """Whether ``gene`` can hold an object, and the evidence, as one block every consumer reads.

    ``{"grips", "closable", "palm", "jaws", "declared", "evidence"}``:

    * ``grips``:     it carries jaws (found structurally), or it DECLARES a holding tool
                       (gripper / hand / suction: what a composer built, or what an importer read off names);
    * ``closable``:  SLIDING jaws sit on the flagged end effector itself, so the compiled body carries the
                       ``grasp_site`` our grasp scripts aim at and they can actually close on an object;
    * ``evidence``:  one sentence saying which of those is true and why, for an issue or a verdict.
    """
    segments = list(getattr(gene, "segments", None) or [])
    declared = str(getattr(gene, "end_effector_type", "") or "none").lower()
    ee = next((s for s in segments if getattr(s, "is_end_effector", False)), None)
    ee_name = ee.name if ee is not None else None
    palm, jaws = palm_and_jaws(segments, ee_name, getattr(gene, "coupled_joints", None))
    names = [j.name for j in jaws]
    sliding = bool(jaws) and all(getattr(j, "joint_type", None) == "prismatic" for j in jaws)
    closable = sliding and palm == ee_name
    out = {"grips": bool(jaws) or declared in HOLDING_TOOLS, "closable": closable, "palm": palm,
           "jaws": names, "declared": declared}
    if jaws:
        kind = "sliding jaws" if sliding else "coupled revolute jaws"
        out["evidence"] = f"{kind} {', '.join(repr(n) for n in names)} on {palm!r}"
        if not sliding:
            out["evidence"] += " (our grasp script closes sliding jaws only)"
        elif not closable:
            out["evidence"] += f" (the end effector is flagged on {ee_name!r}, not on the palm)"
    elif declared in HOLDING_TOOLS:
        out["evidence"] = (f"the model declares end_effector_type {declared!r}, but no jaws were found on its end "
                           f"effector {ee_name!r}")
    else:
        out["evidence"] = (f"no jaws were found on its end effector {ee_name!r}, and the model declares "
                           f"end_effector_type {declared!r}")
    return out
