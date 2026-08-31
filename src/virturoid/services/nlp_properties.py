"""Parse a free-text robot description into TYPED, provenance-tagged properties + edit ops (Part B of the MVP
completion plan). A robotics team importing an existing robot rarely has every fact in the CAD/URDF -- they SAY
it: "the body is aluminum, the legs are carbon fiber, it carries a 5 kg payload, it's a 6-DOF arm". This turns
that sentence into the same TYPED edit operators the assistant already applies (set_material per group,
set_payload), each carrying the exact phrase it came from, so ``ingest_project`` can apply the user's stated
materials + load to the inferred RobotGene. Pure text -> structure; no physics, no LLM (deterministic + testable).

Non-goals (kept honest): DOF is REPORTED, not silently applied -- changing an arm's joint count is a structural
rebuild, not a localized edit, so we surface it as an intake note rather than fabricating joints.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# material phrase -> canonical material understood by edit_operators.set_material
_MATERIAL_ALIASES = {
    "carbon fiber": "carbon_fiber", "carbon-fiber": "carbon_fiber", "carbonfiber": "carbon_fiber",
    "carbon": "carbon_fiber", "cf": "carbon_fiber",
    "aluminium": "aluminum", "aluminum": "aluminum", "alu": "aluminum", "6061": "aluminum", "7075": "aluminum",
    "stainless steel": "steel", "stainless": "steel", "steel": "steel",
    "titanium": "titanium", "ti": "titanium",
    "abs plastic": "abs_plastic", "abs": "abs_plastic", "plastic": "abs_plastic", "polymer": "abs_plastic",
    "pla": "abs_plastic", "nylon": "abs_plastic", "3d printed": "abs_plastic", "3d-printed": "abs_plastic",
}
# body-part phrase -> canonical group understood by edit_operators (_GROUP_WORDS keys + "all")
_GROUP_ALIASES = {
    "legs": "legs", "leg": "legs", "thigh": "legs", "thighs": "legs", "shank": "legs", "shanks": "legs",
    "shin": "legs", "shins": "legs", "calf": "legs", "femur": "legs", "tibia": "legs",
    "arms": "arms", "arm": "arms", "forearm": "arms", "forearms": "arms", "wrist": "arms", "wrists": "arms",
    "elbow": "arms", "shoulder": "arms", "shoulders": "arms", "manipulator": "arms",
    "body": "torso", "torso": "torso", "trunk": "torso", "chest": "torso", "frame": "torso", "chassis": "torso",
    "base": "torso", "spine": "torso", "abdomen": "torso",
    "head": "head", "skull": "head", "snout": "head",
    "neck": "neck", "tail": "tail",
    "foot": "feet", "feet": "feet", "toe": "feet", "toes": "feet",
}
_LB_TO_KG = 0.45359237

# ---- payload vs the robot's OWN mass ---------------------------------------------------------------------
# A stated mass figure is one of three things and they must not be confused, because `set_payload` is not a
# label: it raises joint torques, re-grounds to bigger motors and adds mass, so a misread silently RE-SPECS the
# customer's robot -- and then emits its own "N joint(s) exceed the strongest catalog motor for this payload"
# warning off the number it invented.
_MASS_RE = r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|kilos?|kgs|lbs?|pounds?)\b"
# A payload NOUN may follow the figure ("2 kg payload", "10 kg load capacity"). Anchored, and whitespace-only
# before the noun: a COMMA starts a new clause, so "35 kg, payload 5 kg" must not label the 35.
_PAYLOAD_NOUN_AFTER = re.compile(
    r"^\s*(?:of\s+)?(?:max(?:imum)?\s+|rated\s+|useful\s+)?(payload|load|capacity)\b")
# A payload cue may precede it ("carries 5 kg", "payload: 10 kg", "rated for 8 kg"). Only searched BEFORE the
# figure: a verb AFTER it governs the NEXT number ("a 40 kg base that carries 5 kg" is a 5 kg payload).
_PAYLOAD_CUE_BEFORE = re.compile(
    r"\b(payload|payloads|carry|carries|carrying|carried|lift|lifts|lifting|haul|hauls|hauling|"
    r"hold|holds|holding|tote|totes|load|loads|capacity|rated for|rated to)\b")
# Words that mark the figure as the robot's OWN mass -- never a load it carries.
_SELF_MASS_CUE = re.compile(r"\b(weigh|weighs|weighed|weighing|weight|mass|masses|tare|curb|kerb|unladen)\b")


def _cue_window(text: str, start: int, end: int, back: int = 44, fwd: int = 30) -> tuple[str, str]:
    """The text just before/after a mass figure, TRUNCATED at the nearest OTHER mass figure.

    A cue word only binds to the number it actually sits beside. In "24 kg, 2 kg payload" the word 'payload'
    stands behind the 2 kg, so it must not reach back over that figure and mark the 24 kg as a payload too --
    that reach is exactly what let the robot's own mass win the old ``max()``."""
    before, after = text[max(0, start - back):start], text[end:min(len(text), end + fwd)]
    prev = None
    for mm in re.finditer(_MASS_RE, before):
        prev = mm                                                # keep the last (closest) preceding mass figure
    if prev:
        before = before[prev.end():]
    nxt = re.search(_MASS_RE, after)
    if nxt:
        after = after[:nxt.start()]
    return before, after


def _classify_mass(text: str, start: int, end: int) -> str:
    """``"payload"`` (explicitly labelled), ``"self_mass"`` (the robot's own weight) or ``"bare"`` (unmarked).

    Order matters: an anchored payload NOUN right after the figure is the strongest signal ("2 kg payload"),
    then an explicit weight word in front ("weighs 12 kg"), then a carry/lift cue in front ("carries 5 kg")."""
    before, after = _cue_window(text, start, end)
    if _PAYLOAD_NOUN_AFTER.match(after):
        return "payload"
    if _SELF_MASS_CUE.search(before):
        return "self_mass"
    if _PAYLOAD_CUE_BEFORE.search(before):
        return "payload"
    if _SELF_MASS_CUE.search(after):
        return "self_mass"
    return "bare"


@dataclass
class ExtractedProperties:
    """Typed, provenance-tagged read of a description + the edit ops that realize it."""
    materials: list[dict] = field(default_factory=list)          # [{group, material, evidence}]
    payload_kg: float | None = None
    payload_evidence: str | None = None
    dof: int | None = None
    dof_evidence: str | None = None
    ops: list[dict] = field(default_factory=list)                # set_material per group + set_payload -> apply_ops
    notes: list[str] = field(default_factory=list)               # honest caveats (e.g. DOF not auto-applied)

    def to_dict(self) -> dict:
        return {"materials": self.materials, "payload_kg": self.payload_kg, "payload_evidence": self.payload_evidence,
                "dof": self.dof, "dof_evidence": self.dof_evidence, "ops": self.ops, "notes": self.notes}


_GROUP_RE = "|".join(re.escape(w) for w in sorted(_GROUP_ALIASES, key=len, reverse=True))


def _find_group_near(text: str, start: int, end: int, window: int = 46) -> tuple[str | None, str | None]:
    """The body part a material phrase refers to, using ENGLISH ORDER, not raw distance (raw distance mis-pairs
    "aluminum body, carbon-fiber legs"). Priority: (1) adjective "<material> <group>" -- a group right after the
    material; (2) copula "<group> is/are/made of <material>" -- a group before, grammatically bound; (3) nearest
    within the window. Returns (canonical_group, matched_word) or (None, None) if the material stands alone."""
    after = text[end:min(len(text), end + window)]
    before = text[max(0, start - window):start]
    # 1) adjective: a group word immediately after the material (allow small fillers: "for the", "on its")
    ma = re.match(r"(?:\s+(?:for|on|the|its|of|a|an)\b)*\s*\b(" + _GROUP_RE + r")\b", after)
    if ma:
        return _GROUP_ALIASES[ma.group(1)], ma.group(1)
    # 2) copula: the closest group BEFORE the material, bound to it by is/are/made/of/from
    bound = None
    for m in re.finditer(r"\b(" + _GROUP_RE + r")\b", before):
        if re.search(r"\b(is|are|made|of|from|in|uses?|using)\b", before[m.end():]):
            bound = m                                            # keep the last (closest) copula-bound group
    if bound:
        return _GROUP_ALIASES[bound.group(1)], bound.group(1)
    return None, None                                            # unbound material ("titanium quadruped") -> whole robot


def extract_properties(description: str) -> ExtractedProperties:
    """Deterministic NLP -> typed properties + edit ops. Materials are paired to the nearest body part; a lone
    material ("made of aluminum") applies to the whole robot. DOF is parsed with unit handling and REPORTED.

    Payload is only applied when the description actually MARKS a figure as a load; the robot's own mass is
    never adopted, and an unmarked figure is surfaced in ``notes`` instead of guessed (see ``_classify_mass``)."""
    out = ExtractedProperties()
    if not description or not description.strip():
        return out
    text = " " + description.lower().strip() + " "

    # ---- materials, each paired to a body part by grammar ------------------------------------------------
    seen_groups: dict[str, str] = {}                             # group -> material (first mention wins)
    consumed: list[tuple[int, int]] = []                         # char spans already claimed by a longer phrase
    # longest aliases first so "carbon fiber" wins over "carbon", "stainless steel" over "steel"
    for phrase in sorted(_MATERIAL_ALIASES, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            if any(m.start() < ce and cs < m.end() for cs, ce in consumed):
                continue                                         # this is a substring of an already-matched material
            consumed.append((m.start(), m.end()))
            material = _MATERIAL_ALIASES[phrase]
            group, _gword = _find_group_near(text, m.start(), m.end())
            key = group or "all"
            if key in seen_groups:
                continue
            evidence = text[max(0, m.start() - 24):min(len(text), m.end() + 24)].strip()
            seen_groups[key] = material
            out.materials.append({"group": key, "material": material, "evidence": evidence})
    # if a lone-material "all" AND specific groups both exist, the specific groups are more informative; keep both
    for entry in out.materials:
        out.ops.append({"op": "set_material", "args": {"group": entry["group"], "material": entry["material"]}})

    # ---- payload -----------------------------------------------------------------------------------------
    # An EXPLICITLY LABELLED payload always beats a bare mass figure. This used to score each mass as
    # (cue-word-within-the-window?, kilograms) and take the ``max`` -- "among equals, the larger load (the
    # binding requirement)" -- but both numbers in a sentence fall inside the same +/-30 / 18-char window, so
    # the labelled value LOST to the bigger unlabelled one. Measured: "A telescoping arm, 24 kg, 2 kg payload."
    # applied 24 kg, and "Our Unitree G1 humanoid, 1.3 m tall, 35 kg, 29 joints." applied the robot's own 35 kg.
    # A bare figure is NEVER adopted silently: it is surfaced as a note (the ingest path relays `notes` to the
    # customer), the same way input_evidence turns a self-weight budget into a conflict + an intake question
    # instead of a silent value. Guessing here re-specs someone's actuators.
    candidates = []                                              # (kind, kg, evidence)
    for m in re.finditer(_MASS_RE, text):
        val = float(m.group(1)); unit = m.group(2)
        kg = round(val * _LB_TO_KG, 2) if unit.startswith(("lb", "pound")) else val
        ctx = text[max(0, m.start() - 30):min(len(text), m.end() + 18)].strip()
        candidates.append((_classify_mass(text, m.start(), m.end()), kg, ctx))
    labelled = [c for c in candidates if c[0] == "payload"]
    if labelled:
        # among GENUINE payloads the larger is still the binding requirement -- the original intent, now applied
        # only to figures that are actually marked as loads
        _, kg, ctx = max(labelled, key=lambda c: c[1])
        if 0.1 <= kg <= 50.0:
            out.payload_kg = kg
            out.payload_evidence = ctx
            out.ops.append({"op": "set_payload", "args": {"payload_kg": kg}})
        elif kg > 50.0:
            out.notes.append(f"stated payload {kg} kg exceeds the safe amend range (0.1-50 kg); needs a larger "
                             "actuator class / gearbox -- not auto-applied")
    elif any(c[0] == "bare" for c in candidates):
        figs = ", ".join(f"{c[1]:g} kg" for c in candidates if c[0] == "bare")
        out.notes.append(
            f"{figs} is stated, but nothing marks it as a PAYLOAD rather than the robot's own mass -- NOT "
            f"applied, because set_payload upsizes real actuators and adds mass. Say '<n> kg payload' or "
            f"'carries <n> kg' to apply it as a load, or 'weighs <n> kg' if it is the robot's own mass.")
    elif candidates:
        figs = ", ".join(f"{c[1]:g} kg" for c in candidates)
        out.notes.append(f"read {figs} as the robot's OWN mass, not a payload -- no payload applied")

    # ---- DOF (reported, not auto-applied) ----------------------------------------------------------------
    md = re.search(r"(\d+)\s*[-\s]?\s*(dof|d\.o\.f|degrees?\s+of\s+freedom|axis|axes)\b", text)
    if md:
        out.dof = int(md.group(1))
        out.dof_evidence = md.group(0).strip()
        out.notes.append(f"description states {out.dof} DOF -- reported for reconciliation with the imported model; "
                         "changing joint count is a structural rebuild, not applied as a localized edit")
    return out
