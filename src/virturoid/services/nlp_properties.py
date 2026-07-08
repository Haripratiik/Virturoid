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
    material ("made of aluminum") applies to the whole robot. Payload/DOF are parsed with unit handling."""
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
    # a number + mass unit; prefer one near a carry/lift/payload cue, else the largest stated mass
    candidates = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|kilos?|kgs|lbs?|pounds?)\b", text):
        val = float(m.group(1)); unit = m.group(2)
        kg = round(val * _LB_TO_KG, 2) if unit.startswith(("lb", "pound")) else val
        ctx = text[max(0, m.start() - 30):min(len(text), m.end() + 18)]
        cued = bool(re.search(r"\b(payload|carry|carries|lift|lifts|hold|holds|load|capacity)\b", ctx))
        candidates.append((cued, kg, ctx.strip()))
    if candidates:
        # a cued mass beats an uncued one; among equals, the larger load (the binding requirement)
        cued, kg, ctx = max(candidates, key=lambda c: (c[0], c[1]))
        if 0.1 <= kg <= 50.0:
            out.payload_kg = kg
            out.payload_evidence = ctx
            out.ops.append({"op": "set_payload", "args": {"payload_kg": kg}})
        elif kg > 50.0:
            out.notes.append(f"stated payload {kg} kg exceeds the safe amend range (0.1-50 kg); needs a larger "
                             "actuator class / gearbox -- not auto-applied")

    # ---- DOF (reported, not auto-applied) ----------------------------------------------------------------
    md = re.search(r"(\d+)\s*[-\s]?\s*(dof|d\.o\.f|degrees?\s+of\s+freedom|axis|axes)\b", text)
    if md:
        out.dof = int(md.group(1))
        out.dof_evidence = md.group(0).strip()
        out.notes.append(f"description states {out.dof} DOF -- reported for reconciliation with the imported model; "
                         "changing joint count is a structural rebuild, not applied as a localized edit")
    return out
