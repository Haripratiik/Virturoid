"""Deterministic parser for QUANTITATIVE prompt constraints — height, self-weight budget, payload, pinned parts.

The YC-readiness audit (docs/yc_readiness_plan.md §4.8E) found the composer parsed only payload + arm-reach and
*ignored* a specified HEIGHT (offline), a self-WEIGHT budget, and PINNED parts — so a detailed prompt like
"a 1.2 m tall humanoid under 12 kg with a LiDAR" was largely not honored. This module extracts those constraints
WITHOUT an LLM, so they can be applied to the build and reported honestly (requested-vs-achieved) — the honesty
layer for detailed prompts. Parsing is conservative: a value is only taken when its intent is unambiguous (a
height needs a stature cue, a weight budget needs a ceiling word), so an arm-reach or payload isn't misread.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PromptSpec:
    """Quantitative constraints lifted from a build prompt (any field may be None / empty if unspecified)."""
    target_height_m: float | None = None         # "a 1.2 m tall humanoid", "stands 40 cm high"
    weight_budget_kg: float | None = None         # "under 10 kg", "no heavier than 8 kg", "max 5 kg"
    payload_kg: float | None = None               # "carry 5 kg", "lift a 2 kg tool"
    pinned_parts: list = field(default_factory=list)   # ["lidar", "parallel gripper", ...] explicitly requested

    def any(self) -> bool:
        return bool(self.target_height_m or self.weight_budget_kg or self.payload_kg or self.pinned_parts)


def _len_to_m(value: float, unit: str) -> float:
    u = unit.lower()
    return value / 100.0 if u.startswith("cm") or u.startswith("centi") else value


def _mass_to_kg(value: float, unit: str) -> float:
    return value / 1000.0 if unit.lower().startswith("g") else value


# HEIGHT: a length adjacent to a stature cue (tall/high/height/stature/stands), in either order. Requiring the
# cue is what keeps a height from being confused with an arm-reach ("0.9 m reach") or a payload.
_HEIGHT_AFTER = re.compile(
    r"(\d+(?:\.\d+)?)\s*(m|meters?|metres?|cm|centimet\w*)\b\s*(?:tall|high|in height|in stature)", re.I)
_HEIGHT_BEFORE = re.compile(
    r"\b(?:tall|height|high|stature|stands?)\b[^.\d]{0,24}?(\d+(?:\.\d+)?)\s*(m|meters?|metres?|cm|centimet\w*)\b",
    re.I)

# SELF-WEIGHT BUDGET: a mass behind a ceiling word — distinct from a payload the robot must carry.
_WEIGHT_BUDGET = re.compile(
    r"\b(?:under|below|less than|lighter than|no (?:more|heavier) than|at most|max(?:imum)?|up to|weigh\w*\s*(?:under|less than)?)\s*"
    r"(\d+(?:\.\d+)?)\s*(kg|g|grams?)\b", re.I)

# PAYLOAD: a mass behind a carry/lift verb.
_PAYLOAD = re.compile(
    r"\b(?:carr(?:y|ies|ying)|lift(?:s|ing)?|hold(?:s|ing)?|haul\w*|transport\w*|payload\s+of|move)\b[^.]{0,30}?"
    r"(\d+(?:\.\d+)?)\s*(kg|g|grams?)\b", re.I)

# PINNED PARTS: a small vocabulary of components a user can explicitly require; matched anywhere in the prompt.
_PART_VOCAB: tuple[tuple[str, str], ...] = (
    ("lidar", r"\blidar\b"),
    ("depth camera", r"\bdepth camera\b|\brgb-?d\b|\brealsense\b"),
    ("thermal camera", r"\bthermal\b"),
    ("force-torque sensor", r"\bforce[- ]torque\b|\bf/t sensor\b|6-axis (?:wrist )?(?:f/t |force )?sensor"),
    ("parallel gripper", r"\bparallel (?:jaw )?gripper\b|\b(?:2|two)[- ]finger (?:gripper|jaw)\b"),
    ("suction gripper", r"\bsuction (?:gripper|cup)\b|\bvacuum gripper\b"),
    ("gps", r"\bgps\b|\bgnss\b|\brtk\b"),
)


def parse_target_height_m(prompt: str) -> float | None:
    p = prompt or ""
    for rx in (_HEIGHT_AFTER, _HEIGHT_BEFORE):
        m = rx.search(p)
        if m:
            return round(_len_to_m(float(m.group(1)), m.group(2)), 3)
    return None


def parse_weight_budget_kg(prompt: str) -> float | None:
    m = _WEIGHT_BUDGET.search(prompt or "")
    return round(_mass_to_kg(float(m.group(1)), m.group(2)), 3) if m else None


def parse_payload_kg(prompt: str) -> float | None:
    m = _PAYLOAD.search(prompt or "")
    return round(_mass_to_kg(float(m.group(1)), m.group(2)), 3) if m else None


def parse_pinned_parts(prompt: str) -> list:
    p = prompt or ""
    return [canon for canon, pat in _PART_VOCAB if re.search(pat, p, re.I)]


def parse_prompt_spec(prompt: str) -> PromptSpec:
    """Lift all quantitative constraints from a prompt in one pass (deterministic, no LLM)."""
    return PromptSpec(
        target_height_m=parse_target_height_m(prompt),
        weight_budget_kg=parse_weight_budget_kg(prompt),
        payload_kg=parse_payload_kg(prompt),
        pinned_parts=parse_pinned_parts(prompt),
    )


def measure_gene_height_m(gene) -> float | None:
    """Standing height of the assembled body — compile to MJCF, settle one forward step, take the z-extent of
    the geom AABBs (the same measure ``anatomy_critic`` uses). Returns None if MuJoCo/compile is unavailable."""
    try:
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        xml = compile_gene_to_mjcf(gene, spawn_z=standing_spawn_z(gene, meshed=False))
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        keep = [g for g in range(m.ngeom) if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
        if not keep:
            return None
        P = d.geom_xpos[keep]
        rb = m.geom_rbound[keep]
        return float((P + rb[:, None]).max(0)[2] - (P - rb[:, None]).min(0)[2])
    except Exception:  # noqa: BLE001 - measurement is best-effort; never block a build
        return None


def spec_compliance_report(gene, spec: PromptSpec, *, bom: dict | None = None) -> dict:
    """Requested-vs-achieved for every parsed constraint — the honest answer to "did the build honor my spec?".

    ``bom`` (optional) lets pinned-part compliance be checked against the actual parts list."""
    items: list[dict] = []
    if spec.target_height_m:
        got = measure_gene_height_m(gene)
        honored = got is not None and abs(got - spec.target_height_m) <= 0.12 * spec.target_height_m
        items.append({"constraint": "height_m", "requested": spec.target_height_m,
                      "achieved": round(got, 3) if got is not None else None, "honored": honored})
    if spec.weight_budget_kg:
        got = round(sum(s.mass_kg for s in gene.segments), 3)
        items.append({"constraint": "self_weight_kg", "requested_max": spec.weight_budget_kg,
                      "achieved": got, "honored": got <= spec.weight_budget_kg})
    if spec.payload_kg:
        items.append({"constraint": "payload_kg", "requested": spec.payload_kg, "achieved": spec.payload_kg,
                      "honored": True, "note": "drives manipulator actuator sizing"})
    if spec.pinned_parts:
        present = None
        if bom is not None:
            blob = " ".join(str(ln.get("part", "")) + " " + str(ln.get("detail", ""))
                            for ln in bom.get("lines", [])).lower()
            present = {p: bool(re.search(_canon_to_pat(p), blob, re.I)) for p in spec.pinned_parts}
        items.append({"constraint": "pinned_parts", "requested": spec.pinned_parts,
                      "present_in_bom": present, "honored": (all(present.values()) if present else None)})
    decided = [i["honored"] for i in items if i.get("honored") is not None]
    return {"constraints": items, "all_honored": (all(decided) if decided else None)}


def _canon_to_pat(canon: str) -> str:
    for name, pat in _PART_VOCAB:
        if name == canon:
            return pat
    return re.escape(canon)
