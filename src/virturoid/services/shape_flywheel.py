"""Shape-word flywheel — the SHAPE dialect of the corpus (Thesis A × Thesis B, docs/robotics_language_gen3d_research.md).

Thesis A: the flywheel is a physics-VERIFIED, embodiment-keyed program CORPUS, and every write is gated by a
verdict pinned to its config. Thesis B: the robot BODY is a language of parametric shape PROGRAMS. This module
is their intersection, concrete and testable WITHOUT an LLM:

  * bank_shape   — a build123d shape program (a tentacle, a mantle, a wing) that PASSES an un-gameable GEOMETRY
                   verdict (realizes a valid, positive-volume, bounded solid) is banked as a reusable CORPUS
                   WORD keyed by (role, body-morphology embedding), with the verdict + realizer CONFIG pinned.
  * recall_shape — a future body recalls the best VERIFIED program for a role on a structurally-similar body,
                   so a good tentacle COMPOUNDS across octopus builds instead of being re-authored from scratch.

The gate is the geometry verdict (a degenerate program can never enter the bank); retrieval is the grounding
(a role's best verified program is offered to the next body). Deterministic; CPU; standard MemoryDB persistence.
"""
from __future__ import annotations

import hashlib
import json
import re

SHAPE_TASK = "shape_program"
_MIN_SHAPE_SIMILARITY = 0.5   # a shape word only transfers to a structurally-similar body (mirror recall_gait)

# functional part roles a shape word can fill — the corpus's vocabulary. A segment name like ``tentacle1_l_0``
# resolves to the role ``tentacle`` so one octopus's verified tentacle grounds the NEXT octopus's tentacles.
_KNOWN_ROLES = {"tentacle", "leg", "arm", "torso", "body", "mantle", "head", "neck", "tail", "wing", "foot",
                "fin", "gripper", "hand", "finger", "hip", "thigh", "shank", "claw", "spine", "chassis",
                "wheel", "rotor", "shell", "carapace", "abdomen", "thorax", "beak", "snout", "flipper"}


def _segment_role(name: str) -> str:
    """Normalize a segment NAME to its functional ROLE (side/index-stripped): ``tentacle1_l_0`` -> ``tentacle``,
    ``leg2_r_1`` -> ``leg``, ``upper_arm_l`` -> ``arm``, ``mantle`` -> ``mantle``. A known role token wins over a
    positional prefix, so the corpus keys on WHAT the part is, not where it sits. Falls back to the first stem."""
    keep = []
    for tok in re.split(r"[_\s]+", str(name).lower()):
        stem = re.sub(r"\d+$", "", tok)                # tentacle1 -> tentacle, leg2 -> leg
        if not stem or stem in ("l", "r"):             # a pure side / index marker carries no role
            continue
        keep.append(stem)
    for t in keep:                                     # a real role token beats a positional prefix (upper/lower/…)
        if t in _KNOWN_ROLES:
            return t
    return keep[0] if keep else "part"


def _realizer_config() -> dict:
    """The config the geometry verdict was rendered under — PINNED with every write so a corpus row's verdict is
    reproducible (dossier risk #10: verdicts are config-relative). For shapes the 'sim' is the CAD realizer."""
    cfg = {"realizer": "build123d", "scale_mm": 1000, "verdict": "geometry"}
    try:
        import build123d
        cfg["version"] = getattr(build123d, "__version__", "?")
    except Exception:  # noqa: BLE001
        cfg["version"] = "absent"
    return cfg


def _declared_shape_is_empty(program: dict) -> bool:
    """A structural spec whose defining collection is empty/too-small would ONLY survive via realize_shape's
    capsule fallback — i.e. it would bank as a mislabeled primitive, not the shape it declares. Catch that so a
    ``loft`` word actually lofts and an ``extrude`` word actually has a polygon (corpus honesty, cheap, no CAD)."""
    fam = str(program.get("family") or "").lower()
    need = {"extrude": ("profile", 3), "revolve": ("profile", 2), "loft": ("sections", 2), "compound": ("parts", 1)}
    if fam in need:
        key, n = need[fam]
        return len(program.get(key) or []) < n
    return False   # parametric families (tapered/box/cylinder/capsule/sphere/role) carry safe defaults


def shape_verdict(program: dict, *, min_volume_cm3: float = 0.05, max_span_m: float = 3.0) -> dict:
    """UN-GAMEABLE geometry verdict for a shape program: it must realize a VALID, positive-volume, finitely-
    bounded solid AS THE SHAPE IT DECLARES. A degenerate / empty / exploded program (or one that would only
    survive as a capsule fallback) scores ``credible=False`` and can never be banked."""
    from virturoid.services.cad_geometry import realize_shape
    if not isinstance(program, dict) or not program.get("family"):
        return {"credible": False, "volume_cm3": 0.0, "span_m": 0.0, "reason": "no shape family declared"}
    if _declared_shape_is_empty(program):
        return {"credible": False, "volume_cm3": 0.0, "span_m": 0.0,
                "reason": f"empty/insufficient '{program.get('family')}' spec (would fall back to a primitive)"}
    try:
        solid = realize_shape(program)
        vol_mm3 = float(solid.volume)
        bb = solid.bounding_box()
        span = max(bb.size.X, bb.size.Y, bb.size.Z) / 1000.0   # mm -> m
        credible = bool(getattr(solid, "is_valid", True)) and vol_mm3 > min_volume_cm3 * 1000.0 and 0 < span < max_span_m
        return {"credible": credible, "volume_cm3": round(vol_mm3 / 1000.0, 3),
                "span_m": round(span, 3), "reason": "" if credible else "degenerate/exploded/empty solid"}
    except Exception as e:  # noqa: BLE001 - a program that won't even realize is not a word
        return {"credible": False, "volume_cm3": 0.0, "span_m": 0.0, "reason": f"{type(e).__name__}: {str(e)[:60]}"}


def _program_hash(role: str, program: dict) -> str:
    return hashlib.sha1(json.dumps([role, program], sort_keys=True, default=str).encode()).hexdigest()[:10]


def bank_shape(db, role: str, program: dict, *, gene=None, notes: str = "") -> str | None:
    """Bank a role's shape program as a corpus word IFF it passes the geometry verdict. Keyed by role; indexed by
    the body's morphology embedding for cross-body recall. Verdict + realizer config are pinned in base_config.
    Returns the skill_id, or ``None`` when the program is not a credible solid (the bank holds only real words)."""
    role = (role or "part").lower()
    v = shape_verdict(program)
    if not v["credible"]:
        return None
    skill_id = f"shape::{role}::{_program_hash(role, program)}"[:96]
    db.record_skill(
        skill_id, role, SHAPE_TASK, success_rate=min(1.0, v["volume_cm3"] / 100.0 + 0.5),
        species=(getattr(gene, "species", None) or role), gene_id=getattr(gene, "id", None),
        base_config={"shape_program": program, "controller": "shape_program", "role": role,
                     "verdict": v, "sim_config": _realizer_config()},
        notes=notes or "verified shape word (geometry-gated); base_config.shape_program IS the program")
    # index by THIS body's morphology so a structurally-similar body recalls this role's verified program
    if gene is not None:
        try:
            from virturoid.services.robotics_vector_memory import (SKILL, RoboticsVectorMemory, _body_latent,
                                                                    embed_skill)
            vm = RoboticsVectorMemory(db)
            vec = embed_skill(f"{SHAPE_TASK} {role}", gene, success_rate=1.0, latent=_body_latent(gene))
            vm.upsert(SKILL, skill_id, vec, {"robot_class": role, "task_type": SHAPE_TASK, "role": role})
        except Exception:  # noqa: BLE001 - vector index is an accelerant; the DB bank is the source of truth
            pass
    return skill_id


def _program_for_skill(db, skill_id: str) -> dict | None:
    row = db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
    if not row or not row["base_config"]:
        return None
    try:
        bc = json.loads(row["base_config"]) if isinstance(row["base_config"], str) else row["base_config"]
    except json.JSONDecodeError:
        return None
    return bc.get("shape_program") if isinstance(bc, dict) and bc.get("controller") == "shape_program" else None


def recall_shape(db, role: str, *, gene=None) -> dict | None:
    """Recall the best VERIFIED shape program for ``role``. Morphology-embedding retrieval FIRST (a
    structurally-similar body's verified program for this role), then the exact role match. None if nothing banked."""
    role = (role or "part").lower()
    if gene is not None:                                       # tokenized cross-body retrieval
        try:
            from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
            for hit in RoboticsVectorMemory(db).nearest_skills(gene, SHAPE_TASK, k=6, min_sim=_MIN_SHAPE_SIMILARITY):
                prog = _program_for_skill(db, hit.get("obj_id", ""))
                if prog and (hit.get("role") == role or True):
                    # keep only same-role hits
                    row = db.conn.execute("SELECT robot_class FROM skills WHERE skill_id=?",
                                          (hit.get("obj_id", ""),)).fetchone()
                    if row and (row["robot_class"] or "").lower() == role:
                        return prog
        except Exception:  # noqa: BLE001 - fall back to the exact role match
            pass
    sk = db.recall_skill(role, SHAPE_TASK)
    if sk:
        return _program_for_skill(db, sk.get("skill_id", ""))
    return None


def bank_body_shapes(db, gene, *, max_roles: int = 16) -> list[str]:
    """SELF-MANUFACTURING corpus write: bank one VERIFIED shape word per functional ROLE of a built body, keyed by
    that body's morphology. This is how the corpus grows on the ordinary build loop (the shape analog of
    ``_auto_bank_gait``): a body that composed a valid lofted mantle + tapered tentacles contributes a ``mantle``
    word and a ``tentacle`` word that the NEXT octopus recalls instead of re-deriving. Dedup by role (one
    representative program per role, not eight near-identical tentacles); each still passes ``shape_verdict`` at
    ``bank_shape``. Best-effort + keep-best; returns the banked skill_ids."""
    banked, seen = [], set()
    for s in getattr(gene, "segments", []) or []:
        program = getattr(s, "geometry", None)
        if not isinstance(program, dict) or not program:       # only segments carrying a shape PROGRAM
            continue
        role = _segment_role(getattr(s, "name", "") or "part")
        if role in seen or len(seen) >= max_roles:
            continue
        seen.add(role)
        sid = bank_shape(db, role, program, gene=gene, notes=f"auto-banked from {getattr(gene, 'id', '?')}")
        if sid:
            banked.append(sid)
    return banked


def _default_db():
    from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return MemoryDB(DEFAULT_DB_PATH)


def auto_bank_body_shapes(gene) -> list[str]:
    """Open the default corpus DB and bank a built body's verified shape words. The single call a build seam makes;
    never raises (a missing build123d / awkward solid just banks fewer words). Returns the banked skill_ids."""
    try:
        with _default_db() as db:
            return bank_body_shapes(db, gene)
    except Exception:  # noqa: BLE001 - corpus growth is an accelerant, never a build blocker
        return []


# ── Agent-facing grounding surface (Thesis A: retrieval = runtime grounding) ─────────────────────────────────
# The corpus is exposed to the EXTERNAL LLM (Claude/Codex via MCP) as two tools: recall a role's best VERIFIED
# shape program to GROUND its next design (an exemplar to adapt, not a blank prompt — the RoboMorph anti-mode-
# collapse point), and bank a shape it authored (gated by the same un-gameable geometry verdict, which is also
# honest FEEDBACK: "your program is degenerate/exploded"). Zero product-side LLM tokens; the DB is the corpus.

def _recall_shape_word(args: dict) -> dict:
    role = _segment_role(str(args.get("role") or "part"))
    gene = None
    rid = args.get("robot_id")
    if rid:                                                    # morphology-nearest recall against a held body
        try:
            from virturoid.services import session_state as S
            held = S.get_robot(rid) if hasattr(S, "get_robot") else None
            gene = held.get("gene") if isinstance(held, dict) else (getattr(held, "gene", None) or held)
        except Exception:  # noqa: BLE001 - fall back to role-exact recall
            gene = None
    with _default_db() as db:
        prog = recall_shape(db, role, gene=gene)
        if prog is None:
            return {"role": role, "found": False,
                    "note": f"no verified '{role}' shape word banked yet — author one and it will ground future builds"}
        v = shape_verdict(prog)
        return {"role": role, "found": True, "shape_program": prog, "verdict": v,
                "note": "a physics/geometry-VERIFIED exemplar for this role — adapt it, don't start blank"}


def _bank_shape_word(args: dict) -> dict:
    role = _segment_role(str(args.get("role") or "part"))
    program = args.get("shape_program") or args.get("program")
    if not isinstance(program, dict) or not program:
        return {"banked": False, "error": "shape_program (a build123d shape-program dict) is required"}
    v = shape_verdict(program)
    with _default_db() as db:
        sid = bank_shape(db, role, program)
    return {"banked": bool(sid), "skill_id": sid, "role": role, "verdict": v,
            "note": ("banked as a corpus word — it will ground future builds of this role" if sid
                     else f"rejected by the geometry verdict: {v.get('reason')}")}


SHAPE_TOOLS = {
    "recall_shape_word": {
        "description": "Thesis A grounding: recall the best PHYSICS-VERIFIED shape program for a functional part "
                       "role (tentacle/leg/mantle/wing/...) to ground a new design with a proven exemplar instead "
                       "of authoring blind. Keyed by role + (optional) a held robot's morphology. Zero LLM tokens.",
        "parameters": {"type": "object", "required": ["role"], "properties": {
            "role": {"type": "string", "description": "functional part role, e.g. 'tentacle', 'leg', 'mantle'"},
            "robot_id": {"type": "string", "description": "optional held robot for morphology-nearest recall"}}},
        "handler": _recall_shape_word,
    },
    "bank_shape_word": {
        "description": "Bank a build123d shape PROGRAM you authored as a reusable corpus word for a part role. "
                       "Gated by an un-gameable geometry verdict (must realize a valid, positive-volume, bounded "
                       "solid) — the verdict is also honest feedback when a program is degenerate.",
        "parameters": {"type": "object", "required": ["role", "shape_program"], "properties": {
            "role": {"type": "string"}, "shape_program": {"type": "object",
                "description": "a shape-program dict: {family: tapered|loft|extrude|revolve|compound, ...}"}}},
        "handler": _bank_shape_word,
    },
}
