"""LLM ANATOMY DESIGNER — the intelligent front of the general path: the LLM, which actually KNOWS what a dog
(or bird, horse, lizard, crab) looks like, describes the creature's ANATOMY as a structured graph; the general
``anatomy_compiler`` then builds the robot geometry around it. No per-species code.

The LLM owns STRUCTURE (which parts, their roles, how they attach, relative proportions, joints, symmetry) —
the thing LLMs are genuinely good at (semantic knowledge of anatomy). It never emits a coordinate, so it can't
produce the beaded-noodle / scattered-block failures that raw-geometry generation did. This is "the LLM
outputs a dog's anatomy and our agents build parts around it," made reliable.
"""

from __future__ import annotations

_PART = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "role": {"type": "string"},
        "parent": {"type": "string"},
        "attach": {"type": "string"},
        "aim": {"type": "string"},
        "size": {"type": "number"},
        "girth": {"type": "number"},
        "segments": {"type": "integer"},
        "joint": {"type": "string", "enum": ["fixed", "revolute"]},
        "symmetry": {"type": "string", "enum": ["none", "left_right"]},
        "aspect": {"type": "string", "enum": ["long", "wide", "flat", "round"]},
        "curl": {"type": "number"},
        "detail": {"type": "string", "enum": ["smooth", "paneled", "vented", "rugged"]},
        "thickness": {"type": "number"},
        "chamfer": {"type": "number"},
    },
    "required": ["name", "role"],
}
ANATOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "robot_class": {"type": "string"},
        "name": {"type": "string"},
        "parts": {"type": "array", "items": _PART},
    },
    "required": ["robot_class", "parts"],
}

ANATOMY_SYSTEM = (
    "You are Virturoid's anatomy designer. You design ROBOTS — actual machines — in the FORM of the requested "
    "creature. A 'dog' means a ROBOT dog (think Boston Dynamics Spot or Unitree Go1), a 'humanoid' means a "
    "robot like Figure/Atlas — NOT a living animal. So translate biology into MECHANISM: the torso is a "
    "rigid CHASSIS that houses the battery and electronics; the head is a SENSOR module carrying cameras (use "
    "role 'head' — our compiler renders it as a mechanical sensor head, not an organic skull); every movable "
    "joint is a motor/ACTUATOR. Do NOT add purely decorative organic parts (a soft snout, floppy ears, fur, "
    "eyes as eyeballs) — only include a part if it serves a robotic function (a sensor housing, a gripper, a "
    "balancing tail, a stabilising antenna). Keep the limb/segment COUNT close to the real animal's so it is "
    "recognisable, but every part is hardware.\n"
    "Describe the robot's BODY as a structured ANATOMY GRAPH — a list of parts. Our compiler turns your graph "
    "into connected, credible ORIGINAL ROBOT geometry; you ONLY describe the anatomy, NEVER coordinates or meshes.\n"
    "Each part has: name; role; parent (the part it attaches to — OMIT for the single root 'body'); attach "
    "(where ON the parent: front/rear/top/bottom/left/right, the longitudinal anchors front/front_mid/mid/"
    "rear_mid/rear, or combos front_bottom/front_mid_bottom/rear_mid_bottom/rear_bottom/front_top/tip); "
    "aim (world direction it points: forward/back/up/down/forward_up/forward_down/back_up/back_down, or the "
    "diagonal-lateral fans forward_out/back_out/forward_down_out/back_down_out/down_out/out); size (its length "
    "in METERS); girth (its thickness in meters); segments (number of joints in a LIMB — a walking leg = 3); "
    "joint (revolute=movable, fixed=rigid); symmetry (left_right to MIRROR a paired limb to both sides); "
    "aspect (ROOT BODY ONLY — the body plan: omit/'long' = a normal long-front-to-back animal barrel; 'wide' "
    "= a broad carapace, e.g. crab/beetle; 'flat' = a low wide disc, e.g. turtle/ray; 'round' = a bulbous sac, "
    "e.g. an octopus mantle or a spider's abdomen); curl (a MULTI-SEGMENT part's total rest-bend in radians so "
    "a chain arches instead of going straight — e.g. a scorpion's raised tail ~2.0, a curled trunk ~1.2; "
    "omit/0 = straight).\n"
    "CAD CONTROL — shape each part to fit the robot AND its task: detail (a part's mechanical finish: 'smooth' "
    "= sleek/clean for a drone or light robot; 'paneled' = chamfered machined edges; 'vented' = chamfered + "
    "lightening/cooling cutouts; 'rugged' = heavy industrial); thickness (cross-section multiplier — 1.0 "
    "normal, >1 (up to ~2) for a LOAD-BEARING limb on a heavy-lift/strong robot, <1 for a light/agile/flying "
    "one); chamfer (explicit edge bevel in m). Use these intentionally: a heavy loader's legs = thickness ~1.6 "
    "+ detail 'rugged'; a racing drone's arms = thickness ~0.7 + detail 'smooth'.\n"
    "ROLE vocabulary (reusable across all animals): body, neck, head, snout, jaw, ear, eye, horn, beak, tail, "
    "leg, arm, tentacle, trunk, paw, foot, hand, wing, fin, flipper, claw, antenna, shell.\n"
    "RULES: exactly ONE root part with role 'body' and no parent. For a normal animal the body is LONG "
    "front-to-back and only modestly wide (never a cube/tower) — for a crab/turtle/ray/octopus set the body's "
    "aspect instead. Use symmetry:left_right for paired limbs (give ONE 'leg' — it becomes both sides; do NOT "
    "also list the mirror). A walking leg = role 'leg', segments 3, joint revolute, aim 'down', attach "
    "front_bottom (front pair) or rear_bottom (hind pair). A MANY-LEGGED creature (spider/crab/insect = 4 leg "
    "pairs) spreads its pairs across attach front_bottom, front_mid_bottom, rear_mid_bottom, rear_bottom and "
    "fans them outward with forward_down_out / down_out / back_down_out. Keep appendage sizes SMALLER than the "
    "body. Adapt the proportions to the animal (a bird has wings+beak; a horse longer legs; a lizard a long "
    "tail + low stance).\n"
    "Follow this EXACT structure (this is a ROBOT dog — note: a sensor head, NO snout/ears, a short stabilising "
    "neck, an antenna; ADAPT roles/sizes for the requested robot, do not copy blindly):\n"
    "{\"robot_class\":\"quadruped\",\"name\":\"dog\",\"parts\":[\n"
    "  {\"name\":\"chassis\",\"role\":\"body\",\"size\":0.55,\"girth\":0.16},\n"
    "  {\"name\":\"neck\",\"role\":\"neck\",\"parent\":\"chassis\",\"attach\":\"front_top\",\"aim\":\"forward_up\",\"size\":0.1,\"girth\":0.06},\n"
    "  {\"name\":\"sensor_head\",\"role\":\"head\",\"parent\":\"neck\",\"attach\":\"tip\",\"aim\":\"forward\",\"size\":0.13,\"girth\":0.08},\n"
    "  {\"name\":\"antenna\",\"role\":\"antenna\",\"parent\":\"sensor_head\",\"attach\":\"top\",\"aim\":\"up\",\"size\":0.06,\"girth\":0.01},\n"
    "  {\"name\":\"tail\",\"role\":\"tail\",\"parent\":\"chassis\",\"attach\":\"rear_top\",\"aim\":\"back_up\",\"size\":0.14,\"girth\":0.025,\"joint\":\"revolute\"},\n"
    "  {\"name\":\"front_leg\",\"role\":\"leg\",\"parent\":\"chassis\",\"attach\":\"front_bottom\",\"aim\":\"down\",\"size\":0.26,\"girth\":0.05,\"segments\":3,\"symmetry\":\"left_right\",\"joint\":\"revolute\"},\n"
    "  {\"name\":\"hind_leg\",\"role\":\"leg\",\"parent\":\"chassis\",\"attach\":\"rear_bottom\",\"aim\":\"down\",\"size\":0.28,\"girth\":0.055,\"segments\":3,\"symmetry\":\"left_right\",\"joint\":\"revolute\"}]}\n"
    "Output ONLY the JSON anatomy graph for the requested robot."
)


def propose_anatomy(prompt: str, plan, req, llm) -> dict:
    """Ask the LLM for a creature's anatomy graph; normalize/clamp it so the compiler always gets a sane graph.
    Raises on a hard LLM/parse failure (the caller falls back to the coarse intent / keyword builders)."""
    user = (f"Creature / robot requested: {prompt}\n"
            f"(nearest robot_class from the planner: {getattr(plan, 'robot_class', 'quadruped')}). "
            f"Describe its full anatomy as the JSON graph.")
    raw = llm.complete_json(ANATOMY_SYSTEM, user, ANATOMY_SCHEMA, max_tokens=1600) or {}
    parts_in = raw.get("parts") or []
    parts: list[dict] = []
    seen = set()
    for p in parts_in[:40]:                       # cap the part count (a creature is < ~40 parts)
        if not isinstance(p, dict) or not p.get("role"):
            continue
        name = str(p.get("name") or p.get("role"))
        if name in seen:
            name = f"{name}_{len(parts)}"
        seen.add(name)
        parts.append({
            "name": name,
            "role": str(p.get("role")).lower().strip(),
            "parent": (str(p["parent"]).strip() if p.get("parent") else None),
            "attach": str(p.get("attach") or "mid").lower().strip(),
            "aim": str(p.get("aim") or "").lower().strip(),
            "size": _clamp(p.get("size"), 0.02, 1.5, 0.12),
            "girth": _clamp(p.get("girth"), 0.004, 0.5, 0.0),
            "segments": int(p.get("segments")) if str(p.get("segments") or "").isdigit() else 1,
            "joint": "revolute" if str(p.get("joint") or "").lower() == "revolute" else (
                "fixed" if str(p.get("joint") or "").lower() == "fixed" else None),
            "symmetry": "left_right" if str(p.get("symmetry") or "").lower() == "left_right" else "none",
            "aspect": (str(p.get("aspect")).lower().strip()
                       if str(p.get("aspect") or "").lower().strip() in ("wide", "flat", "round", "long") else None),
            "curl": _clamp(p.get("curl"), -3.0, 3.0, 0.0),
            "detail": (str(p.get("detail")).lower().strip()
                       if str(p.get("detail") or "").lower().strip() in ("smooth", "paneled", "vented", "rugged") else None),
            "thickness": _clamp(p.get("thickness"), 0.4, 2.5, 1.0),
            "chamfer": _clamp(p.get("chamfer"), 0.0, 0.05, 0.0),
        })
    # Guarantee exactly one root body (the compiler needs it): if none has a null parent, force the first
    # 'body' (or the first part) to be the root.
    if parts and not any(not p["parent"] for p in parts):
        root = next((p for p in parts if p["role"] == "body"), parts[0])
        root["parent"] = None
    return {"robot_class": str(raw.get("robot_class") or getattr(plan, "robot_class", "quadruped")),
            "name": str(raw.get("name") or "creature"), "parts": parts, "source": "llm_anatomy"}


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default
