"""In-app AI ASSISTANT core (docs/ai_native_plan.md P3) - the grounded, scoped-intent loop behind the chat
panel. A turn: (1) GROUND in the current robot/scene state, (2) classify the request's SCOPE and map it to
TYPED edit ops (parameter/feature -> localized edit, NEVER regenerate; structure -> confirm; scene -> theme;
question -> answer), (3) execute via the shared tool registry so the edit lands as one undoable step with a
DIFF. Returns a structured turn the UI renders (restatement + diff + artifacts + confirm/undo) - the
Cursor/Figma copilot contract.

Uses the configured LLM (``llm_client.get_llm('assistant')``) with a schema-constrained intent call when
available; falls back to a DETERMINISTIC keyword intent parser so the core capability ("make it taller",
"house instead of warehouse") works with a weak local model or none at all (and is testable offline). This is
the design mitigation for Ollama being weak: the LLM only has to pick an operator + args, or we infer them.
"""
from __future__ import annotations

import re

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {"type": "string", "enum": ["parameter", "feature", "structure", "scene", "question", "task"]},
        "target": {"type": "string", "enum": ["robot", "scene"]},
        "restatement": {"type": "string"},
        "ops": {"type": "array", "items": {"type": "object",
                "properties": {"op": {"type": "string"}, "args": {"type": "object"}}}},
        "needs_confirmation": {"type": "boolean"},
        "answer": {"type": "string"},
    },
    "required": ["scope", "restatement"],
}

_MATERIALS = {"carbon fiber": "carbon_fiber", "carbon-fiber": "carbon_fiber", "carbon": "carbon_fiber",
              "steel": "steel", "aluminum": "aluminum", "aluminium": "aluminum", "titanium": "titanium",
              "plastic": "abs_plastic"}


def _magnitude(msg: str, base: float = 1.2) -> float:
    """How much? 'a little/slightly' -> gentle, 'much/a lot/way' -> strong. Returns a >1 factor."""
    if re.search(r"\b(a little|slightly|a bit|a touch|tad)\b", msg):
        return 1.12
    if re.search(r"\b(much|a lot|way|far|significantly|really)\b", msg):
        return 1.4
    m = re.search(r"(\d+)\s*%|\bby\s+(\d+)", msg)
    if m:
        pct = int(m.group(1) or m.group(2)); return 1.0 + pct / 100.0
    return base


def _fallback_intent(msg: str, robot_id: str | None, scene_id: str | None) -> dict:
    """Deterministic keyword -> scoped intent (works with no LLM). Covers the common design-edit vocabulary."""
    m = msg.lower().strip()
    if re.search(r"\bundo\b", m):
        return {"scope": "feature", "target": "scene" if scene_id and not robot_id else "robot",
                "restatement": "Undo the last change", "ops": [{"op": "undo"}], "needs_confirmation": False}
    # SCENE theme swap. Prefer the theme named as the TARGET ("change to a HOUSE instead of a warehouse") over
    # the one it's replacing — parse the theme right after to/into/make-it; else the first theme mentioned.
    from virturoid.services.scene_themes import THEMES
    themes_in = [th for th in THEMES if re.search(rf"\b{th}\b", m)]
    if themes_in and (scene_id or re.search(r"\b(scene|environment|room|setting)\b", m)):
        tm = re.search(r"(?:to|into|make it)\s+(?:a\s+|an\s+)?(" + "|".join(THEMES) + r")\b", m)
        target_theme = tm.group(1) if tm else themes_in[0]
        return {"scope": "scene", "target": "scene", "restatement": f"Change the scene to a {target_theme}",
                "ops": [{"op": "swap_theme", "args": {"theme": target_theme}}], "needs_confirmation": False}
    # ROBOT edits
    if re.search(r"\b(taller|higher|longer legs|raise)\b", m):
        return {"scope": "parameter", "target": "robot", "restatement": "Make the robot taller (lengthen legs)",
                "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": _magnitude(m)}}],
                "needs_confirmation": False}
    if re.search(r"\b(shorter|lower legs|lower it)\b", m):
        return {"scope": "parameter", "target": "robot", "restatement": "Make the robot shorter",
                "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": 1.0 / _magnitude(m)}}],
                "needs_confirmation": False}
    if re.search(r"\b(thicker|stockier|sturdier|beefier|wider legs)\b", m):
        return {"scope": "parameter", "target": "robot", "restatement": "Make the robot's legs thicker",
                "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "girth", "factor": _magnitude(m)}}],
                "needs_confirmation": False}
    if re.search(r"\b(thinner|slimmer|leaner)\b", m):
        return {"scope": "parameter", "target": "robot", "restatement": "Make the robot's legs thinner",
                "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "girth", "factor": 1.0 / _magnitude(m)}}],
                "needs_confirmation": False}
    if re.search(r"\b(bigger|larger|scale up|grow)\b", m):
        return {"scope": "parameter", "target": "robot", "restatement": "Make the whole robot bigger",
                "ops": [{"op": "scale_robot", "args": {"factor": _magnitude(m)}}], "needs_confirmation": False}
    if re.search(r"\b(smaller|scale down|shrink)\b", m):
        return {"scope": "parameter", "target": "robot", "restatement": "Make the whole robot smaller",
                "ops": [{"op": "scale_robot", "args": {"factor": 1.0 / _magnitude(m)}}], "needs_confirmation": False}
    mh = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|metre|cm)\b.*\b(tall|high)|make it\s+(\d+(?:\.\d+)?)\s*(m|cm)", m)
    if mh:
        val = float(mh.group(1) or mh.group(4)); unit = (mh.group(2) or mh.group(5) or "m")
        tgt = val / 100.0 if unit.startswith("cm") else val
        return {"scope": "parameter", "target": "robot", "restatement": f"Make the robot ~{tgt} m tall",
                "ops": [{"op": "set_height", "args": {"target_m": tgt}}], "needs_confirmation": False}
    # CAPABILITY amend: "make it lift / carry N kg", "heavier payload", "carry more" -> upsize the actuators
    # (set_payload). Guard against the pure size words (bigger/taller) which are geometry, not load.
    if re.search(r"\b(lift|lifts|carry|carries|carrying|payload|heavier|stronger)\b", m) \
            and not re.search(r"\b(taller|shorter|bigger|larger|smaller)\b", m):
        mk = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilo|kilogram|kilograms|kgs|lb|lbs|pound|pounds)\b", m)
        if mk:
            val = float(mk.group(1)); unit = mk.group(2)
            payload = round(val * 0.4536, 2) if unit.startswith(("lb", "pound")) else round(val, 2)
        else:
            payload = 5.0                                       # "heavier / stronger" with no number -> a real default load
        payload = min(50.0, max(0.1, payload))
        return {"scope": "feature", "target": "robot", "restatement": f"Upsize the actuators to carry {payload} kg",
                "ops": [{"op": "set_payload", "args": {"payload_kg": payload}}], "needs_confirmation": False}
    for phrase, mat in _MATERIALS.items():
        if phrase in m:
            grp = "legs" if "leg" in m else ("torso" if re.search(r"\b(body|torso|frame)\b", m) else "all")
            return {"scope": "feature", "target": "robot", "restatement": f"Make the {grp} {mat.replace('_', ' ')}",
                    "ops": [{"op": "set_material", "args": {"group": grp, "material": mat}}], "needs_confirmation": False}
    mlc = re.search(r"\b(six|eight|four|\d+)\s*legs?\b|make it a (hexapod|octopod|quadruped)", m)
    if mlc:
        word = {"four": 2, "six": 3, "eight": 4, "quadruped": 2, "hexapod": 3, "octopod": 4}
        n = word.get(mlc.group(1) or mlc.group(2))
        if n is None and (mlc.group(1) or "").isdigit():
            n = max(1, int(mlc.group(1)) // 2)
        if n:
            return {"scope": "structure", "target": "robot", "restatement": f"Rebuild with {n*2} legs (structural)",
                    "ops": [{"op": "set_leg_count", "args": {"n_pairs": n}}], "needs_confirmation": True}
    return {"scope": "question", "target": "robot", "restatement": msg, "ops": [],
            "needs_confirmation": False, "answer": ""}


def _ground(robot_id: str | None, scene_id: str | None) -> dict:
    """Snapshot the CURRENT state the assistant edits against (Onshape/Figma: ground in what's open/selected)."""
    from virturoid.services import session_state as S
    ctx = {}
    if robot_id and S.get_robot(robot_id) is not None:
        from virturoid.services.ai_native_tools import _summary
        ctx["robot"] = {**_summary(S.get_robot(robot_id), robot_id), **S.robot_meta(robot_id)}
    if scene_id and S.get_scene(scene_id) is not None:
        ctx["scene"] = S.scene_meta(scene_id)
    return ctx


def _llm_intent(message: str, ctx: dict, llm) -> dict | None:
    """Ask the LLM to map the request to a scoped intent + typed ops (schema-constrained). None on any failure."""
    try:
        from virturoid.services.edit_operators import op_specs
        system = ("You edit an EXISTING robot/scene in a design tool. Map the user's request to a SCOPE and "
                  "TYPED ops over the current design - NEVER regenerate for a small edit. Robot ops: "
                  f"{op_specs()}. Scene op: swap_theme(theme in warehouse|house|kitchen|lab|yard). "
                  "structure scope needs_confirmation=true. For a question, set scope=question + answer.")
        user = f"Current design: {ctx}\nUser: {message}\nReturn the intent JSON."
        out = llm.complete_json(system, user, _INTENT_SCHEMA, max_tokens=400)
        return out if isinstance(out, dict) and out.get("scope") else None
    except Exception:  # noqa: BLE001
        return None


def handle_turn(message: str, *, robot_id: str | None = None, scene_id: str | None = None, llm="auto") -> dict:
    """One assistant turn. Grounds in current state, resolves a scoped intent (LLM if available else keyword
    fallback), executes typed ops via the shared tools, and returns a structured turn for the UI:
    ``{reply, restatement, scope, ops, applied, diffs, needs_confirmation, artifacts, robot_id, scene_id}``.
    A ``needs_confirmation`` op is NOT executed - the UI confirms, then re-calls with ``confirm=true`` in ops."""
    from virturoid.services.agent_tools import call_tool
    ctx = _ground(robot_id, scene_id)
    if llm == "auto":
        try:
            from virturoid.services.llm_client import get_llm
            llm = get_llm("assistant")
        except Exception:  # noqa: BLE001
            llm = None
    intent = (_llm_intent(message, ctx, llm) if llm else None) or _fallback_intent(message, robot_id, scene_id)
    scope = intent.get("scope", "question")
    target = intent.get("target", "robot")
    ops = intent.get("ops") or []
    turn = {"restatement": intent.get("restatement", message), "scope": scope, "ops": ops,
            "needs_confirmation": bool(intent.get("needs_confirmation")), "applied": False,
            "diffs": [], "artifacts": [], "robot_id": robot_id, "scene_id": scene_id}

    if scope == "question" or not ops:
        turn["reply"] = intent.get("answer") or (
            f"Current robot: {ctx.get('robot')}. Ask me to make it taller, thicker, a different material, "
            "give it more legs, or change the scene theme." if ctx else
            "Create a robot first (create_robot), then I can edit it incrementally.")
        return turn
    if turn["needs_confirmation"]:
        turn["reply"] = f"{turn['restatement']} - this is a STRUCTURAL change (bigger than a tweak). Confirm to apply."
        return turn

    # execute
    if target == "scene" or scope == "scene":
        if ops and ops[0].get("op") == "undo":
            r = call_tool("undo_robot" if robot_id and not scene_id else "edit_scene", {"scene_id": scene_id})
        else:
            r = call_tool("edit_scene", {"scene_id": scene_id, "ops": ops})
        if r.get("ok") and r["result"].get("ok"):
            turn["applied"] = True
            turn["reply"] = f"Done - {turn['restatement']}. Scene now themed '{r['result'].get('theme')}'."
        else:
            turn["reply"] = f"Could not apply: {(r.get('result') or {}).get('error') or r.get('error')}"
        return turn

    if ops and ops[0].get("op") == "undo":
        r = call_tool("undo_robot", {"robot_id": robot_id})
        turn["applied"] = bool(r.get("ok") and r["result"].get("ok"))
        turn["reply"] = "Reverted the last change." if turn["applied"] else "Nothing to undo."
        return turn

    r = call_tool("edit_robot", {"robot_id": robot_id, "ops": ops})
    res = r.get("result") or {}
    if r.get("ok") and res.get("ok"):
        turn["applied"] = True
        turn["diffs"] = res.get("diffs", [])
        turn["artifacts"] = res.get("artifacts", [])
        h = next((d.get("standing_height_m") for d in turn["diffs"] if d.get("standing_height_m")), None)
        extra = f" Height {h[0]} -> {h[1]} m." if h else ""
        turn["reply"] = f"Done - {turn['restatement']}.{extra} (one undo step; say 'undo' to revert)."
    else:
        turn["reply"] = f"Could not apply: {res.get('error') or r.get('error')}"  # teaching error surfaced
    return turn
