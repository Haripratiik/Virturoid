"""Design-intent compiler — the reliable, general way the LLM designs a robot body.

THE PROBLEM (measured across many gpt-5.2 runs): when the LLM hand-places every capsule's length / radius /
offset / shape, it produces a DIFFERENT defect every run — a 38-segment "beaded noodle", scattered
disconnected blocks, or five bare finger boxes welded onto a foot. Raw-geometry LLM design is fundamentally
unreliable, and the coarse geometric critic rubber-stamps the broken bodies.

THE FIX (the architecture both CAD deep-research reports converge on — report 7's "intent layer → custom-part
generation layer"): the LLM makes the HIGH-LEVEL design decision — robot class, overall scale, how many of
which appendages, an end effector — a small, structured, reliable choice. A DETERMINISTIC builder then
synthesizes the geometry from our proven parametric library (the anthropometric-humanoid loft builder, the
N-legged composed template, the spider/snake/frog archetypes). The body is connected, correctly-proportioned,
and role-shaped BY CONSTRUCTION. The LLM still DESIGNS (it picks the body plan); it just doesn't get to weld a
finger to a foot. This is general across the morphology families AND credible — the combination raw geometry
could never reliably hit. ORIGINAL-generation mandate is preserved: every builder emits our own geometry, no
copied real-robot meshes.
"""

from __future__ import annotations

import math as _m

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "robot_class": {"type": "string",
                        "enum": ["humanoid", "biped", "quadruped", "legged", "manipulator", "mobile_base"]},
        "archetype": {"type": "string", "enum": ["spider", "snake", "frog", "none"]},
        "scale_m": {"type": "number"},          # characteristic size (humanoid stature / animal body length / arm reach)
        "n_legs": {"type": "integer"},          # legged bodies: 2/4/6/8 (spider=8, insect/hexapod=6, dog=4)
        "n_arms": {"type": "integer"},          # humanoid manipulating arms (usually 2)
        "arm_dof": {"type": "integer"},         # manipulator reaching DOF (2-7)
        "has_head": {"type": "boolean"},
        "end_effector": {"type": "string", "enum": ["gripper", "hand", "spray", "suction", "hook", "none"]},
        "payload_kg": {"type": "number"},       # what it must lift/carry -> seeds real actuator sizing
        # OPTIONAL distinctive EXTRA appendages on top of the base body's legs/arms — what makes a crab a crab
        # (claws), a scorpion a scorpion (claws + tail). The deterministic builder owns all geometry/placement;
        # the LLM only names the kind/count/mount. Legs themselves stay in robot_class/n_legs, not here.
        "appendages": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["claw", "tail", "antenna", "mandible", "stinger"]},
            "count": {"type": "integer"},
            "mount": {"type": "string", "enum": ["front", "rear", "top"]},
            "scale": {"type": "number"}}}},
        # OPTIONAL per-segment length nudges in mm (each independently clamped by the builder); omit to use
        # builder defaults. Keys are class-specific; unknown/absent keys are ignored.
        "nominal_dims": {"type": "object", "properties": {
            "torso_mm": {"type": "number"}, "upper_arm_mm": {"type": "number"}, "forearm_mm": {"type": "number"},
            "thigh_mm": {"type": "number"}, "shin_mm": {"type": "number"},
            "body_mm": {"type": "number"}, "leg_mm": {"type": "number"}, "reach_mm": {"type": "number"}}},
    },
    "required": ["robot_class", "scale_m"],
}

# Lean, reasoning-model-friendly: the injected schema already carries every enum/field, so the prompt only
# adds the one-line gloss the schema can't express. ASCII bullets (Unicode bullets tokenize as standalone
# tokens). ~230 system tokens, paid on every product build.
INTENT_SYSTEM = (
    "You are Virturoid's morphology designer. Choose a BODY PLAN for the requested robot - never raw geometry. "
    "Our deterministic builders turn your plan into credible, connected, ORIGINAL geometry, so pick only the "
    "high-level design. The design must be original and generic: never name, copy, or imitate a specific real "
    "or branded robot.\n\n"
    "Fill the schema fields (their enums are authoritative). Guidance only where the schema cannot say it:\n"
    "- robot_class: the nearest body family. 'legged' = any non-quadruped multi-legged animal (spider, insect, "
    "octopod, crab).\n"
    "- archetype: name spider/snake/frog ONLY when the request clearly IS one (dedicated builder); else 'none'.\n"
    "- scale_m: characteristic size in meters - a humanoid's standing height, an animal's body length, or a "
    "manipulator's reach.\n"
    "- n_legs for legged/quadruped; arm_dof for a manipulator; payload_kg if it must lift/carry something.\n"
    "- nominal_dims (optional): nudge individual limb lengths in mm ONLY when the request implies unusual "
    "proportions (a long-reach arm, a compact picker); omit otherwise.\n"
    "- appendages: ALWAYS give an animal its DEFINING extra parts, even for a terse prompt — a crab or lobster "
    "ALWAYS has 2 'claw' (mount front); a scorpion ALWAYS 2 'claw' + a 'tail' + a 'stinger' (rear); a mantis 2 "
    "'claw' (raptorial); ants/beetles/insects 'antenna' (front). Omit only for a plain machine with none. The "
    "walking legs go in n_legs, NOT here (a crab/scorpion has 8 legs -> n_legs 8, class 'legged').\n"
    "Output ONLY the JSON plan."
)


def propose_intent(prompt: str, plan, req, llm, *, errors: list[str] | None = None) -> dict:
    """Ask the LLM for a structured body-plan intent, defaulted from the build planner + requirements so a
    sparse answer is still complete. Raises on a hard LLM/parse failure (the caller then uses the keyword
    builder). The planner's class is the default; the LLM may refine it. ``errors`` (from a prior plan that
    the validation gate flagged) drives a one-shot REPAIR — they are fed back so the LLM corrects the plan."""
    user = (f"Request: {prompt}\n"
            f"(build planner suggests class={getattr(plan, 'robot_class', '?')}, "
            f"morphology={getattr(plan, 'morphology', '')!r}; reach≈{getattr(req, 'reach_m', None)} m). "
            f"Design the body plan.")
    if errors:
        user += ("\nYour PREVIOUS plan built a body that FAILED validation for: "
                 + "; ".join(errors[:4]) + ". Return a corrected plan (e.g. adjust scale/proportions, joint "
                 "count, or payload) that avoids these.")
    # The intent is a tiny structured decision — ask the reasoning model for a SHORT hidden chain (low effort)
    # and honor the small token budget, so it stays fast/cheap without losing judgement.
    raw = llm.complete_json(INTENT_SYSTEM, user, INTENT_SCHEMA, max_tokens=600, reasoning_effort="low") or {}
    cls = str(raw.get("robot_class") or getattr(plan, "robot_class", "") or "manipulator").lower()
    if cls not in ("humanoid", "biped", "quadruped", "legged", "manipulator", "mobile_base"):
        cls = getattr(plan, "robot_class", "manipulator")
    arche = str(raw.get("archetype") or "none").lower()
    return {
        "robot_class": cls,
        "archetype": arche if arche in ("spider", "snake", "frog") else "none",
        "scale_m": _num(raw.get("scale_m"), None),
        "n_legs": _int(raw.get("n_legs"), None),
        "n_arms": _int(raw.get("n_arms"), 2),
        "arm_dof": _int(raw.get("arm_dof"), None),
        "has_head": bool(raw.get("has_head", True)),
        "end_effector": str(raw.get("end_effector") or "").lower() or None,
        "payload_kg": _num(raw.get("payload_kg"), None),
        "nominal_dims": _clean_dims(raw.get("nominal_dims")),
        "appendages": _clean_appendages(raw.get("appendages")),
        "source": "llm",
    }


_APPENDAGE_KINDS = ("claw", "tail", "antenna", "mandible", "stinger")


def _clean_appendages(raw) -> list:
    """Keep only known appendage kinds with clamped count/scale + a valid mount (the LLM names the part; the
    builder owns all geometry). Bounded so a hallucinated list can't explode the body."""
    if not isinstance(raw, list):
        return []
    out = []
    for a in raw[:6]:                                    # at most 6 appendage GROUPS
        if not isinstance(a, dict):
            continue
        kind = str(a.get("kind") or "").lower()
        if kind not in _APPENDAGE_KINDS:
            continue
        mount = str(a.get("mount") or "front").lower()
        out.append({"kind": kind,
                    "count": max(1, min(4, _int(a.get("count"), 2 if kind in ("claw", "antenna", "mandible") else 1))),
                    "mount": mount if mount in ("front", "rear", "top") else "front",
                    "scale": max(0.4, min(2.0, _num(a.get("scale"), 1.0)))})
    return out


# Per-segment nominal-length clamps (metres) — the LLM may NUDGE a length within a sane band; the builder
# still owns every other dimension (radius, axis, offset, profile), so geometry stays credible by construction.
_DIM_CLAMP_M = {
    "torso_mm": (0.25, 0.75), "upper_arm_mm": (0.15, 0.45), "forearm_mm": (0.12, 0.40),
    "thigh_mm": (0.15, 0.55), "shin_mm": (0.15, 0.55), "body_mm": (0.06, 1.20),
    "leg_mm": (0.05, 0.60), "reach_mm": (0.20, 2.00),
}


def _clean_dims(raw) -> dict:
    """Keep only known nominal-dimension keys, convert mm->m, and clamp each INDEPENDENTLY to its sane band.
    A hallucinated value can't escape the band; unknown/absent keys are dropped (builder uses its default)."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for k, (lo, hi) in _DIM_CLAMP_M.items():
        v = _num(raw.get(k), None)
        if v is None:
            continue
        m = v / 1000.0 if v > 5.0 else float(v)          # accept mm (typical) or already-metres
        out[k] = max(lo, min(hi, m))
    return out


def _attach_appendages(spec: dict, appendages: list) -> dict:
    """Add distinctive EXTRA appendages (claws/tail/antennae) onto a legged body's torso. The builder OWNS all
    geometry, placement (flush within the torso radius), and orientation; the intent only named kind/count/
    mount. Each appendage is original parametric geometry (tapered lofts + extruded plates), connected by
    construction — so a crab gets recognizable claws, a scorpion a curling tail, without the LLM ever placing
    a vertex. Quietly no-op if the spec has no torso to attach to."""
    base = spec.get("base") or {}
    if base.get("name") != "torso" or not appendages:
        return spec
    tr = float(base.get("radius", 0.16))
    tl = float(base.get("length", 0.08))
    limbs = spec.setdefault("limbs", [])
    for gi, ap in enumerate(appendages):
        limbs.extend(_appendage_limbs(ap["kind"], ap["count"], ap["mount"], ap["scale"], tr, tl, gi))
    return spec


def _appendage_limbs(kind: str, count: int, mount: str, sc: float, tr: float, tl: float, gi: int) -> list:
    """Build the limb-spec(s) for one appendage group. +x is forward; mounts sit ON the torso surface."""
    out: list = []
    # mount station on the torso surface (offsets are relative to the torso's distal tip at z=tl).
    fx = {"front": tr * 0.85, "rear": -tr * 0.85, "top": 0.0}.get(mount, tr * 0.85)
    fz = -tl * 0.5 if mount != "top" else -tl * 0.15
    # local +z aim: forward (front), up-and-back (rear), up-and-forward (top).
    euler = {"front": (0.0, _m.pi / 2, 0.0), "rear": (0.0, -_m.pi / 4, 0.0),
             "top": (0.0, _m.pi / 4, 0.0)}.get(mount, (0.0, _m.pi / 2, 0.0))

    def _seg(name, length, r, joint, geom, **extra):
        d = {"name": name, "length": round(length, 4), "radius": round(r, 4), "joint": joint,
             "axis": (0.0, 1.0, 0.0), "geometry": geom}
        d.update(extra)
        return d

    if kind in ("claw", "mandible"):
        # a pincer ARM: a tapered upper segment + a flat opening claw plate. count -> mirrored ±y pairs.
        base_len, base_r, claw_len = 0.10 * sc, 0.026 * sc, 0.075 * sc
        cw = 0.05 * sc                                          # claw half-width
        pincer = {"family": "extrude", "height": round(claw_len, 4), "fillet": round(0.006 * sc, 4),
                  "profile": [[-0.45 * cw, -cw], [0.45 * cw, -cw], [cw, 0.2 * cw], [0.35 * cw, cw],
                              [-0.35 * cw, cw], [-cw, 0.2 * cw]]}   # a fat base narrowing to a split tip
        sides = [1.0, -1.0] if count >= 2 else [1.0]
        for i, sy in enumerate(sides[:count]):
            out.append({"prefix": f"claw{gi}_{i}", "parent": "torso",
                        "mount_offset": (fx, sy * tr * 0.5, fz), "mount_euler": euler,
                        "links": [
                            _seg("base", base_len, base_r, "revolute", torque=6.0 * sc,
                                 geom={"family": "role", "role": "actuator_segment",
                                       "length": round(base_len, 4), "radius": round(base_r, 4)}),
                            _seg("pincer", claw_len, cw, "fixed", shape="box", geom=pincer)]})
    elif kind in ("tail", "stinger"):
        # a tapering, gently up-curling chain of N segments (a real-looking tail), one per group (count -> N).
        n = max(2, min(6, count if kind == "tail" else 2))
        seg_len = 0.09 * sc
        links = []
        for i in range(n):
            r0 = (0.030 - 0.004 * i) * sc
            links.append(_seg(f"t{i}", seg_len, max(0.008, r0), "revolute",
                              lower=-0.6, upper=0.6, torque=round(4.0 * sc, 2), mount_euler=(0.0, 0.35, 0.0),
                              geom={"family": "role", "role": "actuator_segment",
                                    "length": round(seg_len, 4), "radius": round(max(0.008, r0), 4)}))
        if kind == "stinger" or True:                          # cap the tail in a pointed revolve stinger
            tip = (0.024) * sc
            links.append(_seg("tip", 0.05 * sc, max(0.006, 0.012 * sc), "fixed",
                              geom={"family": "revolve",
                                    "profile": [[round(tip, 4), 0.0], [round(0.4 * tip, 4), round(0.035 * sc, 4)],
                                                [round(0.04 * sc, 4), round(0.05 * sc, 4)]]}))
        out.append({"prefix": f"tail{gi}", "parent": "torso", "mount_offset": (fx, 0.0, fz),
                    "mount_euler": euler, "links": links})
    elif kind == "antenna":
        a_len, a_r = 0.12 * sc, 0.008 * sc
        sides = [1.0, -1.0] if count >= 2 else [0.0]
        for i, sy in enumerate(sides[:max(1, count)]):
            out.append({"prefix": f"ant{gi}_{i}", "parent": "torso",
                        "mount_offset": (tr * 0.6, sy * tr * 0.35, -tl * 0.1),
                        "mount_euler": (0.0, _m.pi / 4, 0.0),
                        "links": [_seg("a", a_len, a_r, "fixed",
                                       geom={"family": "tapered", "length": round(a_len, 4),
                                             "r0": round(a_r, 4), "r1": round(0.3 * a_r, 4)})]})
    return out


def build_from_intent(intent: dict, prompt: str, req):
    """Synthesize a connected, credible ``RobotGene`` from a body-plan intent using our parametric builders.
    Returns ``None`` if the intent cannot be realized (caller falls back to the keyword builder)."""
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements

    cls = intent.get("robot_class", "manipulator")
    arche = intent.get("archetype", "none")
    scale = intent.get("scale_m")
    nd = intent.get("nominal_dims") or {}
    payload = intent.get("payload_kg")
    req_payload = float(payload) if payload else (getattr(req, "payload_kg", 0.25) or 0.25)

    # 1) Distinctive exotic archetypes get their dedicated, recognizable builder (spider/snake/frog).
    if arche in ("spider", "snake", "frog"):
        from virturoid.services import novel_anatomy
        if arche == "spider":
            g = novel_anatomy.build_spider(int(intent.get("n_legs") or 8))
        elif arche == "snake":
            g = novel_anatomy.build_snake()
        else:
            g = novel_anatomy.build_frog()
        g.design_source = "intent"
        return g

    # 2) Humanoid/biped -> the anthropometric loft humanoid, scaled by stature (~7.7 head-heights tall). Optional
    #    per-segment length nudges (nominal_dims) let the LLM ask for a long-armed reacher vs a compact picker.
    if cls in ("humanoid", "biped"):
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        H = max(0.12, min(0.30, (float(scale) / 7.7) if scale else 0.205))
        g = build_anthropometric_humanoid(H, dims=nd or None)
        g.design_source = "intent"
        return g

    # 3) Quadruped / legged animal -> the N-legged composed template (one loft trunk + N role-built legs +
    #    a forward head). The LLM's leg count drives it (so a crab/octopod gets 8 legs, an insect 6); scale_m
    #    (animal body length) + payload size the body+actuators (previously scale_m was a no-op here).
    if cls in ("quadruped", "legged"):
        n_legs = intent.get("n_legs")
        if not n_legs:
            from virturoid.services.morphology_composer import _leg_count
            n_legs = _leg_count(prompt)
        n_legs = max(2, min(8, int(n_legs)))
        spec = morphology_from_requirements(getattr(req, "reach_m", 0.6) or 0.6, req_payload, prompt=prompt,
                                            environment=getattr(req, "environment", None),
                                            robot_class="quadruped" if n_legs == 4 else "legged",
                                            n_legs=n_legs, scale_m=_num(scale, None), nominal_dims=nd)
        _attach_appendages(spec, intent.get("appendages") or [])   # crab claws / scorpion tail / insect antennae
        g = compose_from_spec(spec)
        g.design_source = "intent"
        return g

    # 4) Mobile base -> the wheeled rover.
    if cls == "mobile_base":
        g = compose_from_spec(morphology_from_requirements(0.6, req_payload, prompt=prompt,
                                                           robot_class="mobile_base"))
        g.design_source = "intent"
        return g

    # 5) Manipulator -> the anthropomorphic serial arm. Honor reach + grasp style from the intent; reach is
    #    UPPER-clamped so a hallucinated scale_m=50 can't build a 50 m arm with absurd actuators.
    reach = nd.get("reach_mm") or getattr(req, "reach_m", None) or (float(scale) if scale else 0.65)
    reach = max(0.2, min(2.0, float(reach)))
    ee = intent.get("end_effector")
    p = prompt
    if ee == "hand":
        p = prompt + " dexterous multi-finger hand"
    elif ee == "spray":
        p = prompt + " spray paint"
    dof = intent.get("arm_dof")
    if dof and dof >= 6:
        p = p + " precise 6-dof"
    spec = morphology_from_requirements(reach, req_payload, prompt=p,
                                        environment=getattr(req, "environment", None),
                                        robot_class="manipulator")
    g = compose_from_spec(spec)
    g.design_source = "intent"
    return g


def _num(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
