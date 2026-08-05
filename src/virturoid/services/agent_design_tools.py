"""Agent-first DESIGN + STATEFUL-CHAIN tools (docs/agent_first_plan.md P1: G-A + G-C). These are the tools
that let an EXTERNAL frontier agent (Claude Code/Codex over MCP) be the whole brain against our substrate,
with ZERO internal LLM spend: the agent AUTHORS the robot's anatomy graph itself (``submit_design``) instead
of prompting our internal generator, then drives the full loop on THE gene it made (``evaluate_held`` /
``train_held`` / ``export_held``) instead of recomposing from a prompt. Grounded by ``get_design_schema``
(the anatomy-graph language + worked examples) so the agent knows what to submit. Every result is the compact
verdict contract with teaching errors (SWE-agent ACI). Registered into ``agent_tools.TOOLS``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# M11 honesty: goals that name a domain NONE of our tiers (land/mobile/legged/aerial/aquatic/manip) can run.
# "fly to the target" is charitably a go-to for a legged body (handled by the proposer); "fly to the MOON",
# "reach orbit", "teleport" have no honest reinterpretation, so run_task must say so instead of scoring the
# body's default task 1.0.
_OUT_OF_DOMAIN = re.compile(
    r"\b(moon|outer ?space|to space|orbit|orbital|mars|jupiter|galaxy|interstellar|cosmos|"
    r"teleport|time ?travel|through (?:the )?walls?|phase through|underground|to the stars)\b", re.I)

# The REAL anatomy-graph vocabulary (extracted from anatomy_compiler.build_from_anatomy), so the schema we
# teach the agent is accurate, not hallucinated. A part = one body/limb node; symmetry mirrors it to +-y.
_ROLES = ["body", "neck", "head", "tail", "leg", "arm", "tentacle", "trunk", "wheel", "wing", "fin", "flipper", "foot", "hand",
          "claw", "paw", "ear", "eye", "horn", "antenna", "shell", "beak", "snout"]
_ROLESET = frozenset(_ROLES)
_ATTACH = ["front_top", "front_bottom", "front_mid_bottom", "mid_bottom", "rear_mid_bottom", "rear_bottom",
           "rear_top", "tip"]
_AIM = ["forward", "back", "up", "down", "out", "forward_up", "forward_out", "forward_down_out",
        "back_up", "back_out", "back_down_out", "down_out", "up_out"]

_EXAMPLE_QUAD = {
    "robot_class": "quadruped", "name": "agent_lynx",
    "parts": [
        {"name": "torso", "role": "body", "size": 0.55, "girth": 0.14},
        {"name": "neck", "role": "neck", "parent": "torso", "attach": "front_top", "aim": "forward_up", "size": 0.12, "girth": 0.05},
        {"name": "head", "role": "head", "parent": "neck", "attach": "tip", "aim": "forward", "size": 0.14, "girth": 0.055},
        {"name": "tail", "role": "tail", "parent": "torso", "attach": "rear_top", "aim": "back_up", "size": 0.25, "girth": 0.03, "joint": "revolute"},
        {"name": "leg1", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out", "size": 0.40, "girth": 0.018, "segments": 4, "symmetry": "left_right", "joint": "revolute"},
        {"name": "leg2", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down_out", "size": 0.40, "girth": 0.018, "segments": 4, "symmetry": "left_right", "joint": "revolute"},
    ]}
_EXAMPLE_HEX = {
    "robot_class": "quadruped", "name": "agent_hexapod",
    "parts": [{"name": "torso", "role": "body", "size": 0.5, "girth": 0.13}] + [
        {"name": f"leg{i+1}", "role": "leg", "parent": "torso",
         "attach": ["front_bottom", "mid_bottom", "rear_bottom"][i], "aim": "down_out",
         "size": 0.34, "girth": 0.016, "segments": 4, "symmetry": "left_right", "joint": "revolute"} for i in range(3)]}
_EXAMPLE_ROVER = {
    "robot_class": "mobile_base", "name": "agent_rover",
    # a flat DECK chassis (aspect 'deck' = a low rectangular slab, not a rounded pod) with wheel pairs at the
    # FRONT and REAR bottom corners -> a credible flat-deck rover that drives (verified ~0.55 m, 0% slip).
    "parts": [{"name": "chassis", "role": "body", "size": 0.8, "girth": 0.34, "aspect": "deck", "detail": "smooth"}] + [
        {"name": f"wheel{i+1}", "role": "wheel", "parent": "chassis",
         "attach": ["front_bottom", "rear_bottom"][i],
         "size": 0.2, "girth": 0.08, "symmetry": "left_right"} for i in range(2)]}   # 2 pairs = 4 wheels at corners

# NON-ANIMAL worked examples. The three above are all creatures, and every example an agent had was a creature,
# so the machine half of the vocabulary (prismatic joints, declared axes, open roles, parametric attach, a rest
# stance) had no precedent showing how the pieces go together. These two are the load-bearing shapes: a SCARA is
# the "declared axis + prismatic" case, an excavator the "articulated boom that must REST somewhere" case.
#
# `rest` is the part that is easy to skip and impossible to un-see once rendered: with every joint at 0 a chain
# of forward-aimed links is a straight horizontal line, so the excavator below WITHOUT its rest angles lies flat
# on the ground (measured: 0.025 m tall) and WITH them stands 1.248 m.
_EXAMPLE_SCARA = {
    "robot_class": "manipulator", "name": "agent_scara",
    # 2R in a horizontal plane about Z, then a prismatic quill. `axis` matters: the role-derived default is a
    # limb's (0,1,0) pitch, which would make this an elbow that bends the wrong way.
    "parts": [
        {"name": "base", "role": "body", "size": 0.22, "girth": 0.11},
        # The COLUMN is the part that is easy to leave out and obvious once rendered: a SCARA's links work in a
        # horizontal plane, so without a vertical standoff the whole arm sits at z~0 and reads as a pipe lying on
        # the floor. Structure a machine needs to look like itself is the designer's job, not the compiler's.
        {"name": "column", "role": "column", "like": "neck", "parent": "base", "attach": "front_top",
         "aim": "up", "size": 0.45, "girth": 0.06},
        # AXIS IS IN THE SEGMENT'S OWN FRAME, and a segment's local +z runs ALONG the part. So for a link aimed
        # "forward", [0,0,1] is a ROLL joint about the arm's own length -- not a SCARA elbow, which turns about
        # the vertical. Measured: with [0,0,1] both elbows came out horizontal (world axis 1,0,0) and driving
        # link2 through 1.2 rad moved the tip 0.0000 m. The arm could not fold at all, while still compiling,
        # passing every gate, and rendering as a plausible SCARA. Here [-1,0,0] is the local direction that maps
        # to world +z for a forward-aimed link, and the tip travels 0.0837 m.
        {"name": "link1", "role": "arm", "parent": "column", "aim": "forward",
         "size": 0.35, "girth": 0.045, "joint": "revolute", "axis": [-1, 0, 0], "lower": -2.6, "upper": 2.6},
        {"name": "link2", "role": "arm", "parent": "link1", "aim": "forward",
         "size": 0.28, "girth": 0.038, "joint": "revolute", "axis": [-1, 0, 0], "lower": -2.6, "upper": 2.6,
         "rest": 0.9},
        {"name": "quill", "role": "quill", "like": "arm", "parent": "link2", "aim": "down",
         "size": 0.18, "girth": 0.022, "joint": "prismatic", "axis": [0, 0, 1], "lower": -0.15, "upper": 0.0},
    ]}
_EXAMPLE_EXCAVATOR = {
    "robot_class": "manipulator", "name": "agent_excavator",
    # open roles (slew_ring / boom / stick / bucket) each declaring what they ARE via `like`, plus the rest
    # stance that puts the machine in a working posture instead of flat on the floor.
    "parts": [
        {"name": "house", "role": "body", "size": 0.9, "girth": 0.30},
        {"name": "slew", "role": "slew_ring", "like": "neck", "parent": "house", "attach": "front_top",
         "aim": "up", "size": 0.16, "girth": 0.20, "joint": "revolute", "axis": [0, 0, 1],
         "lower": -3.1, "upper": 3.1},
        {"name": "boom", "role": "boom", "like": "arm", "parent": "slew", "aim": "forward",
         "size": 0.95, "girth": 0.07, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -0.2, "upper": 1.2, "rest": -0.15},
        {"name": "stick", "role": "stick", "like": "arm", "parent": "boom", "aim": "forward",
         "size": 0.70, "girth": 0.05, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -2.2, "upper": 0.2, "rest": -1.6},
        {"name": "bucket", "role": "bucket", "like": "hand", "parent": "stick", "aim": "forward",
         "size": 0.32, "girth": 0.15, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -2.6, "upper": 0.4, "rest": -1.2},
    ]}


_EXAMPLE_DELTA = {
    # A PARALLEL mechanism — the one shape a tree genuinely cannot express on its own, and the reason
    # `loop_closures` exists. Three arms drive ONE shared platform, so two of the three joins are loops.
    #
    # The trap that cost two attempts: MuJoCo's `connect` locks in whatever offset the two parts have AT BUILD
    # TIME. Declaring a loop between parts that are apart does not pull them together — it welds the gap. The
    # first version's arm tips sat 0.5045 m from the platform before stepping and 0.5045 m after 2000 steps,
    # with nu=3 / neq=2 and a clean validate the whole time. So every forearm's `aim` and `size` here are
    # computed to land EXACTLY on the platform rim; `validate_gene_design` reports `loop_closures_meet` and
    # will say so if an edit breaks that.
    "robot_class": "delta", "name": "agent_delta", "base_height_m": 1.05,
    "loop_closures": [{"a": "fore1", "b": "platform"}, {"a": "fore2", "b": "platform"}],
    "parts": [
        {"name": "plate", "role": "frame", "like": "body", "size": 0.40, "girth": 0.24},
        # The STEM is structure we add, not a delta part: a tree needs ONE parent for the platform, and the
        # platform has to be built where the arms will meet it. Drawn thin so it reads as a mast.
        {"name": "stem", "role": "stem", "like": "neck", "parent": "plate", "joint": "fixed",
         "attach": {"along": 0.5, "lateral": 0.0, "height": 0.0}, "aim": [0, 0, -1],
         "size": 0.55, "girth": 0.018},
        # SHORT disc at the stem's tip. Using a long `length` to position it instead draws a fat column down
        # the middle — length both places a part and draws it.
        {"name": "platform", "role": "platform", "like": "hand", "parent": "stem", "joint": "fixed",
         "aim": [0, 0, -1], "size": 0.045, "girth": 0.105},
        # Three actuated upper arms at 120 degrees. Only an explicit aim VECTOR can do this: every aim token
        # has y >= 0, so a radial fan is unreachable through the named directions.
        {"name": "up0", "role": "delta_upper", "like": "arm", "parent": "plate",
         "attach": {"along": 0.5, "lateral": 0.0, "height": 0.0}, "aim": [0.800, 0.000, -0.600],
         "size": 0.30, "girth": 0.026, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -0.9, "upper": 0.9, "rest": 0.0},
        {"name": "up1", "role": "delta_upper", "like": "arm", "parent": "plate",
         "attach": {"along": 0.5, "lateral": 0.0, "height": 0.0}, "aim": [-0.400, 0.693, -0.600],
         "size": 0.30, "girth": 0.026, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -0.9, "upper": 0.9, "rest": 0.0},
        {"name": "up2", "role": "delta_upper", "like": "arm", "parent": "plate",
         "attach": {"along": 0.5, "lateral": 0.0, "height": 0.0}, "aim": [-0.400, -0.693, -0.600],
         "size": 0.30, "girth": 0.026, "joint": "revolute", "axis": [0, 1, 0],
         "lower": -0.9, "upper": 0.9, "rest": 0.0},
        # Forearms converge INWARD and down; aim = (platform rim - upper arm's tip), size = that distance.
        {"name": "fore0", "role": "delta_fore", "like": "arm", "parent": "up0",
         "aim": [-0.636, 0.000, -0.772], "size": 0.4795, "girth": 0.018},
        {"name": "fore1", "role": "delta_fore", "like": "arm", "parent": "up1",
         "aim": [0.318, -0.551, -0.772], "size": 0.4795, "girth": 0.018},
        {"name": "fore2", "role": "delta_fore", "like": "arm", "parent": "up2",
         "aim": [0.318, 0.551, -0.772], "size": 0.4795, "girth": 0.018},
    ]}


def _corpus_grounding(args: dict) -> dict:
    """Thesis A — RETRIEVAL = RUNTIME GROUNDING. Return the best PHYSICS-VERIFIED shape programs the
    self-manufactured corpus holds for the roles this design will use, so the agent ADAPTS a proven precedent
    (that already realizes a valid solid) instead of authoring geometry blind — the RoboMorph anti-mode-collapse
    point. Roles are taken from ``args['roles']`` or inferred from ``args['robot_class']``. Zero product LLM
    tokens: the corpus is the DB, retrieval is a lookup. Empty (omitted) until the corpus has banked words."""
    roles = args.get("roles")
    if not roles:
        cls = str(args.get("robot_class") or "").lower()
        roles = {"quadruped": ["body", "leg", "foot"], "hexapod": ["body", "leg"],
                 "humanoid": ["body", "leg", "arm"], "biped": ["body", "leg"], "arm": ["arm", "gripper"],
                 "octopus": ["mantle", "tentacle"]}.get(cls,
                 ["body", "leg", "arm", "tentacle", "wing", "mantle", "head", "tail"])
    exemplars: dict = {}
    try:
        from virturoid.services.memory_db import DEFAULT_DB_PATH, MemoryDB
        from virturoid.services.shape_flywheel import recall_shape
        if not DEFAULT_DB_PATH.exists():
            return {}
        with MemoryDB(DEFAULT_DB_PATH) as db:
            for role in roles:
                prog = recall_shape(db, str(role).lower())
                if prog:
                    exemplars[str(role).lower()] = prog
    except Exception:  # noqa: BLE001 - grounding is an accelerant; a missing corpus never blocks design
        return {}
    if not exemplars:
        return {}
    return {"shape_exemplars": exemplars,
            "note": "PHYSICS-VERIFIED shape programs recalled from the corpus for these roles — drop one into a "
                    "part's `geometry` field and adapt it, rather than authoring blind. Each already realizes a "
                    "valid solid (retrieval = runtime grounding).",
            "source": "shape_flywheel corpus (self-manufactured from prior verified builds)"}


def get_design_schema(args: dict) -> dict:
    """The anatomy-graph LANGUAGE the agent authors a robot in: part fields + the roles/attach/aim vocabularies
    + two worked examples that compile. Call this before submit_design.

    Thesis A (retrieval = runtime grounding): also returns ``corpus_grounding`` — the best PHYSICS-VERIFIED shape
    exemplars the self-manufactured corpus holds for the roles this design will use (pass ``roles`` and/or
    ``robot_class`` to target them) — so the agent adapts a proven precedent, not just the static dimension bands."""
    schema = {"ok": True, "format": "anatomy_graph",
            "top_level": {"robot_class": "quadruped|hexapod|humanoid|biped|legged", "name": "str", "parts": "[part...]"},
            "part_fields": {
                "name": "unique str (required)", "role": f"one of {_ROLES} (required)",
                "parent": "another part's name; omit for the root body",
                "attach": f"where on the parent it mounts: {_ATTACH}",
                "aim": (f"the direction it points: {_AIM} — or an explicit [x, y, z] vector for anything those "
                        "cannot say. Every token has y >= 0, because they were written for ANIMALS whose limbs "
                        "come in mirrored pairs (see `symmetry`), so a single part pointing at -y, or a RADIAL "
                        "fan like three delta arms at 120 degrees, needs the vector. An unrecognised token is "
                        "refused, never silently treated as 'forward'"),
                "size": "length in metres (the segment's long axis)", "girth": "radius in metres",
                "aspect": "BODY shape (root part only): 'deck'/'chassis' = a flat rectangular slab for a "
                          "rover/AGV (wheels mount at its corners); 'wide' = a broad low pod; 'round' = a "
                          "bulbous sac; omit for the default sleek barrel torso",
                "segments": "int; a leg with 4 = 3 actuated joints + a welded foot (Go2-class)",
                "symmetry": "'left_right' mirrors the part to a +y/-y PAIR (so one leg entry = two legs)",
                "joint": "'revolute' to actuate it, 'prismatic' for a SLIDING axis (a gantry ram, a rail "
                         "carriage, a SCARA quill); omit for a welded/fixed part",
                "axis": "[x,y,z] the joint's axis IN THE PART'S OWN FRAME, where local +z runs ALONG the part "
                        "and the part is oriented by its `aim`. This is the easiest thing here to get wrong: for "
                        "a link aimed 'forward', [0,0,1] is a ROLL about the arm's own length, NOT a vertical "
                        "yaw — a SCARA built that way compiles, passes every check, renders correctly, and "
                        "cannot fold (measured: tip travel 0.0000 m). For a forward-aimed link the vertical is "
                        "[-1,0,0]; for an up-aimed one it is [0,0,1]. Check it with probe_robot rather than "
                        "reasoning about frames. Omitted, the axis is derived from the ANIMAL role, which is "
                        "right for a limb and wrong for a machine",
                "lower": "float; joint travel limit (radians for revolute, metres for prismatic)",
                "upper": "float; the other limit. Declare both for anything with a real stroke or range",
                "rest": "float; the angle/extension this joint RESTS at, within [lower, upper]. Machines have no "
                        "animal default, so every joint sits at 0 and a chain of forward-aimed links comes out as "
                        "a straight horizontal line lying on the ground — declaring the working stance is what "
                        "stands it up (a measured excavator went 0.025 -> 1.248 m tall). Legs keep their derived "
                        "knee bend when this is omitted",
                "curl": "float; a resting curl spread across a multi-segment part (a curved tail/neck)",
                "geometry": "OPTIONAL shape program to AUTHOR a single-segment part; its build/export approximation "
                            "is kept with the segment and the high-fidelity mesh path realizes the full shape. See geometry_families."},
            # T4: author arbitrary part geometry (visual). Single-segment parts only; the collider is untouched.
            "geometry_families": {
                "extrude": "{family:'extrude', profile:[[x,y],...], height} - extrude a 2-D polygon (plates, "
                           "brackets, L-shapes)",
                "revolve": "{family:'revolve', profile:[[r,z],...]} - revolve a profile about z (domes, cones, "
                           "nozzles, shells)",
                "tapered": "{family:'tapered', length, r0, r1} - a frustum (tapered shaft)",
                "loft": "{family:'loft', sections:[[z,half_y,half_x],...]} - an elliptical loft (organic bodies)",
                "modifiers": "any family also takes fillet (mm, round edges), chamfer (mm, bevel), cutouts (list)",
                "example": {"name": "dome", "role": "head", "parent": "torso", "attach": "front_top",
                            "aim": "forward", "size": 0.14, "girth": 0.06,
                            "geometry": {"family": "revolve",
                                         "profile": [[0.0, 0.0], [0.06, 0.0], [0.055, 0.05], [0.0, 0.08]]}}},
            # T8: real dimension bands (metres) grounded in production robots (Go2/Spot legs, UR5e links,
            # rover wheels) so the agent authors CREDIBLE sizes, not a tiny/toy or absurd body. size = the
            # part's long axis; girth = its radius (half-width). Stay within these unless the prompt is explicit.
            "typical_dimensions_m": {
                "body": {"size(length)": "0.3-1.2", "girth(half_width)": "0.08-0.30",
                         "note": "the torso/chassis; a dog ~0.5, a humanoid torso ~0.5, a rover chassis 0.6-1.0"},
                "leg": {"size(length)": "0.25-0.6", "girth": "0.02-0.05", "segments": "4 (=3 DOF + foot)",
                        "note": "SLENDER: length/diameter >= ~2.5 (a real leg, not a sausage stub)"},
                "wheel": {"size(diameter)": "0.10-0.40", "girth(tread_width)": "0.03-0.10",
                          "note": "a rover wheel ~0.15-0.25 dia; keep wheel dia < ~0.4x the chassis length so "
                                  "the wheels don't dwarf the body"},
                "arm": {"size(length)": "0.4-1.0", "girth": "0.03-0.06", "segments": "4-6",
                        "note": "a tabletop arm ~0.6-0.8 reach; the hand/gripper is a short terminal part"},
                "head": {"size": "0.08-0.2", "girth": "0.03-0.08"}, "neck": {"size": "0.08-0.25"},
                "tail": {"size": "0.1-0.4", "girth": "0.015-0.04"}},
            "proportion_rules": [
                "a limb/wheel should be a FRACTION of the body, not larger than it — size appendages relative "
                "to the body you chose (a wheel radius > the chassis half-width reads as wheels-with-a-box)",
                "parts outside 0.005-6.0 m (size) / 0.002-1.2 m (girth) are rejected as absurd (M16 gate)"],
            "rules": ["exactly one root part with role 'body' and no parent",
                      "a WALKING leg needs segments>=4 and joint='revolute' and aim 'down_out' for a stable stance",
                      "a WHEEL (role 'wheel') is a rolling cylinder: size=diameter, girth=tread width; the axle "
                      "is laid lateral automatically, so a mobile_base with wheel pairs DRIVES (not walks)",
                      "symmetry:'left_right' counts as TWO of the part — 2 leg parts = a quadruped, 3 = a hexapod, "
                      "3 wheel parts = a six-wheeled rover",
                      "roles MUST come from the roles list above; an unknown role is rejected with a teaching "
                      "error (it is NOT silently compiled into a limb)",
                      "the compiler + validity gates run on submit; a broken graph returns a teaching error"],
            # Every example here used to be a CREATURE, so the machine half of the vocabulary had no precedent.
            "examples": {"quadruped": _EXAMPLE_QUAD, "hexapod": _EXAMPLE_HEX, "rover": _EXAMPLE_ROVER,
                         "scara_arm": _EXAMPLE_SCARA, "excavator": _EXAMPLE_EXCAVATOR,
                         "delta_parallel": _EXAMPLE_DELTA},
            # HONEST out-of-box capability of each example (measured, not assumed): so the agent knows what
            # walks/drives immediately vs what needs training. The scripted wave gait is a strong PRIOR for a
            # 4-leg body but marginal for 6+ legs (a known frontier — the learned residual/train_held closes it).
            "example_capability": {
                "quadruped": "VERIFIED walker out of the box (scripted crawl gait, ~1.7 m in verify)",
                "rover": "VERIFIED driver out of the box (torque wheels, ~0.4 m in verify)",
                "hexapod": "compiles + is structurally correct (6 legs), but the SCRIPTED gait is marginal for "
                           "6+ legs (~0 net) — call train_held to find a credible gait, or expect a weak verdict"}}
    g = _corpus_grounding(args or {})                          # Thesis A: retrieved verified exemplars for the roles
    if g:
        schema["corpus_grounding"] = g
    try:                                                       # WS-B.2: if a DRAFT graph is supplied, add the
        from virturoid.services.exemplar_retrieval import exemplar_grounding   # query-specific nearest VERIFIED
        eg = exemplar_grounding(args or {})                    # bodies (scores omitted — mode-collapse guard)
        if eg:
            schema.update(eg)
    except Exception:  # noqa: BLE001 - retrieval grounding is value-add; never blocks the schema
        pass
    return schema


# M16 buildable-scale bands (metres). Generous — an elephant leg is ~1.5 m, an industrial arm link ~1.2 m —
# so a real design never trips them, but a 30 m leg (which held a 19.7 m / 130 kg "robot" silently) is caught
# with a teaching error instead of compiling. Function-agnostic: applies to any part in any body.
_SIZE_BAND = (0.005, 6.0)        # a single segment's long axis
_GIRTH_BAND = (0.002, 1.2)       # a single segment's radius
_MAX_PARTS = 80


def _proportion_warnings(graph: dict) -> list[str]:
    """T8: NON-blocking advisories when an appendage is out of proportion to the body — the exact class that
    made a rover read as 'a box with oversized wheels'. Teaches without rejecting (a real design may be odd)."""
    parts = graph.get("parts") or []
    body = next((p for p in parts if p.get("role") == "body" and not p.get("parent")), None)
    if not body:
        return []
    bsize = float(body.get("size") or 0.5)
    bgirth = float(body.get("girth") or 0.42 * bsize)
    warns = []
    for p in parts:
        role, nm = p.get("role"), p.get("name", "?")
        sz = float(p.get("size") or 0.0)
        if role == "wheel":
            if 0.5 * sz > bgirth * 1.2:                        # wheel RADIUS (0.5*size) vs chassis half-width
                warns.append(f"wheel '{nm}' radius {0.5*sz:.2f} m is larger than the chassis half-width "
                             f"{bgirth:.2f} m — it will dwarf the body; shrink the wheel or widen the chassis")
        elif role in ("leg", "arm") and sz > 1.4 * bsize:
            warns.append(f"{role} '{nm}' length {sz:.2f} m exceeds the body length {bsize:.2f} m — a limb longer "
                         f"than the whole body reads as disproportionate")
    return warns[:4]


# T4 shape-programs: the geometry families the mesh layer (cad_geometry.realize_shape) renders, + their
# required fields. A part may carry an optional ``geometry`` to author its OWN visual shape.
_GEO_FIELDS = {"extrude": ["profile", "height"], "revolve": ["profile"],
               "tapered": ["length", "r0", "r1"], "loft": ["sections"]}


def _check_geometry(graph: dict) -> str | None:
    """Return a teaching error if any part's optional ``geometry`` spec is malformed (T4) — so a bad shape
    program is REJECTED, not silently rendered as a fallback capsule. None if all geometry is valid/absent."""
    for p in graph.get("parts") or []:
        g = p.get("geometry")
        if g is None:
            continue
        if not isinstance(g, dict):
            return f"part '{p.get('name','?')}' geometry must be an object {{family, ...}}"
        fam = str(g.get("family") or "").lower()
        if fam not in _GEO_FIELDS:
            return (f"part '{p.get('name','?')}' geometry.family '{fam}' is not one of {sorted(_GEO_FIELDS)}; "
                    f"see get_design_schema.geometry_families")
        missing = [k for k in _GEO_FIELDS[fam] if k not in g]
        if missing:
            return f"part '{p.get('name','?')}' {fam} geometry needs {missing} (see get_design_schema)"
    return None


def _check_scale(graph: dict) -> str | None:
    """Return a teaching error if any part's size/girth is outside the buildable band (M16), else None."""
    parts = graph.get("parts") or []
    if len(parts) > _MAX_PARTS:
        return f"{len(parts)} parts exceeds the buildable limit ({_MAX_PARTS}); simplify the design"
    for p in parts:
        nm = p.get("name", "?")
        sz = p.get("size")
        if sz is not None and not (_SIZE_BAND[0] <= float(sz) <= _SIZE_BAND[1]):
            return (f"part '{nm}' size {sz} m is outside the buildable band {_SIZE_BAND} m — real robot "
                    f"segments are sub-metre to a few metres; scale it down (or split into segments)")
        gr = p.get("girth")
        if gr is not None and not (_GIRTH_BAND[0] <= float(gr) <= _GIRTH_BAND[1]):
            return (f"part '{nm}' girth {gr} m (radius) is outside the buildable band {_GIRTH_BAND} m; "
                    f"pick a realistic limb/body thickness")
    return None


def _bank_to_flywheel(gene, *, prompt: str, task: str, success_rate: float | None,
                      source: str = "agent") -> None:
    """B4: write an agent-authored design/outcome into the design flywheel (memory_db + species memory) so
    recall_knowledge / nearest_bodies / search_memory surface the agent's OWN prior work — the moat compounds
    from exactly the usage the pivot created. Best-effort: memory is value-add, never blocks the design loop.

    ``success_rate=None`` = HONEST "not yet evaluated" (v7-C2, master_plan_v7 §2): submit-time was hardcoding
    ``0.0``, which recorded a fake MEASURED FAILURE for every submitted design (measured: 1441 placeholder-zero
    rows vs 51 real) — poisoning any success-ranked retrieval. An unevaluated design now records provenance with
    a NULL rate + NULL succeeded, and it never enters the species best-of ledger (that ledger is for MEASURED
    outcomes only). Real rates keep flowing from the train/eval paths unchanged."""
    try:
        from virturoid.services.agent_tools import safe_build_path
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.task_matched_eval import robot_kind
        mem_dir = safe_build_path(None, "memory")
        mem_dir.mkdir(parents=True, exist_ok=True)
        unevaluated = success_rate is None
        with MemoryDB(mem_dir / "virturoid_memory.db") as db:
            db.record_run(prompt=prompt or f"[{source}] {gene.robot_class}", robot_class=gene.robot_class,
                          task_type=task or robot_kind(gene), converged_design=gene.to_dict(),
                          success_rate=None if unevaluated else float(success_rate),
                          species=None if unevaluated else gene.robot_class,
                          succeeded=None if unevaluated else bool(success_rate >= 0.5), design_source=source)
    except Exception:  # noqa: BLE001
        pass


def _dropped_part_coercions(graph: dict, gene) -> list:
    """Authored parts that produced NO segment in the built body — reported as coercions, never silently.

    MEASURED (MVP red-team): a graph with a CYCLE, or simply authored bottom-up (a part listed before its
    parent), compiles to `ok: True` with the offending limbs missing — `_emit_chain` skips any part whose parent
    is not built yet. One probe produced a legless torso (n_seg=1) with `coercions: None`: the product silently
    shipped a different robot than the one authored. A >80-part graph is also clamped to 8 without a word.

    A part legitimately expands into several segments (a leg -> leg_0, leg_1...), so this matches by name prefix
    and only flags parts that contributed NOTHING.
    """
    try:
        parts = [p for p in (graph.get("parts") or []) if isinstance(p, dict) and p.get("name")]
        seg_names = [s.name for s in getattr(gene, "segments", [])]
        dropped = [str(p["name"]) for p in parts
                   if not any(n == p["name"] or n.startswith(f"{p['name']}_") for n in seg_names)]
        if not dropped:
            return []
        return [{"field": "parts", "part": nm, "from": "authored", "to": "omitted",
                 "why": ("this part produced no body: its parent was not built before it (a cycle, or the part "
                         "listed before its parent), or the graph exceeded the part cap. Reorder so every part "
                         "follows its parent, and check the root has no parent.")}
                for nm in dropped[:12]]
    except Exception:  # noqa: BLE001 - provenance is additive; never block a valid design
        return []


def _robotics_grounding(gene) -> dict:
    """The robotics AI GROUNDING the LLM's design (the whole point of having one): the nearest PHYSICS-VERIFIED
    prior bodies by the morphology embedding + whether a banked gait is likely to warm-start this body. So the
    model reasons about its just-authored design against WHAT HAS ACTUALLY WORKED — not blind, and not a
    deterministic template. High similarity to a verified walker => a gait likely transfers (warm-start); low
    similarity => expect to train a fresh controller. Best-effort; a missing corpus never blocks the design."""
    try:
        from virturoid.services.agent_tools import safe_build_path
        from virturoid.services.gait_flywheel import recall_gait
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import BODY, RoboticsVectorMemory
        mem = safe_build_path(None, "memory") / "virturoid_memory.db"
        if not mem.exists():
            return {}
        with MemoryDB(mem) as db:
            vm = RoboticsVectorMemory(db)
            if vm.count(BODY) == 0:
                vm.index_species_bodies()
            near = vm.nearest_bodies(gene, k=3, min_sim=0.0)
            gait = recall_gait(db, gene)
        top = near[0]["similarity"] if near else 0.0
        return {
            "nearest_verified_bodies": [{"body": h.get("obj_id"),
                                         "similarity": round(float(h.get("similarity", 0)), 3),
                                         "class": (h.get("meta") or {}).get("robot_class")} for h in near],
            "warm_start_gait_available": bool(gait),
            "transfer_outlook": ("a verified precedent is close — a banked gait is likely to warm-start this body"
                                 if top >= 0.85 and gait else
                                 "no close verified precedent — expect to TRAIN a fresh controller (train_held)"),
            "note": "grounded by the robotics embedding on PHYSICS-VERIFIED prior bodies — the moat grounding "
                    "your design, not a template."}
    except Exception:  # noqa: BLE001
        return {}


def submit_design(args: dict) -> dict:
    """The agent AUTHORS a robot: compile its anatomy graph, run the validity gates, and HOLD it under a
    robot_id for the rest of the loop (simulate/edit/train/export). No prompt, no internal generator — the
    external agent is the designer. Returns the id + summary + render + robotics_grounding, or a teaching error."""
    from virturoid.services import session_state as S
    from virturoid.services.anatomy_compiler import build_from_anatomy
    graph = args.get("graph")
    if not isinstance(graph, dict) or not graph.get("parts"):
        return {"ok": False, "error": "provide graph:{robot_class, parts:[...]}; call get_design_schema for the language"}
    roots = [p for p in graph["parts"] if (p.get("role") == "body") and not p.get("parent")]
    if len(roots) != 1:
        return {"ok": False, "error": f"a design needs EXACTLY ONE root part with role 'body' and no parent "
                f"(found {len(roots)}); see get_design_schema examples"}
    # OPEN ROLE VOCABULARY, still nothing guessed. The 23 roles are an ANIMAL vocabulary, so a gantry, SCARA,
    # turret, boom, rail carriage, track or rotor was not merely awkward to express -- it was rejected outright,
    # and no amount of geometry authoring helped because the rejection happens before geometry is read. But the
    # original guard is right that an unknown role must not be silently turned into a limb, so the rule becomes:
    # any role NAME is allowed, and an unfamiliar one must say which known role it behaves like structurally.
    # The agent states the semantics; the compiler still never guesses.
    unknown = {}
    for p in graph["parts"]:
        r = str(p.get("role") or "").lower()
        if not r or r in _ROLESET:
            continue
        like = str(p.get("like") or "").lower()
        if like in _ROLESET:
            p["role"] = like                                   # compile AS the declared structure...
            p.setdefault("role_label", r)                      # ...while keeping what the designer called it
        else:
            unknown[r] = p.get("name")
    if unknown:
        return {"ok": False, "error": (
            f"part role(s) {sorted(unknown)} are outside the built-in vocabulary. That is allowed, but you must "
            f"say what each one IS structurally: add \"like\": one of {sorted(_ROLESET)}. For example a gantry "
            f"column or an excavator boom is like an 'arm' (a serial chain), a track is like a 'wheel' (ground "
            f"drive), a turret is like a 'neck' (a rotating mount). An unknown role is never silently turned "
            f"into a limb, which is why this asks instead of assuming.")}
    scale_err = _check_scale(graph)                            # M16: reject absurd proportions with a teaching error
    if scale_err:
        return {"ok": False, "error": scale_err}
    geo_err = _check_geometry(graph)                           # T4: reject a malformed shape program (no silent fallback)
    if geo_err:
        return {"ok": False, "error": geo_err}
    try:
        gene = build_from_anatomy(graph)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"graph did not compile ({type(exc).__name__}: {exc}); check get_design_schema examples"}
    if gene is None:
        return {"ok": False, "error": "graph compiled to nothing; need one root 'body' part + at least one limb"}
    issues = gene.validate()
    if issues:
        return {"ok": False, "error": f"design failed the validity gate: {'; '.join(issues[:3])}"}
    try:
        from virturoid.services.grounded_physics import ground_gene
        ground_gene(gene, material=str(args.get("material") or "aluminum"), fill=0.25)
    except Exception:  # noqa: BLE001 - grounding is value-add; a valid gene is still usable
        pass
    # This is a true build gate, not a visual score: a contact-visible foot/wheel may not disagree with the
    # collider that simulation trains against, and a free body may not start airborne. Return a teaching error
    # before the invalid design is held or banked so an agent can amend the graph and resubmit it.
    try:
        from virturoid.services.visual_physics_gate import audit_gene

        visual_physics = audit_gene(gene)
        if not visual_physics.ok:
            reasons = "; ".join(issue.detail for issue in visual_physics.issues[:3])
            return {"ok": False, "error": f"design failed the visual/physics alignment gate: {reasons}",
                    "visual_physics": visual_physics.to_dict()}
    except ImportError:  # MuJoCo is optional for schema-only clients; the later verify step remains mandatory
        visual_physics = None
    except Exception as exc:  # a compiler/load failure is not safe to silently bank as a usable body
        return {"ok": False, "error": f"design could not clear the visual/physics gate: {exc}"}
    try:
        from virturoid.services.structural_assertions import evaluate_structural_assertions

        structural_contract = evaluate_structural_assertions(gene)
        if not structural_contract.ok:
            reasons = "; ".join(a.detail for a in structural_contract.assertions if not a.ok)
            return {"ok": False, "error": f"design failed structural seam assertions: {reasons}",
                    "structural_contract": structural_contract.to_dict()}
    except ImportError:
        structural_contract = None
    except Exception as exc:
        return {"ok": False, "error": f"design could not execute structural assertions: {exc}"}
    # NB (flywheel_breakthrough §3.M/§5d): in-place stance_repair was tried on this path and REVERTED — 0/5
    # measured product walk-rate lift (dominant failure is fore-aft LURCH, not lateral roll-over). Module kept
    # for the factory verify-build only.
    from virturoid.services.ai_native_tools import _render_gene, _summary
    rid = S.put_robot(gene, prompt=f"[submitted:{graph.get('name', 'design')}]", label="submitted")
    # B4 provenance; success_rate=None = "not yet evaluated" (v7-C2) — NEVER a fake measured-0.0 failure
    _bank_to_flywheel(gene, prompt=f"[agent] {graph.get('name', 'design')}", task="", success_rate=None)
    try:                                                       # Thesis A WRITE-side: this design's VERIFIED shape
        from virturoid.services.shape_flywheel import auto_bank_body_shapes   # words enter the corpus, so the NEXT
        banked = auto_bank_body_shapes(gene)                   # get_design_schema recalls them (self-manufacture)
    except Exception:  # noqa: BLE001 - corpus growth is an accelerant, never blocks a valid design
        banked = []
    out = {"ok": True, **_summary(gene, rid), "name": graph.get("name")}
    # Expose the same evidence used by the inline see→critique→fix loop. This keeps an accepted design useful
    # to the external reasoning model: it sees which engineering checks passed and any non-blocking anatomy
    # observations instead of receiving only a pretty render.
    try:
        from virturoid.services.gene_validation import validate_gene_design
        from virturoid.services.anatomy_critic import critique_gene

        out["design_review"] = {
            "engineering": validate_gene_design(gene, material=str(args.get("material") or "aluminum")),
            "anatomy": critique_gene(gene),
            "visual_physics": visual_physics.to_dict() if visual_physics is not None else {"status": "not_run"},
            "structural_contract": (structural_contract.to_dict()
                                    if structural_contract is not None else {"status": "not_run"}),
        }
    except Exception:  # noqa: BLE001 - evidence is additive after the hard alignment gate
        pass
    rg = _robotics_grounding(gene)                             # the robotics AI grounds the LLM's design in verified precedent
    if rg:
        out["robotics_grounding"] = rg
    if banked:
        out["corpus_shape_words"] = len(banked)               # retrieval grounding now has this design's words
    warns = _proportion_warnings(graph)                        # T8: non-blocking proportion advisories
    if warns:
        out["proportion_warnings"] = warns
    try:                                                       # WS-B.4: surface EVERY compiler coercion of the
        from virturoid.services.coercion_audit import detect_coercions   # authored graph — no silent rewrites
        coer = list(detect_coercions(graph) or [])
        coer.extend(_dropped_part_coercions(graph, gene))       # parts that produced NO body at all (see helper)
        if coer:
            out["coercions"] = coer
            out["coercions_note"] = ("the compiler clamped/defaulted these fields of your design (buildability / "
                                     "omitted-field defaults) — surfaced so nothing is changed silently; author "
                                     "them explicitly to keep full control")
    except Exception:  # noqa: BLE001 - the audit is grounding value-add; a valid design never blocks on it
        pass
    img = _render_gene(gene, rid)
    if img:
        out["artifacts"] = [img]
    return out


def critique_design(args: dict) -> dict:
    """Render and measure a held design so the external model can propose a localized edit.

    The model is never the acceptance gate: this tool exposes deterministic engineering/anatomy/contact
    findings plus the actual render, while ``edit_robot`` applies a proposed patch and the same gates can be
    rerun. Keeping critique separate also lets callers stop when a round is non-improving.
    """
    from virturoid.services import session_state as S
    from virturoid.services.ai_native_tools import _render_gene
    from virturoid.services.anatomy_critic import critique_gene
    from virturoid.services.gene_validation import validate_gene_design
    from virturoid.services.structural_assertions import evaluate_structural_assertions
    from virturoid.services.visual_physics_gate import audit_gene

    rid = args.get("robot_id")
    round_number = int(args.get("round") or 1)
    if round_number < 1 or round_number > 4:
        return {"ok": False, "error": "critique round must be 1..4; stop after the hard cap instead of over-repairing"}
    gene = S.get_robot(rid)
    if gene is None:
        return {"ok": False, "error": f"no robot '{rid}'"}
    engineering = validate_gene_design(
        gene, material=str(args.get("material") or "aluminum"),
        payload_kg=float(args.get("payload_kg") or 0.0),
    )
    anatomy = critique_gene(gene)
    visual_physics = audit_gene(gene).to_dict()
    structural_contract = evaluate_structural_assertions(gene).to_dict()
    findings = [
        {"source": "engineering", "severity": f["severity"], "detail": f["detail"]}
        for f in engineering["risk_flags"] if f["severity"] in ("fatal", "high", "med")
    ] + [
        {"source": "anatomy", "severity": f["severity"], "detail": f["detail"]}
        for f in anatomy["issues"] if f["severity"] in ("fatal", "high", "med")
    ] + [
        {"source": "visual_physics", "severity": "high", "detail": f["detail"]}
        for f in visual_physics["issues"]
    ] + [
        {"source": "structural_contract", "severity": "high", "detail": f["detail"]}
        for f in structural_contract["assertions"] if not f["ok"]
    ]
    img = _render_gene(
        gene, f"{rid}_critique", azimuth=float(args.get("azimuth", 50.0)),
        elevation=float(args.get("elevation", -16.0)),
    )
    return {
        "ok": True,
        "accepted": (engineering["ok"] and not anatomy["issues"] and visual_physics["ok"]
                     and structural_contract["ok"]),
        "findings": findings,
        "engineering": engineering,
        "anatomy": anatomy,
        "visual_physics": visual_physics,
        "structural_contract": structural_contract,
        "artifacts": [img] if img else [],
        "repair_contract": (
            "Use edit_robot for one localized patch, rerun critique_design, and keep the edit only if fatal/high "
            "findings do not increase. Stop after three total critique rounds (hard cap four)."
        ),
        "round": round_number,
    }


def submit_scene_spec(args: dict) -> dict:
    """The agent AUTHORS a scene directly: a list of objects (name/category/size_xyz/pose/material) + a robot
    spawn -> a validated SceneGraph held under a scene_id. Replaces the internal scene-author LLM role."""
    from virturoid.schemas.scenes import SceneGraph, SceneObject
    from virturoid.services import session_state as S
    objs_in = args.get("objects")
    if not isinstance(objs_in, list) or not objs_in:
        return {"ok": False, "error": "provide objects:[{name, category, size_xyz:[x,y,z], pose_xyz_rpy:[6], material}]"}
    try:
        objs = [SceneObject(name=o["name"], object_type=o.get("object_type", "obstacle"),
                            category=o.get("category"), material=o.get("material"),
                            size_xyz=tuple(o["size_xyz"]) if o.get("size_xyz") else None,
                            pose_xyz_rpy=tuple(o.get("pose_xyz_rpy") or (0, 0, 0, 0, 0, 0))) for o in objs_in]
        spawn = tuple(args.get("robot_spawn_xyz_rpy") or (0.0, 0.0, 0.1, 0, 0, 0))
        sg = SceneGraph(id=f"scene_agent_{len(objs)}", name=str(args.get("name") or "agent_scene"),
                        backend_targets=["mujoco"], robot_spawn_xyz_rpy=spawn, objects=objs,
                        variation_parameters={"task": args.get("task", "custom"), "theme": args.get("theme", "custom")})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"scene did not build ({type(exc).__name__}: {exc}); each object needs name + size_xyz"}
    issues = sg.validate()
    ok = getattr(issues, "ok", True)
    if not ok:
        return {"ok": False, "error": f"scene failed validation: {[i.code for i in getattr(issues, 'issues', [])][:3]}"}
    sid = S.put_scene(sg.to_dict(), task=args.get("task", "custom"), theme=args.get("theme", "custom"))
    return {"ok": True, "scene_id": sid, "n_objects": len(objs), "valid": True}


def evaluate_held(args: dict) -> dict:
    """Score the HELD robot on its morphology-implied task (NOT a recompose from prompt) — real MuJoCo. The
    agent evaluates the exact gene it designed/edited."""
    from virturoid.services import session_state as S
    from virturoid.services.task_matched_eval import evaluate_robot, robot_kind
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'; submit_design or create_robot first"}
    res = evaluate_robot(gene, prompt=(S.robot_meta(args["robot_id"]) or {}).get("prompt", ""))
    return {"ok": True, "kind": robot_kind(gene), "task": res.get("task"), "metric": res.get("metric"),
            "value": res.get("value"), "scored_gait": (res.get("detail") or {}).get("scored_gait")}


_EXPORT_FORMATS = ("mjcf", "cad", "urdf", "ros2", "bom", "spec", "usd", "isaac_lab", "certificate")


def export_held(args: dict) -> dict:
    """Export the HELD robot to real, buildable files. ``formats`` (default all): mjcf (runnable sim model) |
    cad (meshes) | urdf (ROS/Gazebo robot description) | ros2 (installable ament_python package) | bom (real
    sized bill-of-materials: motors/sensors/battery, json+md) | spec (a spec sheet) | usd (OpenUSD physics for
    NVIDIA Isaac Sim) | isaac_lab (a full Isaac Lab hand-off: USD + ArticulationCfg + spawn/train scaffolds).
    B3: the whole buildable-robot bundle, not just sim files. usd/isaac_lab need the ``usd-core`` package."""
    from virturoid.services import session_state as S
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'"}
    # B3c (2026-07-24 audit): GROUND the body up front through the SAME helper build_gene_package uses
    # (ground_and_repair), so every exported format (mjcf, urdf, cad, bom, spec) describes the SAME buildable
    # robot -- and it matches what the package builder would ship for this prompt. Before this, export_held
    # exported the ungrounded held gene for the URDF/sim while the BOM grounded separately -> the two exit doors
    # shipped physically different robots (3.57 kg URDF vs grounded ~8 kg BOM) for one prompt. Ground a COPY so
    # the held session gene is untouched; grounding is idempotent so a pre-grounded (amended) body is unaffected.
    #
    # ...and PROVE it is still the same robot. ``ground_and_repair`` now reproduces the body's own recorded
    # grounding instead of hardcoding carbon fibre, and preserves an imported robot's manufacturer masses
    # outright, so this is normally a genuine no-op. It is not guaranteed to be (the structural-repair and
    # housing-fit passes can still thicken an under-margined link), and when it is not, the customer must be
    # told rather than shipped a package whose certificate claims deploy==measure. Measured before the fix: a
    # Go2 verified at 27.362 kg exported at 19.151 kg, silently, on all 13 links.
    import copy

    from virturoid.services.grounded_physics import fingerprint_delta, physical_fingerprint
    held_fp = physical_fingerprint(gene)
    gene = copy.deepcopy(gene)
    try:
        from virturoid.services.gene_build import ground_and_repair
        ground_and_repair(gene)
    except Exception:  # noqa: BLE001 - grounding is the consistency layer; a failure still exports the raw body
        pass
    body_parity = fingerprint_delta(held_fp, physical_fingerprint(gene))
    fmts = args.get("formats") or list(_EXPORT_FORMATS)
    bad = [f for f in fmts if f not in _EXPORT_FORMATS]
    if bad:
        return {"ok": False, "error": f"unknown format(s) {bad}; choose from {list(_EXPORT_FORMATS)}"}
    from virturoid.services.agent_tools import safe_build_path  # H2: confine writes under build/
    out_dir = safe_build_path(args.get("out_dir"), "agent_exports") / args["robot_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    task = str(args.get("task") or (S.robot_meta(args["robot_id"]) or {}).get("prompt", ""))
    artifacts: dict = {}
    if "mjcf" in fmts:
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        p = out_dir / "robot.xml"
        p.write_text(compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)), encoding="utf-8")
        artifacts["mjcf"] = str(p)
    if "cad" in fmts:
        try:
            from virturoid.services.cad_geometry import export_gene_cad
            cad = export_gene_cad(gene, str(out_dir / "cad"))
            artifacts["cad"] = cad if isinstance(cad, dict) else str(out_dir / "cad")
        except Exception as exc:  # noqa: BLE001
            artifacts["cad_error"] = f"{type(exc).__name__}: {exc}"
    if "urdf" in fmts or "ros2" in fmts:
        # _write_genome_and_urdf produces robot_genome.json + robot.urdf + (best-effort) the installable ROS2 pkg
        try:
            from virturoid.services.gene_build import _write_genome_and_urdf
            _write_genome_and_urdf(gene, out_dir)
            urdf = out_dir / "robot" / "robot.urdf"
            if "urdf" in fmts and urdf.exists():
                artifacts["urdf"] = str(urdf)
            ros2_root = out_dir / "export" / "ros2"
            if "ros2" in fmts and ros2_root.exists():
                pkgs = [str(p) for p in ros2_root.glob("*") if p.is_dir()]
                artifacts["ros2"] = pkgs[0] if pkgs else str(ros2_root)
        except Exception as exc:  # noqa: BLE001
            artifacts["urdf_error"] = f"{type(exc).__name__}: {exc}"
    if "bom" in fmts:
        try:
            from virturoid.services.bom_builder import build_bom, format_bom_markdown
            bom = build_bom(gene, task=task)
            (out_dir / "bom.json").write_text(json.dumps(bom, indent=2, default=str), encoding="utf-8")
            (out_dir / "bom.md").write_text(format_bom_markdown(bom), encoding="utf-8")
            artifacts["bom"] = str(out_dir / "bom.json")
        except Exception as exc:  # noqa: BLE001
            artifacts["bom_error"] = f"{type(exc).__name__}: {exc}"
    if "usd" in fmts:
        # OpenUSD physics articulation for NVIDIA Isaac Sim (transcribed from the simulated MuJoCo model)
        try:
            from virturoid.services.usd_exporter import export_usd
            um = export_usd(gene, str(out_dir / "robot.usda"))
            artifacts["usd"] = um["usd_path"]
        except Exception as exc:  # noqa: BLE001 - usd-core may be absent; degrade honestly
            artifacts["usd_error"] = f"{type(exc).__name__}: {exc}"
    if "isaac_lab" in fmts:
        # a full Isaac Lab hand-off package: USD + ArticulationCfg + spawn/train scaffolds + README
        try:
            from virturoid.services.isaac_lab_exporter import export_isaac_lab
            im = export_isaac_lab(gene, str(out_dir / "isaac_lab"))
            artifacts["isaac_lab"] = im["files"].get("readme", str(out_dir / "isaac_lab"))
        except Exception as exc:  # noqa: BLE001
            artifacts["isaac_lab_error"] = f"{type(exc).__name__}: {exc}"
    if "certificate" in fmts:
        # the moat's honesty, TRAVELLING with the export: certificate v2 (WS-E) — the un-gameable verdict PLUS
        # tiered sim-to-real evidence (model-sanity VOID gate, actuator-fidelity level, per-joint margins, and a
        # frozen-policy DR sweep with a Clopper-Pearson bound when ``dr_sweep`` is requested). "Arrives in Isaac
        # already verified" now MEANS something to a hardware team. Margins are cheap (1 rollout); the DR sweep is
        # opt-in (many rollouts) so the standard export stays fast.
        try:
            from virturoid.services.ai_native_tools import verify_robot
            from virturoid.services.certificate_v2 import build_certificate_v2
            # MEASURE THE BODY THIS PACKAGE SHIPS. This used to verify the SESSION robot_id and then staple the
            # result onto the exported gene, so cert["verdict"]["checks"] described one robot while
            # cert["model_sanity"]/["margins"] described another. Verify the exported body itself, under a
            # scratch id so the customer's session is untouched, and hand the certificate the held-vs-shipped
            # comparison so it can only claim deploy==measure when that is actually true.
            _vid = f"__export__{args['robot_id']}"
            S.put_robot(gene, robot_id=_vid, prompt=task, label="export-verify")
            try:
                v = verify_robot({"robot_id": _vid, "mode": "quick"})
            finally:
                S.forget_robot(_vid)
            cert = build_certificate_v2(gene, v, task=task, robot_id=args["robot_id"],
                                        body_parity=body_parity,
                                        run_margins=bool(args.get("certificate_margins", True)),
                                        run_dr=bool(args.get("dr_sweep", False)),
                                        dr_draws=int(args.get("dr_draws", 12)))
            # Say WHERE these numbers came from. They are a fresh rollout taken at export time on the shipped
            # body -- not a transcript of whatever `verify_robot` the customer ran earlier, which may have used
            # a different budget (full = 1500 steps vs quick = 800) and would legitimately read differently.
            cert["measured_on"] = {"body": "the body in this package", "when": "export",
                                   "mode": "quick", "same_as_held_body": bool(body_parity.get("same")),
                                   "note": "re-measured here so the verdict and the shipped model are one "
                                           "robot; an earlier interactive verdict at a different rollout "
                                           "budget can differ and is not what this certificate signs"}
            (out_dir / "verification_certificate.json").write_text(
                json.dumps(cert, indent=2, default=str), encoding="utf-8")
            artifacts["certificate"] = str(out_dir / "verification_certificate.json")
        except Exception as exc:  # noqa: BLE001 - a certificate failure must never sink the buildable export
            artifacts["certificate_error"] = f"{type(exc).__name__}: {exc}"
    # THE SPEC SHEET GOES LAST, because it is a pure read-over-artifacts summary of everything above it. It used
    # to be written between 'bom' and 'usd' -- i.e. BEFORE the certificate existed -- so the sheet in every
    # exported package reported `task: "quadruped"` with no verdict, no forward distance and no cross-check
    # against the certificate's motor selection, while a fully populated verification_certificate.json sat
    # beside it. Nothing here computes; it only aggregates, so it must run once every input has been written.
    if "spec" in fmts:
        try:
            from virturoid.services.spec_sheet import write_spec_sheet
            sp = write_spec_sheet(out_dir)
            if sp:
                artifacts["spec"] = str(sp)
        except Exception as exc:  # noqa: BLE001
            artifacts["spec_error"] = f"{type(exc).__name__}: {exc}"
    real = {k: v for k, v in artifacts.items() if not k.endswith("_error")}
    out = {"ok": bool(real), "artifacts": artifacts, "out_dir": str(out_dir),
           # The customer should not have to open a json to learn whether they were shipped the robot they
           # verified. ``same_as_verified`` is the whole invariant in one boolean.
           "body_parity": {"same_as_verified": bool(body_parity.get("same")),
                           "total_mass_kg": body_parity.get("total_mass_kg"),
                           "n_links_changed": body_parity.get("n_links_changed")},
           "note": "MJCF runs in sim; URDF/ROS2 deploy; BOM is the real sized parts list; spec is the datasheet; "
                   "usd/isaac_lab hand off to NVIDIA Isaac Sim/Lab; verification_certificate.json is the "
                   "un-gameable physics verdict that travels with the design"}
    if not body_parity.get("same"):
        out["warnings"] = [
            f"EXPORT CHANGED THE BODY: {body_parity.get('n_links_changed')} link(s) differ from the robot you "
            f"verified (total mass {body_parity.get('total_mass_kg')} kg, delta "
            f"{body_parity.get('delta_mass_kg')} kg). The certificate in this package was re-measured on the "
            "SHIPPED body and does not claim deploy==measure against your earlier verdict."]
        out["body_parity"]["changed"] = body_parity.get("changed")
    return out


def export_isaac(args: dict) -> dict:
    """Package the HELD robot for NVIDIA Isaac Sim / Isaac Lab: a physics USD (transcribed from the simulated
    MuJoCo model + re-read/validated with OpenUSD) plus an ArticulationCfg (real per-joint motor limits), a
    standalone spawn/smoke script, a velocity-locomotion env (legged) subclassing Isaac Lab's own task, a README,
    and a manifest. The 'front of funnel -> Isaac back of funnel' hand-off. Needs the ``usd-core`` package."""
    from virturoid.services import session_state as S
    gene = S.get_robot(args["robot_id"])
    if gene is None:
        return {"ok": False, "error": f"no robot '{args['robot_id']}'"}
    from virturoid.services.agent_tools import safe_build_path  # H2: confine writes under build/
    out_dir = safe_build_path(args.get("out_dir"), "agent_exports") / args["robot_id"] / "isaac_lab"
    try:
        from virturoid.services.isaac_lab_exporter import export_isaac_lab
        man = export_isaac_lab(gene, str(out_dir), robot_name=args.get("robot_name"))
    except ImportError as exc:
        return {"ok": False, "error": str(exc),
                "hint": "pip install usd-core (pure-CPU OpenUSD; no GPU/Omniverse needed)"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": bool(man.get("validated")), "manifest": man, "files": man["files"],
            "dof": man["dof"], "base": man["base"], "is_legged": man["is_legged"],
            "usd_validated": man["validated"], "isaac_lab_target": man["isaac_lab_target"],
            "note": "USD re-read + frame-consistency validated offline; NOT run in Isaac here (needs RTX/Omniverse) "
                    "— the README states what to verify in Isaac. Front of funnel; Isaac Lab does the training."}


def train_held(args: dict) -> dict:
    """Optimize/train a controller for the HELD robot and return a job_id (poll get_job). Default mode
    'gait_search' = the bounded, VERIFIED CPG search (CPU, reliable) on THIS gene; 'gpu_rl' = MJX PPO on the
    box when reachable. The long-job handle pattern (start now, poll later)."""
    from virturoid.services import job_registry as J
    from virturoid.services import session_state as S
    rid = args.get("robot_id")
    if not rid or S.get_robot(rid) is None:
        return {"ok": False, "error": f"no robot '{rid}'; submit_design/create_robot first"}
    from virturoid.services.agent_tools import safe_build_path  # H2: confine writes under build/
    job = J.create("train_gene", {"robot_id": rid, "mode": args.get("mode", "gait_search"),
                                  "max_evals": int(args.get("max_evals", 8)), "iters": int(args.get("iters", 200))},
                   safe_build_path(args.get("build_root"), "agent_builds"))
    return {"ok": True, "job_id": job.get("id"), "status": job.get("status"),
            "note": "poll get_job(job_id, since) for progress + the honest gait verdict"}


def run_train_gene_job(args: dict, progress=None) -> dict:
    """The train_gene job WORKER (called by job_registry). Reads the held gene, runs the search/train, and
    returns the honest best verdict. In-process, so it reads session_state directly."""
    from virturoid.services import session_state as S
    def say(stage, msg):
        if progress:
            progress({"stage": stage, "message": msg})
    # Accept a HELD robot_id OR a prompt (compose one) — never crash with a raw KeyError on a missing robot_id.
    rid = args.get("robot_id")
    if rid:
        gene = S.get_robot(rid)
        if gene is None:
            return {"error": f"no robot '{rid}' held; create_robot/submit_design first, or pass a 'prompt'"}
    elif str(args.get("prompt") or "").strip():
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot(str(args["prompt"]).strip(), ensure_walkable=True)
        rid = S.put_robot(gene, prompt=str(args["prompt"]).strip(), label="train_gene")
    else:
        return {"error": "provide 'robot_id' (a held robot) or 'prompt' (to compose one) to train"}
    mode = args.get("mode", "gait_search")
    if mode == "gpu_rl":
        from virturoid.services.gpu_trainer import default_training_recipe, gpu_available, train_gene_on_gpu
        if gpu_available(timeout=20):
            recipe = default_training_recipe(gene)           # AUTO recipe per body (cpg/adaptive/deploy deltas)
            say("train", f"GPU reachable — MJX PPO on the held gene (auto recipe: adaptive={recipe['adaptive']}, "
                         f"cpg={recipe['cpg']})")
            out = Path("build/agent_builds") / rid / "policy.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            npz = train_gene_on_gpu(gene, out_path=str(out), iters=int(args.get("iters", 200)), envs=512,
                                    progress=lambda m: say("train", m), **recipe)
            return {"mode": "gpu_rl", "policy": npz, "trained": bool(npz)}
        say("train", "GPU not reachable — falling back to the CPU gait search")
    # One gait path owns recall, bounded search, classify()-credible early-stop, deploy comparison and banking.
    say("search", "recalling prior gait hints, then physics-evaluating a bounded per-body search")
    from virturoid.services.agent_tools import safe_build_path
    from virturoid.services.gait_flywheel import learn_gait_flywheel
    from virturoid.services.memory_db import MemoryDB
    max_evals = max(1, int(args.get("max_evals", 8)))
    mem = safe_build_path(None, "memory")
    mem.mkdir(parents=True, exist_ok=True)
    with MemoryDB(mem / "virturoid_memory.db") as db:
        learned = learn_gait_flywheel(
            gene, db, generations=max_evals, pop=min(8, max_evals), steps=600, deploy_steps=600,
            seed=int(args.get("seed", 0)), workers=1, max_evals=max_evals, stop_on_credible=True,
        )
    solved = bool(learned.get("survived")) and bool(learned.get("beats_default"))
    say("done", f"searched {learned['n_evals']} configs; credible stop="
                f"{learned['stopped_reason'] == 'credible_walk'}; beats default={learned['beats_default']}")
    if learned.get("banked_skill"):
        _bank_to_flywheel(gene, prompt=f"[agent-trained] {gene.robot_class}", task="locomotion",
                          success_rate=min(1.0, max(0.0, abs(float(learned["forward_m"])) / 0.5)),
                          source="agent_trained")
    return {"mode": "gait_search", "solved": solved, "credible": bool(learned.get("survived")),
            "n_evals": learned["n_evals"], "stopped_reason": learned["stopped_reason"],
            "reused_prior": learned["reused_prior"], "banked_gait": learned["banked_skill"],
            "default_forward_m": learned["default_forward_m"], "beats_default": learned["beats_default"],
            "best": {"params": learned["params"], "forward_m": learned["forward_m"],
                     "height_ratio": learned["height_ratio"]}}


def list_skills(_args: dict) -> dict:
    """T5: the general TASK vocabulary — the skills an agent can sequence + the closed predicate ops a goal
    scores. The task layer is morphology-agnostic: a spec is VERIFIED against the body (an arm asked to walk is
    honestly infeasible), then executed with real skills. Call before run_task/submit_task."""
    from virturoid.schemas.task_spec import PredicateOp
    from virturoid.services.capability_registry import REGISTRY
    skills = [{"skill": s.skill, "summary": s.summary, "morphologies": list(s.morphologies),
               "establishes": [o.value for o in s.establishes]} for s in REGISTRY.values()]
    return {"ok": True, "skills": skills, "predicate_ops": [op.value for op in PredicateOp],
            "note": "run_task {robot_id, goal} to plan+verify+run from a goal; submit_task to author the "
                    "skill sequence yourself"}


def run_task(args: dict) -> dict:
    """T5: give the HELD robot a goal in open language; the substrate PROPOSES a skill sequence, VERIFIES it
    against the morphology (honest infeasible on a mismatch), RUNS the real skills, and scores the goal
    predicates. The general 'any task' capability — no per-task hard-coding (task_executor.evaluate_task)."""
    from virturoid.services import session_state as S
    from virturoid.services.task_executor import evaluate_task
    gene = S.get_robot(args.get("robot_id"))
    if gene is None:
        return {"ok": False, "error": f"no robot '{args.get('robot_id')}'; submit_design/create_robot first"}
    raw = args.get("goal")
    if raw is not None and not str(raw).strip():           # an EXPLICIT empty/blank goal is a mistake, not a default
        return {"ok": False, "error": "the goal is empty — give a plain-language task, e.g. 'walk to the goal' "
                                      "or 'pick up the block and place it on the target'"}
    goal = (str(raw).strip() if raw else "") or (S.robot_meta(args["robot_id"]) or {}).get("prompt", "")
    if not goal:
        return {"ok": False, "error": "provide goal: a plain-language task (e.g. 'navigate to the far corner')"}
    if _OUT_OF_DOMAIN.search(goal):                            # M11: honest infeasible, never a fake 1.0
        return {"ok": True, "goal": goal, "feasible": False, "success": False, "score": 0.0,
                "goal_met": 0, "goal_total": 0, "task": "out_of_domain", "planned_skills": [], "steps": [],
                "issues": ["the goal names an out-of-domain intent this platform has no tier for "
                           "(no aerospace / teleportation / phase-through capability) — rephrase as a "
                           "ground locomotion, navigation, or manipulation task"]}
    r = evaluate_task(str(goal), gene, llm="auto")             # llm None under NO_INTERNAL_LLM -> heuristic plan
    return {"ok": True, "goal": goal, "feasible": bool(r.get("feasible")), "success": bool(r.get("success")),
            "score": round(float(r.get("score", 0.0)), 3), "goal_met": r.get("goal_met"),
            "goal_total": r.get("goal_total"), "task": r.get("task"), "planned_skills": r.get("steps_planned"),
            "steps": [{"skill": s.get("skill"), "success": s.get("success"), "detail": s.get("detail")}
                      for s in (r.get("steps") or [])], "issues": r.get("issues") or []}


def submit_task(args: dict) -> dict:
    """T5: the agent AUTHORS the task itself — an explicit skill sequence + goal predicates (the pivot: you are
    the planner). Builds a TaskSpec, VERIFIES it against the held morphology (teaching error on an unknown skill
    or an infeasible plan), runs it, and scores the predicates. See list_skills for the vocabulary."""
    from virturoid.schemas.task_spec import Entity, Predicate, PredicateOp, SkillCall, TaskSpec
    from virturoid.services import session_state as S
    from virturoid.services.task_executor import run_task as _run
    from virturoid.services.task_verifier import verify_task
    gene = S.get_robot(args.get("robot_id"))
    if gene is None:
        return {"ok": False, "error": f"no robot '{args.get('robot_id')}'; submit_design/create_robot first"}
    steps_in = args.get("steps")
    if not isinstance(steps_in, list) or not steps_in:
        return {"ok": False, "error": "provide steps:[{skill, params?}] + goal:[{op, args?}]; call list_skills"}
    try:
        steps = [SkillCall(str(s.get("step_id") or f"s{i}"), str(s["skill"]), dict(s.get("params") or {}))
                 for i, s in enumerate(steps_in)]
        goal = [Predicate(PredicateOp(g["op"]), list(g.get("args") or [])) for g in (args.get("goal") or [])]
    except (KeyError, ValueError) as exc:
        return {"ok": False, "error": f"bad step/goal ({exc}); each step needs a 'skill', each goal a valid "
                f"'op' — call list_skills for the skill + predicate_op vocabulary"}
    ents = [Entity(str(e["id"]), str(e.get("kind", "object"))) for e in (args.get("entities") or []) if e.get("id")]
    spec = TaskSpec(name=str(args.get("name") or "agent_task"), prompt=str(args.get("goal_text") or ""),
                    entities=ents, steps=steps, goal=goal, source="agent")
    v = verify_task(spec, gene)
    if not v.get("ok"):
        return {"ok": False, "feasible": False, "error": "task INFEASIBLE for this morphology: "
                + "; ".join(v.get("issues", [])[:4]), "issues": v.get("issues")}
    r = _run(spec, gene)
    return {"ok": True, "feasible": True, "success": bool(r.get("success")),
            "score": round(float(r.get("score", 0.0)), 3), "goal_met": r.get("goal_met"),
            "goal_total": r.get("goal_total"), "steps": [{"skill": s.get("skill"), "success": s.get("success")}
                                                          for s in (r.get("steps") or [])]}


def llm_spend(_args: dict) -> dict:
    """The internal-LLM spend ledger (G-D). Proof of the zero-our-tokens pitch: after an agent-driven loop,
    ``totals.internal_calls`` should be 0 (every role fell through to the deterministic substrate or the
    connected agent). ``blocked`` counts roles the no-internal-LLM switch denied. Read this to VERIFY, don't
    trust the claim."""
    from virturoid.services.llm_client import spend_snapshot
    snap = spend_snapshot()
    zero = snap["totals"]["internal_calls"] == 0
    return {"ok": True, "zero_internal_spend": zero, **snap,
            "note": ("no internal LLM has fired this process — the connected agent + deterministic substrate did "
                     "all the work" if zero else "internal LLM calls occurred; set VIRTUROID_NO_INTERNAL_LLM=1 "
                     "to force the zero-spend, agent-only mode")}


AGENT_DESIGN_TOOLS: dict[str, dict] = {
    "llm_spend": {"description": "The internal-LLM SPEND LEDGER — verify the zero-our-tokens promise. Returns "
                  "per-role calls/blocked/tokens + totals; internal_calls==0 after your loop = proof we spent "
                  "nothing. No args.", "heavy": False, "handler": llm_spend,
                  "parameters": {"type": "object", "properties": {}}},
    "get_design_schema": {"description": "The anatomy-graph LANGUAGE to author a robot (part fields + roles/"
                          "attach/aim vocab + 2 worked examples). Call before submit_design.", "heavy": False,
                          "handler": get_design_schema, "parameters": {"type": "object", "properties": {}}},
    "submit_design": {"description": "AUTHOR a robot: compile YOUR anatomy graph, gate it, hold it under a "
                      "robot_id (no prompt, no internal generator — you are the designer). Returns id+summary+render "
                      "or a teaching error.", "heavy": True, "handler": submit_design,
                      "parameters": {"type": "object", "required": ["graph"], "properties": {
                          "graph": {"type": "object", "description": "anatomy graph: {robot_class, name, parts:[...]}"},
                          "material": {"type": "string"}}}},
    "critique_design": {"description": "SEE + CRITIQUE a held design: returns an inline render plus deterministic "
                        "engineering, anatomy, and visible-contact-vs-physics findings. The model proposes edits; "
                        "Python remains the gate. Use at most three rounds (hard cap four).", "heavy": True,
                        "handler": critique_design,
                        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                            "robot_id": {"type": "string"}, "material": {"type": "string"},
                            "payload_kg": {"type": "number"}, "azimuth": {"type": "number"},
                            "elevation": {"type": "number"}, "round": {"type": "integer", "minimum": 1,
                            "maximum": 4}}}},
    "submit_scene_spec": {"description": "AUTHOR a scene: a list of objects + a robot spawn -> a validated held "
                          "scene (you are the scene designer).", "heavy": False, "handler": submit_scene_spec,
                          "parameters": {"type": "object", "required": ["objects"], "properties": {
                              "objects": {"type": "array", "items": {"type": "object"}}, "task": {"type": "string"},
                              "robot_spawn_xyz_rpy": {"type": "array"}}}},
    "evaluate_held": {"description": "Score the HELD robot on its task (real MuJoCo) — the exact gene you "
                      "designed/edited, not a recompose. The task is IMPLIED BY THE MORPHOLOGY — it is not an "
                      "input; use run_task/submit_task to give the robot a goal you choose.",
                      "heavy": True, "handler": evaluate_held,
                      # `task` used to be advertised here and was never read: `evaluate_held` scores the
                      # morphology-implied task and derives the prompt from the held robot's own metadata, so an
                      # agent that passed `task:'transport'` got the same verdict it would have got anyway and no
                      # indication its argument had been dropped. A schema must not promise a lever that moves
                      # nothing — removed rather than implemented, because task CHOICE already has two tools.
                      "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                          "robot_id": {"type": "string"}}}},
    "export_held": {"description": "Export the HELD robot to real files. formats (default all): mjcf | cad | urdf | "
                    "ros2 | bom | spec | usd (OpenUSD physics for Isaac Sim) | isaac_lab (full Isaac Lab hand-off). "
                    "Returns paths.", "heavy": True, "handler": export_held,
                    "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                        "robot_id": {"type": "string"}, "formats": {"type": "array", "items": {"type": "string"}}}}},
    "export_isaac": {"description": "Package the HELD robot for NVIDIA Isaac Sim / Isaac Lab: a validated physics "
                     "USD + ArticulationCfg (real motor limits) + spawn/train scaffolds + README. The hand-off so "
                     "an Isaac/NVIDIA engineer can import it and train. Needs usd-core.", "heavy": True,
                     "handler": export_isaac, "parameters": {"type": "object", "required": ["robot_id"],
                     "properties": {"robot_id": {"type": "string"},
                                    "robot_name": {"type": "string", "description": "name for the generated cfg/files"}}}},
    "train_held": {"description": "Optimize/train a controller for the HELD robot; returns a job_id (poll "
                   "get_job). mode 'gait_search'(CPU, default) or 'gpu_rl'(MJX PPO when the box is up).", "heavy": False,
                   "handler": train_held, "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                       "robot_id": {"type": "string"}, "mode": {"type": "string"}, "max_evals": {"type": "integer"}}}},
    "list_skills": {"description": "The general TASK vocabulary: skills you can sequence + the predicate ops a "
                    "goal scores. Call before run_task/submit_task. No args.", "heavy": False,
                    "handler": list_skills, "parameters": {"type": "object", "properties": {}}},
    "run_task": {"description": "Give the HELD robot a GOAL in plain language: the substrate proposes a skill "
                 "sequence, VERIFIES it against the morphology (honest infeasible on a mismatch), runs the real "
                 "skills, and scores the goal — the general 'any task' capability (real physics).", "heavy": True,
                 "handler": run_task, "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                     "robot_id": {"type": "string"}, "goal": {"type": "string", "description": "plain-language task"}}}},
    "submit_task": {"description": "AUTHOR the task yourself: an explicit skill sequence + goal predicates "
                    "(you are the planner). Verified against the held morphology, run, scored. See list_skills.",
                    "heavy": True, "handler": submit_task, "parameters": {"type": "object",
                    "required": ["robot_id", "steps"], "properties": {"robot_id": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "object"}},
                    "goal": {"type": "array", "items": {"type": "object"}},
                    "entities": {"type": "array", "items": {"type": "object"}}}}},
}
