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
from pathlib import Path

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
                "aim": f"the direction it points: {_AIM}",
                "size": "length in metres (the segment's long axis)", "girth": "radius in metres",
                "aspect": "BODY shape (root part only): 'deck'/'chassis' = a flat rectangular slab for a "
                          "rover/AGV (wheels mount at its corners); 'wide' = a broad low pod; 'round' = a "
                          "bulbous sac; omit for the default sleek barrel torso",
                "segments": "int; a leg with 4 = 3 actuated joints + a welded foot (Go2-class)",
                "symmetry": "'left_right' mirrors the part to a +y/-y PAIR (so one leg entry = two legs)",
                "joint": "'revolute' to actuate it; omit for a welded/fixed part",
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
            "examples": {"quadruped": _EXAMPLE_QUAD, "hexapod": _EXAMPLE_HEX, "rover": _EXAMPLE_ROVER},
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


def _bank_to_flywheel(gene, *, prompt: str, task: str, success_rate: float, source: str = "agent") -> None:
    """B4: write an agent-authored design/outcome into the design flywheel (memory_db + species memory) so
    recall_knowledge / nearest_bodies / search_memory surface the agent's OWN prior work — the moat compounds
    from exactly the usage the pivot created. Best-effort: memory is value-add, never blocks the design loop."""
    try:
        from virturoid.services.agent_tools import safe_build_path
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.task_matched_eval import robot_kind
        mem_dir = safe_build_path(None, "memory")
        mem_dir.mkdir(parents=True, exist_ok=True)
        with MemoryDB(mem_dir / "virturoid_memory.db") as db:
            db.record_run(prompt=prompt or f"[{source}] {gene.robot_class}", robot_class=gene.robot_class,
                          task_type=task or robot_kind(gene), converged_design=gene.to_dict(),
                          success_rate=float(success_rate), species=gene.robot_class,
                          succeeded=success_rate >= 0.5, design_source=source)
    except Exception:  # noqa: BLE001
        pass


def submit_design(args: dict) -> dict:
    """The agent AUTHORS a robot: compile its anatomy graph, run the validity gates, and HOLD it under a
    robot_id for the rest of the loop (simulate/edit/train/export). No prompt, no internal generator — the
    external agent is the designer. Returns the id + summary + render, or a teaching error."""
    from virturoid.services import session_state as S
    from virturoid.services.anatomy_compiler import build_from_anatomy
    graph = args.get("graph")
    if not isinstance(graph, dict) or not graph.get("parts"):
        return {"ok": False, "error": "provide graph:{robot_class, parts:[...]}; call get_design_schema for the language"}
    roots = [p for p in graph["parts"] if (p.get("role") == "body") and not p.get("parent")]
    if len(roots) != 1:
        return {"ok": False, "error": f"a design needs EXACTLY ONE root part with role 'body' and no parent "
                f"(found {len(roots)}); see get_design_schema examples"}
    bad_roles = sorted({str(p.get("role") or "").lower() for p in graph["parts"]} - _ROLESET - {""})
    if bad_roles:                                              # T3: teach, never silently mis-compile an unknown role
        return {"ok": False, "error": f"unknown part role(s) {bad_roles}; use only {sorted(_ROLESET)} "
                f"(an unknown role is NOT silently turned into a limb)"}
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
    from virturoid.services.ai_native_tools import _render_gene, _summary
    rid = S.put_robot(gene, prompt=f"[submitted:{graph.get('name', 'design')}]", label="submitted")
    _bank_to_flywheel(gene, prompt=f"[agent] {graph.get('name', 'design')}", task="", success_rate=0.0)  # B4 provenance
    try:                                                       # Thesis A WRITE-side: this design's VERIFIED shape
        from virturoid.services.shape_flywheel import auto_bank_body_shapes   # words enter the corpus, so the NEXT
        banked = auto_bank_body_shapes(gene)                   # get_design_schema recalls them (self-manufacture)
    except Exception:  # noqa: BLE001 - corpus growth is an accelerant, never blocks a valid design
        banked = []
    out = {"ok": True, **_summary(gene, rid), "name": graph.get("name")}
    if banked:
        out["corpus_shape_words"] = len(banked)               # retrieval grounding now has this design's words
    warns = _proportion_warnings(graph)                        # T8: non-blocking proportion advisories
    if warns:
        out["proportion_warnings"] = warns
    img = _render_gene(gene, rid)
    if img:
        out["artifacts"] = [img]
    return out


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


_EXPORT_FORMATS = ("mjcf", "cad", "urdf", "ros2", "bom", "spec", "usd", "isaac_lab")


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
    if "spec" in fmts:
        try:
            from virturoid.services.spec_sheet import write_spec_sheet
            sp = write_spec_sheet(out_dir)
            if sp:
                artifacts["spec"] = str(sp)
        except Exception as exc:  # noqa: BLE001
            artifacts["spec_error"] = f"{type(exc).__name__}: {exc}"
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
    real = {k: v for k, v in artifacts.items() if not k.endswith("_error")}
    return {"ok": bool(real), "artifacts": artifacts, "out_dir": str(out_dir),
            "note": "MJCF runs in sim; URDF/ROS2 deploy; BOM is the real sized parts list; spec is the datasheet; "
                    "usd/isaac_lab hand off to NVIDIA Isaac Sim/Lab"}


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
    # gait_search: the bounded, honesty-gated CPG search on THIS gene (the verified multi-step search)
    say("search", "sweeping CPG gait params, physics-evaluating each")
    from virturoid.services.design_search import run_design_search
    from virturoid.services.search_adapters import cpg_grid_proposer, make_cpg_evaluate
    evaluate = make_cpg_evaluate(gene, steps=600)
    gates = {"forward_m": 0.12, "cadence": 3.0, "upright": 0.5}
    rep = run_design_search(propose=cpg_grid_proposer(), evaluate=evaluate, task_type="locomotion",
                            max_evals=int(args.get("max_evals", 8)), gates=gates)
    b = rep.best
    say("done", f"searched {rep.n_evals} configs; solved={rep.solved}")
    # B4: bank the real trained OUTCOME (walking distance -> success) into the flywheel, source=agent
    _fwd = float(b.result.get("forward", 0)) if b else 0.0
    _bank_to_flywheel(gene, prompt=f"[agent-trained] {gene.robot_class}", task="locomotion",
                      success_rate=min(1.0, max(0.0, _fwd / 0.5)), source="agent_trained")
    return {"mode": "gait_search", "solved": rep.solved, "n_evals": rep.n_evals,
            "best": ({"params": b.spec.get("params"), "forward_m": round(float(b.result.get("forward", 0)), 3),
                      "cadence": round(float(b.result.get("cadence", 0)), 1),
                      "failure_mode": b.artifact.get("failure_mode")} if b else None)}


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
    "submit_scene_spec": {"description": "AUTHOR a scene: a list of objects + a robot spawn -> a validated held "
                          "scene (you are the scene designer).", "heavy": False, "handler": submit_scene_spec,
                          "parameters": {"type": "object", "required": ["objects"], "properties": {
                              "objects": {"type": "array", "items": {"type": "object"}}, "task": {"type": "string"},
                              "robot_spawn_xyz_rpy": {"type": "array"}}}},
    "evaluate_held": {"description": "Score the HELD robot on its task (real MuJoCo) — the exact gene you "
                      "designed/edited, not a recompose.", "heavy": True, "handler": evaluate_held,
                      "parameters": {"type": "object", "required": ["robot_id"], "properties": {
                          "robot_id": {"type": "string"}, "task": {"type": "string"}}}},
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
