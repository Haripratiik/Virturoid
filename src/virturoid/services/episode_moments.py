"""Thread C — semantic MOMENT search over episodes: "when did the gripper fail?" without an LLM per frame.

The thesis (docs/robotics_language_gen3d_research.md §4): in a SIMULATOR, semantic moments are EMITTABLE, not
inferrable — the physics already computes, every step, exactly what a temporal question is about (contacts,
grasp attach/detach, body height, goal reach, joint saturation). ``extract_events`` replays a recorded episode
(a compiled model + its qpos frames) and emits a compact TEMPORAL SENTENCE of typed transition events
(~10^2 per episode, not 10^5 raw steps). ``ask_episode`` compiles a natural-language question into ONE predicate
query over that event log — one call per QUESTION, ZERO LLM calls per frame — a runtime-monitor / text2SQL move
(REFLECT, NL2TL, MCAP). Questions outside the closed vocabulary return ``matched=False`` so a caller can escalate
to the external LLM (which authors a predicate over the SAME logged channels) or a windowed embedding fallback —
never a per-frame VLM scan.

This module is deliberately self-contained (mujoco + numpy only): it never re-runs control, only re-reads the
already-recorded trajectory, so it works for ANY rollout that recorded qpos frames (locomotion, manipulation,
flight) and for imported episodes. Deterministic; CPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Closed event vocabulary — the "words" of the temporal dialect. Every event is a TRANSITION (an edge), so an
# episode is a short sentence, not a per-step dump.
CONTACT_BEGIN = "contact_begin"
CONTACT_END = "contact_end"
FOOT_DOWN = "foot_down"          # a foot touched the floor (a step's stance onset)
FOOT_UP = "foot_up"             # a foot left the floor (swing onset)
GRASP_ATTACH = "grasp_attach"    # a gripper geom made contact with a payload
GRASP_DETACH = "grasp_detach"    # ...and later lost it
FELL = "fell"                    # base height dropped below fall_ratio * initial (a topple / collapse)
RECOVERED = "recovered"          # ...and climbed back up
GOAL_REACHED = "goal_reached"    # base entered the goal radius
JOINT_SATURATED = "joint_saturated"  # an actuated joint pinned at a limit


@dataclass
class Event:
    step: int
    t: float                      # seconds
    type: str
    subject: str = ""             # e.g. the foot / gripper / joint name
    obj: str = ""                 # the other party (floor / payload)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"step": self.step, "t": round(self.t, 3), "type": self.type,
                "subject": self.subject, "obj": self.obj, **({"detail": self.detail} if self.detail else {})}


def _geom_body_name(model, gi, mujoco):
    b = int(model.geom_bodyid[gi])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def _classify_geoms(model, mujoco, *, grippers=None, payloads=None, floor=None):
    """Best-effort role tags for geoms so events read in human terms without the caller wiring names.
    floor = plane geoms; gripper = geoms on a hand/gripper/finger/claw body; payload = free non-robot bodies."""
    floor_g, grip_g, pay_g, foot_g = set(), set(), set(), set()
    grip_kw = tuple(g.lower() for g in (grippers or ("gripper", "hand", "finger", "claw", "palm")))
    pay_kw = tuple(p.lower() for p in (payloads or ("box", "block", "cube", "object", "payload", "ball", "can")))
    for gi in range(model.ngeom):
        gt = int(model.geom_type[gi])
        if gt == 0:                               # mjGEOM_PLANE
            floor_g.add(gi); continue
        bn = _geom_body_name(model, gi, mujoco).lower()
        if any(k in bn for k in grip_kw):
            grip_g.add(gi)
        if any(k in bn for k in pay_kw):
            pay_g.add(gi)
        if any(k in bn for k in ("foot", "paw", "toe")) or "foot" in bn:
            foot_g.add(gi)
    if floor is not None:                         # explicit floor geom name override
        for gi in range(model.ngeom):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or "") == floor:
                floor_g.add(gi)
    return floor_g, grip_g, pay_g, foot_g


def extract_events(model, qpos_frames, *, dt: float = 0.01, frame_every: int = 1,
                   grippers=None, payloads=None, goal_xy=None, goal_radius: float = 0.3,
                   fall_ratio: float = 0.55) -> list[Event]:
    """Replay a recorded episode and emit its typed transition events. ``qpos_frames`` = the per-recorded-frame
    qpos arrays (e.g. ``crawl_gait_rollout(record_qpos=True)['qpos_frames']``); ``frame_every`` = how many sim
    steps each recorded frame spans (for the ``t`` timestamp). Robust to partial qpos rows."""
    import numpy as np
    import mujoco

    if not qpos_frames:
        return []
    data = mujoco.MjData(model)
    floor_g, grip_g, pay_g, foot_g = _classify_geoms(model, mujoco, grippers=grippers, payloads=payloads)
    # base height reference = the first frame's base z (free-joint qpos[2]); height_ratio = z / z0
    q0 = np.asarray(qpos_frames[0], float)
    z0 = float(q0[2]) if model.nq > 2 and q0.shape[0] > 2 else 1.0
    z0 = z0 if abs(z0) > 1e-6 else 1.0

    def name(gi):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or _geom_body_name(model, gi, mujoco)

    events: list[Event] = []
    prev_pairs: set = set()
    prev_grasp = False
    fallen = False
    reached = False
    sat_prev: set = set()
    for fi, q in enumerate(qpos_frames):
        q = np.asarray(q, float)
        n = min(q.shape[0], model.nq)
        data.qpos[:n] = q[:n]
        mujoco.mj_forward(model, data)
        step = fi * max(1, frame_every)
        t = step * dt
        # --- contacts this frame (unordered geom pairs) ---
        pairs = set()
        grasp_now = False
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            pairs.add((min(g1, g2), max(g1, g2)))
            if (g1 in grip_g and g2 in pay_g) or (g2 in grip_g and g1 in pay_g):
                grasp_now = True
        for pr in pairs - prev_pairs:
            a, b = pr
            on_floor = a in floor_g or b in floor_g
            foot = (a in foot_g or b in foot_g)
            typ = FOOT_DOWN if (on_floor and foot) else CONTACT_BEGIN
            events.append(Event(step, t, typ, name(a if b in floor_g else b), name(b if b in floor_g else a)))
        for pr in prev_pairs - pairs:
            a, b = pr
            on_floor = a in floor_g or b in floor_g
            foot = (a in foot_g or b in foot_g)
            typ = FOOT_UP if (on_floor and foot) else CONTACT_END
            events.append(Event(step, t, typ, name(a if b in floor_g else b), name(b if b in floor_g else a)))
        prev_pairs = pairs
        # --- grasp attach/detach ---
        if grasp_now and not prev_grasp:
            events.append(Event(step, t, GRASP_ATTACH, "gripper", "payload"))
        elif prev_grasp and not grasp_now:
            events.append(Event(step, t, GRASP_DETACH, "gripper", "payload",
                                {"note": "gripper lost the payload"}))
        prev_grasp = grasp_now
        # --- fall / recover ---
        if model.nq > 2 and n > 2:
            hr = float(q[2]) / z0
            if not fallen and hr < fall_ratio:
                fallen = True
                events.append(Event(step, t, FELL, "base", "", {"height_ratio": round(hr, 3)}))
            elif fallen and hr > fall_ratio + 0.15:
                fallen = False
                events.append(Event(step, t, RECOVERED, "base", "", {"height_ratio": round(hr, 3)}))
        # --- goal reached ---
        if goal_xy is not None and not reached and model.nq > 1:
            d2 = ((float(q[0]) - goal_xy[0]) ** 2 + (float(q[1]) - goal_xy[1]) ** 2) ** 0.5
            if d2 < goal_radius:
                reached = True
                events.append(Event(step, t, GOAL_REACHED, "base", "", {"dist": round(d2, 3)}))
        # --- joint saturation (actuated hinge/slide pinned at a limit) ---
        sat_now = set()
        for j in range(model.njnt):
            if not bool(model.jnt_limited[j]) or int(model.jnt_type[j]) not in (2, 3):
                continue
            adr = int(model.jnt_qposadr[j])
            if adr >= n:
                continue
            lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
            val = float(q[adr])
            span = max(1e-6, hi - lo)
            if val <= lo + 0.02 * span or val >= hi - 0.02 * span:
                sat_now.add(j)
        for j in sat_now - sat_prev:
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            events.append(Event(step, t, JOINT_SATURATED, jn, "", {}))
        sat_prev = sat_now
    return events


# ----------------------------- the query compiler (rule-based; LLM fallback for the open tail) -----------------

_QUERY_RULES = [
    # (regex over the lowercased question, event predicate over the log) -> the FIRST matching event is the answer
    ("fall|fell|topple|collaps|tip over|tipped|fall over|knock", lambda ev: [e for e in ev if e.type == FELL]),
    ("recover|got back up|stood back", lambda ev: [e for e in ev if e.type == RECOVERED]),
    ("grip.*fail|grasp.*fail|drop|dropped|lose.*grip|lost.*grip|let go|release|slipped out",
     lambda ev: [e for e in ev if e.type == GRASP_DETACH]),
    ("grasp|grab|grip|pick up|picked up|grabbed|latch", lambda ev: [e for e in ev if e.type == GRASP_ATTACH]),
    ("reach.*goal|reach.*target|arrive|arrived|get to the|got to the|reach the",
     lambda ev: [e for e in ev if e.type == GOAL_REACHED]),
    ("first step|start walking|first contact|touch.*ground|first.*foot",
     lambda ev: [e for e in ev if e.type in (FOOT_DOWN, CONTACT_BEGIN)]),
    ("saturat|hit.*limit|max out|maxed|pinned|joint.*limit",
     lambda ev: [e for e in ev if e.type == JOINT_SATURATED]),
]


def compile_query(question: str):
    """Compile a natural-language temporal question into a predicate over the event log. Returns
    ``(predicate, matched_rule)`` or ``(None, None)`` when the question is outside the closed vocabulary — the
    caller then escalates to the external LLM (author a predicate over the same channels) or the embedding
    fallback. This is one call PER QUESTION, never per frame."""
    q = (question or "").lower()
    for pat, pred in _QUERY_RULES:
        if re.search(pat, q):
            return pred, pat
    return None, None


def ask_episode(events, question: str, *, first: bool = True) -> dict:
    """Answer a temporal question over a pre-extracted event log. Deterministic, zero LLM/VLM calls.

    Returns ``{matched, question, when_s, when_step, event, count, all, explanation}``. ``matched=False`` means
    the question fell outside the closed vocabulary (escalate to the LLM query-author / embedding fallback)."""
    pred, rule = compile_query(question)
    if pred is None:
        return {"matched": False, "question": question,
                "explanation": "outside the closed event vocabulary — escalate to the LLM query-compiler or the "
                               "windowed-embedding fallback (no per-frame scan needed)"}
    hits = pred(list(events))
    if not hits:
        return {"matched": True, "question": question, "found": False, "rule": rule, "count": 0,
                "explanation": "the queried moment never occurred in this episode"}
    hits = sorted(hits, key=lambda e: e.step)
    pick = hits[0] if first else hits[-1]
    return {"matched": True, "question": question, "found": True, "rule": rule,
            "when_s": round(pick.t, 3), "when_step": pick.step, "event": pick.to_dict(),
            "count": len(hits), "all": [e.to_dict() for e in hits],
            "explanation": f"{pick.type} at t={pick.t:.2f}s (step {pick.step})"
                           + (f" — {pick.detail}" if pick.detail else "")}


def summarize_episode(events) -> dict:
    """A compact, human/LLM-legible timeline of an episode (the ~10^2-token temporal sentence). Cheap to store
    alongside the raw trace; this is what a caller passes to the LLM when it must author an out-of-vocabulary
    predicate, instead of the frames."""
    from collections import Counter
    counts = Counter(e.type for e in events)
    return {"n_events": len(events), "types": dict(counts),
            "timeline": [e.to_dict() for e in events],
            "first_fall_s": next((round(e.t, 3) for e in events if e.type == FELL), None),
            "grasped": any(e.type == GRASP_ATTACH for e in events),
            "grasp_failed": any(e.type == GRASP_DETACH for e in events),
            "reached_goal": any(e.type == GOAL_REACHED for e in events)}


# ----------------------------- product tool: ask_episode over a HELD robot's episode --------------------------

_EVENT_CACHE: dict = {}   # (robot_id, steps) -> (events, summary): compile the event log ONCE, reuse per query


def _run_and_extract(gene, *, steps: int = 1500):
    """Run one locomotion episode for ``gene`` (recording qpos) and extract its event log. Legged bodies use the
    deployable crawl rollout; other kinds return an empty log until their episode type is wired (honest)."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    m = mujoco.MjModel.from_xml_string(
        compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene)))
    try:
        from virturoid.services.morph_policy import crawl_gait_rollout
        r = crawl_gait_rollout(gene, steps=steps, record_qpos=True, frame_every=5)
        frames = r.get("qpos_frames") or []
    except Exception:  # noqa: BLE001 - non-legged / rollout failure -> no locomotion episode
        frames = []
    return extract_events(m, frames, dt=float(m.opt.timestep), frame_every=5)


def ask_episode_tool(args: dict) -> dict:
    """Agent/MCP tool: answer a temporal question about a HELD robot's episode ("when did it fall?", "when did
    the gripper fail?") by compiling ONE predicate query over the sim-emitted event log — no LLM call per frame.
    The event log is extracted once per (robot, horizon) and cached, so repeated questions are instant."""
    rid = args.get("robot_id")
    question = args.get("question") or args.get("prompt") or ""
    if not rid or not question:
        return {"error": "provide robot_id and question (e.g. 'when did it fall?')"}
    try:
        from virturoid.services import session_state as S
        gene = S.get_robot(rid)
    except Exception:  # noqa: BLE001
        gene = None
    if gene is None:
        return {"error": f"no held robot {rid}"}
    steps = int(args.get("steps", 1500))
    key = (rid, steps)
    if key not in _EVENT_CACHE or args.get("refresh"):
        events = _run_and_extract(gene, steps=steps)
        _EVENT_CACHE[key] = (events, summarize_episode(events))
    events, summary = _EVENT_CACHE[key]
    ans = ask_episode(events, question)
    ans["episode"] = {k: summary[k] for k in ("n_events", "types", "grasped", "grasp_failed", "reached_goal")}
    ans["method"] = "sim-emitted event log + predicate query (0 LLM calls; 0 per-frame VLM)"
    return ans


EPISODE_TOOLS = {
    "ask_episode": {
        "description": "Answer a TEMPORAL question about a held robot's episode — 'when did it fall?', 'when did "
                       "the gripper fail?', 'when did it first step?'. The simulator EMITS typed events "
                       "(contacts, grasp attach/detach, falls, goal-reach, joint saturation); the question "
                       "compiles to ONE predicate query over that log — no LLM call per frame. Returns the "
                       "matched moment (t, step, event) + the episode's event summary.",
        "heavy": True, "handler": ask_episode_tool,
        "parameters": {"type": "object", "required": ["robot_id", "question"], "properties": {
            "robot_id": {"type": "string"}, "question": {"type": "string"},
            "steps": {"type": "integer", "description": "episode horizon (default 1500)"},
            "refresh": {"type": "boolean", "description": "re-run the episode instead of using the cached log"}}},
    },
}
