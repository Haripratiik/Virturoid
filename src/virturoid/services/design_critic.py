"""Reasoned design improvement: diagnose WHY a robot body fails, then fix it on purpose.

This is the body half of the "robotics builder agent" the platform is for. When a task reveals
a limit *no controller can overcome* — the arm is too short, the shoulder actuator too weak —
you cannot train your way out of it; you have to change the morphology. The question is HOW:

- **Not randomly** (a genetic algorithm / CEM mutates blindly and re-simulates thousands of
  bodies — slow, and it never learns *why* a change helped). ``codesign_optimizer`` is that
  random-search baseline; this module is the deliberate alternative.
- **Not by a human** eyeballing link lengths.
- **Reasoned**: physics measures *what failed and by how much* in physical units (reach short by
  0.12 m; shoulder torque 1.8x over its hold limit); a reasoner (LLM, or a deterministic fallback)
  infers the *directed* fix from that diagnosis (short reach -> lengthen the upper arm; over-torque
  -> stronger actuator / lighter links); and physics then **verifies** the fix actually worked.
  That is the physics + LLM combo: physics is the ground truth and the measurement, the reasoner
  turns a measurement into a directed change instead of a guess.
- **Remembered across customers** (the flywheel): a proven (diagnosis -> fix) becomes a *lesson*
  in ``MemoryDB.lessons`` keyed by robot_class + failure mode. The next customer whose robot hits
  the same diagnosis retrieves the fix instantly instead of re-deriving it.

Diagnosis is analytic **statics** (no MuJoCo, deterministic, offline): for a manipulator the
dominant body-sizing criteria are reach (kinematic chain length vs the task's farthest point) and
joint torque (gravity hold-torque of the downstream links + payload). The dynamic simulator
(scripted controller or RL) is the *verifier* in the loop, used when available; the diagnosis
itself never needs it, so the whole reasoning loop runs on a laptop with no GPU and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from virturoid.schemas.gene import GeneSegment, RobotGene, amend_gene

G = 9.81

# Thresholds (physical, conservative). REACH_MARGIN: the fully-extended arm should clear the
# farthest target by this much. TORQUE_MARGIN: a joint whose required hold torque exceeds this
# fraction of its actuator is at risk (1.0 = literally cannot hold the pose statically).
REACH_MARGIN_M = 0.03
TORQUE_RISK_RATIO = 0.85
STRUCT_MIN_SF = 2.0


# --------------------------------------------------------------------------- physics (statics)
def chain_to_ee(gene: RobotGene) -> list:
    """Segments on the kinematic path root -> end-effector (the active manipulation chain)."""
    ee = gene.end_effector()
    if ee is None:
        return []
    chain, node, seen = [], ee, set()
    while node is not None and node.name not in seen:
        chain.append(node)
        seen.add(node.name)
        node = gene.segment(node.parent) if node.parent else None
    chain.reverse()
    return chain


def kinematic_reach(gene: RobotGene) -> float:
    """Max distance the ee can extend from the base mount = chain length past the welded root."""
    return round(sum(s.length_m for s in chain_to_ee(gene) if s.parent is not None), 5)


def joint_hold_torques(gene: RobotGene, payload_kg: float = 0.0) -> list[dict]:
    """Worst-case (horizontal) static gravity torque each revolute joint on the ee chain must hold.

    For each actuated revolute joint: torque = g * (Σ downstream link mass * lever arm to its COM
    + payload * lever arm to the ee tip), with lever arms measured along the extended chain. This
    is the textbook joint-sizing load for a manipulator and needs no simulation.
    """
    chain = chain_to_ee(gene)
    # cumulative start distance of each chain segment from the base, and its COM distance.
    start, com, dist = [], [], 0.0
    for s in chain:
        start.append(dist)
        com.append(dist + s.length_m / 2.0)
        dist += s.length_m
    tip = dist
    out = []
    for i, s in enumerate(chain):
        if s.joint_type != "revolute":
            continue
        torque = 0.0
        for j in range(i, len(chain)):  # this link + everything distal to the joint
            torque += chain[j].mass_kg * max(0.0, com[j] - start[i])
        torque += payload_kg * max(0.0, tip - start[i])
        required = round(G * torque, 4)
        cap = float(s.actuator_torque_nm or 0.0)
        out.append({
            "joint": s.name,
            "required_nm": required,
            "actuator_nm": round(cap, 4),
            "ratio": round(required / cap, 4) if cap > 0 else float("inf"),
        })
    return out


# --------------------------------------------------------------------------- reachability (M3 G3)
def _ik_distances(gene: RobotGene, points, ik_iters: int = 6, ik_candidates: int = 40):
    """Best achievable ee-to-point distance for each point, via the FK-IK planner. Needs MuJoCo.

    This is what made the push skill work (M2): the analytic ``kinematic_reach`` says how FAR the arm
    extends, but not whether it can FOLD to a low tabletop point — only IK reveals the true reachable
    set. Generalizes ``scripts/codesign_reach.py`` into a reusable platform capability.
    """
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.pick_place_controller import plan_joint_targets

    try:
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))
    except Exception:  # noqa: BLE001 - uncompilable gene -> nothing reachable
        return [float("inf")] * len(points)
    return [plan_joint_targets(mj, tuple(p), iterations=ik_iters, candidates=ik_candidates)[1] for p in points]


def reach_feasibility(gene: RobotGene, points, margin: float = 0.06) -> float:
    """Fraction of ``points`` (each (x,y,z)) the arm can reach within ``margin`` (i.e. can contact)."""
    if not points:
        return 0.0
    d = _ik_distances(gene, points)
    return round(sum(1 for v in d if v < margin) / len(points), 3)


def reachable_workspace(gene: RobotGene, xy_lo, xy_hi, z: float = 0.05, grid: int = 7,
                        margin: float = 0.06) -> dict:
    """Map where on the table the arm can actually contact (reach within ``margin``).

    Returns the reachable grid points + a bounding box + the feasibility fraction. The task runtime can
    use the bbox to spawn objects where the body can reach (auto task↔body matching — M2's lesson) and
    ``design_critic`` can co-design reach when the region is too small.
    """
    import numpy as np

    xs = np.linspace(xy_lo[0], xy_hi[0], grid)
    ys = np.linspace(xy_lo[1], xy_hi[1], grid)
    pts = [(float(x), float(y), float(z)) for x in xs for y in ys]
    dists = _ik_distances(gene, pts)
    reachable = [(p[0], p[1]) for p, dv in zip(pts, dists) if dv < margin]
    if reachable:
        rx = [p[0] for p in reachable]; ry = [p[1] for p in reachable]
        bbox = {"x": [round(min(rx), 4), round(max(rx), 4)], "y": [round(min(ry), 4), round(max(ry), 4)]}
    else:
        bbox = {"x": None, "y": None}
    return {"reachable": reachable, "bbox": bbox,
            "feasibility": round(len(reachable) / len(pts), 3), "n_points": len(pts), "margin": margin}


# --------------------------------------------------------------------------- diagnosis
@dataclass
class FailureMode:
    code: str                       # reach_limited | torque_limited | structural_weak | ...
    severity: float                 # 0..1, higher = more limiting
    root_cause: str                 # physical, human-readable
    evidence: dict = field(default_factory=dict)
    levers: list[str] = field(default_factory=list)   # operator names that address it, best first


# Which amendment operators address each failure mode, in preference order.
_LEVERS = {
    "reach_limited": ["lengthen_reach", "widen_joint_ranges"],
    "torque_limited": ["upgrade_actuators", "lighten_links"],
    "structural_weak": ["thicken_links", "lighten_links"],
}


def diagnose(gene: RobotGene, *, reach_required_m: float | None = None, payload_kg: float = 0.0,
             components=None, episode: dict | None = None) -> list[FailureMode]:
    """Rank the physical reasons this body can't do the task (most limiting first)."""
    modes: list[FailureMode] = []

    if reach_required_m is not None:
        reach = kinematic_reach(gene)
        deficit = reach_required_m - reach
        if deficit > -REACH_MARGIN_M:  # can't clear the farthest target with margin
            sev = min(1.0, max(0.05, (deficit + REACH_MARGIN_M) / max(reach_required_m, 1e-6)))
            modes.append(FailureMode(
                "reach_limited", round(sev, 3),
                f"reach {reach:.3f} m vs required {reach_required_m:.3f} m "
                f"(short by {max(0.0, deficit):.3f} m incl. {REACH_MARGIN_M:.2f} m margin)",
                {"reach_m": reach, "required_m": round(reach_required_m, 4), "deficit_m": round(deficit, 4)},
                list(_LEVERS["reach_limited"])))

    torques = joint_hold_torques(gene, payload_kg=payload_kg)
    over = [t for t in torques if t["ratio"] >= TORQUE_RISK_RATIO]
    if over:
        worst = max(over, key=lambda t: t["ratio"])
        sev = min(1.0, max(0.05, (worst["ratio"] - TORQUE_RISK_RATIO) / 1.0 + 0.2))
        ratio_txt = "inf" if worst["ratio"] == float("inf") else f"{worst['ratio']:.2f}x"
        modes.append(FailureMode(
            "torque_limited", round(sev, 3),
            f"joint {worst['joint']!r} needs {worst['required_nm']:.1f} Nm to hold its load but "
            f"has {worst['actuator_nm']:.1f} Nm ({ratio_txt} of capacity)"
            + (f" under {payload_kg:.2f} kg payload" if payload_kg else ""),
            {"joint": worst["joint"], "required_nm": worst["required_nm"],
             "actuator_nm": worst["actuator_nm"], "ratio": worst["ratio"], "payload_kg": payload_kg,
             "all_joints": torques},
            list(_LEVERS["torque_limited"])))

    if components is not None:
        try:
            from virturoid.services.structural_validation import validate_structure
            structural = validate_structure(gene, payload_kg=payload_kg)
            sf = structural.get("min_safety_factor")
            if sf is not None and sf < STRUCT_MIN_SF:
                sev = min(1.0, max(0.05, (STRUCT_MIN_SF - sf) / STRUCT_MIN_SF))
                modes.append(FailureMode(
                    "structural_weak", round(sev, 3),
                    f"min structural safety factor {sf:.2f} < {STRUCT_MIN_SF:.1f} (a link may yield under load)",
                    {"min_safety_factor": sf}, list(_LEVERS["structural_weak"])))
        except Exception:  # noqa: BLE001 - structural check is advisory; never break diagnosis
            pass

    # A scripted-episode failure_label reinforces the analytic signal (missed_grasp on a
    # reach-tight body is almost always reach; otherwise it's control, not body — left to RL).
    if episode and episode.get("failure_label") == "missed_grasp":
        for m in modes:
            if m.code == "reach_limited":
                m.evidence["episode_failure"] = "missed_grasp"
                m.severity = round(min(1.0, m.severity + 0.1), 3)

    modes.sort(key=lambda m: m.severity, reverse=True)
    return modes


# --------------------------------------------------------------------------- amendment operators
def _scale_overrides(gene, field_name, factor, *, include=None):
    """Build segment_overrides scaling ``field_name`` for the selected segments."""
    if include is None:
        include = lambda s: s.parent is not None and not s.is_end_effector  # noqa: E731
    ov = {}
    for s in gene.segments:
        if include(s) and getattr(s, field_name) is not None:
            ov[s.name] = {field_name: round(getattr(s, field_name) * factor, 5)}
    return ov


def lengthen_reach(gene: RobotGene, *, factor: float = 1.25, new_id=None, species=None) -> RobotGene:
    """Extend the actuated arm links so the ee can reach farther (the reach fix)."""
    chain = {s.name for s in chain_to_ee(gene)}
    inc = lambda s: s.name in chain and s.parent is not None and not s.is_end_effector and s.joint_type == "revolute"  # noqa: E731
    return _amend(gene, _scale_overrides(gene, "length_m", factor, include=inc), new_id, species, "lengthen_reach")


def upgrade_actuators(gene: RobotGene, *, factor: float = 1.5, new_id=None, species=None) -> RobotGene:
    """Step every actuated joint up to a stronger actuator (the torque fix)."""
    ov = {s.name: {"actuator_torque_nm": round((s.actuator_torque_nm or 10.0) * factor, 4)}
          for s in gene.actuated_joints()}
    return _amend(gene, ov, new_id, species, "upgrade_actuators")


def widen_joint_ranges(gene: RobotGene, *, delta: float = 0.35, new_id=None, species=None) -> RobotGene:
    """Open up joint limits so the arm can fold into more of its workspace."""
    ov = {}
    for s in gene.actuated_joints():
        if s.joint_lower is not None and s.joint_upper is not None:
            ov[s.name] = {"joint_lower": round(s.joint_lower - delta, 4),
                          "joint_upper": round(s.joint_upper + delta, 4)}
    return _amend(gene, ov, new_id, species, "widen_joint_ranges")


def lighten_links(gene: RobotGene, *, factor: float = 0.8, new_id=None, species=None) -> RobotGene:
    """Reduce link mass to cut the gravity load the joints must hold (the torque/structure fix)."""
    return _amend(gene, _scale_overrides(gene, "mass_kg", factor), new_id, species, "lighten_links")


def thicken_links(gene: RobotGene, *, factor: float = 1.3, new_id=None, species=None) -> RobotGene:
    """Increase link cross-section so it survives bending load (the structural fix)."""
    return _amend(gene, _scale_overrides(gene, "radius_m", factor), new_id, species, "thicken_links")


def _amend(gene, overrides, new_id, species, op):
    new_id = new_id or f"{gene.id}__{op}"
    species = species or (gene.species + f".{op}")
    child = amend_gene(gene, new_id=new_id, species=species, segment_overrides=overrides,
                       metadata={"amend_operator": op})
    issues = child.validate()
    if issues:
        raise ValueError(f"{op} produced an invalid gene: {'; '.join(issues)}")
    return child


# Operator registry: name -> (callable, default kwargs, the failure modes it addresses).
OPERATORS = {
    "lengthen_reach": (lengthen_reach, {"factor": 1.25}, {"reach_limited"}),
    "upgrade_actuators": (upgrade_actuators, {"factor": 1.5}, {"torque_limited"}),
    "widen_joint_ranges": (widen_joint_ranges, {"delta": 0.35}, {"reach_limited"}),
    "lighten_links": (lighten_links, {"factor": 0.8}, {"torque_limited", "structural_weak"}),
    "thicken_links": (thicken_links, {"factor": 1.3}, {"structural_weak"}),
}


def apply_operator(gene: RobotGene, operator: str, *, params: dict | None = None,
                   new_id=None, species=None) -> RobotGene:
    if operator not in OPERATORS:
        raise KeyError(f"unknown amendment operator {operator!r}; known: {sorted(OPERATORS)}")
    fn, defaults, _ = OPERATORS[operator]
    return fn(gene, **{**defaults, **(params or {})}, new_id=new_id, species=species)


def add_parallel_gripper(gene: RobotGene, *, span: float = 0.10, finger_len: float = 0.06,
                         finger_radius: float = 0.012, mass: float = 0.03, close: float = 0.045,
                         torque: float = 8.0, new_id=None, species=None) -> RobotGene:
    """Amend a gene with an actuated parallel-jaw gripper (M3 G1): two fingers on the current ee (the
    palm), each a prismatic closing joint, mounted side-by-side in y via ``mount_offset``. Enables a
    real friction grasp — MJX-JAX has no adhesion / ignores ``eq_active``, so a weld-grasp won't port.

    ``span`` is the rest center-to-center finger separation; the open inner gap is
    ``span - 2*finger_radius`` and must exceed the object's width to straddle it (the default 0.10
    clears the ~0.05 tabletop cube — a CPU geometry probe showed 0.06 was too narrow to enclose it).
    ``close`` caps each finger's travel short of the centerline so the jaws can't cross on a miss.
    """
    palm = gene.end_effector()
    if palm is None:
        raise ValueError("gene has no end-effector to mount a gripper on")

    def finger(name, y, ax):
        return GeneSegment(name=name, parent=palm.name, shape="box", length_m=finger_len,
                           radius_m=finger_radius, mass_kg=mass, joint_type="prismatic", joint_axis=ax,
                           joint_lower=0.0, joint_upper=close, mount_offset=(0.0, y, 0.0),
                           actuator_torque_nm=torque)

    fingers = [finger("finger_l", span / 2, (0.0, -1.0, 0.0)),    # closes toward center (-y)
               finger("finger_r", -span / 2, (0.0, 1.0, 0.0))]    # closes toward center (+y)
    child = amend_gene(gene, new_id=new_id or f"{gene.id}__gripper",
                       species=species or gene.species + ".gripper", add_segments=fingers,
                       end_effector_type="gripper", metadata={"amend_operator": "add_parallel_gripper"})
    issues = child.validate()
    if issues:
        raise ValueError(f"add_parallel_gripper produced an invalid gene: {'; '.join(issues)}")
    return child


# --------------------------------------------------------------------------- the reasoner
@dataclass
class Amendment:
    operator: str
    params: dict
    addresses: str                  # failure code this targets
    rationale: str
    source: str                     # "lesson" (flywheel reuse) | "llm" | "heuristic"


def propose_amendment(gene: RobotGene, modes: list[FailureMode], *, memory=None, llm=None,
                      blocked: set | None = None) -> Amendment | None:
    """Choose a directed fix for the top failure mode: flywheel lesson first, else reason.

    Order of preference, all grounded by the diagnosis:
      1. **Flywheel** — a proven lesson for (robot_class, failure mode) from a prior customer.
      2. **LLM** — pick operator + magnitude from the constrained menu, with a causal rationale.
      3. **Heuristic** — the diagnosis's preferred lever (deterministic, offline-safe).
    """
    blocked = blocked or set()
    for mode in modes:
        # 1) flywheel: has this class+failure been solved before?
        if memory is not None:
            lesson = memory.recall_lesson(gene.robot_class, mode.code)
            if lesson and lesson["operator"] in OPERATORS and lesson["operator"] not in blocked:
                return Amendment(lesson["operator"], dict(lesson.get("params") or {}), mode.code,
                                 f"reused proven fix for {mode.code} on {gene.robot_class} "
                                 f"(improved success by {lesson.get('improvement')})",
                                 "lesson")
        # 2) LLM reasoning over the constrained operator menu
        if llm is not None:
            amd = _llm_amendment(gene, mode, llm, blocked)
            if amd is not None:
                return amd
        # 3) deterministic heuristic: the diagnosis's first un-blocked lever
        for lever in mode.levers:
            if lever in OPERATORS and lever not in blocked:
                _, defaults, _ = OPERATORS[lever]
                return Amendment(lever, dict(defaults), mode.code,
                                 f"{mode.root_cause} -> {lever}", "heuristic")
    return None


_AMENDMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "operator": {"type": "string", "enum": list(OPERATORS)},
        "factor": {"type": "number"},
        "delta": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["operator", "rationale"],
}


def _llm_amendment(gene, mode: FailureMode, llm, blocked) -> Amendment | None:
    """Ask the reasoner to pick a fix from the menu. Validated; any miss falls back to heuristic."""
    allowed = [op for op in mode.levers if op in OPERATORS and op not in blocked] or \
              [op for op in OPERATORS if op not in blocked]
    system = (
        "You are a robot mechanical designer. Given a PHYSICS diagnosis of why a robot body fails "
        "a task, choose ONE amendment operator from the allowed list that most directly fixes the "
        "root cause, and a sensible magnitude. Reason causally (e.g. short reach -> lengthen arm; "
        "over-torque -> stronger actuator or lighter links). Do not invent operators."
    )
    user = (
        f"robot_class={gene.robot_class}; failure={mode.code}; root_cause={mode.root_cause}\n"
        f"evidence={mode.evidence}\nallowed_operators={allowed}\n"
        "Pick operator and magnitude (factor for scaling ops, delta for widen_joint_ranges)."
    )
    try:
        out = llm.complete_json(system, user, _AMENDMENT_SCHEMA)
    except Exception:  # noqa: BLE001 - the loop must survive a flaky/absent model
        return None
    op = out.get("operator")
    if op not in OPERATORS or op in blocked:
        return None
    _, defaults, _ = OPERATORS[op]
    params = dict(defaults)
    for k in ("factor", "delta"):
        if k in defaults and isinstance(out.get(k), (int, float)):
            params[k] = float(out[k])
    return Amendment(op, params, mode.code, out.get("rationale") or f"LLM fix for {mode.code}", "llm")


# --------------------------------------------------------------------------- the loop
def reach_required_from_scenes(scenes, base_height_m: float = 0.165) -> float:
    """Farthest manipulable distance from the shoulder across FIXED task scenes.

    The reach requirement is set by the *customer's task* (where the objects are), not by the
    current — possibly too-short — robot. Hold these scenes fixed across candidates so the loop
    amends the body to meet the requirement instead of shrinking the task to fit a weak body.
    """
    import math
    farthest = 0.0
    for scene in scenes:
        for obj in getattr(scene, "objects", []):
            if getattr(obj, "object_type", None) in {"cube", "box"}:
                x, y, z = obj.pose_xyz_rpy[0], obj.pose_xyz_rpy[1], obj.pose_xyz_rpy[2]
                farthest = max(farthest, math.hypot(math.hypot(x, y), z - base_height_m))
    return round(farthest, 4)


def task_sim_eval(spec, scenes, params: dict | None = None):
    """Build a ``sim_eval(gene) -> success_rate`` that runs the scripted controller in real MuJoCo.

    This is the physics *verifier* the loop uses when a simulator is present (MuJoCo on the laptop,
    or the RL policy on the GPU as a drop-in). Scenes are passed in fixed so candidates compare
    fairly. MuJoCo is imported lazily inside the closure, so importing this module stays free.
    """
    def _eval(gene: RobotGene) -> float:
        from virturoid.services.task_runtime import evaluate_gene_on_task
        return float(evaluate_gene_on_task(gene, spec, scenes, params=params)["success_rate"])
    return _eval


@dataclass
class ImprovementTrace:
    baseline_success: float
    final_success: float
    final_gene: RobotGene
    rounds: list[dict] = field(default_factory=list)
    accepted: int = 0


def improve_design(gene: RobotGene, sim_eval, *, reach_required_m: float | None = None,
                   payload_kg: float = 0.0, components=None, memory=None, llm=None,
                   max_rounds: int = 4, target_success: float = 0.85,
                   task_type: str | None = None) -> ImprovementTrace:
    """Directed, verified, remembered body improvement (NOT random search).

    Each round: diagnose the current body -> propose a directed fix (flywheel/LLM/heuristic) ->
    apply it -> RE-SIMULATE to verify -> keep the change only if measured success rose, and record
    the proven (failure -> fix) as a reusable lesson. ``sim_eval(gene) -> success_rate`` is the
    physics ground truth (scripted controller or RL); it is the only piece that needs a simulator.
    """
    cur = gene
    cur_score = float(sim_eval(cur))
    trace = ImprovementTrace(round(cur_score, 4), round(cur_score, 4), cur)
    blocked: set = set()

    for r in range(max_rounds):
        if cur_score >= target_success:
            break
        modes = diagnose(cur, reach_required_m=reach_required_m, payload_kg=payload_kg, components=components)
        if not modes:
            break
        amd = propose_amendment(cur, modes, memory=memory, llm=llm, blocked=blocked)
        if amd is None:
            break
        try:
            cand = apply_operator(cur, amd.operator, params=amd.params,
                                  new_id=f"{gene.id}__r{r}_{amd.operator}")
        except (ValueError, KeyError):
            blocked.add(amd.operator)
            continue
        cand_score = float(sim_eval(cand))
        delta = round(cand_score - cur_score, 4)
        accepted = cand_score > cur_score + 1e-9
        trace.rounds.append({
            "round": r, "diagnosis": modes[0].code, "root_cause": modes[0].root_cause,
            "operator": amd.operator, "params": amd.params, "source": amd.source,
            "rationale": amd.rationale, "success_before": round(cur_score, 4),
            "success_after": round(cand_score, 4), "delta": delta, "accepted": accepted,
        })
        if accepted:
            trace.accepted += 1
            if memory is not None:
                memory.record_lesson(gene.robot_class, modes[0].code, amd.operator,
                                     params=amd.params, root_cause=modes[0].root_cause,
                                     improvement=delta, task_type=task_type, source_gene=gene.id)
            cur, cur_score = cand, cand_score
            blocked.clear()  # a fix that worked may expose a new bottleneck; reconsider all levers
        else:
            blocked.add(amd.operator)  # didn't help this body; try the next lever

    trace.final_success = round(cur_score, 4)
    trace.final_gene = cur
    return trace
