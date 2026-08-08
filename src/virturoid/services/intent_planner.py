"""LLM build planner: turn ANY natural-language request into a structured, capability-checked plan.

This is the "brain" the startup plan calls for (§31.1 constrained NL generation, §65.1 user intent): the
composer's keyword routing mis-reads anything outside its word lists (e.g. "a frog-like robot to run a
maze" → a tabletop arm). The planner instead maps a free prompt to a `BuildPlan` — robot class,
canonical task family, environment, a morphology hint over our building blocks — AND honestly checks it
against what we can actually *build and run* today, flagging the gaps (unknown morphology, unsupported
task family, or a physics need MuJoCo can't meet) instead of silently producing the wrong robot.

It uses the project LLM (`llm_client.get_llm`) when available and falls back to a capability-aware
heuristic when not — so it always returns a grounded, validated plan. Feasibility is decided by OUR
capability registry, never the LLM's optimism: the LLM proposes, the registry disposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# These are execution families, not the product taxonomy.  A user can request and
# name any concept; the planner maps its observable body/task cues to one of these
# currently implemented execution routes.  The concept itself is retained in memory
# and later promoted from candidate -> evaluated -> verified by real build evidence.
EXECUTION_FAMILIES = ("manipulator", "mobile_base", "quadruped", "humanoid", "mobile_manipulator")
EVALUATED_TASKS = ("pick_place_sort", "place_to_target", "grasp_lift", "navigation",
                   "spray_coverage", "locomotion")
_TASK_EXECUTION_FAMILY = {  # task family -> the route that has a current evaluator/controller
    "pick_place_sort": "manipulator", "place_to_target": "manipulator", "grasp_lift": "manipulator",
    "spray_coverage": "manipulator", "navigation": "mobile_base", "locomotion": "quadruped",
}
# Physics / capability needs we do NOT have in the MuJoCo core (the honest frontier).
_UNSUPPORTED = {
    # MEASURED: MuJoCo 3.9 flexcomp DOES simulate the deformable fabric (an 8x8 cloth grid drapes correctly), so
    # it is NOT tier-blocked. The open frontier is robust grasp-and-FOLD: a rigid gripper pinching cloth via
    # contact is finicky (a test lift only peeled a corner) — a reliable connect-constraint grasp + a fold task
    # + verdict is the scoped next capability, not an Isaac dependency.
    "cloth": "deformable cloth grasp-and-fold (flexcomp simulates the fabric; robust deformable GRASP is the frontier)",
    "fold": "deformable cloth grasp-and-fold (flexcomp simulates the fabric; robust deformable GRASP is the frontier)",
    "iron": "deformable cloth grasp-and-fold (flexcomp simulates the fabric; robust deformable GRASP is the frontier)",
    "fabric": "deformable cloth grasp-and-fold (flexcomp simulates the fabric; robust deformable GRASP is the frontier)",
    "fluid": "fluid/liquid simulation (needs an Isaac/PhysX-class backend)",
    "liquid": "fluid/liquid simulation (needs an Isaac/PhysX-class backend)",
    "pour": "fluid/liquid simulation (needs an Isaac/PhysX-class backend)",
    # 'maze' is SUPPORTED now: services/maze.py (A* over a generated maze) + the solve_maze skill drive a mobile
    # base through it to the goal (run_task 'solve the maze' -> success). It was a stale under-claim to gap it.
    "stairs": "stair/terrain traversal (no rough-terrain locomotion task yet)",
    "climb": "climbing (no climbing task family yet)",
    "in-hand": "in-hand dexterous manipulation (no dexterous-manipulation task yet)",
    # 'drone'/'flies'/'quadcopter'/'aerial'/... are SUPPORTED now: services/aerial.py composes a quadcopter
    # (hub + 4 rotors) and ai_native_tools._honest_fly flies it with real rotor THRUST forces + a geometric
    # flight controller (measured: reaches arbitrary targets, 0.00 m error). It was a stale gap to list them.
}


@dataclass
class BuildPlan:
    prompt: str
    robot_class: str                       # execution route; not a user-facing taxonomy label
    task_family: str                       # canonical, mapped into EVALUATED_TASKS when possible
    concept: str = ""                       # open-ended user concept, e.g. "trilobite-like rover"
    concept_aliases: list[str] = field(default_factory=list)  # LLM-proposed synonyms for safe exact recall
    environment: str = "tabletop"
    morphology: str = ""                   # short body description (LEGO recipe intent)
    objects: list = field(default_factory=list)
    action_verbs: list = field(default_factory=list)
    buildable: bool = True                 # can we compose AND run it end-to-end today?
    gaps: list = field(default_factory=list)   # specific, actionable capability gaps
    reasoning: str = ""
    source: str = "heuristic"              # "llm" | "heuristic"
    # An arbitrary noun is not a valid reason to quietly choose a default body.
    # A novel concept with enough body/task evidence can be routed and learned;
    # one with no usable evidence is recorded as a candidate then clarified.
    routing_confidence: str = "explicit"    # explicit | task_inferred | novel | uncertain | llm_unavailable

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "prompt", "robot_class", "task_family", "concept", "concept_aliases", "environment", "morphology", "objects",
            "action_verbs", "buildable", "gaps", "reasoning", "source", "routing_confidence")}


_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "robot_class": {"type": "string"},
        "task_family": {"type": "string"},
        "concept": {"type": "string"},
        "concept_aliases": {"type": "array", "items": {"type": "string"}},
        "environment": {"type": "string"},
        "morphology": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "action_verbs": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "routing_confidence": {"type": "string", "enum": ["explicit", "task_inferred", "novel", "uncertain"]},
    },
    "required": ["robot_class", "task_family", "morphology"],
}

_SYSTEM = (
    "You are Virturoid's build planner. Map a natural-language robot request to a structured plan over "
    "our building blocks. robot_class is an EXECUTION ROUTE, not the name of the requested robot: choose "
    "the nearest route among manipulator, mobile_base, quadruped, humanoid, or mobile_manipulator (for "
    "example, a frog or hopper is closest to 'quadruped' legged; a rover "
    "is 'mobile_base'). task_family is the canonical task (e.g. pick_place_sort, place_to_target, "
    "grasp_lift, navigation, spray_coverage, locomotion, or a short new name if none fit). 'morphology' "
    "is a one-line body description. 'concept' is the user's open-ended name for this kind of robot; preserve "
    "a new name rather than forcing it into a known taxonomy. Provide up to five short concept_aliases only when "
    "they truly mean the same body concept. List the action verbs and target objects. Set routing_confidence to "
    "'explicit' when the request names a body class, 'task_inferred' when a supported task implies one, or "
    "'uncertain' when the request does neither. Be concise and literal."
)

# heuristic keyword routing (capability-aware) — far broader than the composer's lists.
_CLASS_WORDS = {
    "quadruped": ("frog", "amphibian", "hop", "jump", "leg", "legged", "quadruped", "dog", "gait",
                  "crawl", "walk", "trot", "march", "stride", "gallop", "amble", "stroll", "traverse",
                  "locomot", "spider", "arachnid", "tarantula", "insect", "lizard", "four-leg",
                  "biped", "hexapod", "octopod", "crab", "crustacean", "mantis", "gecko", "salamander",
                  "newt", "scorpion", "centipede", "beetle", "ant", "turtle", "snake", "serpent",
                  "creature", "animal", "octopus",
                  # common animal nouns so a bare 'a giraffe robot' routes to a legged walker, not a default arm
                  "cat", "horse", "giraffe", "bird", "elephant", "cow", "goat", "deer", "pony", "donkey",
                  "sheep", "pig", "rabbit", "fox", "wolf", "bear", "tiger", "lion", "cheetah", "leopard",
                  "camel", "kangaroo", "ostrich", "penguin", "chicken", "duck", "mammal", "reptile",
                  "dinosaur", "raptor", "axolotl", "rhino", "hippo", "zebra", "antelope", "gazelle",
                  "panther", "puma", "lynx", "mule", "llama", "alpaca", "moose", "elk", "bison", "goose"),
    "mobile_base": ("wheel", "rover", "drive ", "drive to", "mobile", "navigate", "deliver", "patrol",
                    "maze", "waypoint", "destination", "go to", "head to", "roomba", "agv", "trolley",
                    "vehicle", "car-like", "wheeled",
                    # G4: floor-cleaning nouns — 'a robot vacuum' previously matched NOTHING and became a
                    # tabletop sorting ARM (the worst e2e intent failure)
                    "vacuum", "mop", "sweep", "sweeper", "floor-clean", "floor clean"),
    # "torso" is NOT here: it is a body PART every legged animal, humanoid and mobile manipulator has, not a
    # class noun -- and ``humanoid`` is tried FIRST below, so one part word outranked every quadruped cue in
    # the prompt. Measured 2026-08-08: "a four-legged walking robot with a slender torso", "a quadruped robot
    # with a long torso", "a robot dog with a slender torso" and "a hexapod robot with a broad torso" ALL
    # planned humanoid and built the 2-legged humanoid recipe; so did "a torso-mounted arm". This turned
    # load-bearing when #285 taught ``body_proportions`` to honour "a slender torso"/"a long torso" as real
    # proportion asks -- the exact phrases the composer had just learned to build were the ones that lost
    # their body class on the way in. The genuine humanoid phrasings below still match on their own.
    "humanoid": ("humanoid", "bimanual", "two arms", "two-arm", "android", "human-like"),
    "manipulator": ("arm", "grasp", "grip", "pick", "place", "sort", "manipulat", "assemble", "weld",
                    "spray", "paint", "lift", "stack", "insert"),
}
_TASK_WORDS = {
    "spray_coverage": ("spray", "paint", "coat", "weld", "varnish", "lacquer", "glaze", "polish"),
    "pick_place_sort": ("sort", "bin", "into matching"),
    "grasp_lift": ("grasp", "lift", "pick up", "hold"),
    "navigation": ("navigate", "maze", "goal", "through", "deliver", "patrol", "reach", "drive to"),
    "locomotion": ("walk", "hop", "run", "gait", "trot", "crawl"),
    "place_to_target": ("place", "move", "put", "transport", "assemble", "stack"),
}
_CLARIFICATION_GAP = (
    "No recognizable robot body plan or task was found; clarify the body (for example wheels, leg count, arm, "
    "or humanoid) and intended task before building."
)


_CONCEPT_STOP_WORDS = {
    "a", "an", "the", "build", "create", "design", "make", "me", "robot", "new", "custom", "autonomous",
}


def _concept_from_prompt(prompt: str) -> str:
    """Extract a stable, open-ended label without treating it as an execution route.

    This deliberately keeps only a small amount of grammar rather than an
    ever-growing catalogue of robot names.  ``a trilobite-like robot`` becomes
    ``trilobite-like``; unknown labels are useful memory keys even when the
    system must still ask for missing morphology or a task.
    """
    text = " ".join(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (prompt or "").lower()))
    match = re.search(r"\b(?:a|an|the|build|create|design)\s+(.{1,80}?)\s+robot\b", text)
    if not match:
        return ""
    words = [w for w in match.group(1).split() if w not in _CONCEPT_STOP_WORDS]
    # Keep the key compact and deterministic.  It is a recall label, not a
    # claim that the phrase is a recognised scientific category.
    return " ".join(words[:5]).strip()


def _clean_concept_aliases(raw, concept: str) -> list[str]:
    """Bound untrusted LLM aliases; this validates labels but never invents them."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    canonical = " ".join((concept or "").lower().split())
    for value in raw[:5]:
        alias = " ".join(str(value or "").lower().split())
        if 2 <= len(alias) <= 80 and alias != canonical and alias not in out:
            out.append(alias)
    return out


def _kw(word: str, text: str) -> bool:
    # leading WORD-BOUNDARY match (not bare substring), so 'leg'/'ant'/'arm' no longer fire on
    # 'elegant'/'important'/'alarm' while stems still match ('walk'->'walking', 'manipulat'->'manipulate').
    return re.search(rf"\b{re.escape(word)}", text) is not None


def _heuristic_plan(prompt: str) -> BuildPlan:
    p = (prompt or "").lower()
    # robot class: first matching family by priority (legged/mobile/humanoid before generic manipulator)
    robot_class = None
    for cls in ("humanoid", "quadruped", "mobile_base", "manipulator"):
        if any(_kw(w, p) for w in _CLASS_WORDS[cls]):
            robot_class = cls
            break
    class_cue_found = robot_class is not None
    # G4 composite: an explicitly MOBILE platform asked to grasp/carry is a mobile manipulator — never silently
    # drop either half (the e2e test's 'mobile robot that picks boxes and carries them' built a FIXED arm).
    if robot_class == "mobile_base" and any(_kw(w, p) for w in _CLASS_WORDS["manipulator"]):
        robot_class = "mobile_manipulator"
    # task family: prefer the one consistent with the class, else first keyword match
    task = None
    for fam, words in _TASK_WORDS.items():
        if any(_kw(w, p) for w in words):
            task = fam
            break
    task_cue_found = task is not None
    if robot_class is None:
        # nothing in the class lists matched. A prompt with a manipulation/navigation TASK verb is an arm/rover;
        # otherwise a bare noun phrase ('a giraffe robot', 'an axolotl') is most likely a LEGGED creature, NOT a
        # default tabletop arm — this was the offline failure where every un-keyworded creature became an arm.
        if task in ("spray_coverage", "pick_place_sort", "grasp_lift", "place_to_target"):
            robot_class = "manipulator"
        elif task == "navigation":
            robot_class = "mobile_base"
        else:
            robot_class = "quadruped"
    if task is None:
        task = {"manipulator": "place_to_target", "mobile_base": "navigation",
                "quadruped": "locomotion", "humanoid": "place_to_target",
                "mobile_manipulator": "pick_place_sort"}[robot_class]
    morph = {"manipulator": "serial arm + gripper", "mobile_base": "wheeled rover",
             "quadruped": "legged walker", "humanoid": "torso + two arms",
             "mobile_manipulator": "wheeled chassis + grasp arm"}[robot_class]
    if "frog" in p or "hop" in p:
        morph = "legged hopper (frog-like) — nearest buildable: 4-legged walker"
    concept = _concept_from_prompt(prompt)
    routing_confidence = "explicit" if class_cue_found else (
        "task_inferred" if task_cue_found else ("novel" if concept else "uncertain"))
    initial_gaps = []
    if routing_confidence in {"novel", "uncertain"}:
        initial_gaps.append(_CLARIFICATION_GAP)
    return BuildPlan(prompt=prompt, robot_class=robot_class, task_family=task, concept=concept, morphology=morph,
                     environment="maze" if "maze" in p else "tabletop",
                     action_verbs=[w for w in ("grasp", "lift", "sort", "place", "navigate", "walk",
                                   "hop", "run", "spray") if w in p],
                     gaps=initial_gaps, routing_confidence=routing_confidence,
                     reasoning="keyword-routed (no LLM backend)", source="heuristic")


def _assess(plan: BuildPlan) -> BuildPlan:
    """Decide feasibility from OUR capability registry — the LLM proposes, the registry disposes."""
    text = f"{plan.prompt} {plan.task_family} {plan.environment} {plan.morphology}".lower()
    # Preserve planner-reported ambiguity while adding capability gaps below.
    gaps: list[str] = list(plan.gaps)
    if plan.routing_confidence in {"novel", "uncertain"}:
        gaps.append(_CLARIFICATION_GAP)
    for marker, why in _UNSUPPORTED.items():
        if marker in text and why not in gaps:
            gaps.append(why)
    if plan.robot_class not in EXECUTION_FAMILIES:
        gaps.append(f"no available execution route for '{plan.robot_class}'")
    if plan.task_family not in EVALUATED_TASKS:
        gaps.append(f"task family '{plan.task_family}' has no evaluator/controller yet")
    # class/task mismatch (e.g. a quadruped asked to do pick_place) — a real integration gap
    need = _TASK_EXECUTION_FAMILY.get(plan.task_family)
    if need and need != plan.robot_class:
        gaps.append(f"'{plan.task_family}' is implemented for {need}, not {plan.robot_class} "
                    f"(cross-morphology task wiring missing)")
    plan.gaps = list(dict.fromkeys(gaps))
    plan.buildable = not gaps
    return plan


def _llm_unavailable_plan(prompt: str, reason: str) -> BuildPlan:
    """Fail closed for production AI-first paths rather than inventing a template body."""
    return BuildPlan(
        prompt=prompt,
        robot_class="unrouted",
        task_family="unrouted",
        concept="",
        morphology="",
        buildable=False,
        gaps=[reason],
        reasoning="LLM-first planning required; no heuristic route was substituted.",
        source="llm_unavailable",
        routing_confidence="llm_unavailable",
    )


def _plan_build_once(prompt: str, *, llm="auto", require_llm: bool = False) -> BuildPlan:
    """Plan a build from a free prompt.

    ``require_llm=True`` is the production AI-first contract: a missing or
    malformed model response produces an explicit non-buildable plan instead of
    a keyword/template substitute.  The heuristic remains available only for
    tests, offline development, and explicit compatibility callers.
    """
    if llm == "auto":
        try:
            from virturoid.services.llm_client import get_llm
            llm = get_llm("planner")
        except Exception:  # noqa: BLE001
            llm = None
    if llm is not None:
        try:
            raw = llm.complete_json(_SYSTEM, f"Request: {prompt}", _PLAN_SCHEMA) or {}
            requested_route = str(raw.get("robot_class") or "").strip().lower()
            fallback = _heuristic_plan(prompt) if not require_llm else None
            # A model may name a genuinely novel kind of robot.  Preserve that name
            # as the concept, but select an execution route only from evidence in the
            # prompt (or an explicit executable route) — never let an arbitrary
            # label silently become a pretend simulator backend.
            rc = requested_route if requested_route in EXECUTION_FAMILIES else (
                fallback.robot_class if fallback is not None else "")
            concept = str(raw.get("concept") or "").strip() or (
                requested_route if requested_route and requested_route not in EXECUTION_FAMILIES and not require_llm
                else (fallback.concept if fallback is not None else ""))
            concept_aliases = _clean_concept_aliases(raw.get("concept_aliases"), concept)
            task_family = str(raw.get("task_family") or "").strip() or (
                "place_to_target" if not require_llm else "")
            # Ignore structurally empty model output.  A valid open-world proposal
            # supplies either a route, a concept label, or a morphology description;
            # otherwise the deterministic heuristic remains the source of truth.
            if rc and task_family and (requested_route or str(raw.get("concept") or "").strip()
                       or str(raw.get("morphology") or "").strip()):
                raw_confidence = str(raw.get("routing_confidence") or "task_inferred")
                if raw_confidence not in {"explicit", "task_inferred", "novel", "uncertain"}:
                    raw_confidence = "task_inferred"
                if requested_route and requested_route not in EXECUTION_FAMILIES and fallback is not None:
                    raw_confidence = "novel" if fallback.routing_confidence == "uncertain" else fallback.routing_confidence
                plan = BuildPlan(
                    prompt=prompt, robot_class=rc, concept=concept, concept_aliases=concept_aliases,
                    task_family=task_family,
                    environment=str(raw.get("environment") or "tabletop"),
                    morphology=str(raw.get("morphology") or ""),
                    objects=list(raw.get("objects") or []),
                    action_verbs=list(raw.get("action_verbs") or []),
                    reasoning=str(raw.get("reasoning") or ""), source="llm",
                    routing_confidence=raw_confidence)
                return _assess(plan)
        except Exception:  # noqa: BLE001 - any LLM/parse failure falls back, never blocks
            if require_llm:
                return _llm_unavailable_plan(prompt, "LLM planner response was unavailable or invalid; no robot was generated.")
    elif require_llm:
        return _llm_unavailable_plan(prompt, "LLM planner is not configured or reachable; no robot was generated.")
    if require_llm:
        return _llm_unavailable_plan(prompt, "LLM planner did not produce a usable grounded plan; no robot was generated.")
    return _assess(_heuristic_plan(prompt))


def plan_build(prompt: str, *, llm="auto", require_llm: bool = False) -> BuildPlan:
    """Plan a build, offering one grounded LLM self-repair in strict mode.

    The repair does not infer a route in code. The capability layer only reports
    why the first proposal cannot execute; the model makes the revised semantic
    decision. Ambiguous/novel requests retain their clarification path rather than
    being pushed toward an arbitrary known route.
    """
    if not require_llm:
        return _plan_build_once(prompt, llm=llm, require_llm=False)

    if llm == "auto":
        try:
            from virturoid.services.llm_client import get_llm
            llm = get_llm("planner")
        except Exception:  # noqa: BLE001 - the one-shot helper will return fail-closed context
            llm = None
    first = _plan_build_once(prompt, llm=llm, require_llm=True)
    if (first.source != "llm" or first.buildable
            or first.routing_confidence in {"novel", "uncertain"}):
        return first

    feedback = (
        "\nYour previous plan cannot execute in the currently grounded capability layer: "
        + "; ".join(first.gaps[:4])
        + ". Preserve the user's open-ended concept, but return a corrected executable route and task."
    )

    class _GroundedRetryLLM:
        def complete_json(self, system, user, schema, max_tokens=2048, reasoning_effort=None):
            return llm.complete_json(
                system, user + feedback, schema,
                max_tokens=max_tokens, reasoning_effort=reasoning_effort,
            )

    retry_llm = _GroundedRetryLLM()
    retry_llm.name = getattr(llm, "name", "llm")
    return _plan_build_once(prompt, llm=retry_llm, require_llm=True)
