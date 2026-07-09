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

# What the platform can actually compose + run end-to-end today (keep in sync with the composer +
# task_matched_eval). Anything outside these is reported as a gap, not silently mis-built.
SUPPORTED_CLASSES = ("manipulator", "mobile_base", "quadruped", "humanoid", "mobile_manipulator")
SUPPORTED_TASKS = ("pick_place_sort", "place_to_target", "grasp_lift", "navigation",
                   "spray_coverage", "locomotion")
_TASK_CLASS = {  # task family -> the robot class that performs it
    "pick_place_sort": "manipulator", "place_to_target": "manipulator", "grasp_lift": "manipulator",
    "spray_coverage": "manipulator", "navigation": "mobile_base", "locomotion": "quadruped",
}
# Physics / capability needs we do NOT have in the MuJoCo core (the honest frontier).
_UNSUPPORTED = {
    "cloth": "deformable cloth/fabric simulation (needs an Isaac/PhysX-class backend)",
    "fold": "deformable cloth/fabric simulation (needs an Isaac/PhysX-class backend)",
    "iron": "deformable cloth/fabric simulation (needs an Isaac/PhysX-class backend)",
    "fabric": "deformable cloth/fabric simulation (needs an Isaac/PhysX-class backend)",
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
    robot_class: str                       # nearest buildable class
    task_family: str                       # canonical, mapped into SUPPORTED_TASKS when possible
    environment: str = "tabletop"
    morphology: str = ""                   # short body description (LEGO recipe intent)
    objects: list = field(default_factory=list)
    action_verbs: list = field(default_factory=list)
    buildable: bool = True                 # can we compose AND run it end-to-end today?
    gaps: list = field(default_factory=list)   # specific, actionable capability gaps
    reasoning: str = ""
    source: str = "heuristic"              # "llm" | "heuristic"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "prompt", "robot_class", "task_family", "environment", "morphology", "objects",
            "action_verbs", "buildable", "gaps", "reasoning", "source")}


_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "robot_class": {"type": "string", "enum": list(SUPPORTED_CLASSES)},
        "task_family": {"type": "string"},
        "environment": {"type": "string"},
        "morphology": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "action_verbs": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": ["robot_class", "task_family", "morphology"],
}

_SYSTEM = (
    "You are Virturoid's build planner. Map a natural-language robot request to a structured plan over "
    "our building blocks. robot_class MUST be one of: manipulator, mobile_base, quadruped, humanoid — "
    "pick the NEAREST buildable class (e.g. a frog or hopper is closest to 'quadruped' legged; a rover "
    "is 'mobile_base'). task_family is the canonical task (e.g. pick_place_sort, place_to_target, "
    "grasp_lift, navigation, spray_coverage, locomotion, or a short new name if none fit). 'morphology' "
    "is a one-line body description. List the action verbs and target objects. Be concise and literal."
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
    "humanoid": ("humanoid", "bimanual", "two arms", "two-arm", "android", "torso", "human-like"),
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
    return BuildPlan(prompt=prompt, robot_class=robot_class, task_family=task, morphology=morph,
                     environment="maze" if "maze" in p else "tabletop",
                     action_verbs=[w for w in ("grasp", "lift", "sort", "place", "navigate", "walk",
                                   "hop", "run", "spray") if w in p],
                     reasoning="keyword-routed (no LLM backend)", source="heuristic")


def _assess(plan: BuildPlan) -> BuildPlan:
    """Decide feasibility from OUR capability registry — the LLM proposes, the registry disposes."""
    text = f"{plan.prompt} {plan.task_family} {plan.environment} {plan.morphology}".lower()
    gaps: list[str] = []
    for marker, why in _UNSUPPORTED.items():
        if marker in text and why not in gaps:
            gaps.append(why)
    if plan.robot_class not in SUPPORTED_CLASSES:
        gaps.append(f"unknown robot class '{plan.robot_class}'")
    if plan.task_family not in SUPPORTED_TASKS:
        gaps.append(f"task family '{plan.task_family}' has no evaluator/controller yet")
    # class/task mismatch (e.g. a quadruped asked to do pick_place) — a real integration gap
    need = _TASK_CLASS.get(plan.task_family)
    if need and need != plan.robot_class:
        gaps.append(f"'{plan.task_family}' is implemented for {need}, not {plan.robot_class} "
                    f"(cross-morphology task wiring missing)")
    plan.gaps = gaps
    plan.buildable = not gaps
    return plan


def plan_build(prompt: str, *, llm="auto") -> BuildPlan:
    """Plan a build from a free prompt. ``llm``: an object with ``complete_json`` (preferred), ``None``
    to force the heuristic, or ``"auto"`` to use the configured project LLM if one is available."""
    if llm == "auto":
        try:
            from virturoid.services.llm_client import get_llm
            llm = get_llm("planner")
        except Exception:  # noqa: BLE001
            llm = None
    if llm is not None:
        try:
            raw = llm.complete_json(_SYSTEM, f"Request: {prompt}", _PLAN_SCHEMA)
            rc = raw.get("robot_class") if raw.get("robot_class") in SUPPORTED_CLASSES else None
            if rc:
                plan = BuildPlan(
                    prompt=prompt, robot_class=rc,
                    task_family=str(raw.get("task_family") or "").strip() or "place_to_target",
                    environment=str(raw.get("environment") or "tabletop"),
                    morphology=str(raw.get("morphology") or ""),
                    objects=list(raw.get("objects") or []),
                    action_verbs=list(raw.get("action_verbs") or []),
                    reasoning=str(raw.get("reasoning") or ""), source="llm")
                return _assess(plan)
        except Exception:  # noqa: BLE001 - any LLM/parse failure falls back, never blocks
            pass
    return _assess(_heuristic_plan(prompt))
