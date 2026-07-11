"""The versioned Design-Bench prompt battery (master_plan_v6 §8.1 / kickoff #3).

Strict LLM-first means the model's design quality IS the product — so it must be measured on a FIXED, versioned
set of prompts, not ad-hoc. This module owns that battery. Two deliberate design choices, both from the research:

  * **Both phrasing styles for the same concept** — an *appearance* phrasing ("a sleek greyhound built to run")
    and a *construction-order* phrasing ("a quadruped: a slim torso with four 3-joint legs"). Text2CAD-style
    harnesses measured a real style confound; carrying both per concept lets Design-Bench separate "the model
    understands the animal" from "the model can follow an explicit part list".
  * **Machine-checkable constraints per prompt** — the objectively parseable facts the prompt asserts (leg count,
    DOF, wheel count, a size band). These feed the deterministic *spec-faithfulness* metric (§8.1.9): did the
    compiled body actually honour what the prompt literally said?

The battery is DISJOINT from ``heldout_set`` (the compounding-proof eval set) — Design-Bench measures the design
funnel; the held-out set measures transfer compounding. Nothing here designs a robot; it only poses the prompts.
"""
from __future__ import annotations

BATTERY_VERSION = "v1"

# Each concept appears twice — once per phrasing style — sharing a ``concept`` id so the two can be paired.
# constraints (all optional, machine-checkable against the compiled gene):
#   legs:int  arms:int  wheels:int  dof:int / dof_min:int / dof_max:int  kind:str  size_max_m:float
_BATTERY: tuple[dict, ...] = (
    # ---------------- animals (legged) ----------------
    {"concept": "lynx", "family": "animal", "phrasing": "appearance",
     "prompt": "a lean, agile lynx that prowls on four legs",
     "constraints": {"legs": 4, "kind": "legged"}},
    {"concept": "lynx", "family": "animal", "phrasing": "construction",
     "prompt": "a quadruped: a slender torso with four legs, each leg three actuated joints plus a foot",
     "constraints": {"legs": 4, "kind": "legged"}},
    {"concept": "elephant", "family": "animal", "phrasing": "appearance",
     "prompt": "a massive elephant with thick pillar legs and a long trunk",
     "constraints": {"legs": 4, "kind": "legged"}},
    {"concept": "elephant", "family": "animal", "phrasing": "construction",
     "prompt": "a large quadruped with four short thick legs and a long front-mounted trunk appendage",
     "constraints": {"legs": 4, "kind": "legged"}},
    {"concept": "gecko", "family": "animal", "phrasing": "appearance",
     "prompt": "a small gecko with a wide splayed stance close to the ground",
     "constraints": {"legs": 4, "kind": "legged"}},
    {"concept": "gecko", "family": "animal", "phrasing": "construction",
     "prompt": "a low quadruped with four short legs splayed wide to the sides and a long tail",
     "constraints": {"legs": 4, "kind": "legged"}},
    # ---------------- vehicles (mobile) ----------------
    {"concept": "rover", "family": "vehicle", "phrasing": "appearance",
     "prompt": "a six-wheeled planetary rover with a flat instrument deck",
     "constraints": {"wheels": 6, "kind": "mobile"}},
    {"concept": "rover", "family": "vehicle", "phrasing": "construction",
     "prompt": "a mobile base: a flat rectangular deck chassis with three wheel pairs at the corners",
     "constraints": {"wheels": 6, "kind": "mobile"}},
    {"concept": "cart", "family": "vehicle", "phrasing": "appearance",
     "prompt": "a small four-wheeled delivery cart for indoor hallways",
     "constraints": {"wheels": 4, "kind": "mobile"}},
    {"concept": "cart", "family": "vehicle", "phrasing": "construction",
     "prompt": "a mobile base with a low deck and two wheel pairs, front and rear",
     "constraints": {"wheels": 4, "kind": "mobile"}},
    # ---------------- arms (manipulator) ----------------
    {"concept": "welder", "family": "arm", "phrasing": "appearance",
     "prompt": "a benchtop welding arm that reaches across a work table",
     "constraints": {"kind": "manipulator", "dof_min": 4}},
    {"concept": "welder", "family": "arm", "phrasing": "construction",
     "prompt": "a manipulator: a fixed base with five revolute joints in series ending in a gripper",
     "constraints": {"kind": "manipulator", "dof": 5}},
    {"concept": "palletizer", "family": "arm", "phrasing": "appearance",
     "prompt": "a strong six-axis industrial arm for stacking boxes onto pallets",
     "constraints": {"kind": "manipulator", "dof_min": 5}},
    {"concept": "palletizer", "family": "arm", "phrasing": "construction",
     "prompt": "a manipulator with six revolute joints and a parallel-jaw gripper at the end",
     "constraints": {"kind": "manipulator", "dof": 6}},
    # ---------------- hybrids / composites ----------------
    {"concept": "mobile_manip", "family": "hybrid", "phrasing": "appearance",
     "prompt": "a wheeled robot that carries a small arm to pick things up as it drives",
     "constraints": {"kind": "mobile", "arms": 1}},
    {"concept": "mobile_manip", "family": "hybrid", "phrasing": "construction",
     "prompt": "a mobile base on four wheels with a three-joint manipulator arm mounted on the deck",
     "constraints": {"wheels": 4, "arms": 1}},
    # ---------------- novel / open-vocab (the open-world stress) ----------------
    {"concept": "starfish", "family": "novel", "phrasing": "appearance",
     "prompt": "a starfish-like crawler with five arms radiating from a central disc",
     "constraints": {"legs": 5, "kind": "legged"}},
    {"concept": "starfish", "family": "novel", "phrasing": "construction",
     "prompt": "a radially symmetric body: a central disc with five identical actuated limbs spaced evenly",
     "constraints": {"legs": 5, "kind": "legged"}},
    {"concept": "inchworm", "family": "novel", "phrasing": "appearance",
     "prompt": "an inchworm robot that arches and extends to creep forward",
     "constraints": {"kind": "legged"}},
    {"concept": "inchworm", "family": "novel", "phrasing": "construction",
     "prompt": "a serial chain of segments connected by revolute joints that flexes to move",
     "constraints": {"kind": "legged"}},
)


def battery() -> list[dict]:
    """The full versioned battery (list of prompt records). Order is stable across runs."""
    return [dict(r) for r in _BATTERY]


def families() -> list[str]:
    """The distinct concept families in the battery."""
    seen: list[str] = []
    for r in _BATTERY:
        if r["family"] not in seen:
            seen.append(r["family"])
    return seen


def by_family() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in _BATTERY:
        out.setdefault(r["family"], []).append(dict(r))
    return out


def prompt_id(record: dict) -> str:
    """A stable id for a battery record — ``<concept>__<phrasing>`` (used as the cassette key)."""
    return f"{record['concept']}__{record['phrasing']}"


def manifest() -> dict:
    return {"version": BATTERY_VERSION, "n_prompts": len(_BATTERY), "families": families(),
            "phrasings": sorted({r["phrasing"] for r in _BATTERY}),
            "concepts": sorted({r["concept"] for r in _BATTERY}),
            "note": ("each concept carries an appearance and a construction phrasing to expose the measured style "
                     "confound; constraints are machine-checked for spec-faithfulness (master_plan_v6 §8.1).")}
