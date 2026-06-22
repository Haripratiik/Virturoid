"""General task representation: a typed skill-call sequence + a small closed predicate DSL.

The gap-closure plan's task-side generality (research synthesis: a typed skill-call DAG + predicate
monitor beats per-task hard-coded handlers; defer behavior-trees/PDDL/reward-machines). An LLM proposes
a ``TaskSpec`` over the capability registry; a verifier checks it against the robot's morphology and the
skills' pre/postconditions (LLM-Modulo proposer/verifier, SayCan grounding done symbolically); a
deterministic executor runs the steps through existing skill primitives and scores goal predicates.

The DSL is deliberately CLOSED — every ``PredicateOp`` has exactly one meaning a skill can establish and
the executor can score — so any spec is machine-verifiable and a fallback never masquerades as success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PredicateOp(str, Enum):
    """The closed vocabulary of goal/precondition facts skills can establish + the executor can score."""
    REACHED_GOAL = "reached_goal"        # the robot's base reached a goal region (args: [goal_id])
    TRAVELED = "traveled"                # base moved >= D metres while upright (args: [meters])
    UPRIGHT = "upright"                  # base stayed upright
    OBJECT_LIFTED = "object_lifted"      # an object was grasped + lifted off the surface (args: [object])
    OBJECTS_PLACED = "objects_placed"    # objects transported into their target bins/zones
    SURFACE_COVERED = "surface_covered"  # a surface was covered to >= fraction (args: [fraction])
    PATH_FOLLOWED = "path_followed"      # followed a planned path to the goal (maze/route)


@dataclass(frozen=True)
class Predicate:
    op: PredicateOp
    args: list = field(default_factory=list)        # entity ids (str) + numeric literals

    def to_dict(self) -> dict:
        return {"op": self.op.value, "args": list(self.args)}

    @staticmethod
    def from_dict(d: dict) -> "Predicate":
        return Predicate(PredicateOp(d["op"]), list(d.get("args", [])))

    def referenced_entities(self) -> set:
        return {a for a in self.args if isinstance(a, str)}


@dataclass
class Entity:
    """An object / region / surface the task refers to (entities ground the predicates)."""
    id: str
    kind: str = "object"                 # object | region | surface | goal


@dataclass
class SkillCall:
    step_id: str
    skill: str                           # MUST be a key in the capability registry
    params: dict = field(default_factory=dict)
    depends_on: list = field(default_factory=list)


@dataclass
class TaskSpec:
    """A verifiable, executable task: ordered skill calls + goal/failure predicates over state."""
    name: str = ""
    prompt: str = ""
    entities: list = field(default_factory=list)     # list[Entity]
    steps: list = field(default_factory=list)        # list[SkillCall]
    goal: list = field(default_factory=list)         # list[Predicate] (AND)
    failure: list = field(default_factory=list)      # list[Predicate] (any-true -> abort)
    source: str = "heuristic"            # "llm" | "heuristic"

    def validate(self) -> list:
        """Structural issues (semantic feasibility lives in the verifier). Mirrors RobotGene.validate()."""
        issues = []
        if not self.steps:
            issues.append("task has no steps")
        if not self.goal:
            issues.append("task has no goal predicate")
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            issues.append("duplicate step_ids")
        known = {s.step_id for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in known:
                    issues.append(f"step {s.step_id!r} depends on unknown step {dep!r}")
        return issues

    def to_dict(self) -> dict:
        return {
            "name": self.name, "prompt": self.prompt, "source": self.source,
            "entities": [{"id": e.id, "kind": e.kind} for e in self.entities],
            "steps": [{"step_id": s.step_id, "skill": s.skill, "params": s.params,
                       "depends_on": s.depends_on} for s in self.steps],
            "goal": [p.to_dict() for p in self.goal],
            "failure": [p.to_dict() for p in self.failure],
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        return TaskSpec(
            name=d.get("name", ""), prompt=d.get("prompt", ""), source=d.get("source", "heuristic"),
            entities=[Entity(e["id"], e.get("kind", "object")) for e in d.get("entities", [])],
            steps=[SkillCall(s.get("step_id", f"s{i}"), s["skill"], dict(s.get("params", {})),
                             list(s.get("depends_on", []))) for i, s in enumerate(d.get("steps", []))],
            goal=[Predicate.from_dict(p) for p in d.get("goal", [])],
            failure=[Predicate.from_dict(p) for p in d.get("failure", [])])
