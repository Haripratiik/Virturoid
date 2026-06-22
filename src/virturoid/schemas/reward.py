"""RewardSpec: a structured, trainable objective for a task (Language-to-Rewards style).

The research is explicit: the LLM should NOT write free-form control or be trusted to author
arbitrary reward code first. It should emit *parameters* of a structured reward — a set of
residual shaping terms over **named quantities** — plus an explicit **sparse success
criterion** that defines the metric. Candidates are always scored by the sparse criterion,
never by their own dense reward (this is how Eureka avoids reward hacking).

This schema is that structured space:
- ``sparse_success`` / ``failure`` — measurable criteria that DEFINE task success (the metric).
- ``dense_terms`` — bounded, potential-style shaping aids over a fixed vocabulary of quantities;
  only an optimization hint, never the metric.
- ``scene_requirements`` / ``end_effector`` — what the task needs in the world / on the robot.
- ``domain_randomization`` — ranges to randomize (DrEureka-style), for robustness + sim-to-real.

The sparse criterion drives the existing scripted evaluator today; the dense terms + DR feed RL
training (PPO on MJX) when learned skills come online.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The fixed vocabulary the LLM may reference — keeps rewards interpretable + validatable, and
# maps to quantities the simulator can actually measure from the gene + scene.
ALLOWED_QUANTITIES = {
    "ee_to_target",        # end-effector distance to a target pose/zone
    "object_to_target",    # manipulated object distance to its target
    "object_height",       # object z (lift tasks)
    "object_in_zone",      # 0/1 object inside a target zone
    "surface_coverage",    # fraction of a surface covered (spray/paint)
    "grasp_contact",       # gripper-object contact / grasp stability
    "base_upright",        # torso/base orientation (whole-body/locomotion)
    "distance_traveled",   # base displacement toward a goal (carry/navigate)
    "joint_limit_margin",  # stay off joint limits (safety)
    "energy",              # actuation effort (efficiency)
    # --- PROCESS quantities: capture HOW the task is done, so the reward can distinguish a
    # controlled manipulation from a degenerate shortcut (a kicked/launched/thrown object). The
    # push milestone showed an outcome-only reward rewards kicking as much as pushing — these let
    # the agent encode "controlled, not kicked" and gate success on real manipulation.
    "object_speed",        # object linear speed (high = launched/kicked, not controlled)
    "contact_sustained",   # ee maintained contact with the object while moving it (vs an impulsive hit)
    "object_settled",      # 0/1 object at the goal AND at rest (placed, not flying through)
}
# Outcome quantities define WHERE things end up; process quantities define HOW they got there.
# A success criterion built only from outcome quantities is hackable (Goodhart): the shortcut that
# games the proxy (kick the box into the zone) scores as well as the real skill (place it).
OUTCOME_QUANTITIES = {"ee_to_target", "object_to_target", "object_height", "object_in_zone",
                      "surface_coverage", "distance_traveled"}
PROCESS_QUANTITIES = {"grasp_contact", "contact_sustained", "object_speed", "object_settled",
                      "base_upright", "energy", "joint_limit_margin"}
END_EFFECTORS = {"gripper", "suction", "spray_nozzle", "hook", "none"}
TERM_KINDS = {"attract", "repel", "bonus", "penalty"}  # potential-style shaping
_MAX_WEIGHT = 10.0


@dataclass
class RewardTerm:
    quantity: str          # one of ALLOWED_QUANTITIES
    kind: str              # attract | repel | bonus | penalty
    weight: float          # bounded shaping weight
    target: float | None = None  # desired value for attract/repel terms


@dataclass
class Criterion:
    name: str
    expression: str        # a checkable condition over named quantities, e.g. "object_in_zone == 1"


@dataclass
class RewardSpec:
    task_type: str
    sparse_success: Criterion                      # THE metric (what counts as done)
    failure: list[Criterion] = field(default_factory=list)
    dense_terms: list[RewardTerm] = field(default_factory=list)  # shaping aids only
    scene_requirements: list[str] = field(default_factory=list)  # objects/zones the task needs
    end_effector: str = "gripper"
    domain_randomization: dict = field(default_factory=dict)
    rationale: str = ""

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.sparse_success.name or not self.sparse_success.expression:
            issues.append("sparse_success must have a name and a measurable expression")
        if not any("time" in c.expression.lower() or "timeout" in c.name.lower() for c in self.failure):
            issues.append("failure must include an episode timeout")
        if self.end_effector not in END_EFFECTORS:
            issues.append(f"end_effector {self.end_effector!r} not in {sorted(END_EFFECTORS)}")
        for i, t in enumerate(self.dense_terms):
            if t.quantity not in ALLOWED_QUANTITIES:
                issues.append(f"dense_terms[{i}] quantity {t.quantity!r} not in the allowed vocabulary")
            if t.kind not in TERM_KINDS:
                issues.append(f"dense_terms[{i}] kind {t.kind!r} not in {sorted(TERM_KINDS)}")
            if not (abs(t.weight) <= _MAX_WEIGHT):
                issues.append(f"dense_terms[{i}] weight {t.weight} must be finite and |w| <= {_MAX_WEIGHT}")
        return issues

    def hacking_risks(self) -> list[str]:
        """Advisories where this reward is likely GAMEABLE (the kick-vs-place problem), so the agent
        (or a human) can harden it. Not validation errors — reward design is judgment — but the
        system reasoning about whether its own objective rewards the right *process*.
        """
        risks: list[str] = []
        expr = (self.sparse_success.expression or "").lower()
        moves_object = (any(q in expr for q in ("object_to_target", "object_in_zone", "object_height"))
                        or self.end_effector in ("gripper", "suction", "hook"))
        success_outcome_only = (any(q in expr for q in OUTCOME_QUANTITIES)
                                and not any(q in expr for q in PROCESS_QUANTITIES)
                                and not any(w in expr for w in ("settle", "speed", "stable", "held", "grasp")))
        if moves_object and success_outcome_only:
            risks.append("sparse_success is OUTCOME-ONLY: a kicked/launched/thrown object satisfies it as well as a "
                         "placed one. Add a process gate — e.g. 'object_settled == 1', or 'object_speed < <thresh>', "
                         "or 'grasp_contact held' — so only controlled manipulation counts as success.")
        if self.dense_terms and not any(t.quantity in PROCESS_QUANTITIES for t in self.dense_terms):
            risks.append("no process/regularization dense term (object_speed / energy / contact): the shaping can be "
                         "maximized by fast, uncontrolled motion. Add a bounded process penalty.")
        return risks
