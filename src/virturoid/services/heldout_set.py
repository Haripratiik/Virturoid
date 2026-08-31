"""The FROZEN held-out body set — the leak-free foundation of the compounding-proof and the corpus factory.

Master-plan v6 §10.4 / kickoff #1: *reserve the held-out set BEFORE anything is fit.* A growth curve that fits
the embedding/whitener/metric on the very bodies it later scores is a mirage — it measures memorisation, not
transfer. So this module owns a **versioned, deterministic** set of bodies that the factory must never admit as a
training label and the metric must never fit on. It holds out two ways, deliberately:

  * **specific bodies** (interpolation control) — named prompts inside common families, reserved so a warm-start
    curve is scored on bodies the corpus has genuinely never seen;
  * **whole niches** (structural EXTRAPOLATION control) — entire structural families (aquatic undulators, aerial
    rotorcraft, many-limbed radial bodies, wheel+leg hybrids) reserved via a STRUCTURAL predicate, so the curve
    proves generalisation to unseen body *shapes*, not just unseen instances of seen shapes.

The guard is consulted structurally (``is_held_out(gene)``) and by prompt (``is_held_out_prompt``). Membership is
pure, cheap, and import-light (``robot_kind`` is the only lazy dependency) so any fitting/admission path can call
it in a hot loop. Nothing here designs a robot — it only PARTITIONS; the LLM still authors every body.
"""
from __future__ import annotations

import hashlib
import re

# v2 (2026-08-08, task #281) — the many-limb predicate stopped counting NECKS AND TAILS as limbs.
#
# MEASURED before the change: every prompt in the corpus factory's own legged bank composed to a body the guard
# called ``many_limb`` — 18 of 18, 100%. The composer's quadruped is a torso with four legs, a neck and a tail:
# six actuated chains hanging off the root, and ``_leg_chain_count`` counts chains, not legs. So the niche whose
# documented job is "octopod, hexapod-and-up" was reserving THE MAIN FAMILY, and the corpus factory could not
# bank a single quadruped no matter what it proposed.
#
# The fix narrows ``many_limb`` to what it always said it was: >= 6 chains carrying >= 2 actuated joints each.
# Measured on the composer's own bodies, that separates cleanly with room to spare — quadruped 4, hexapod 6,
# spider 8, octopod 8, centipede 14 — because a neck or a tail is a ONE-joint stub and a limb you step with is
# a 3-joint chain. Nothing else about the partition moves: the reserved-prompt list, the aquatic/aerial/wheel-leg
# niches and the manifest are untouched, and every reserved specific body stays reserved through its prompt.
#
# WHAT THE PARTITION LOSES, STATED PLAINLY. A reserved quadruped-shaped body (the fox, the tripod surveyor)
# presented WITHOUT its originating prompt used to be caught structurally; now it is not. That catch was never a
# property of the holdout design — it was the bug, which reserved every quadruped ever composed and so caught the
# fox the way a net that covers the ocean catches one fish. Closing that hole properly needs a structural
# signature for the reserved BODIES (``body_key`` exists and its docstring says so), not a niche predicate that
# swallows the family. Left undone deliberately: the keys can only be computed by composing all 25 reserved
# prompts, which is neither pure nor cheap, and this module's callers consult the guard in a hot loop.
HELD_OUT_VERSION = "v2"

# ------------------------------------------------------------------ reserved specific bodies (interpolation)
# Named prompts inside common families, reserved so the compounding curve is scored on bodies the corpus never
# banked. Kept SMALL and stable; the whole-niche predicates below carry the harder extrapolation holdout.
# Each entry: (prompt, family). ``family`` is descriptive only (reporting/coverage), never used for routing.
_RESERVED_BODIES: tuple[tuple[str, str], ...] = (
    # quadruped / legged animals
    ("a nimble fox with a long bushy tail", "quadruped"),
    ("a heavy-set bulldog with short thick legs", "quadruped"),
    ("a lanky newborn foal on tall spindly legs", "quadruped"),
    ("a badger with a broad low body", "quadruped"),
    ("a greyhound built for sprinting", "quadruped"),
    ("a mountain goat with sturdy short legs", "quadruped"),
    ("a panther with a low prowling stance", "quadruped"),
    # bipeds / humanoids
    ("a compact humanoid torso with two arms", "humanoid"),
    ("a two-legged hopping bird like a small ostrich", "biped"),
    ("a tall thin humanoid with long legs", "humanoid"),
    # manipulators / arms
    ("a 5-DOF benchtop welding arm", "manipulator"),
    ("a stubby 3-joint pick-and-place arm", "manipulator"),
    ("a long reach inspection arm for tight spaces", "manipulator"),
    ("a 6-axis industrial arm for palletizing", "manipulator"),
    ("a delicate lab-sample handling arm", "manipulator"),
    # mobile / vehicles
    ("a six-wheeled exploration rover with a flat deck", "mobile_base"),
    ("a compact two-wheel differential-drive robot", "mobile_base"),
    ("a tracked-style wide chassis hauler on four big wheels", "mobile_base"),
    ("a nimble four-wheel indoor delivery cart", "mobile_base"),
    # hybrids / composites
    ("a wheeled base carrying a small manipulator arm", "hybrid"),
    ("a legged base with a gripper arm on top", "hybrid"),
    # novel / open-vocab concepts (the open-world stress)
    ("a five-legged radially symmetric crawler", "novel"),
    ("a segmented caterpillar-like inching robot", "novel"),
    ("a low pancake-shaped disc robot on three stubby legs", "novel"),
    ("a tripod-legged hopping surveyor", "novel"),
)

# ------------------------------------------------------------------ reserved whole niches (extrapolation)
# Structural predicates over a gene — an entire body FAMILY held out so the curve proves it can extrapolate to
# unseen morphology *classes*. These fire on STRUCTURE (robot_kind / segment topology), so a factory-proposed
# body that lands in one of these niches is excluded even though it was never in ``_RESERVED_BODIES``.


def _kind(gene) -> str:
    from virturoid.services.task_matched_eval import robot_kind
    try:
        return robot_kind(gene)
    except Exception:  # noqa: BLE001 - an un-classifiable body is simply "unknown" for holdout purposes
        return "unknown"


def _leg_chain_count(gene) -> int:
    """Number of multi-joint limb chains hanging off the root — the structural 'how many legs/arms' signal,
    derived from topology (root children that begin an actuated chain), robust to LLM-chosen names."""
    root = gene.root() if hasattr(gene, "root") else None
    if root is None:
        return 0
    chains = 0
    for child in gene.children_of(root.name):
        # a limb chain = a root child that is itself actuated or spawns an actuated descendant
        if child.joint_type in ("revolute", "prismatic"):
            chains += 1
            continue
        stack = list(gene.children_of(child.name))
        while stack:
            s = stack.pop()
            if s.joint_type in ("revolute", "prismatic"):
                chains += 1
                break
            stack.extend(gene.children_of(s.name))
    return chains


# A chain is a LIMB — something the body could step on — rather than a neck/tail/antenna stub when it carries at
# least this many actuated joints. Measured across the composer's own bodies: neck 1, tail 1, walking leg 3,
# spider leg 3, tentacle 4. The threshold sits in the empty gap between 1 and 3, so it is not a knife edge.
LIMB_ARTICULATION_MIN = 2
MANY_LIMB_MIN = 6                  # limbs that make a body "many-limbed" (hexapod-and-up)


def _chain_actuated_joints(gene, child) -> int:
    """Actuated joints in the whole subtree hanging off one root child (its own joint included)."""
    n = 1 if child.joint_type in ("revolute", "prismatic") else 0
    stack = list(gene.children_of(child.name))
    while stack:
        s = stack.pop()
        if s.joint_type in ("revolute", "prismatic"):
            n += 1
        stack.extend(gene.children_of(s.name))
    return n


def locomotor_limb_count(gene) -> int:
    """How many of the root's chains are LIMBS (>= ``LIMB_ARTICULATION_MIN`` actuated joints), as opposed to the
    single-joint appendages — a neck, a tail — that ``_leg_chain_count`` cannot tell apart from a leg.

    Deliberately a SECOND function rather than a fix to ``_leg_chain_count``: that one is a body-descriptor used
    by the MAP-Elites niche key, the exemplar features, the morphology-dynamics vector and design-bench's
    ``legs`` constraint, and it is honestly named — it counts limb CHAINS. Only the holdout predicate needed the
    stricter reading, so only the holdout predicate gets it, and no banked embedding or niche key moves.
    """
    root = gene.root() if hasattr(gene, "root") else None
    if root is None:
        return 0
    return sum(1 for c in gene.children_of(root.name)
               if _chain_actuated_joints(gene, c) >= LIMB_ARTICULATION_MIN)


def _niche_aquatic(gene) -> bool:
    return _kind(gene) == "aquatic"


def _niche_aerial(gene) -> bool:
    return _kind(gene) == "aerial"


def _niche_many_limb(gene) -> bool:
    """Radial / many-limbed bodies (octopod, hexapod-and-up): >= ``MANY_LIMB_MIN`` LIMBS. A frontier for scripted
    control and a genuine structural extrapolation target — held out wholesale.

    Counts limbs (``locomotor_limb_count``), not chains. Counting chains meant a torso with four legs, a neck and
    a tail read as a six-limbed body; see the HELD_OUT_VERSION note."""
    return _kind(gene) == "legged" and locomotor_limb_count(gene) >= MANY_LIMB_MIN


def _niche_wheel_leg_hybrid(gene) -> bool:
    """Bodies that carry BOTH wheels and actuated legs — a composite the corpus is unlikely to cover densely."""
    segs = gene.segments
    has_wheels = any(getattr(s, "shape", None) == "cylinder" and s.joint_type == "revolute" for s in segs)
    n_rev_non_wheel = sum(1 for s in segs if s.joint_type == "revolute"
                          and not (getattr(s, "shape", None) == "cylinder"))
    return has_wheels and n_rev_non_wheel >= 2


# name -> predicate. Ordered; ``niche_of`` returns the first match.
_RESERVED_NICHES: dict[str, callable] = {
    "aquatic": _niche_aquatic,
    "aerial": _niche_aerial,
    "many_limb": _niche_many_limb,
    "wheel_leg_hybrid": _niche_wheel_leg_hybrid,
}


# ------------------------------------------------------------------ structural body key
def body_key(gene) -> str:
    """A STABLE structural signature for a body (robot_class + structural kind + segment/DOF counts + coarse size
    bucket), hashed. Two bodies with the same key are structurally interchangeable for holdout/dedup purposes;
    a horse and a dog differ (size bucket / segment counts) but two re-compositions of 'a dog' collapse. Used to
    (a) dedup the corpus and (b) assert a TRAIN body isn't structurally identical to a held-out one."""
    segs = gene.segments
    n_seg = len(segs)
    dof = sum(1 for s in segs if s.joint_type in ("revolute", "prismatic"))
    total_len = round(sum(float(getattr(s, "length_m", 0.0)) for s in segs), 1)
    sig = f"{getattr(gene, 'robot_class', '?')}:{_kind(gene)}:{n_seg}:{dof}:{total_len}"
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def niche_of(gene) -> str | None:
    """Which reserved whole-niche this body falls in (structural), or None. The extrapolation-holdout signal."""
    for name, pred in _RESERVED_NICHES.items():
        try:
            if pred(gene):
                return name
        except Exception:  # noqa: BLE001 - a predicate must never crash the guard
            continue
    return None


# ------------------------------------------------------------------ prompt membership
def normalize_prompt(prompt: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation — so 'A nimble fox.' == 'a  nimble fox'."""
    p = re.sub(r"[^a-z0-9 ]+", " ", (prompt or "").lower())
    return re.sub(r"\s+", " ", p).strip()


_RESERVED_PROMPT_SET = frozenset(normalize_prompt(p) for p, _ in _RESERVED_BODIES)


def is_held_out_prompt(prompt: str) -> bool:
    """True iff this prompt is one of the reserved specific bodies (normalised exact match — no fuzzy matching,
    same discipline as concept aliasing: a new prompt cannot silently inherit held-out status)."""
    return normalize_prompt(prompt) in _RESERVED_PROMPT_SET


def is_held_out(gene, *, prompt: str | None = None) -> bool:
    """The guard every fitting/admission path consults. Held out iff the body falls in a reserved whole-niche
    (structural extrapolation control) OR its originating prompt is a reserved specific body."""
    if niche_of(gene) is not None:
        return True
    if prompt is not None and is_held_out_prompt(prompt):
        return True
    return False


def explain(gene, *, prompt: str | None = None) -> dict:
    """``is_held_out`` with its REASONS — the surface a proposer steers by.

    A boolean can only be filtered on: the caller learns that a slot was wasted and nothing about what to do
    differently, which is why the corpus factory could resample four times into the same reserved niche and
    report the same waste it was written to prevent. This returns the verdict PLUS the measurements behind it
    (which niche, what the limb count was, whether it was the prompt rather than the structure), so a proposer
    can retarget instead of re-rolling. Same cost as ``is_held_out`` — pure structure, no sim, no LLM.
    """
    niche = niche_of(gene)
    by_prompt = bool(prompt is not None and is_held_out_prompt(prompt))
    limbs = locomotor_limb_count(gene)
    if niche == "many_limb":
        reason = f"{limbs} limbs >= the reserved many-limb floor of {MANY_LIMB_MIN}"
    elif niche:
        reason = f"structurally {niche}, a wholesale-reserved niche"
    elif by_prompt:
        reason = "the originating prompt is a reserved specific body"
    else:
        reason = ""
    return {"held_out": bool(niche is not None or by_prompt), "niche": niche, "by_prompt": by_prompt,
            "kind": _kind(gene), "locomotor_limbs": limbs, "limb_chains": _leg_chain_count(gene),
            "reason": reason, "version": HELD_OUT_VERSION}


def design_constraints() -> dict:
    """What a body must AVOID to be admissible — the guard stated as a design brief rather than a verdict.

    Handed to the proposer up front (``corpus_factory.guard_brief``) so a reserved body is never drawn in the
    first place. Everything here is derived from the predicates themselves, so the brief cannot drift away from
    the guard it describes."""
    return {
        "version": HELD_OUT_VERSION,
        "avoid_niches": {
            "aquatic": "no undulating/swimming body plans (robot_kind 'aquatic')",
            "aerial": "no rotorcraft/winged body plans (robot_kind 'aerial')",
            "many_limb": (f"fewer than {MANY_LIMB_MIN} LIMBS, where a limb is a root chain with >= "
                          f"{LIMB_ARTICULATION_MIN} actuated joints (a neck or a tail is not a limb)"),
            "wheel_leg_hybrid": "do not put revolute wheels and >=2 non-wheel revolute joints on one body",
        },
        "avoid_prompts": held_out_prompts(),
        "max_limbs": MANY_LIMB_MIN - 1,
    }


def held_out_prompts() -> list[str]:
    """The reserved specific-body prompts (for building the held-out eval split)."""
    return [p for p, _ in _RESERVED_BODIES]


def partition(items, *, gene_getter=lambda x: x, prompt_getter=lambda x: None):
    """Split an iterable of things into (train, held_out). ``gene_getter``/``prompt_getter`` extract the gene and
    (optional) originating prompt from each item, so this works over genes, (prompt, gene) pairs, or corpus rows."""
    train, held = [], []
    for it in items:
        g = gene_getter(it)
        p = prompt_getter(it)
        (held if is_held_out(g, prompt=p) else train).append(it)
    return train, held


def manifest() -> dict:
    """A machine-readable description of the frozen split — for the readiness ledger, the compounding report, and
    the published benchmark slice (so the holdout is auditable, not asserted)."""
    return {
        "version": HELD_OUT_VERSION,
        "reserved_bodies": [{"prompt": p, "family": fam} for p, fam in _RESERVED_BODIES],
        "reserved_body_count": len(_RESERVED_BODIES),
        "reserved_niches": sorted(_RESERVED_NICHES),
        "reserved_niche_rules": design_constraints()["avoid_niches"],
        "policy": ("bodies here are reserved BEFORE any embedding/metric fit or factory admission; the metric is "
                   "fit on TRAIN bodies only and whole niches are held out for structural extrapolation "
                   "(master_plan_v6 §10.4). Membership is structural (niche) or exact-prompt (specific body)."),
    }
