"""ONE derivation of "what kind of body is this?", because N of them kept disagreeing.

Which rubric a body is judged by is decided in ~40 places (verify, eval, gait, training recipe, BOM, spec
sheet, export env, template offer, task feasibility). Each of those used to reach for whichever classifier was
nearest, and the classifiers did not agree — which is how the same bug shipped five separate times:

  #214    a fixed-base URDF quadruped read as a manipulator, so verify ran the ARM rubric on it
  #218    an imported Go2's cylinder legs read as wheels -> "TIPPED while driving"
  #244    ``_infer_class`` counted limb BRANCHES, so no humanoid could ever classify as one
  80ec693 a second, coarser classifier downstream hard-coded "quadruped" for any class outside
          ("quadruped","legged","hexapod"), renaming every imported humanoid a quadruped
  2026-08-01 (this) ``robot_kind`` checked HANDS before LEGS, so Talos — a 45-segment floating-base biped —
          came out "manipulator" and was scored on pick-place; a drone with zero joints fell through every
          branch to the same answer; and ``build_appendage_map`` found ZERO legs on a Booster T1, TWO on a
          two-finger gripper, and a SNAKE'S SPINE on a UR5e.

So the pieces live here once and every site derives from them:

  ``family_from_legs``   the leg-count -> family ladder (was written out twice, disagreeing at 3 legs)
  ``FLOATING_BASE_CLASSES``  which classes get a free base (was listed twice; ``robot_import``'s copy was
                         missing legged/biped/aerial/aquatic, welding those imports to a table)
  ``measured_legs`` /    the structural appendage count, from the COMPILED body (never names, never labels)
  ``measured_appendages``
  ``body_kind``          the one ordered rule set that turns all of it into a dispatch kind + family

EVIDENCE ORDER. Structure decides whenever structure is conclusive — a body LABELLED "legged" that rolls on
wheels is still driven, not walked (tests/test_structural_dispatch.py). The ``robot_class`` string is consulted
only where cheap structure genuinely cannot separate two bodies, and then only because on an IMPORT that string
is itself a measurement: ``robot_import._infer_class`` poses the source model in its shipped stance and counts
the chains that carry it.

COST. ``body_kind`` is called on hot paths, and measuring structure means compiling the gene (~90 ms). So the
cheap segment-tree rules answer first and the compiled map is consulted ONLY for the genuinely ambiguous
shapes (a floating body with hands; a bench-mounted body with several limb chains; a floating base whose
wheels the segment test cannot see). Those results are memoised on the body's topology.
"""
from __future__ import annotations

from dataclasses import dataclass

# Classes that already carry legged meaning. ``robot_import._infer_class`` decides the family from the limbs
# that actually hold the body up (#244) and is right across the Menagerie corpus, so nothing downstream may
# overwrite one of these (80ec693).
LEGGED_CLASSES = frozenset({"quadruped", "hexapod", "octopod", "legged", "humanoid", "biped", "bipedal"})

# The walkable reference template is a QUADRUPED fan/crawl recipe — only these families can adopt it.
QUAD_TEMPLATE_CLASSES = frozenset({"quadruped", "hexapod", "octopod", "legged"})

# A robot that MOVES needs a FREE (floating 6-DOF) base. "floor"/"table" weld the base to the world, so the
# body cannot translate at all — every gait rolls out to 0.0 m forward and the verdict is meaningless. This set
# was written out twice (robot_import + anatomy_compiler) and had drifted: the import copy was missing
# "legged", "biped", "aerial" and "aquatic", so a body in one of those families was bolted to a table on the
# way in and then judged for not walking.
FLOATING_BASE_CLASSES = frozenset({"mobile_base", "mobile_manipulator", "quadruped", "hexapod", "octopod",
                                   "legged", "humanoid", "biped", "bipedal", "aerial", "aquatic"})


# WHEN A BOLTED-DOWN BODY STILL COUNTS AS LEGGED. URDF has no floating-base concept, so MuJoCo's loader adds
# no freejoint and an ordinary URDF quadruped -- or one whose static torso got fused into the world, leaving
# four separate leg roots -- presents exactly as "fixed" as a bench-mounted hand (#214). The only honest
# promotion is a real STANCE: several chains reaching down TOGETHER to a common level. ``_infer_class`` tests
# that as a support-polygon AREA on the source model in its shipped keyframe; the compiled twin loses the
# lateral spread (measured: #214's four tips all land at x=y=0), so what survives is the other half of the same
# physics -- feet that carry a body are LEVEL WITH EACH OTHER. Measured on the compiled bodies: #214's quad and
# the nonsense-labelled quad both spread 0.000 of their height, a LEAP hand's four finger chains spread 0.69 of
# its height (three fingertips 31 mm above the thumb) and are not a stance at all.
MIN_STANCE_CHAINS = 3
STANCE_LEVEL_FRAC = 0.10


def family_from_legs(n_legs: int) -> str:
    """THE leg-count -> legged-family ladder. ``""`` when the count carries no family meaning.

    Deliberately conservative at the bottom: 1 leg is not a family, and 0 legs is not evidence of one. Callers
    that know the body IS legged fall back to the honest generic ``"legged"`` rather than inventing a family,
    because the family drives the BOM, the spec sheet, the verify rubric and the walkable-template offer.

    3 legs reads as the quadruped family (a tripod crawls; it is certainly not a biped). The two copies of this
    ladder used to disagree exactly there.
    """
    n = int(n_legs)
    if n >= 6:
        return "hexapod"
    if n >= 3:
        return "quadruped"
    if n == 2:
        return "humanoid"
    return ""


@dataclass(frozen=True)
class BodyKind:
    """One body's answer, from one derivation.

    ``kind``    the coarse dispatch kind: legged | manipulator | mobile | aerial | aquatic | spray.
    ``family``  the legged family (quadruped | hexapod | humanoid | legged), ``""`` for a non-legged body.
    ``n_legs``/``n_wheels``/``n_arms``  structural counts, or ``None`` where the cheap rules were conclusive
                and the body was never compiled. Use ``measured_legs`` when you need the number itself.
    ``evidence`` which rule answered — for debugging a disagreement without re-deriving it.
    """
    kind: str
    family: str = ""
    n_legs: int | None = None
    n_wheels: int | None = None
    n_arms: int | None = None
    evidence: str = ""


def _md(gene) -> dict:
    return getattr(gene, "metadata", None) or {}


def _cls(gene) -> str:
    return (getattr(gene, "robot_class", "") or "").strip().lower()


def has_wheels(segments) -> bool:
    """A wheel is a cylinder on a CONTINUOUS (unbounded) revolute joint — it spins freely. A leg link is also a
    cylinder on a revolute joint, but a BOUNDED one (anatomical limits). Requiring the joint to be unbounded is
    the canonical URDF wheel-vs-leg distinction, and it is exactly how this system emits them: our composed
    wheels carry joint_lower/upper = None, while every leg/arm joint (incl. an imported Unitree Go2's, measured
    hip/knee limits like -2.72..-0.84) is bounded. Before this, a Go2's cylinder legs read as wheels, so the
    quadruped was routed to the DRIVING rubric and verified 'TIPPED while driving' — a dishonest verdict (#218).

    The ``cylinder`` half is load-bearing in the other direction and must stay: measured on the corpus, an
    imported Kinova arm (Tidybot) carries four CONTINUOUS wrist/shoulder joints and Cassie — a biped — carries
    four continuous passive linkage rods. Dropping the shape test to catch imported wheels would call an arm
    and a biped mobile. Imported wheels are found by ``measured_appendages`` instead, which reads ground
    contact off the compiled body.
    """
    return any(getattr(s, "shape", None) == "cylinder" and s.joint_type == "revolute"
               and s.joint_lower is None and s.joint_upper is None for s in segments)


def _topology_key(gene):
    """A stable key for the body's SHAPE — what the structural measurement actually depends on."""
    segs = getattr(gene, "segments", None) or []
    return (getattr(gene, "base_mount", ""), getattr(gene, "end_effector_type", ""), _cls(gene), len(segs),
            tuple((s.name, s.parent, s.joint_type, s.joint_lower is None, s.joint_upper is None,
                   getattr(s, "shape", None), round(float(getattr(s, "length_m", 0.0) or 0.0), 4),
                   round(float(getattr(s, "radius_m", 0.0) or 0.0), 4),
                   tuple(round(float(v), 4) for v in (getattr(s, "mount_offset", None) or (0, 0, 0))))
                  for s in segs))


_APPENDAGE_CACHE: dict = {}
_CACHE_MAX = 512


def measured_appendages(gene) -> dict:
    """``{"legs": n, "wheels": n, "arms": n, "spine": bool}`` measured off the COMPILED body.

    THE structural count. Everything that needs to know how many legs a body has reads it here rather than
    counting chains its own way — ``gait_flywheel`` and ``heldout_set`` each grew their own counter, and
    ``_legged_family``/``_honest_biped`` each called the appendage map directly with their own error handling.
    Returns zeros (never raises) when MuJoCo is absent or the body will not compile, because a classification
    guess must never break an import.
    """
    key = _topology_key(gene)
    hit = _APPENDAGE_CACHE.get(key)
    if hit is not None:
        return hit
    out = {"legs": 0, "wheels": 0, "arms": 0, "spine": False, "measured": False}
    try:
        import mujoco

        from virturoid.services.appendage_map import build_appendage_map
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        amap = build_appendage_map(mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))))
        out = {"legs": int(amap.n_legs), "wheels": int(amap.n_wheels), "arms": int(amap.n_arms),
               "spine": amap.spine is not None, "measured": True}
    except Exception:  # noqa: BLE001 - no MuJoCo / an uncompilable twin -> the cheap rules stand alone
        pass
    if len(_APPENDAGE_CACHE) >= _CACHE_MAX:
        _APPENDAGE_CACHE.clear()
    _APPENDAGE_CACHE[key] = out
    return out


def measured_legs(gene) -> int:
    """How many legs the body actually stands on, measured off the compiled model."""
    return int(measured_appendages(gene)["legs"])


def body_kind(gene) -> BodyKind:
    """THE ordered rule set. Structure first; the class string only where cheap structure cannot separate two
    genuinely different bodies (and there it is itself a measurement — see the module docstring)."""
    segs = getattr(gene, "segments", None) or []
    cls, md = _cls(gene), _md(gene)
    ee = getattr(gene, "end_effector_type", None) or "none"
    free = getattr(gene, "base_mount", "") == "free"
    n_rev = sum(1 for s in segs if s.joint_type == "revolute")
    roots = {s.name for s in segs if s.parent is None}
    limbs = sum(1 for s in segs if s.parent in roots and s.joint_type == "revolute")

    # 1-3. declared envelopes: a quadcopter is flown and an undulator is swum, whatever their limbs look like.
    if cls == "aerial" or md.get("rotor_offsets"):
        return BodyKind("aerial", evidence="rotor metadata / aerial class")
    if cls == "aquatic" or md.get("aquatic"):
        return BodyKind("aquatic", evidence="aquatic metadata / class")
    if ee == "spray_nozzle":
        return BodyKind("spray", evidence="spray end effector")

    # 4. wheels beat everything below: a body that rolls is driven, never walked, even when it is LABELLED
    #    legged (tests/test_structural_dispatch.py) and even when it also carries an arm.
    if has_wheels(segs):
        return BodyKind("mobile", evidence="continuous-hinge cylinder wheels")

    # 5. a free body with nothing to articulate cannot manipulate anything. This used to fall through every
    #    branch to the "manipulator" at the bottom, so an imported Skydio X2 — one rigid hull, zero joints —
    #    was scored on REACH and pick-place.
    if n_rev == 0:
        return BodyKind("mobile" if free else "manipulator", evidence="no articulated joints")

    # 6. THE AMBIGUOUS SHAPES — the only ones worth compiling for, because cheap structure cannot separate
    #    them and getting them wrong is exactly this bug family:
    #      (a) a floating articulated body with an END EFFECTOR is either a biped that also has hands (Talos)
    #          or an arm that happens to float. Checking the hands first is what made Talos a manipulator.
    #      (b) a FIXED-BASE body with several limb chains is either a quadruped whose static torso MuJoCo
    #          fused into the world (#214) or a bench-mounted multi-fingered HAND (LEAP: four chains, and it
    #          was coming out "legged").
    #      (c) a floating base whose wheels the segment test cannot see, because an imported wheel is a mesh
    #          that lands as a capsule, not a cylinder (Tiago, Stretch).
    ambiguous = ((free and ee in ("gripper", "hand", "suction") and n_rev >= 2)
                 or (not free and limbs >= 3)
                 or (free and cls in ("mobile_base", "mobile_manipulator")))
    if ambiguous:
        a = measured_appendages(gene)
        if a["measured"]:
            if a["wheels"] >= 2:
                return BodyKind("mobile", n_legs=a["legs"], n_wheels=a["wheels"], n_arms=a["arms"],
                                evidence="measured wheels on the compiled body")
            if a["legs"] >= 2:
                return BodyKind("legged", family=family_from_legs(a["legs"]) or "legged", n_legs=a["legs"],
                                n_wheels=a["wheels"], n_arms=a["arms"], evidence="measured legs carry the body")
            return BodyKind("mobile" if (free and cls == "mobile_base" and not a["arms"]) else "manipulator",
                            n_legs=a["legs"], n_wheels=a["wheels"], n_arms=a["arms"],
                            evidence="measured: nothing carries the body")

    # 7. an end effector means an arm, once the locomotor question above has been settled.
    if ee in ("gripper", "hand", "suction"):
        return BodyKind("manipulator", evidence="end effector, not a locomotor")

    # 8. free-floating, jointed, no wheels, no hands -> it walks on something.
    if free and n_rev >= 2:
        return BodyKind("legged", family=cls if cls in LEGGED_CLASSES else "legged",
                        evidence="floating jointed body")

    # 9. #214, when the compiled measurement was unavailable: many symmetric limb chains off a common root.
    if limbs >= 3 and ee == "none":
        return BodyKind("legged", family=cls if cls in LEGGED_CLASSES else "legged",
                        evidence="multiple limb chains off a common base")
    return BodyKind("manipulator", evidence="fixed base + articulation")
