"""WS-A.1 — the frozen held-out set is the leak-free foundation (master_plan_v6 §10.4)."""
from __future__ import annotations

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import heldout_set as H


def _legged(n_legs: int, *, segs_per_leg: int = 3, total_len: float = 2.0) -> RobotGene:
    """A minimal free-floating jointed body with ``n_legs`` actuated limb chains (robot_kind -> 'legged')."""
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.5, radius_m=0.1, mass_kg=2.0)]
    for i in range(n_legs):
        prev = "torso"
        for j in range(segs_per_leg):
            nm = f"leg{i}_{j}"
            segs.append(GeneSegment(name=nm, parent=prev, joint_type="revolute",
                                    length_m=total_len / (n_legs * segs_per_leg), radius_m=0.02, mass_kg=0.1))
            prev = nm
    return RobotGene(id="t", species="test.legged", robot_class="quadruped", segments=segs, base_mount="free",
                     end_effector_type="none")


def _aerial() -> RobotGene:
    g = RobotGene(id="a", species="test.aerial", robot_class="aerial",
                  segments=[GeneSegment(name="body", parent=None, joint_type=None)])
    g.metadata = {"rotor_offsets": [(0.1, 0.1, 0), (-0.1, -0.1, 0)]}
    return g


def _aquatic() -> RobotGene:
    return RobotGene(id="q", species="test.aquatic", robot_class="aquatic",
                     segments=[GeneSegment(name="s0", parent=None, joint_type=None),
                               GeneSegment(name="s1", parent="s0", joint_type="revolute")],
                     base_mount="free")


def test_manifest_meets_volume_and_is_versioned():
    m = H.manifest()
    assert m["version"] == H.HELD_OUT_VERSION
    # §10.4: >=20-30 reserved specific bodies, plus whole-niche extrapolation holdout
    assert m["reserved_body_count"] >= 20
    assert len(H.held_out_prompts()) == m["reserved_body_count"]
    assert {"aquatic", "aerial", "many_limb"} <= set(m["reserved_niches"])


def test_normalize_and_prompt_membership_is_exact_not_fuzzy():
    p = H.held_out_prompts()[0]
    assert H.is_held_out_prompt(p)
    assert H.is_held_out_prompt("  " + p.upper() + " .")          # normalised exact match
    assert not H.is_held_out_prompt(p + " and also a jetpack")    # a superset prompt does NOT inherit holdout
    assert not H.is_held_out_prompt("a totally different robot")


def test_body_key_is_stable_and_discriminating():
    a = _legged(4)
    a2 = _legged(4)
    b = _legged(4, total_len=6.0)                                 # much bigger -> different size bucket
    assert H.body_key(a) == H.body_key(a2)                       # structurally identical -> same key
    assert H.body_key(a) != H.body_key(b)                        # different size -> different key


def test_whole_niche_holdout_fires_structurally():
    assert H.niche_of(_aerial()) == "aerial"
    assert H.niche_of(_aquatic()) == "aquatic"
    assert H.niche_of(_legged(6)) == "many_limb"                 # >=6 limb chains
    assert H.niche_of(_legged(4)) is None                        # an ordinary quadruped is NOT held out by niche
    assert H.is_held_out(_aerial())
    assert not H.is_held_out(_legged(4))                         # ...unless its prompt is reserved
    assert H.is_held_out(_legged(4), prompt="a nimble fox with a long bushy tail")


def test_partition_splits_train_and_held():
    items = [("a nimble fox with a long bushy tail", _legged(4)),   # reserved prompt -> held
             ("a random walking bot", _legged(4)),                  # ordinary -> train
             ("a swimmer", _aquatic())]                             # aquatic niche -> held
    train, held = H.partition(items, gene_getter=lambda it: it[1], prompt_getter=lambda it: it[0])
    assert len(held) == 2 and len(train) == 1
    assert train[0][0] == "a random walking bot"


def _quad_with_neck_and_tail() -> RobotGene:
    """The composer's actual quadruped shape: a torso carrying four 3-joint legs, a 1-joint neck (+ welded head)
    and a 1-joint tail. Six chains hang off the root; four of them are limbs."""
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.16, radius_m=0.08, mass_kg=3.0)]
    segs.append(GeneSegment(name="neck", parent="torso", joint_type="revolute", length_m=0.1))
    segs.append(GeneSegment(name="head", parent="neck", joint_type=None, length_m=0.08))
    segs.append(GeneSegment(name="tail", parent="torso", joint_type="revolute", length_m=0.11))
    for i in range(4):
        prev = "torso"
        for j in range(3):
            nm = f"leg{i}_{j}"
            segs.append(GeneSegment(name=nm, parent=prev, joint_type="revolute", length_m=0.08, radius_m=0.02))
            prev = nm
        segs.append(GeneSegment(name=f"leg{i}_foot", parent=prev, joint_type=None, length_m=0.07))
    return RobotGene(id="q", species="test.quad", robot_class="quadruped", segments=segs, base_mount="free",
                     end_effector_type="none")


def test_a_neck_and_a_tail_are_not_limbs(monkeypatch):
    """#281's mechanism, pinned. ``many_limb`` reserves hexapod-and-up; it used to count CHAINS, so the
    composer's four-legged body — which also has a neck and a tail — read as six-limbed and the corpus factory
    could not admit a single quadruped (measured: 18 of 18 of its own legged prompts held out)."""
    g = _quad_with_neck_and_tail()
    assert H._leg_chain_count(g) == 6                  # the body DESCRIPTOR is unchanged: six chains
    assert H.locomotor_limb_count(g) == 4              # ...of which four are limbs
    assert H.niche_of(g) is None and not H.is_held_out(g)
    why = H.explain(g)
    assert why["held_out"] is False and why["limb_chains"] == 6 and why["locomotor_limbs"] == 4


def test_the_reserved_many_limb_families_still_fire():
    """The narrowing must not let a genuinely many-limbed body through — that is the leak that would destroy the
    extrapolation control. Joint counts here are the ones the composer actually builds (measured 2026-08-08:
    hexapod/spider leg 3 joints, octopod tentacle 4, neck 1, tail 1)."""
    assert H.niche_of(_legged(6, segs_per_leg=3)) == "many_limb"          # hexapod
    assert H.niche_of(_legged(8, segs_per_leg=4)) == "many_limb"          # octopod / tentacled
    assert H.niche_of(_legged(14, segs_per_leg=3)) == "many_limb"         # centipede
    assert H.niche_of(_legged(6, segs_per_leg=2)) == "many_limb"          # 2 joints is still a limb
    assert H.niche_of(_legged(5, segs_per_leg=3)) is None                 # five legs: not reserved
    # ...and a many-limbed body with a neck and a tail on top is still many-limbed
    many = _legged(6, segs_per_leg=3)
    many.segments.append(GeneSegment(name="tail", parent="torso", joint_type="revolute", length_m=0.1))
    assert H.niche_of(many) == "many_limb"


def test_explain_names_the_wall_it_hit():
    """A boolean cannot steer a proposer. ``explain`` is the reason the wrapper feeds back."""
    why = H.explain(_legged(7, segs_per_leg=3))
    assert why["held_out"] and why["niche"] == "many_limb" and "7 limbs" in why["reason"]
    byp = H.explain(_legged(4), prompt="A nimble fox with a long bushy tail.")
    assert byp["held_out"] and byp["by_prompt"] and byp["niche"] is None
    assert byp["version"] == H.HELD_OUT_VERSION


def test_design_constraints_are_derived_from_the_predicates_not_restated():
    """The brief handed to the proposer must not be able to drift away from the guard enforcing it."""
    c = H.design_constraints()
    assert c["max_limbs"] == H.MANY_LIMB_MIN - 1
    assert set(c["avoid_niches"]) == set(H._RESERVED_NICHES)
    assert H.niche_of(_legged(c["max_limbs"], segs_per_leg=3)) is None            # at the brief's ceiling: fine
    assert H.niche_of(_legged(c["max_limbs"] + 1, segs_per_leg=3)) == "many_limb"  # one over: reserved
    assert set(c["avoid_prompts"]) == set(H.held_out_prompts())


@pytest.mark.slow
def test_the_factory_prompt_bank_does_not_compose_into_the_held_out_partition():
    """THE #281 regression. Composes the corpus factory's own legged targets through the product composer and
    asserts the guard does not reserve them — the measurement that was 18/18 held out before the fix.

    Deliberately runs the composer rather than a fixture: the defect lived in the interaction between what the
    composer builds (a quadruped WITH a neck and a tail) and what the guard counts, and no hand-built gene would
    have caught it."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from corpus_factory_night import _PROMPT_BANK

    from virturoid.services.morphology_composer import compose_robot
    bank = _PROMPT_BANK["legged"]
    sampled = [bank[0], bank[len(bank) // 3], bank[2 * len(bank) // 3], bank[-1]]
    held = []
    for prompt in sampled:
        g = compose_robot(prompt, llm=None, ensure_walkable=False)
        if H.is_held_out(g, prompt=prompt):
            held.append((prompt, H.explain(g, prompt=prompt)))
    assert not held, f"the factory's own bank still composes into reserved niches: {held}"


def test_compose_smoke_offline_paths_classify():
    """Integration: the offline composer builds bodies whose niche the guard reads without crashing."""
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    assert isinstance(H.body_key(g), str) and len(H.body_key(g)) == 12
    assert H.niche_of(g) in (None, "many_limb", "wheel_leg_hybrid")   # a real classification, never a crash
