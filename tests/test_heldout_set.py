"""WS-A.1 — the frozen held-out set is the leak-free foundation (master_plan_v6 §10.4)."""
from __future__ import annotations

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


def test_compose_smoke_offline_paths_classify():
    """Integration: the offline composer builds bodies whose niche the guard reads without crashing."""
    from virturoid.services.morphology_composer import compose_robot
    g = compose_robot("a simple four-legged walking robot", ensure_walkable=True)
    assert isinstance(H.body_key(g), str) and len(H.body_key(g)) == 12
    assert H.niche_of(g) in (None, "many_limb", "wheel_leg_hybrid")   # a real classification, never a crash
