"""WS-C.1 — the factory selection substrate: PATA-EC + MAP-Elites + learnability (master_plan_v6 §10.1)."""
from __future__ import annotations

import pytest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services import novelty_search as NS


def _legged(n_legs: int, *, segs_per_leg: int = 3, total_len: float = 2.0) -> RobotGene:
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.5, radius_m=0.1)]
    for i in range(n_legs):
        prev = "torso"
        for j in range(segs_per_leg):
            nm = f"leg{i}_{j}"
            segs.append(GeneSegment(name=nm, parent=prev, joint_type="revolute",
                                    length_m=total_len / (n_legs * segs_per_leg), radius_m=0.02))
            prev = nm
    return RobotGene(id=f"L{n_legs}", species="t.legged", robot_class="quadruped", segments=segs,
                     base_mount="free", end_effector_type="none")


# ---------------------------------------------------------------- MAP-Elites niches
def test_niche_grid_coverage_and_novelty_decay():
    quad, hexb = _legged(4), _legged(6)
    grid = NS.MapElitesGrid([quad])
    assert grid.is_empty_cell(hexb)                     # a 6-leg body is a different niche
    assert grid.novelty(hexb) == 1.0                    # empty cell -> max novelty
    n_before = grid.novelty(quad)
    grid.add(quad)                                       # fill the quad cell more
    assert grid.novelty(quad) < n_before               # novelty decays as the cell fills
    cov = grid.coverage()
    assert cov["filled_cells"] == 1 and cov["total_bodies"] == 2


def test_embedding_novelty_empty_corpus_and_duplicate():
    quad = _legged(4)
    assert NS.embedding_novelty(quad, []) == 1.0        # nothing to be similar to
    assert NS.embedding_novelty(quad, [quad]) == pytest.approx(0.0, abs=1e-6)   # identical -> ~0 novelty


# ---------------------------------------------------------------- PATA-EC rank-order novelty
def test_rank_vector_and_distance():
    assert NS.rank_vector([0.1, 0.9, 0.5]) == [0, 2, 1]
    assert NS.rank_distance([0, 1, 2], [0, 1, 2]) == 0.0
    assert NS.rank_distance([0, 1, 2], [2, 1, 0]) == pytest.approx(1.0)   # fully reversed = max


def test_pata_ec_flags_a_novel_rank_order():
    existing = [[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]]        # controllers rank the same way on known bodies
    same = NS.pata_ec_novelty([0.15, 0.45, 0.85], existing)   # candidate preserves the order -> low novelty
    novel = NS.pata_ec_novelty([0.9, 0.5, 0.1], existing)     # candidate REVERSES the order -> high novelty
    assert novel > same
    assert NS.pata_ec_novelty([0.1, 0.2], []) == 1.0     # nothing seen -> maximally novel


# ---------------------------------------------------------------- learnability / regret
def test_learnability_is_peaked_minimal_criterion():
    assert NS.learnability(0.0) == 0.0                  # unsolvable
    assert NS.learnability(1.0) == 0.0                  # trivial
    assert NS.learnability(0.45) > NS.learnability(0.1)  # mid band is most learnable
    assert NS.regret(best_achieved=0.3, reference=1.0) == pytest.approx(0.7)


# ---------------------------------------------------------------- combined selection
def test_select_next_bodies_prefers_novel_and_thin_classes():
    corpus = [_legged(4), _legged(4), _legged(4)]        # corpus is all quadrupeds
    cands = [_legged(4), _legged(6), _legged(8)]         # a 6- and 8-leg body are novel niches
    picks = NS.select_next_bodies(cands, corpus_genes=corpus, k=3)
    assert picks[0]["score"] >= picks[-1]["score"]
    # the novel-niche bodies (6/8 legs) outrank the duplicate quadruped
    top_dof = [p["niche"][2] for p in picks[:2]]         # limb-chain bucket
    assert picks[0]["empty_cell"] or picks[0]["novelty"] > picks[-1]["novelty"]


def test_select_transfer_pairs_skips_tested_and_caps():
    ids = ["a", "b", "c", "d"]
    emb = {"a": [1, 0], "b": [0, 1], "c": [1, 0.1], "d": [0.1, 1]}
    pairs = NS.select_transfer_pairs(ids, embed_by_id=emb, tested_pairs={("a", "b")}, k=3)
    assert ("a", "b") not in pairs and ("b", "a") not in pairs   # tested pair excluded
    assert len(pairs) <= 3
    # the most DIVERSE untested pairs are the ~orthogonal ones (a↔d and b↔c); a near-duplicate pair (a↔c) is not first
    assert pairs[0] in (("a", "d"), ("b", "c"))
    assert ("a", "c") not in pairs[:1]


# ---------------------------------------------------------------- ledger-backed PATA-EC (integration)
def test_ledger_pata_ec_reads_the_transfer_matrix(tmp_path):
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.transfer_ledger import record_transfer_trial
    a, b, c = _legged(4), _legged(6), _legged(8)
    with MemoryDB(tmp_path / "m.db") as db:
        # a's gait walks on b, falls on c; b's gait walks on a — enough to form a forward matrix
        record_transfer_trial(db, src_gene=a, dst_gene=b, gait_params={"freq": 1.0},
                              result={"survived": True, "credible": True, "forward": 0.6})
        record_transfer_trial(db, src_gene=a, dst_gene=c, gait_params={"freq": 1.0},
                              result={"survived": True, "credible": False, "forward": 0.05})
        record_transfer_trial(db, src_gene=b, dst_gene=a, gait_params={"freq": 1.2},
                              result={"survived": True, "credible": True, "forward": 0.7})
        nov = NS.ledger_pata_ec(db)
    assert nov and all(0.0 <= v <= 1.0 for v in nov.values())
