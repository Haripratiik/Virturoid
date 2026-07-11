"""Flywheel breakthrough R0 — retrieval hygiene: exact-structure cache + leg-count hard filter.

Measured failure (flywheel_breakthrough_plan §3.I2): at cosine 0.9939 a hexapod recalled a QUADRUPED's gait over
the real hexapod (0.9291) — the 29-D structural vector blurs leg count, so a 4-leg gait seeded a 6-leg body.
These tests prove the two corrections in ``gait_flywheel._structural_recall``.
"""
from __future__ import annotations

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.gait_flywheel import LOCOMOTION, _structural_recall, recall_gait


def _legged(n_legs, tag, *, seg_per_leg=3, total_len=2.0):
    # realistic legs: (seg_per_leg-1) revolute joints + a WELDED foot pad (joint None) — so _leg_count sees them
    segs = [GeneSegment(name="torso", parent=None, joint_type=None, length_m=0.5, radius_m=0.1, mass_kg=2.0)]
    for i in range(n_legs):
        prev = "torso"
        for j in range(seg_per_leg):
            nm = f"leg{i}_{j}"
            welded_foot = j == seg_per_leg - 1
            segs.append(GeneSegment(name=nm, parent=prev, joint_type=(None if welded_foot else "revolute"),
                                    length_m=total_len / (n_legs * seg_per_leg), radius_m=0.02, mass_kg=0.1))
            prev = nm
    segs[-1].is_end_effector = True
    return RobotGene(id=tag, species=f"t.{tag}", robot_class="quadruped", segments=segs, base_mount="free",
                     end_effector_type="none")


def _bank(db, gene, params):
    """Bank a gait skill for ``gene`` with inline body meta (mirrors bank_gait's vector write)."""
    import types

    from virturoid.services.gait_flywheel import bank_gait
    r = types.SimpleNamespace(best_survived=True, best_forward=0.8, best_credible=True,
                              best_height_ratio=0.8, best_params=params)
    return bank_gait(db, gene, r)


def test_leg_count_filter_blocks_the_measured_hex_to_quad_blur(tmp_path):
    from virturoid.services.memory_db import MemoryDB
    quad = _legged(4, "quad")
    hexb = _legged(6, "hexbody")
    with MemoryDB(tmp_path / "m.db") as db:
        _bank(db, quad, {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5})
        _bank(db, hexb, {"freq": 0.9, "hip_amp": 0.4, "knee_amp": 0.5, "duty": 0.5, "kp": 60.0, "kd": 3.0})
        # a FRESH 6-leg body (distinct from the banked hex) must recall the HEX gait, never the quad
        q6 = _legged(6, "fresh_hex", total_len=2.4)
        params = recall_gait(db, q6)
        assert params is not None, "a leg-count-matching gait exists and must be recalled"
        assert abs(params["freq"] - 0.9) < 1e-6, f"recalled the wrong-leg-count gait: {params}"
        assert params["duty"] == 0.5                          # the hex gait's signature, not the quad's 0.25


def test_exact_structure_cache_returns_verbatim(tmp_path):
    from virturoid.services.memory_db import MemoryDB
    quad = _legged(4, "quad")
    with MemoryDB(tmp_path / "m.db") as db:
        p = {"freq": 1.42, "hip_amp": 0.7, "knee_amp": 0.8, "duty": 0.3, "kp": 40.0, "kd": 2.0}
        _bank(db, quad, p)
        # the SAME structural body recalls its own gait as an exact-cache hit
        hit = _structural_recall(db, _legged(4, "quad_rebuild"), LOCOMOTION)
        assert hit is not None and hit[2] == "exact_cache"
        assert hit[0]["freq"] == 1.42


def test_no_structural_match_returns_none(tmp_path):
    from virturoid.services.memory_db import MemoryDB
    with MemoryDB(tmp_path / "m.db") as db:
        _bank(db, _legged(4, "quad"), {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 32.0, "kd": 1.5})
        # a 9-leg body has no leg-count match and no exact structure -> honest None (search, don't borrow wrong)
        assert recall_gait(db, _legged(9, "nonapod")) is None
