"""CPU<->GPU parity for the morph-attention forward (the FiLM/topo-bias GPU port's correctness guarantee).

The GPU (Flax/jax) trainer and the CPU MorphPolicy MUST run byte-identical math, or a policy trained on the
GPU mis-controls on CPU. Both now call the SAME ``morph_forward.attention_forward``; this test proves (a) that
shared forward matches the shipped ``MorphPolicy.act`` (proprioception-only), and (b) it gives identical
results under numpy and jax — so ``trainer(jax) == act(numpy)`` transitively, for every FiLM/topo-bias combo.
"""

import unittest

import numpy as np

from virturoid.services.morph_forward import attention_forward
from virturoid.services.morph_policy import MorphPolicy
from virturoid.services.topo_pe import hop_distance_matrix

_HOP5 = np.array(hop_distance_matrix([-1, 0, 1, 0, 2]))          # 5 tokens on a small tree


def _att(F, H, rng, film, topo):
    a = {"We": rng.normal(0, 0.3, (F, H)), "be": rng.normal(0, 0.1, H),
         "Wq": rng.normal(0, 0.3, (H, H)), "Wk": rng.normal(0, 0.3, (H, H)),
         "Wv": rng.normal(0, 0.3, (H, H)), "Wo": rng.normal(0, 0.3, (H, H)),
         "Wh": rng.normal(0, 0.3, (H, 1)), "bh": rng.normal(0, 0.1, 1)}
    if film:
        a["Wfilm"] = rng.normal(0, 0.3, (F, 2 * H))
    if topo:
        a["Wtopo"] = rng.normal(0, 0.5, (9,))                   # 8 buckets + 1
    return a


def _att_percept(F, H, rng, film, topo):
    """``_att`` + the perception encoder (Wrange/brange) so the percept-token path can be exercised."""
    a = _att(F, H, rng, film, topo)
    a["Wrange"] = rng.normal(0, 0.3, (MorphPolicy.PERCEPT_DIM, H))
    a["brange"] = rng.normal(0, 0.1, H)
    return a


def _policy_from_att(att, F, H, film, topo):
    p = MorphPolicy(F, hidden=H, film=film, topo_bias=topo, topo_buckets=8)
    for k, v in att.items():
        p._arrs[k] = np.asarray(v, dtype=float)
    return p


class MorphGpuParityTests(unittest.TestCase):
    def test_shared_forward_matches_cpu_policy_act(self):
        F, H = 24, 16
        rng = np.random.default_rng(0)
        obs = rng.normal(0, 1, (5, F))
        for film in (False, True):
            for topo in (False, True):
                att = _att(F, H, rng, film, topo)
                p = _policy_from_att(att, F, H, film, topo)
                cpu = p.act(obs, hop=_HOP5 if topo else None)
                shared, _ = attention_forward(att, obs, H, xp=np, film=film,
                                              hop=_HOP5 if topo else None, topo_bias=topo)
                self.assertTrue(np.allclose(cpu, shared), f"mismatch film={film} topo={topo}")

    def test_shared_forward_matches_cpu_policy_act_with_percept(self):
        """The percept-token path (Wrange global perception/clock token) must ALSO byte-match MorphPolicy.act,
        so a GPU-trained conditioned/clocked policy deploys identically on CPU — the P1/P2/P6 enabler."""
        F, H = 24, 16
        rng = np.random.default_rng(2)
        obs = rng.normal(0, 1, (5, F))
        ranges = rng.normal(1.0, 0.3, 8)                        # 8 rangefinder distances
        cmd = [0.4, -0.2]                                       # [vx, wz] command (or a phase clock's sin/cos)
        PD = MorphPolicy.PERCEPT_DIM
        percept = np.concatenate([np.asarray(ranges), np.asarray(cmd)])[:PD]   # packed exactly as act packs it
        for film in (False, True):
            for topo in (False, True):
                att = _att_percept(F, H, rng, film, topo)
                p = _policy_from_att(att, F, H, film, topo)
                cpu = p.act(obs, ranges=ranges, cmd=cmd, hop=_HOP5 if topo else None)
                shared, _ = attention_forward(att, obs, H, xp=np, film=film,
                                              hop=_HOP5 if topo else None, topo_bias=topo, percept=percept)
                self.assertEqual(shared.shape, (obs.shape[0],))   # percept token carries no action
                self.assertTrue(np.allclose(cpu, shared), f"percept mismatch film={film} topo={topo}")

    def test_numpy_and_jax_forward_are_identical(self):
        try:
            import jax.numpy as jp
        except Exception:  # noqa: BLE001 - jax is the GPU-box dep; skip locally if absent
            self.skipTest("jax not installed")
        F, H = 24, 16
        rng = np.random.default_rng(1)
        obs = rng.normal(0, 1, (5, F))
        att = _att_percept(F, H, rng, film=True, topo=True)     # the hardest case: FiLM + topo + percept all on
        percept = rng.normal(0, 1, MorphPolicy.PERCEPT_DIM)
        np_out, np_pool = attention_forward(att, obs, H, xp=np, film=True, hop=_HOP5, topo_bias=True,
                                            percept=percept)
        jatt = {k: jp.asarray(v) for k, v in att.items()}
        j_out, j_pool = attention_forward(jatt, jp.asarray(obs), H, xp=jp, film=True,
                                          hop=jp.asarray(_HOP5), topo_bias=True, percept=jp.asarray(percept))
        self.assertTrue(np.allclose(np_out, np.asarray(j_out), atol=1e-5))   # the GPU==CPU control guarantee
        self.assertTrue(np.allclose(np_pool, np.asarray(j_pool), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
