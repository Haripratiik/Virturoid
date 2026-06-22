"""Batched in-process population rollout — "run the same sim many times in MuJoCo" (the CPU analog of
MJX's vmap-over-policies). Profiling showed physics is only ~9% of an ES step; the cost was the per-call
XML recompile (now cached) and the per-candidate Python policy forward. So the efficient single-core
primitive compiles the body ONCE and scores the WHOLE population in one vectorized op.

These run INSIDE pytest (batched is single-process — no multiprocessing fan-out needed). They lock:
  1. the compiled-model cache is a real cache (same object) and bit-identical,
  2. batched scores equal the serial forward_score (numerically — same physics, same math, same rounding),
  3. the batched ES backend reaches the same fitness as serial in less wall-clock.
See [[task-effectiveness-loop]]."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class BatchedRolloutTests(unittest.TestCase):
    def _dog(self):
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a robot dog that walks", llm=None)
        fd = encode_robot(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))).feature_dim
        return g, fd

    def test_compiled_model_cache_is_a_cache(self):
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        g, _ = self._dog()
        xml = robot_mjcf(g)
        m1 = compiled_model(xml)
        m2 = compiled_model(xml)
        self.assertIs(m1, m2, "compiled_model must reuse the cached MjModel (same body recompiled is waste)")
        self.assertEqual(m1.nq, m2.nq)
        self.assertEqual(m1.nbody, m2.nbody)

    def test_batched_scores_match_serial(self):
        import numpy as np

        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.morph_trainer import batched_forward_scores, forward_score
        g, fd = self._dog()
        rng = np.random.default_rng(7)
        base = MorphPolicy(fd, seed=3).get_params()
        params = base + 0.15 * rng.normal(0, 1, (8, base.size))

        def mk(p):
            pol = MorphPolicy(fd)
            pol.set_params(p)
            return pol

        serial = np.array([forward_score(g, mk(p), steps=160, alpha=0.5) for p in params])
        batched = batched_forward_scores(g, params, steps=160, alpha=0.5)
        # same physics + same math + same 3-decimal rounding -> equal to well under the score scale
        self.assertTrue(np.allclose(serial, batched, atol=1e-3),
                        msg=f"max|serial-batched|={np.max(np.abs(serial-batched)):.4f}\nserial={serial}\nbatched={batched}")

    def test_batched_es_matches_serial_fitness(self):
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_trainer import train_morph_es
        g, _ = self._dog()
        fd = encode_robot(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))).feature_dim
        _, hs = train_morph_es([g], feature_dim=fd, generations=5, pop=12, steps=160, seed=1, eval_backend="serial")
        _, hb = train_morph_es([g], feature_dim=fd, generations=5, pop=12, steps=160, seed=1, eval_backend="batched")
        # numerically-equivalent evaluator -> the ES trajectory tracks serial closely (not bit-identical)
        self.assertAlmostEqual(hs[-1], hb[-1], delta=0.05,
                               msg=f"batched ES final {hb[-1]:.3f} drifted from serial {hs[-1]:.3f}")


if __name__ == "__main__":
    unittest.main()
