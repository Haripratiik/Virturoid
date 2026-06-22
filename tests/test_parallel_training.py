"""Parallel ES training: the population is evaluated across CPU cores (each candidate's rollout is an
independent process), so the same budget trains ~Ncores faster. This MUST stay bit-for-bit identical to the
serial path (the eps are drawn in the parent) — a stronger policy in the same wall-clock, not a different one.

Run in a SUBPROCESS (with a real ``__main__`` guard) because multiprocessing spawn can't fan out from inside
the pytest runner. See [[task-effectiveness-loop]]."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None

_CHECK = textwrap.dedent('''
    if __name__ == "__main__":
        import time, warnings
        warnings.filterwarnings("ignore")
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_trainer import train_morph_es
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a robot dog that walks", llm=None)
        fd = encode_robot(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))).feature_dim
        t0 = time.time(); _, hs = train_morph_es([g], feature_dim=fd, generations=6, pop=16, steps=180, seed=1, workers=1); ts = time.time() - t0
        t0 = time.time(); _, hp = train_morph_es([g], feature_dim=fd, generations=6, pop=16, steps=180, seed=1, workers=4); tp = time.time() - t0
        assert abs(hs[-1] - hp[-1]) < 1e-9, f"parallel != serial: {hs[-1]} vs {hp[-1]}"
        print(f"PARALLEL_OK identical serial={ts:.1f}s parallel={tp:.1f}s speedup={ts/tp:.1f}x")
''')


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ParallelTrainingTests(unittest.TestCase):
    def test_parallel_es_is_deterministic_and_faster(self):
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "parallel_es_check.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(_CHECK)
            env = {**os.environ, "PYTHONPATH": "src", "VIRTUROID_NO_LOCAL_ENV": "1"}
            r = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env, timeout=400)
            self.assertIn("PARALLEL_OK", r.stdout, msg=f"stdout={r.stdout}\nstderr={r.stderr[-1500:]}")


if __name__ == "__main__":
    unittest.main()
