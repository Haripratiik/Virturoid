"""CPU params-mode flywheel (Theme 2): ``transfer_train_morph`` recalls a banked MorphPolicy, WARM-STARTS
ES from it, trains, and re-banks — so the next related build learns from the last. The orchestration is
proven fast with injected deps; the warm-start *mechanism* (init_policy actually seeds the params) is
proven against real MuJoCo on a tiny body."""

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class _FakePolicy:
    """Stand-in returned by a fake trainer — only needs ``to_npz`` for the orchestration test."""
    def __init__(self, warm):
        self.warm = warm

    def to_npz(self, path, score: float = 0.0):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(f"fake:{self.warm}:{score}", encoding="utf-8")


class TransferOrchestrationTests(unittest.TestCase):
    """No MuJoCo: prove the read->train->write wiring and the warm-start decision."""

    def test_cold_then_warm_across_two_builds(self):
        from virturoid.services import policy_flywheel as pf

        calls = []

        def fake_train(genes, *, feature_dim, generations, pop, steps, seed, init_policy):
            calls.append({"init_policy": init_policy})
            return _FakePolicy(warm=init_policy is not None), [0.1, 0.4, 0.7]

        banked_policy = {}  # one slot shared across the two builds, like the real DB bank

        def fake_recall(gene, db, *, task_type="locomotion"):
            return banked_policy.get("p")

        def fake_bank(npz_path, gene, db, *, task_type="locomotion"):
            banked_policy["p"] = _FakePolicy(warm=True)   # now there is a policy to recall next time
            return {"skill_id": f"morph_legged_{task_type}"}

        with tempfile.TemporaryDirectory() as tmp:
            common = dict(db=None, feature_dim=24, generations=1, pop=2, steps=10,
                          train_fn=fake_train, recall_fn=fake_recall, bank_fn=fake_bank)
            first = pf.transfer_train_morph(["bodyA"], params_path=Path(tmp) / "a.npz", **common)
            second = pf.transfer_train_morph(["bodyB"], params_path=Path(tmp) / "b.npz", **common)
            banked_files = (Path(tmp) / "a.npz").exists() and (Path(tmp) / "b.npz").exists()

        self.assertFalse(first["warm_started"])           # nothing banked -> cold start
        self.assertIsNone(calls[0]["init_policy"])
        self.assertTrue(second["warm_started"])           # recalled the first build's policy
        self.assertIsNotNone(calls[1]["init_policy"])     # ...and fed it to the trainer as init
        self.assertEqual(second["skill_id"], "morph_legged_locomotion")
        self.assertTrue(banked_files)                     # both runs banked a params file

    def test_reported_final_is_the_elite_fitness_not_last_generation(self):
        # The ES is elitist (returns the BEST-seen policy), so transfer_train_morph must report the elite's
        # fitness (max history), not the last generation's — else it understates the policy it actually banked.
        from virturoid.services import policy_flywheel as pf

        def peaky_train(genes, *, feature_dim, generations, pop, steps, seed, init_policy):
            return _FakePolicy(warm=False), [0.10, 0.90, 0.40]   # peak in the MIDDLE, lower at the end

        with tempfile.TemporaryDirectory() as tmp:
            res = pf.transfer_train_morph(
                ["b"], db=None, feature_dim=24, generations=2, pop=2, steps=5,
                params_path=Path(tmp) / "p.npz", train_fn=peaky_train,
                recall_fn=lambda *a, **k: None, bank_fn=lambda *a, **k: {"skill_id": "s"})
        self.assertAlmostEqual(res["baseline"], 0.10)     # gen-0
        self.assertAlmostEqual(res["final"], 0.90)        # the elite (banked) fitness, NOT 0.40 (last gen)


def _radial(n):
    from virturoid.services.morphology_builder import MorphologyBuilder
    b = MorphologyBuilder("legged", base_mount="free", species=f"r{n}")
    b.base("torso", shape="box", length=0.08, radius=0.18, mass=3.0)
    for i in range(n):
        a = 2 * math.pi * i / n
        b.limb(f"leg{i}", [{"length": 0.12, "axis": (0, 1, 0), "torque": 14.0},
                           {"length": 0.12, "axis": (0, 1, 0), "torque": 12.0}],
               parent="torso", mount_offset=(0.16 * math.cos(a), 0.16 * math.sin(a), -0.04),
               mount_euler=(math.pi, 0, 0))
    return b.build()


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class WarmStartMechanismTests(unittest.TestCase):
    """Real physics: ``init_policy`` must actually seed the starting params (generation-0 fitness == the
    banked policy's fitness), so warm-start begins from prior learning, not from random weights."""

    def test_init_policy_seeds_the_starting_params(self):
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.morph_trainer import forward_score, train_morph_es
        from virturoid.services.policy_flywheel import _feature_dim_for

        body = _radial(4)
        fdim = _feature_dim_for(body)
        prior = MorphPolicy(fdim, seed=3)                 # a specific "banked" policy

        # generations=0 -> history is just the baseline fitness of the (warm-started) params.
        _, warm_hist = train_morph_es([body], feature_dim=fdim, generations=0, pop=2,
                                      steps=120, init_policy=prior)
        self.assertAlmostEqual(warm_hist[0], forward_score(body, prior, steps=120), places=5)

        _, cold_hist = train_morph_es([body], feature_dim=fdim, generations=0, pop=2,
                                      steps=120, seed=7)   # fresh random params, different seed
        cold_baseline = forward_score(body, MorphPolicy(fdim, seed=7), steps=120)
        self.assertAlmostEqual(cold_hist[0], cold_baseline, places=5)
        # The warm baseline comes from the banked policy, the cold one from random init -> distinct.
        self.assertNotAlmostEqual(warm_hist[0], cold_hist[0], places=6)

    def test_elitism_never_returns_worse_than_the_warm_start_seed(self):
        # The flywheel must be monotonic: warm-starting from a banked policy and training must hand back a
        # policy at least as good as the seed, never a regression (ES can drift down over a generation).
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.morph_trainer import forward_score, train_morph_es
        from virturoid.services.policy_flywheel import _feature_dim_for

        body = _radial(4)
        fdim = _feature_dim_for(body)
        seed_policy = MorphPolicy(fdim, seed=3)
        seed_fit = forward_score(body, seed_policy, steps=120)
        trained, hist = train_morph_es([body], feature_dim=fdim, generations=4, pop=6,
                                       steps=120, init_policy=seed_policy, seed=1)
        # The returned (elite) policy is >= the seed and >= every generation's fitness in the history.
        self.assertGreaterEqual(forward_score(body, trained, steps=120) + 1e-6, seed_fit)
        self.assertGreaterEqual(forward_score(body, trained, steps=120) + 1e-6, max(hist))


if __name__ == "__main__":
    unittest.main()
