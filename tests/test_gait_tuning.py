"""Un-gameable walk verdict + honest per-body gait tuning.

Two guards born from a measured near-miss (2026-07-09): a per-body gait tune found a hexapod config that cleared
every SCALAR gate (forward 0.38, upright 0.94) — but the render showed it REARING/pitching to fake the forward
metric. So (1) the credible-walk gate now also requires a LEVEL body (bounded roll/pitch), rejecting the lurch;
and (2) the tuner CACHES ONLY genuinely-credible configs, so it can never write a gamed gait onto a robot.
"""
import importlib.util
import math
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


def _quat_frames(*, roll_deg=0.0, pitch_deg=0.0):
    """A qpos trace [x,y,z, w,x,y,z] whose base quaternion has the given PURE roll or pitch (deg)."""
    a = math.radians(roll_deg) / 2.0
    b = math.radians(pitch_deg) / 2.0
    if pitch_deg:
        f = [0.0, 0.0, 0.3, math.cos(b), 0.0, math.sin(b), 0.0]
    else:
        f = [0.0, 0.0, 0.3, math.cos(a), math.sin(a), 0.0, 0.0]
    return [f, f, f]


class ClassifyGateTests(unittest.TestCase):
    """Pure — no MuJoCo. The un-gameable orientation gate on the walk verdict."""

    def _r(self, **kw):
        base = {"survived": True, "upright_frac": 1.0, "height_ratio": 0.9, "cadence": 8.0,
                "support_frac": 0.5, "forward": 0.8}
        base.update(kw)
        return base

    def test_clean_level_walk_is_credible(self):
        from virturoid.services.gait_quality import classify
        r = self._r(qpos_frames=_quat_frames(roll_deg=8))          # a clean quad-like trot
        self.assertEqual(classify(r), "CREDIBLE WALK")

    def test_pitching_lurch_is_rejected(self):
        # clears every scalar gate (forward/upright/cadence/support) but REARS ~25deg -> not a clean walk
        from virturoid.services.gait_quality import classify
        r = self._r(upright_frac=0.94, cadence=15.0, support_frac=0.47, forward=0.38,
                    qpos_frames=_quat_frames(pitch_deg=25))
        v = classify(r)
        self.assertNotEqual(v, "CREDIBLE WALK")
        self.assertIn("LURCH", v)

    def test_metric_only_call_keeps_scalar_verdict(self):
        # no qpos trace -> orientation can't be judged -> the scalar verdict stands (back-compat)
        from virturoid.services.gait_quality import classify
        self.assertEqual(classify(self._r()), "CREDIBLE WALK")


@unittest.skipUnless(_MUJOCO, "the gait tuner runs real MuJoCo rollouts")
class GaitTunerTests(unittest.TestCase):
    def _build(self, prompt, tune=False):
        from virturoid.services import session_state as S
        from virturoid.services.ai_native_tools import create_robot
        return S.get_robot(create_robot({"prompt": prompt, "tune_gait": tune})["robot_id"])

    def test_shipped_quad_is_a_credible_walk(self):
        """The quad the product SHIPS must be a credible walk at the DEPLOY horizon, under the op-point it
        ships with. This is the product's actual claim, and it is asserted against whatever body and whatever
        controller ``create_robot`` chose — not against one particular controller family.

        This used to be half of ``test_quad_stays_credible_and_short_circuits``, which asked ``tune_crawl_gait``
        (the 6-row scripted grid, re-centred on ``default_gait_for``) whether the built quad was credible. That
        was a valid question only while ``create_robot`` composed with ``ensure_walkable=True``, i.e. while it
        FANNED the body's stance before anyone measured it — the fanned template is the one body the shipped
        freq-1.5 op-point was hand-tuned for ([[one-controller-one-body-is-the-blocker]]). ``create_robot`` now
        defers that decision: it composes the AUTHORED body, grounds it, fits an op-point to IT, and only then
        asks whether it can walk. So the subject of the old assertion silently changed from the fanned template
        to the customer's own dog, and the authored dog does not walk at the template's op-point — that IS the
        co-tuning wall.

        REWRITTEN 2026-08-02 (task #267). This asserted that the shipped body CARRIES ``gait_params``, which is
        an implementation detail and not the claim: a body whose SHIPPED DEFAULT already walks is deliberately
        never searched (``fit_gait_for_body`` short-circuits) and correctly carries no fitted op-point. Once the
        walk verdict was moved to the settling horizon that branch started firing for this prompt, and the test
        went red for a body that MEASURABLY WALKS — +6.594 m over 6000 steps at a travel rate of
        1.342 -> 1.166 -> 1.096 m per 1000 steps, i.e. still walking at the end.

        So it now asserts THE CLAIM, and asserts it at the horizon the claim is made at. The old form could not
        have caught the defect it was written around: the grounded authored cat cleared exactly this assertion
        at 1500 steps while falling at step 2014.
        """
        from virturoid.services.gait_flywheel import _SETTLE_STEPS
        from virturoid.services.gait_quality import classify, settling
        from virturoid.services.morph_policy import crawl_gait_rollout
        gene = self._build("a quadruped robot dog that walks")
        fit = (gene.metadata or {}).get("gait_fit") or {}
        self.assertTrue(fit, "a shipped legged body must disclose how its operating point was decided")
        # EITHER a fitted op-point, OR an explicit finding that the shipped default already walks. What is not
        # allowed is shipping a walk claim nobody measured.
        self.assertTrue((gene.metadata or {}).get("gait_params") or not fit.get("searched"),
                        f"searched but adopted nothing, and still shipping: {fit.get('reason')}")
        r = crawl_gait_rollout(gene, steps=_SETTLE_STEPS, record_qpos=True)   # reads metadata['gait_params']
        v = classify(r)
        self.assertTrue(v.startswith("CREDIBLE"),
                        f"the shipped quad must still be walking at step {_SETTLE_STEPS}, got {v}")
        s = settling(r)
        self.assertIsNotNone(s, "the shipped claim must be made at a horizon that can be judged")
        self.assertTrue(s["holds_rate"],
                        f"a walk whose travel rate decays has not walked, it has drifted: {s['rates']}")

    def test_tuner_short_circuits_on_a_body_whose_default_walks(self):
        """A quad that already walks must keep its DEFAULT op-point — the tuner must not churn a working gait.

        This is the other half of the old ``test_quad_stays_credible_and_short_circuits``, and the property it
        was really guarding: ``tune_crawl_gait`` evaluates its grid ROW 0 (the body's own ``default_gait_for``)
        first and, when that row is robustly credible, breaks immediately and caches exactly those numbers. The
        tuner itself is unchanged; only the body ``create_robot`` hands back moved (see the test above), so the
        contract is asserted here on the body it was written for — a FANNED quad, which is what
        ``compose_robot(ensure_walkable=True)`` builds and what ``create_robot`` used to build inline.

        The default is derived from the body's own size rather than being the literal 1.5 Hz of one reference
        robot (``morph_policy.default_gait_for``): stride frequency scales as sqrt(g/L), so a slightly
        shorter-legged dog settles a little faster. Pinning the constant tested the reference body, not the
        behaviour; pinning "it short-circuits to THIS body's default" tests what the short-circuit is for.
        """
        from virturoid.services.gene_build import ground_and_repair
        from virturoid.services.morph_policy import default_gait_for, tune_crawl_gait
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot("a quadruped robot dog that walks", ensure_walkable=True)
        ground_and_repair(gene)                                  # judge it at the mass it would ship at
        best = tune_crawl_gait(gene)
        self.assertTrue(best["verdict"].startswith("CREDIBLE"))
        self.assertAlmostEqual((gene.metadata or {}).get("gait_params", {}).get("freq"),
                               default_gait_for(gene)["freq"], places=3)

    def test_tuner_never_caches_a_noncredible_gait(self):
        # a hexapod has no robustly-CLEAN scripted crawl -> the tuner must cache NOTHING (never a gamed lurch)
        from virturoid.services.morph_policy import tune_crawl_gait
        gene = self._build("a hexapod that walks")
        best = tune_crawl_gait(gene)
        if best.get("untuned"):
            self.assertIsNone((gene.metadata or {}).get("gait_params"))
        else:
            self.assertTrue(best["verdict"].startswith("CREDIBLE"))   # if it cached, it MUST be credible

    def test_crawl_rollout_honors_cached_params(self):
        # an explicit cache flows into the rollout (deploy uses the tuned op-point); explicit kwargs still win
        from virturoid.services.morph_policy import crawl_gait_rollout
        gene = self._build("a quadruped robot dog that walks")
        gene.metadata = dict(gene.metadata or {}, gait_params={"freq": 0.7, "hip_amp": 0.5, "knee_amp": 0.4})
        r_cached = crawl_gait_rollout(gene, steps=200)
        r_explicit = crawl_gait_rollout(gene, steps=200, freq=1.5, hip_amp=0.9, knee_amp=1.0)
        # different drive params -> different forward travel (proves the cache is actually read)
        self.assertNotAlmostEqual(r_cached["forward"], r_explicit["forward"], places=3)


if __name__ == "__main__":
    unittest.main()
