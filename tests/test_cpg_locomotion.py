"""Trot-CPG locomotion prior (dossier P1) — guards the mechanism that turns the CPU recipe's slide/collapse into
an upright stepping walk. The CPG injects a diagonal-trot stepping rhythm into the PD target so CPU-ES learns only
a propulsion RESIDUAL (it cannot discover a gait from a stand-still). These are FAST, training-free assertions on
the CPG prior itself (deterministic single rollouts); the full learned walk is exercised by test_task_effectiveness.
"""

import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class TrotCpgPriorTests(unittest.TestCase):
    def _quad(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a quadruped robot that walks", llm=None)

    def test_cpg_keeps_the_quad_upright_where_the_scalar_recipe_collapses(self):
        # Same untrained policy (seed 0), same body: the CPG prior must hold the quad UPRIGHT and SURVIVING,
        # giving ES a stable stepping base instead of the scalar recipe's slide-then-collapse.
        from virturoid.services.morph_policy import CPG_DEFAULT, recipe_rollout_morph
        g = self._quad()
        scalar = recipe_rollout_morph(g, None, steps=900, seed=0, cpg=None)
        cpg = recipe_rollout_morph(g, None, steps=900, seed=0, cpg=CPG_DEFAULT)
        self.assertGreater(cpg["height_ratio"], scalar["height_ratio"],
                           f"CPG must hold the body more upright (cpg {cpg['height_ratio']} vs scalar {scalar['height_ratio']})")
        self.assertTrue(cpg["survived"], "the CPG prior must keep an untrained quad upright for the whole horizon")

    def test_cpg_generalizes_to_a_six_leg_body(self):
        # the N-leg generalization: a hexapod (leg{i}_{j} naming, 6 legs) must ALSO get a stepping rhythm
        # (alternating-tripod), not fall back to the scalar-recipe shuffle — the locomotion-generality fix.
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import (CPG_DEFAULT, _trot_cpg_tokens, compiled_model,
                                                     robot_mjcf)
        from virturoid.services.steerable_body import steerable_quadruped
        gr = encode_robot(compiled_model(robot_mjcf(steerable_quadruped(n_legs=6, dofs_per_leg=3))))
        amp, phase, gate = _trot_cpg_tokens(compiled_model(robot_mjcf(
            steerable_quadruped(n_legs=6, dofs_per_leg=3))), gr, CPG_DEFAULT)
        self.assertTrue(gate, "a 6-leg body must be a recognized legged body (N-leg CPG generalization)")
        self.assertGreaterEqual(sum(1 for a in amp if a != 0), 6, "each leg's hip+knee should oscillate")

    def test_dr_robustness_eval_and_perturb_hook_off_is_identical(self):
        # SIM2REAL: recipe_rollout_morph's model_perturb hook is opt-in — None leaves the rollout byte-identical
        # (no regression). recipe_robustness runs N rollouts under randomized dynamics and reports a valid
        # walk-survival rate in [0, 1] (the CPU mirror of the GPU --dr robustness eval).
        from virturoid.services.morph_policy import recipe_robustness, recipe_rollout_morph
        g = self._quad()
        a = recipe_rollout_morph(g, None, steps=150, seed=0)
        b = recipe_rollout_morph(g, None, steps=150, seed=0, model_perturb=None)
        self.assertEqual(a["forward"], b["forward"], "model_perturb=None must leave the rollout byte-identical")
        self.assertEqual(a["alive"], b["alive"])
        rb = recipe_robustness(g, None, n=2, steps=150, seed=0)
        self.assertEqual(rb["n"], 2)
        self.assertTrue(0.0 <= rb["survival_rate"] <= 1.0)
        self.assertIn("mean_forward", rb)

    def test_cpg_gate_is_quadruped_only(self):
        # The trot-CPG is gated to 4-leg bodies; a manipulator arm must get NO CPG (amp all zero) so non-quads
        # fall back to the scalar recipe unchanged.
        import mujoco

        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import CPG_DEFAULT, _trot_cpg_tokens, robot_mjcf
        from virturoid.services.morphology_composer import compose_robot
        arm = compose_robot("a tabletop robot arm", llm=None)
        m = mujoco.MjModel.from_xml_string(robot_mjcf(arm))
        amp, _phase, gate = _trot_cpg_tokens(m, encode_robot(m), CPG_DEFAULT)
        self.assertFalse(gate, "an arm is not a quadruped — the CPG gate must be off")
        self.assertEqual(0.0, float(abs(amp).max()), "no CPG amplitude on a non-quad body")

    def test_phase_obs_flag_round_trips_through_npz(self):
        # P1: a phase-clock-trained policy is wrong without the clock fed at deploy, so the flag MUST persist
        # through bank/reload (meta[9]) — the deploy==train guarantee, exactly like the CPG/decimation flags.
        import mujoco
        import os
        import tempfile

        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import CPG_DEFAULT, MorphPolicy, robot_mjcf
        g = self._quad()
        fd = encode_robot(mujoco.MjModel.from_xml_string(robot_mjcf(g))).feature_dim
        self.assertFalse(MorphPolicy(fd, seed=0).phase_obs, "phase_obs defaults OFF (perception-blind)")
        pol = MorphPolicy(fd, seed=0); pol.cpg = CPG_DEFAULT; pol.phase_obs = True
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.npz"); pol.to_npz(p, 0.0, normalizer=None)
            self.assertTrue(MorphPolicy.from_npz(p).phase_obs, "a phase-clock policy must reload with phase_obs ON")
            off = MorphPolicy(fd, seed=0); off.cpg = CPG_DEFAULT           # phase_obs left False -> meta[9]=0.0
            p2 = os.path.join(d, "off.npz"); off.to_npz(p2, 0.0, normalizer=None)
            self.assertFalse(MorphPolicy.from_npz(p2).phase_obs, "a non-clock policy must reload with phase_obs OFF")
            # WS8/P6: the velocity-command flag (meta[10]) must ALSO round-trip so a conditioned policy deploys
            # tracking (recipe_rollout_morph reads policy.command_conditioned). Independent of phase_obs (meta[9]).
            self.assertFalse(MorphPolicy(fd, seed=0).command_conditioned, "command_conditioned defaults OFF")
            self.assertFalse(MorphPolicy.from_npz(p2).command_conditioned, "an unconditioned policy reloads OFF")
            cc = MorphPolicy(fd, seed=0); cc.cpg = CPG_DEFAULT; cc.command_conditioned = True
            p3 = os.path.join(d, "cc.npz"); cc.to_npz(p3, 0.0, normalizer=None)
            self.assertTrue(MorphPolicy.from_npz(p3).command_conditioned, "a command-conditioned policy reloads ON")

    def test_phase_obs_deploy_feeds_the_gait_clock(self):
        # P1 deploy parity: when phase_obs is ON, recipe_rollout_morph must FEED the [sin,cos] gait clock through
        # the percept token so the policy's actions depend on the cycle phase; when OFF it is perception-blind.
        # Amplify Wrange so the clock's effect on the (untrained) residual is unmistakable, then assert the two
        # deploys diverge — proving the clock is actually plumbed AND correctly gated by the flag.
        import mujoco

        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import CPG_DEFAULT, MorphPolicy, recipe_rollout_morph, robot_mjcf
        g = self._quad()
        fd = encode_robot(mujoco.MjModel.from_xml_string(robot_mjcf(g))).feature_dim
        pol = MorphPolicy(fd, seed=0); pol.cpg = CPG_DEFAULT
        pol._arrs["Wrange"] = pol._arrs["Wrange"] * 5.0                    # make the clock's effect clearly visible
        pol.phase_obs = True
        on = recipe_rollout_morph(g, pol, steps=200, seed=0, cpg=CPG_DEFAULT)
        pol.phase_obs = False
        off = recipe_rollout_morph(g, pol, steps=200, seed=0, cpg=CPG_DEFAULT)
        self.assertNotEqual(on["forward"], off["forward"],
                            "the fed phase clock must change the deployed trajectory (and OFF must not feed it)")

    def test_fanned_quad_walks_under_the_crawl_gait(self):
        # BREAKTHROUGH (walking-breakthrough-abduction): a FANNED wide-stance quad walks UPRIGHT under the
        # statically-stable CRAWL gait (one leg lifts at a time -> 3 always planted -> the CoM stays inside the
        # support triangle -> it cannot roll). The diagonal TROT rolls this same tippy body over for want of
        # active balance; the fan (wide stance) + crawl (static support) together give a real open-loop walk.
        from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
        from virturoid.services.morph_policy import crawl_gait_rollout
        wide = crawl_gait_rollout(build_from_anatomy(_generic_legged_graph(n_pairs=2, girth=0.22, fan=True)),
                                  steps=1200)
        self.assertTrue(wide["survived"], "the fanned quad must survive the whole horizon under the crawl")
        self.assertGreater(wide["forward"], 0.3, "must WALK forward (not creep in place)")
        self.assertGreater(wide["support_frac"], 0.5, "must genuinely STEP (partial support), not slide/stand")
        self.assertGreater(wide["height_ratio"], 0.6, "must stay TALL/upright, not collapse or roll")

    def test_fan_option_defaults_off_and_widens_when_on(self):
        # The fan is OPT-IN: default OFF leaves the generic quad graph byte-identical (narrow 'down' legs); ON
        # flips the quad leg aim to 'down_out' (wide roll-stable stance). Guards the byte-identical guarantee.
        from virturoid.services.anatomy_compiler import _generic_legged_graph
        a = _generic_legged_graph(n_pairs=2)
        self.assertEqual(a, _generic_legged_graph(n_pairs=2, fan=False), "fan defaults OFF -> graph unchanged")
        aims_off = [p["aim"] for p in a["parts"] if p["role"] == "leg"]
        aims_on = [p["aim"] for p in _generic_legged_graph(n_pairs=2, fan=True)["parts"] if p["role"] == "leg"]
        self.assertEqual(aims_off, ["down", "down"], "narrow quad legs aim straight down by default")
        self.assertEqual(aims_on, ["down_out", "down_out"], "fan=True fans the quad legs OUT for a wide stance")

    def test_cpg_flag_round_trips_through_npz(self):
        # A CPG-trained residual is wrong without the CPG, so the flag MUST persist through bank/reload.
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import CPG_DEFAULT, MorphPolicy, robot_mjcf
        import mujoco
        import tempfile
        g = self._quad()
        fd = encode_robot(mujoco.MjModel.from_xml_string(robot_mjcf(g))).feature_dim
        pol = MorphPolicy(fd, seed=0); pol.cpg = CPG_DEFAULT
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.npz"); pol.to_npz(p, 0.0, normalizer=None)
            reloaded = MorphPolicy.from_npz(p)
        self.assertIsNotNone(reloaded.cpg, "a CPG-trained policy must reload with its CPG prior")


if __name__ == "__main__":
    unittest.main()
