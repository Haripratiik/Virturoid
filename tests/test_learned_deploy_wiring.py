"""v7-P0a (master_plan_v7 §1 F1-F4): the learned-control DEPLOY wiring — the measured product half of the
WS-F deploy gap. The deltas were symmetric and policies were banked, but (F1) the legged product verdict never
recalled them, (F2) learned deploys skipped classify()'s ROLL/PITCH gate (no qpos trace — an EASIER bar than
scripted = gameable), (F3) nothing invoked the parity harness before training spend, and (F4) an interactive
GPU launch had no memory guard / deadline kill. These tests lock all four in.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from virturoid.services import ai_native_tools as AIT
from virturoid.services import gpu_trainer as GT
from virturoid.services import morph_policy as MP
from virturoid.services import policy_flywheel as PF


@pytest.fixture(scope="module")
def quad():
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot("a four-legged walking robot", ensure_walkable=True)


# A rollout that passes the FULL un-gameable bar (classify == CREDIBLE WALK): all five scalar gates + a level
# body (identity quaternions -> roll/pitch 0). deployment_controller/upright evidence make it bankable too.
def _credible_rollout():
    return {"survived": True, "upright_frac": 0.9, "height_ratio": 0.9, "cadence": 2.1, "support_frac": 0.8,
            "forward": 0.82, "speed": 0.31, "alive": 900, "deployment_controller": "recipe_cpg",
            "qpos_frames": [np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])] * 4}


def _failed_rollout():
    return {"survived": False, "upright_frac": 0.2, "height_ratio": 0.4, "cadence": 0.3, "support_frac": 0.1,
            "forward": 0.04, "speed": 0.01, "alive": 300, "qpos_frames": []}


# ---------------------------------------------------------------------------- F2: same bar for learned
def test_deploy_rollout_records_qpos_by_default(monkeypatch):
    """Without qpos frames classify() silently skips ROLL/PITCH for learned policies — deploy must record them."""
    seen = {}

    def fake_recipe(gene, policy, *, steps=900, record_qpos=False, **kw):
        seen["record_qpos"] = record_qpos
        return _credible_rollout()
    monkeypatch.setattr(MP, "recipe_rollout_morph", fake_recipe)
    r = MP.rollout_deployed_morph_policy(object(), SimpleNamespace(obs_mean=[0.0], cpg=None))
    assert seen["record_qpos"] is True                       # the honest default
    assert r["deployment_controller"] == "recipe_cpg"
    MP.rollout_deployed_morph_policy(object(), SimpleNamespace(obs_mean=[0.0], cpg=None), record_qpos=False)
    assert seen["record_qpos"] is False                      # explicit override still possible (training inner loops)


# ------------------------------------------------------------- #257: the artifact DECLARES its controller
# scripts/mjx_morph_attention.py::_save's meta layout, mirrored here so the routing contract is pinned against the
# EXACT array the GPU trainer writes: [F, H, NT, fwd, adaptive@4, cpg@5, decimation@6, action_lpf@7, sphere_feet@8,
# phase_obs@9, command@10, recipe@11]. That trainer's ONLY control law is the recipe one, but a run WITHOUT --cpg
# banks no obs_mean and no cpg_arr -- and gpu_trainer.default_training_recipe sets cpg only for >=3 legs, so every
# biped / humanoid / non-legged run lands here.
def _gpu_artifact(path, feature_dim, *, cpg=0.0, decimation=1.0, sphere_feet=0.0, marked=True):
    pol = MP.MorphPolicy(feature_dim, seed=0)                 # hidden=32 == meta[1] below
    meta = [float(feature_dim), 32.0, 12.0, 0.4, 0.0, cpg, decimation, 0.0, sphere_feet, 0.0]
    if marked:                                                # a post-fix artifact: command spacer + the marker
        meta += [0.0, 1.0]
    np.savez(str(path), **{k: pol._arrs[k] for k in ("We", "be", "Wq", "Wk", "Wv", "Wo", "Wh", "bh")},
             meta=np.asarray(meta))
    return str(path)


def _route(monkeypatch, policy):
    """Route ``policy`` with BOTH rollouts stubbed, so only the controller CHOICE is under test."""
    seen = {}
    monkeypatch.setattr(MP, "recipe_rollout_morph",
                        lambda g, p, **kw: seen.update(via="recipe") or _credible_rollout())
    monkeypatch.setattr(MP, "rollout_morph", lambda g, p, **kw: seen.update(via="residual") or _failed_rollout())
    r = MP.rollout_deployed_morph_policy(object(), policy)
    assert seen["via"] == ("recipe" if r["deployment_controller"] == "recipe_cpg" else "residual")
    return r["deployment_controller"]


def test_non_cpg_recipe_artifact_deploys_under_the_recipe_controller(monkeypatch, tmp_path):
    """The measured bug: a recipe-trained artifact with NEITHER obs_mean NOR cpg was deployed under the legacy
    torque-residual controller it was never trained with (which also never terminates on a fall). The explicit
    meta[11] marker -- not the incidental presence of a normalizer -- decides."""
    npz = _gpu_artifact(tmp_path / "gpu_biped.npz", 27)
    pol = MP.MorphPolicy.from_npz(npz)
    assert pol.obs_mean is None and pol.cpg is None           # exactly what the old inference looked at: nothing
    assert pol.recipe_control is True                          # ...but the artifact SAYS what it was trained with
    assert _route(monkeypatch, pol) == "recipe_cpg"


def test_unmarked_artifact_routes_exactly_as_it_does_today(monkeypatch, tmp_path):
    """BACKWARD COMPAT: checkpoints already on disk carry no meta[11] and can never be retro-marked, so an
    UNMARKED artifact must keep the legacy obs_mean/cpg inference verbatim -- in both directions."""
    old = MP.MorphPolicy.from_npz(_gpu_artifact(tmp_path / "old.npz", 27, marked=False))
    assert old.recipe_control is None                          # unknown, NOT False: absence of evidence
    assert _route(monkeypatch, old) == "residual"              # ...so it routes precisely as it did before
    old_cpg = MP.MorphPolicy.from_npz(_gpu_artifact(tmp_path / "old_cpg.npz", 27, cpg=1.0, marked=False))
    assert old_cpg.recipe_control is None and old_cpg.cpg is not None
    assert _route(monkeypatch, old_cpg) == "recipe_cpg"        # the inference still carries the CPG artifacts
    plain = MP.MorphPolicy(27, seed=0)                         # a fresh in-memory policy is unmarked too
    assert plain.recipe_control is None and _route(monkeypatch, plain) == "residual"


def test_recipe_marker_survives_banking_and_an_unmarked_bank_stays_unmarked(monkeypatch, tmp_path):
    """to_npz is the flywheel's banking path: a marked artifact must stay marked through it, and a policy of
    UNKNOWN provenance must bank a 0.0 that still means "unmarked" (never an explicit residual claim)."""
    marked = MP.MorphPolicy.from_npz(_gpu_artifact(tmp_path / "gpu.npz", 27))
    reloaded = MP.MorphPolicy.from_npz(marked.to_npz(str(tmp_path / "banked.npz"), 0.5))
    assert reloaded.recipe_control is True and _route(monkeypatch, reloaded) == "recipe_cpg"
    plain = MP.MorphPolicy(27, seed=0)
    unmarked = MP.MorphPolicy.from_npz(plain.to_npz(str(tmp_path / "plain.npz"), 0.1))
    assert float(np.load(tmp_path / "plain.npz")["meta"][11]) == 0.0
    assert unmarked.recipe_control is None and _route(monkeypatch, unmarked) == "residual"
    # a normalizer is proof of recipe training on its own -> banking one MARKS the file (same criterion as meta[2])
    norm = (np.zeros(27), np.ones(27))
    rec = MP.MorphPolicy.from_npz(plain.to_npz(str(tmp_path / "norm.npz"), 0.1, normalizer=norm))
    assert rec.recipe_control is True and _route(monkeypatch, rec) == "recipe_cpg"


def test_newly_routed_artifact_honours_its_own_decimation_and_sphere_feet(monkeypatch, quad, tmp_path):
    """The deploy==train knobs the legacy residual path silently DROPPED. The recipe rollout adopts the
    artifact's own meta[6]/meta[8], so the newly-routed policy runs the contact model and control rate it was
    trained at -- assert on the real rollout, not on the routing tag alone."""
    import mujoco

    from virturoid.services import sphere_feet as SF
    from virturoid.services.morph_graph import encode_robot
    fd = encode_robot(mujoco.MjModel.from_xml_string(MP.robot_mjcf(quad))).feature_dim
    pol = MP.MorphPolicy.from_npz(_gpu_artifact(tmp_path / "gpu.npz", fd, decimation=6.0, sphere_feet=1.0))
    assert pol.decimation == 6 and pol.sphere_feet is True
    seen = {"sphere": 0, "act": 0}
    real_sf, real_act = SF.apply_sphere_feet, MP.MorphPolicy.act
    monkeypatch.setattr(SF, "apply_sphere_feet", lambda m: seen.update(sphere=seen["sphere"] + 1) or real_sf(m))
    monkeypatch.setattr(MP.MorphPolicy, "act",
                        lambda self, *a, **k: (seen.update(act=seen["act"] + 1), real_act(self, *a, **k))[1])
    r = MP.rollout_deployed_morph_policy(quad, pol, steps=12)
    assert r["deployment_controller"] == "recipe_cpg"
    assert seen["sphere"] == 1                                # meta[8]: the trained contact model is rebuilt
    assert seen["act"] == 2                                   # meta[6]: 12 steps at D=6 -> 2 control updates, not 12


# ------------------------------------------- #257 (cont.): the SAME question, asked in three more places
# "Is this a recipe policy?" is also decided in learn_locomotion.locomotion_episode, learn_locomotion.rollout_view
# and gene_build._build_legged_package. Each decides something DIFFERENT (which rollout scores the learned arm of
# an episode / which control law the viewport replays / whether the banked policy is used at all), and each
# predates the marker — so each mis-answers it for the artifacts the GPU trainer actually writes: no obs_mean
# ever, and no cpg unless the body has >=3 legs.
def _marked(tmp_path, feature_dim, name="gpu.npz", **kw):
    return MP.MorphPolicy.from_npz(_gpu_artifact(tmp_path / name, feature_dim, **kw))


def test_locomotion_episode_scores_a_marked_artifact_under_the_recipe_controller(monkeypatch, tmp_path, quad):
    """SITE 1 (learn_locomotion.py:209) chooses the rollout that produces the LEARNED arm of the episode result —
    the forward_m/upright the capability registry and the MCP locomotion tool report, and the number that decides
    whether "learned" beats "scripted". A marked non-CPG artifact carries neither obs_mean nor cpg, so the legacy
    inference sent it to rollout_morph: the legacy gravity-comp torque residual, which never terminates on a fall
    and drops the artifact's own decimation/sphere_feet."""
    from virturoid.services import learn_locomotion as LL
    from virturoid.services import locomotion_controller as LC

    pol = _marked(tmp_path, 27)
    assert pol.obs_mean is None and pol.cpg is None            # what the old inference looked at: nothing
    assert pol.recipe_control is True                          # ...while the artifact states its controller
    seen = {}
    monkeypatch.setattr(LC, "run_locomotion_episode",
                        lambda model, **kw: {"distance_m": 0.0, "forward_m": 0.0, "upright": False, "status": "fell"})
    monkeypatch.setattr(LL, "banked_policy_for", lambda gene, **kw: pol)
    monkeypatch.setattr(MP, "recipe_rollout_morph",
                        lambda g, p, **kw: seen.update(via="recipe") or _credible_rollout())
    monkeypatch.setattr(MP, "rollout_morph", lambda g, p, **kw: seen.update(via="residual") or _failed_rollout())
    ep = LL.locomotion_episode(quad, horizon=120)
    assert seen["via"] == "recipe"                             # the controller it was TRAINED with
    assert ep["source"] == "learned" and ep["upright"] is True  # ...and the episode reports that rollout's verdict
    # BACKWARD COMPAT: an unmarked artifact keeps the legacy obs_mean/cpg inference verbatim, in both directions.
    seen.clear()
    monkeypatch.setattr(LL, "banked_policy_for", lambda gene, **kw: _marked(tmp_path, 27, "old.npz", marked=False))
    LL.locomotion_episode(quad, horizon=120)
    assert seen["via"] == "residual"
    seen.clear()
    monkeypatch.setattr(LL, "banked_policy_for",
                        lambda gene, **kw: _marked(tmp_path, 27, "old_cpg.npz", cpg=1.0, marked=False))
    LL.locomotion_episode(quad, horizon=120)
    assert seen["via"] == "recipe"


def test_rollout_view_replays_a_marked_artifact_under_the_recipe_control_law(monkeypatch, tmp_path, quad):
    """SITE 2 (learn_locomotion.py:246) chooses the CONTROL LAW the desktop viewport replays with: the recipe
    PD-to-default (+ the artifact's own CPG prior and gains) or the legacy gravity-comp torque residual, which is
    the only path that calls MorphGraph.apply. Its legacy inference is NARROWER than the deploy router's — obs_mean
    only, never cpg — so at HEAD EVERY GPU artifact, CPG or not, was replayed under a controller it never trained
    with, and the user watched a motion that is not the gait that was banked."""
    import mujoco

    from virturoid.services import learn_locomotion as LL
    from virturoid.services import morph_graph as MG
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z

    xml = compile_gene_to_mjcf(quad, include_floor=True, spawn_z=standing_spawn_z(quad, meshed=False))
    fd = MG.encode_robot(mujoco.MjModel.from_xml_string(xml)).feature_dim
    pol = _marked(tmp_path, fd)
    assert pol.obs_mean is None and pol.recipe_control is True
    calls = {"apply": 0}
    real_apply = MG.MorphGraph.apply
    monkeypatch.setattr(MG.MorphGraph, "apply",
                        lambda self, *a, **k: (calls.update(apply=calls["apply"] + 1), real_apply(self, *a, **k))[1])
    view, _ = LL.rollout_view(quad, pol, steps=20, frame_every=10)   # must NOT raise: a recipe artifact has no
    assert calls["apply"] == 0                                       #   normalizer to divide by (obs_mean is None)
    assert len(view["frames"]) >= 2
    # BACKWARD COMPAT: unmarked still replays exactly as before — normalizer -> recipe, bare policy -> residual.
    plain = MP.MorphPolicy(fd, seed=0)
    LL.rollout_view(quad, plain, steps=20, frame_every=10)
    assert calls["apply"] == 20
    plain.obs_mean, plain.obs_std = np.zeros(fd), np.ones(fd)
    LL.rollout_view(quad, plain, steps=20, frame_every=10)
    assert calls["apply"] == 20                                      # still 20: a normalizer still means "recipe"


def test_the_build_headline_scores_the_banked_marked_policy_not_a_random_one(monkeypatch, tmp_path):
    """SITE 3 (gene_build.py:802) is NOT a controller router — both branches already run recipe_rollout_morph. It
    decides whether the banked policy is USED AT ALL, and ``policy=None`` there is not a zero-residual baseline:
    recipe_rollout_morph CONSTRUCTS A RANDOM MorphPolicy (its own docstring). So a marked artifact was silently
    discarded and the build's headline locomotion number — plus the qpos trace the viewer replays — described a
    random policy instead of the species' banked one."""
    from virturoid.fixtures.gene_library import quadruped_gene
    from virturoid.services import gene_build as GB
    from virturoid.services import learn_locomotion as LL

    pol = _marked(tmp_path, 27)
    calls = []

    def fake_recipe(g, p, **kw):
        calls.append((p, kw))
        return _credible_rollout()

    def headline():                                           # the exported-bare-gait check rolls out too, with its
        return [p for p, kw in calls if kw.get("record_qpos")]  # own throwaway policy — the HEADLINE call is the one
    monkeypatch.setattr(MP, "recipe_rollout_morph", fake_recipe)   # that records the qpos trace the viewer replays
    monkeypatch.setattr(MP, "crawl_gait_rollout", lambda g, **kw: _failed_rollout())
    monkeypatch.setattr(LL, "banked_policy_for", lambda gene, **kw: pol)
    GB._build_legged_package(quadruped_gene(), "a quadruped that walks forward", tmp_path / "pkg", None)
    assert headline() and headline()[0] is pol                # the banked artifact, not a random stand-in
    # BACKWARD COMPAT: an UNMARKED policy with no normalizer is still discarded, exactly as today.
    calls.clear()
    monkeypatch.setattr(LL, "banked_policy_for", lambda gene, **kw: _marked(tmp_path, 27, "old.npz", marked=False))
    GB._build_legged_package(quadruped_gene(), "a quadruped that walks forward", tmp_path / "pkg2", None)
    assert headline() and headline()[0] is None


def test_deploy_quality_scores_a_marked_artifact_under_the_recipe_controller(monkeypatch, tmp_path):
    """SITE 4 (skill_library.py:55) scores BANKED SKILLS — the number ``acquire_skill`` compares against
    ``target`` to decide REUSE-vs-retrain, the ``success_rate`` stamped into ``db.record_skill``, and therefore
    what the cross-user flywheel RANKS and RECALLS for every later body. Its legacy inference is the narrow one
    (obs_mean only, never cpg), so a marked GPU artifact — which banks no normalizer — was scored under the
    legacy torque-residual controller it never trained with, and that wrong score was PERSISTED into the bank."""
    from virturoid.services import skill_library as SL

    pol = _marked(tmp_path, 27)
    assert pol.obs_mean is None and pol.cpg is None             # what the old inference looked at: nothing
    assert pol.recipe_control is True                            # ...while the artifact states its controller
    seen = {}
    monkeypatch.setattr(MP, "recipe_rollout_morph",
                        lambda g, p, **kw: seen.update(via="recipe") or {"finite": True, "gait": 0.71})
    monkeypatch.setattr(MP, "rollout_morph",
                        lambda g, p, **kw: seen.update(via="residual") or {"finite": True, "gait": 0.12})
    assert SL.deploy_quality(object(), pol) == pytest.approx(0.71)
    assert seen["via"] == "recipe"                               # the controller it was TRAINED with
    # BACKWARD COMPAT: an unmarked policy keeps the legacy obs_mean-only inference VERBATIM, in both directions.
    seen.clear()
    old = _marked(tmp_path, 27, "old.npz", marked=False)
    assert old.recipe_control is None                            # unknown, NOT False: absence of evidence
    assert SL.deploy_quality(object(), old) == pytest.approx(0.12) and seen["via"] == "residual"
    seen.clear()
    old.obs_mean, old.obs_std = np.zeros(27), np.ones(27)        # a normalizer still means "recipe", as today
    assert SL.deploy_quality(object(), old) == pytest.approx(0.71) and seen["via"] == "recipe"
    seen.clear()
    old.obs_mean = None
    old.cpg = MP.CPG_DEFAULT                                     # ...and cpg alone still does NOT, as today
    assert SL.deploy_quality(object(), old) == pytest.approx(0.12) and seen["via"] == "residual"


def test_deploy_quality_of_a_marked_artifact_needs_no_normalizer(quad, tmp_path):
    """The COUPLING that the site-2 fix forced out, asserted here UNSTUBBED: a marked GPU artifact legitimately
    has obs_mean None, so routing it to the recipe branch must feed RAW observations instead of dividing by a
    normalizer that isn't there. ``recipe_rollout_morph`` already decouples control law from normalization
    (``normalizer`` stays None -> no division), so this site needs no separate flag — prove it, don't assume it."""
    import mujoco

    from virturoid.services import skill_library as SL
    from virturoid.services.morph_graph import encode_robot

    fd = encode_robot(mujoco.MjModel.from_xml_string(MP.robot_mjcf(quad))).feature_dim
    pol = _marked(tmp_path, fd)
    assert pol.obs_mean is None and pol.recipe_control is True
    q = SL.deploy_quality(quad, pol, steps=12)                  # must NOT raise TypeError on the absent normalizer
    assert isinstance(q, float) and q > -1.0                    # a real recipe score, not the not-finite sentinel


# ------------------------------------ #258: the GPU trainer's EXPLORATION SIGMA is bounded, and recorded
# scripts/mjx_morph_attention.py has essentially no coverage, and `mjx` cannot be imported on a CPU box, so its
# `main()` is untestable here. These pin the parts of the fix that are pure numerics (the sigma bound), pure
# argparse (the flag) and pure I/O (the meta layout). The rollout/print plumbing that carries the mean-episode-
# length readout is NOT covered: it lives inside jitted MJX rollouts and needs a real GPU run.
def _trainer():
    import scripts.mjx_morph_attention as T                # a repo-root namespace package; import per test so a
    return T                                               # failure here can never silently skip this whole file


def test_exploration_sigma_is_bounded_at_every_read():
    """The measured bug (#258): ``ent = (logstd + const).sum()`` is LINEAR in logstd, so the PPO loss carries a
    CONSTANT -ENT = -5e-3 gradient pushing sigma UP forever, unscheduled, with nothing opposing it. From the -0.5
    init (sigma 0.6065) that truncated episodes to ~150 of 500 control steps, so the objective never observed the
    fall — and the measured sweep showed sigma >~0.3 INVERTS the objective's ranking of two checkpoints."""
    T = _trainer()
    exp, bs = np.exp, T.bounded_sigma
    assert exp(-0.5) == pytest.approx(0.6065, abs=1e-4)          # the trainer's logstd INIT, unbounded
    assert bs(np.array([-0.5]))[0] == pytest.approx(exp(-0.7))   # ...capped from the very first iteration
    for raw in (-0.5, 0.0, 2.0, 50.0):                           # the RUNAWAY: wherever the constant entropy
        assert bs(np.array([raw]))[0] == pytest.approx(exp(T.LOGSTD_MAX))   # gradient drives it, sigma is bounded
    assert bs(np.array([-9.0]))[0] == pytest.approx(exp(T.LOGSTD_MIN))      # and it can't collapse to 0 either
    assert 0.100 == pytest.approx(exp(T.LOGSTD_MIN), abs=5e-4)   # the documented [0.10, 0.50] band
    assert 0.497 == pytest.approx(exp(T.LOGSTD_MAX), abs=5e-4)
    for raw in (-2.0, -1.5, -0.9):                               # INSIDE the band it is the EXACT identity, so a
        assert bs(np.array([raw]))[0] == float(exp(raw))         # run that never hits the cap is unperturbed
    # the sweep's GOOD region (sigma <= 0.15, where the objective ranked the good checkpoint first) is reachable
    assert bs(np.array([0.0]), hi=-1.9)[0] == pytest.approx(0.1496, abs=1e-4)


def test_the_sigma_bound_is_a_cli_flag_with_the_old_behaviour_reachable():
    """A hard-coded guess is not falsifiable; a flag is. The parser also has to RENDER — an unescaped '%' in any
    help string makes ``--help`` raise, which is how a CLI silently stops being usable."""
    T = _trainer()
    ap = T.build_parser()
    d = ap.parse_args([])
    assert (d.logstd_min, d.logstd_max) == (T.LOGSTD_MIN, T.LOGSTD_MAX)
    assert d.ep_len == 200                                       # horizon stays a RUN-TIME choice; default untouched
    wide = ap.parse_args(["--logstd-min", "-20", "--logstd-max", "20"])
    for raw in (-0.5, 0.0, 2.0):                                 # a wide range == the exact PRE-BOUND behaviour
        assert T.bounded_sigma(np.array([raw]), wide.logstd_min, wide.logstd_max)[0] == float(np.exp(raw))
    tight = ap.parse_args(["--logstd-max", "-1.9"])
    assert T.bounded_sigma(np.array([0.0]), tight.logstd_min, tight.logstd_max)[0] < 0.15
    assert "--logstd-max" in ap.format_help()                    # renders at all (the unescaped-'%' trap)


def _npz_with_meta(path, meta):
    pol = MP.MorphPolicy(27, seed=0)
    np.savez(str(path), **{k: pol._arrs[k] for k in ("We", "be", "Wq", "Wk", "Wv", "Wo", "Wh", "bh")},
             meta=np.asarray(meta))
    return MP.MorphPolicy.from_npz(str(path))


def test_the_banked_artifact_records_the_sigma_it_was_trained_under(tmp_path):
    """meta[12] follows the meta[11] convention EXACTLY: APPENDED (never reordered) so every existing reader is
    untouched, and ABSENT rather than 0.0 when unknown — a 0.0 would be a positive claim of sigma 1.0. Without it
    nothing on disk says which of two checkpoints was explored at a sigma that inverts the objective's ranking."""
    T = _trainer()
    m = T.trainer_meta(27, 32, 12, 0.4, decimation=6, sphere_feet=True, logstd=-0.7)
    assert len(m) == 13 and m[11] == 1.0                         # the #257 recipe marker is untouched
    assert float(np.exp(m[12])) == pytest.approx(0.4966, abs=1e-4)     # exp(meta[12]) = the training sigma
    assert T.trainer_meta(27, 32, 12, 0.4, decimation=6, sphere_feet=True) == m[:12]   # unknown -> NO slot at all
    # a 13-slot artifact must load EXACTLY as the 12-slot one: a tail entry is invisible to every existing reader
    a, b = _npz_with_meta(tmp_path / "m13.npz", m), _npz_with_meta(tmp_path / "m12.npz", m[:12])
    for attr in ("feature_dim", "hidden", "adaptive_gains", "decimation", "action_lpf", "sphere_feet",
                 "phase_obs", "command_conditioned", "recipe_control"):
        assert getattr(a, attr) == getattr(b, attr), attr
    assert a.recipe_control is True and a.decimation == 6 and a.sphere_feet is True


def test_the_entropy_bonus_pushes_sigma_up_forever_and_the_bound_stops_it():
    """The MECHANISM, not just the clamp. ``loss = pg + VF*vf - ENT*ent`` with ``ent = (logstd + const).sum()`` is
    LINEAR in logstd, so its gradient is the CONSTANT -ENT — an unscheduled upward push on sigma every minibatch,
    for the whole run. Reading the CLIPPED logstd in the entropy term makes that gradient ZERO past the cap, so
    the push stops AT the bound instead of accumulating forever in a parameter the rollout no longer obeys."""
    jax = pytest.importorskip("jax")                             # the GPU-box dep; skip on a box without it
    jp = jax.numpy
    T = _trainer()
    ENT = 5e-3                                                   # the trainer's entropy coefficient

    def loss_unbounded(ls):
        return -ENT * (ls + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()

    def loss_bounded(ls):
        return -ENT * (T.bounded_logstd(ls, T.LOGSTD_MIN, T.LOGSTD_MAX, xp=jp)
                       + 0.5 * jp.log(2 * jp.pi * jp.e)).sum()
    for raw in (-0.5, 0.0, 2.0):                                 # unbounded: the SAME constant push, everywhere
        assert float(jax.grad(loss_unbounded)(jp.array([raw]))[0]) == pytest.approx(-ENT)
    assert float(jax.grad(loss_bounded)(jp.array([-1.5]))[0]) == pytest.approx(-ENT)   # inside: unchanged
    for raw in (-0.5, 0.0, 2.0):                                 # at/past the cap: the push is switched OFF
        assert float(jax.grad(loss_bounded)(jp.array([raw]))[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(jax.grad(loss_bounded)(jp.array([-9.0]))[0]) == pytest.approx(0.0, abs=1e-12)   # and below it
    # and the SAME expression is what the rollout samples with, under jit (this is how the trainer reads it)
    assert float(jax.jit(lambda p: T.bounded_sigma(p, T.LOGSTD_MIN, T.LOGSTD_MAX, xp=jp)[0])(jp.array([-0.5]))) \
        == pytest.approx(float(np.exp(T.LOGSTD_MAX)), abs=1e-6)


def test_this_files_gpu_artifact_mirror_still_matches_the_real_writer(tmp_path):
    """``_gpu_artifact`` above hand-mirrors the trainer's meta array. A mirror that DRIFTS from the writer turns
    every routing test in this file into a test of the mirror instead of the contract, so pin them together."""
    T = _trainer()
    mirrored = list(np.load(_gpu_artifact(tmp_path / "mirror.npz", 27, cpg=1.0, decimation=6.0,
                                          sphere_feet=1.0))["meta"])
    real = T.trainer_meta(27, 32, 12, 0.4, cpg=True, decimation=6, sphere_feet=True)
    assert [float(x) for x in mirrored] == [float(x) for x in real]


# ---------------------------------------------------------------------------- F1: recall seam + product verdict
def test_recall_with_rollout_returns_the_screen_rollout(monkeypatch, quad):
    """The credibility screen already runs a full deployed rollout — the verdict path reuses that exact evidence."""
    rollout = _credible_rollout()
    monkeypatch.setattr(MP.MorphPolicy, "from_npz", staticmethod(lambda p: SimpleNamespace(obs_mean=[0.0])))
    monkeypatch.setattr(MP, "rollout_deployed_morph_policy", lambda g, pol, steps=900: rollout)
    db = SimpleNamespace(recall_skill=lambda kind, task: {"params_path": "banked.npz"})
    policy, r = PF.recall_morph_policy(quad, db, with_rollout=True)
    assert policy is not None and r is rollout               # ONE rollout serves screen + verdict
    assert PF.recall_morph_policy(quad, SimpleNamespace(recall_skill=lambda k, t: None),
                                  with_rollout=True) == (None, None)


def test_learned_attempt_builds_a_credible_verdict(monkeypatch, quad, tmp_path):
    from virturoid.services import memory_db as MDB
    dbfile = tmp_path / "m.db"; dbfile.write_bytes(b"")
    monkeypatch.setattr(MDB, "DEFAULT_DB_PATH", dbfile)

    class _StubDB:
        def __init__(self, path): ...
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(MDB, "MemoryDB", _StubDB)
    monkeypatch.setattr(PF, "recall_morph_policy",
                        lambda g, db, task_type="locomotion", with_rollout=False:
                        (SimpleNamespace(), _credible_rollout()) if with_rollout else SimpleNamespace())
    got = AIT._learned_gait_attempt(quad)
    assert got is not None
    out = got["out"]
    assert out["gait_source"] == "learned_policy" and out["verdict"] == "CREDIBLE WALK"
    assert out["roll_max_deg"] == 0.0 and out["pitch_max_deg"] == 0.0   # judged WITH the orientation gate
    # nothing banked / not credible on this body -> honest None
    monkeypatch.setattr(PF, "recall_morph_policy",
                        lambda g, db, task_type="locomotion", with_rollout=False: (None, None))
    assert AIT._learned_gait_attempt(quad) is None


def test_honest_gait_adopts_learned_when_scripted_fails(monkeypatch, quad):
    """The product verdict uses the robot's BEST controller: scripted fails -> a credible banked policy walks it."""
    from virturoid.services import gait_hints as GH
    monkeypatch.setattr(MP, "crawl_gait_rollout", lambda gene, **kw: _failed_rollout())
    monkeypatch.setattr(GH, "mine_gait_hints", lambda db, gene=None: {"n": 0})       # hermetic: no live-DB hints
    banked = {"called": False}

    def fake_attempt(gene):
        banked["called"] = True
        return {"out": {"kind": "legged", "verdict": "CREDIBLE WALK", "survived": True,
                        "gait_source": "learned_policy", "forward_m": 0.82, "speed_mps": 0.31, "cadence": 2.1,
                        "support_frac": 0.8, "height_ratio": 0.9, "roll_max_deg": 0.0, "pitch_max_deg": 0.0,
                        "note": "learned"},
                "rollout": _credible_rollout()}
    monkeypatch.setattr(AIT, "_learned_gait_attempt", fake_attempt)
    poisoned = {"banked": False}
    monkeypatch.setattr(AIT, "_auto_bank_gait", lambda *a, **k: poisoned.update(banked=True))
    out = AIT._honest_gait(quad, steps=60)
    assert banked["called"] and out["gait_source"] == "learned_policy"
    assert str(out["verdict"]).startswith("CREDIBLE")
    # the learned rollout must NOT be re-banked as crawl-gait params (corpus poisoning guard)
    assert poisoned["banked"] is False


def test_honest_gait_skips_learned_when_scripted_credible(monkeypatch, quad):
    """Never-regress + cheap fast path: a credible scripted walk never pays the learned recall."""
    from virturoid.services import gait_hints as GH
    monkeypatch.setattr(MP, "crawl_gait_rollout", lambda gene, **kw: _credible_rollout())
    monkeypatch.setattr(GH, "mine_gait_hints", lambda db, gene=None: {"n": 0})
    monkeypatch.setattr(AIT, "_auto_bank_gait", lambda *a, **k: None)

    def must_not_run(gene):
        raise AssertionError("learned recall must not run when the scripted gait is credible")
    monkeypatch.setattr(AIT, "_learned_gait_attempt", must_not_run)
    out = AIT._honest_gait(quad, steps=60)
    # The contract is SCRIPTED-not-learned, not one particular scripted label. verify now reports the more
    # precise provenance when the body carries its own tuned op-point ("tuned_for_this_body") instead of
    # flattening everything scripted to "default_crawl" -- both are the scripted path, and the thing this test
    # guards (the learned recall never running) is asserted by the must_not_run monkeypatch above.
    assert str(out["verdict"]).startswith("CREDIBLE")
    assert out["gait_source"] in ("default_crawl", "tuned_for_this_body"), out["gait_source"]


# ---------------------------------------------------------------------------- F3: parity as an enforced gate
def test_parity_gate_blocks_training_on_red(monkeypatch, quad, tmp_path):
    monkeypatch.delenv("VIRTUROID_SKIP_PARITY_GATE", raising=False)
    monkeypatch.setattr(GT, "_sync_repo", lambda say: None)
    monkeypatch.setattr(GT, "parity_gate_for_gene", lambda g, **kw: {"parity_ok": False, "note": "sign flip"})
    launched = {"hit": False}
    monkeypatch.setattr(GT, "_launch_ok", lambda cmd, name: launched.update(hit=True) or False)
    monkeypatch.setattr(GT, "_ssh", lambda cmd, **kw: SimpleNamespace(stdout=b"", stderr=b"", returncode=0))
    assert GT.train_gene_on_gpu(quad, out_path=str(tmp_path / "p.npz")) is None
    assert launched["hit"] is False                          # red gate -> not a single training step spent


def test_parity_gate_fails_closed_when_box_unreachable(monkeypatch, quad):
    GT._PARITY_GATE_CACHE.clear()
    monkeypatch.setattr(MP, "crawl_gait_rollout",
                        lambda gene, **kw: {"ctrl_trace": np.zeros((60, 4)), "forward": 0.5})
    def boom(cmd, **kw):
        raise OSError("no tailscale")
    monkeypatch.setattr(GT, "_ssh", boom)
    gate = GT.parity_gate_for_gene(quad)
    assert gate["parity_ok"] is False                        # fail CLOSED: never train unverified
    assert not GT._PARITY_GATE_CACHE                          # a transient failure is never cached


def _fixed_leg(*, parity_ok, diff=0.004, fa=0.001, fb=0.001, sign_flip=False):
    return {"leg": "fixed_ctrl", "parity_ok": parity_ok, "backends": ["cpu_mujoco", "mjx_single"],
            "comparisons": [{"pair": "cpu_mujoco vs mjx_single", "max_state_abs_diff": diff,
                             "forward_a": fa, "forward_b": fb, "sign_flip": sign_flip, "parity_ok": parity_ok}]}


def _gait_leg(*, fa=0.2983, fb=0.3024, sign_flip=False, vacuous=False):
    return {"leg": "gait_trace", "parity_ok": not sign_flip, "vacuous": vacuous,
            "comparison": {"pair": "cpu_mujoco vs mjx_single", "max_state_abs_diff": 0.73,
                           "forward_a": fa, "forward_b": fb, "sign_flip": sign_flip,
                           "parity_ok": not sign_flip}}


def test_parity_policy_measured_calibration_cases():
    """The verdict policy, pinned to the FIRST LIVE GATE FIRING's measured numbers (2026-07-12): a ×3-mass quad
    diverged 0.004 in state (chaos on violent contact) while behavior agreed to 1.4% — behavioral GREEN; a sign
    flip anywhere or missing behavioral evidence stays RED (fail-closed)."""
    # strict green: state tol holds
    v = GT.combine_parity_legs([_fixed_leg(parity_ok=True, diff=1.7e-05), _gait_leg()])
    assert v["parity_ok"] and v["mode"] == "strict"
    # the HEAVY case: leg-1 state chaos + exact forward agreement + 1.4% gait agreement -> behavioral green
    v = GT.combine_parity_legs([_fixed_leg(parity_ok=False), _gait_leg()])
    assert v["parity_ok"] and v["mode"] == "behavioral"
    # the WS-F bug: a sign flip ANYWHERE is red, no matter how good the other leg looks
    v = GT.combine_parity_legs([_fixed_leg(parity_ok=False, sign_flip=True), _gait_leg()])
    assert not v["parity_ok"] and "SIGN FLIP" in v["why"]
    v = GT.combine_parity_legs([_fixed_leg(parity_ok=True), _gait_leg(fa=0.3, fb=-0.3, sign_flip=True)])
    assert not v["parity_ok"]
    # state divergence with NO behavioral confirmation (vacuous gait / big magnitude gap / no gait leg) -> red
    assert not GT.combine_parity_legs([_fixed_leg(parity_ok=False), _gait_leg(vacuous=True)])["parity_ok"]
    assert not GT.combine_parity_legs([_fixed_leg(parity_ok=False), _gait_leg(fa=0.40, fb=0.10)])["parity_ok"]
    assert not GT.combine_parity_legs([_fixed_leg(parity_ok=False)])["parity_ok"]
    # leg-1 forwards CONTRADICTING (delta > 0.05 m) blocks the behavioral fallback even with a good gait leg
    assert not GT.combine_parity_legs([_fixed_leg(parity_ok=False, fa=0.30, fb=0.10), _gait_leg()])["parity_ok"]


# ---------------------------------------------------------------------------- F4: launch safety
def test_launch_cmd_has_memory_guard_and_gate_is_skippable(monkeypatch, quad, tmp_path):
    monkeypatch.setenv("VIRTUROID_SKIP_PARITY_GATE", "1")
    monkeypatch.setattr(GT, "_sync_repo", lambda say: None)
    monkeypatch.setattr(GT, "_ssh", lambda cmd, **kw: SimpleNamespace(stdout=b"", stderr=b"", returncode=0))
    gate = {"hit": False}
    monkeypatch.setattr(GT, "parity_gate_for_gene", lambda g, **kw: gate.update(hit=True) or {"parity_ok": True})
    seen = {}
    monkeypatch.setattr(GT, "_launch_ok", lambda cmd, name: seen.update(cmd=cmd) or False)
    assert GT.train_gene_on_gpu(quad, out_path=str(tmp_path / "p.npz")) is None
    assert gate["hit"] is False                              # the explicit escape hatch skipped the gate
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=.85" in seen["cmd"]   # the leak guard rides EVERY interactive launch


# ---------------------------------------------------------------------------- F3b: the non-vacuous parity leg
def test_crawl_records_ctrl_trace_and_it_replays(quad):
    """The gait ctrl trace is the ONLY control that displaces a stable body past the sign-check floor — it must
    record the applied torques and replay through the parity harness's CPU leg."""
    from virturoid.services.parity_harness import rollout_cpu_mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    r = MP.crawl_gait_rollout(quad, steps=250, record_ctrl=True)
    tr = r.get("ctrl_trace")
    assert tr is not None and tr.ndim == 2 and tr.shape[1] >= 1 and len(tr) >= 50
    assert np.all(np.isfinite(tr)) and float(np.abs(tr).max()) > 0.0    # real torques, not zeros
    xml = compile_gene_to_mjcf(quad, include_floor=True, spawn_z=standing_spawn_z(quad))
    rep = rollout_cpu_mujoco(xml, tr)
    assert rep["finite"] and rep["n_steps"] == len(tr)                  # identical bytes replay cleanly
