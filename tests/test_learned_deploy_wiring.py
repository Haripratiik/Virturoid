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
