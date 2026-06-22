"""Learn-on-request locomotion: when the user asks, TRAIN a control policy for *this* body, then bank it
and commit species-specific tips so the next build of that species reuses it (the flywheel).

This is the product loop the user asked for — control isn't pre-baked; it's learned on demand and banked.
The in-app trainer is a CPU evolution strategy over the morphology-agnostic MorphPolicy (real, but modest
— stable bodies like quadrupeds pick up forward motion; humanoid balance is a frontier and trains far
better on the GPU/MJX path, whose policies bank into the SAME flywheel). Relates to [[morph-policy-theme1]],
[[reasoned-redesign-loop]] (commit tips to the species).
"""

from __future__ import annotations

from pathlib import Path


def learn_locomotion(robot, *, generations: int = 8, pop: int = 12, steps: int = 180,
                     models_dir: str = "models", progress=None, warm_start=None, species: str = None,
                     workers: int | None = None, seeds: int = 1, eval_steps: int | None = None,
                     recipe: bool = False, adaptive: bool = False, cpg="auto") -> dict:
    """Train a MorphPolicy for ``robot`` (a composed RobotGene OR a raw MJCF string for an IMPORTED model)
    and save it. ``warm_start`` (a banked policy from a similar species) seeds training so it converges
    faster. ``workers`` evaluates the ES population in PARALLEL across that many CPU processes (default: all
    but one core).

    ``seeds`` > 1 runs BEST-OF-N restarts — ES is noisy and the short-horizon training fitness doesn't
    perfectly predict the long-horizon deployment gait, so we train ``seeds`` independent restarts and KEEP
    the one with the best ``eval_steps``-horizon gait reward (the deployment-aligned metric). This is the
    reliability fix from the locomotion research: good upright gaits exist but aren't found every seed.

    ``recipe=True`` (the product locomotion path) trains under the ANTI-COLLAPSE RECIPE: position-PD to the
    default standing pose + obs normalization + terminate-on-fall + velocity-tracking reward — the control that
    actually WALKS (the default torque-residual fitness collapses to a fall-forward flop). The obs normalizer is
    BANKED with the policy so every deploy/replay drives it the same way it learned.
    Returns ``{policy, npz_path, score, baseline, warm_started, species, feature_dim, generations, seeds}``."""
    import multiprocessing as mp
    import sys

    import mujoco
    if workers is None:
        # parallel in production; serial under a test runner (don't spawn a process fan-out inside pytest,
        # which is slow + spawn-fragile on Windows). The parallel path is covered by a subprocess test.
        workers = 1 if "pytest" in sys.modules else max(1, mp.cpu_count() - 1)

    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import MorphPolicy, recipe_obs_normalizer, recipe_rollout_morph, rollout_morph
    from virturoid.services.morph_trainer import forward_score, recipe_forward_score, train_morph_es

    # Train on a STANDING-spawn body so it starts on its feet (genes); build the MJCF string once so the
    # many rollouts don't recompile. Imported MJCF strings are used as-is.
    if isinstance(robot, str):
        train_robot = robot
    else:
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        train_robot = compile_gene_to_mjcf(robot, include_floor=True, spawn_z=standing_spawn_z(robot, meshed=False))
    model = mujoco.MjModel.from_xml_string(train_robot)
    fd = encode_robot(model).feature_dim
    warm = warm_start if (warm_start is not None and getattr(warm_start, "feature_dim", None) == fd) else None
    if warm is None and not isinstance(robot, str):          # FLYWHEEL: warm-start from this species' banked policy
        try:                                                  # (a fresh "Learn to move" should BUILD ON the bank,
            _banked = banked_policy_for(robot, models_dir=models_dir)   # not retrain a weak policy from scratch)
            if _banked is not None and getattr(_banked, "feature_dim", None) == fd:
                warm = _banked
        except Exception:  # noqa: BLE001
            pass
    # The obs normalizer is collected ONCE on this body (recipe mode) and banked with the winning policy.
    normalizer = recipe_obs_normalizer(train_robot) if recipe else None
    if recipe and adaptive == "auto":                        # auto-gate: adapt only bodies far from the reference quad
        from virturoid.services.morph_policy import adaptive_recommended
        adaptive = adaptive_recommended(model)
    adaptive = bool(adaptive and recipe)                     # adaptive gains only meaningful on the recipe path
    # TROT-CPG PRIOR: a feed-forward stepping rhythm so CPU-ES learns only a propulsion residual (it can't
    # discover a gait from a stand-still — it slides/collapses). "auto" enables it for recognizable quadrupeds
    # (the _trot_cpg_tokens 4-leg gate); other morphologies fall back to the scalar recipe. See CPG_DEFAULT.
    if recipe and cpg == "auto":
        from virturoid.services.morph_policy import CPG_DEFAULT, _trot_cpg_tokens
        _, _, _gate = _trot_cpg_tokens(model, encode_robot(model), CPG_DEFAULT)
        cpg = CPG_DEFAULT if _gate else None
    elif cpg == "auto" or not recipe:
        cpg = None
    score_fn = ((lambda rb, p, **kw: recipe_forward_score(rb, p, normalizer=normalizer, adaptive=adaptive,
                                                          cpg=cpg, **kw)) if recipe else forward_score)
    baseline = score_fn(train_robot, warm or MorphPolicy(fd, seed=0), steps=steps)  # warm-start = the start point
    if eval_steps is None:
        eval_steps = min(1500, max(steps, 4 * steps))        # select on a LONGER horizon than we trained on

    def deploy_quality(pol, norm=None):                      # deployment-aligned selection: long-horizon gait
        if recipe:
            r = recipe_rollout_morph(train_robot, pol, steps=eval_steps, normalizer=(norm or normalizer))
        else:
            r = rollout_morph(train_robot, pol, steps=eval_steps)
        if not r.get("finite"):
            return -1.0
        # Reward FORWARD DISTANCE (so a stand-still can't win) AND SURVIVAL (so a fast fall-forward that topples
        # can't out-score a slower STABLE upright walk — the demo must look like walking, not a face-plant).
        forward = max(0.0, float(r.get("forward", 0.0)))
        survived = bool(r.get("survived")) or float(r.get("height_ratio", 0.0)) > 0.6
        return float(r.get("gait", 0.0)) + 0.6 * forward + (0.8 if survived else 0.0)

    if progress:
        progress((f"warm-starting from banked tips — " if warm else "training on this body — ")
                 + (f"[recipe] " if recipe else "")
                 + f"{generations}gen × {pop} × {('%d restarts ' % seeds) if seeds > 1 else ''}on {workers} cores…")
    best_policy, best_q, history = None, float("-inf"), []
    for s in range(max(1, seeds)):
        pol, hist = train_morph_es([train_robot], feature_dim=fd, generations=generations, pop=pop, steps=steps,
                                   init_policy=warm, workers=workers, seed=s, recipe=recipe, normalizer=normalizer,
                                   adaptive=adaptive, cpg=cpg)
        pol.adaptive_gains = bool(adaptive)                  # so deploy_quality + banking drive it with the SAME gains
        pol.cpg = cpg                                        # so deploy_quality/replay/banking use the SAME CPG prior
        q = deploy_quality(pol)
        if progress and seeds > 1:
            progress(f"  restart {s + 1}/{seeds}: deploy-quality {q:+.3f}")
        if q > best_q:
            best_policy, best_q, history = pol, q, hist
    policy = best_policy
    if recipe and normalizer is not None:                    # carry the normalizer on the policy object too
        policy.obs_mean, policy.obs_std = normalizer[0], normalizer[1]
    sp = species or getattr(robot, "species", None) or getattr(robot, "robot_class", None) or "imported"
    npz = str(Path(models_dir) / ("learned_" + sp.replace("/", "_") + ".npz"))
    # KEEP-BEST: never let this run clobber a STRONGER policy already banked for this species. A weak from-scratch
    # run banking a stand-still over a banked walker is exactly what made the app "stop walking" (the flywheel must
    # be monotonic — banking only ever improves the species' best).
    kept_banked = False
    if not isinstance(robot, str):
        try:
            existing = banked_policy_for(robot, models_dir=models_dir)
        except Exception:  # noqa: BLE001
            existing = None
        if existing is not None:
            # deploy the banked one under ITS OWN normalizer (recipe_rollout_morph falls back to policy.obs_mean
            # when norm is None); the forward-distance weight dominates, so this comparison is robust.
            ex_norm = (existing.obs_mean, existing.obs_std) if getattr(existing, "obs_mean", None) is not None else None
            eq = deploy_quality(existing, norm=ex_norm)
            if eq >= best_q:
                policy, best_q, kept_banked = existing, eq, True
    score = score_fn(train_robot, policy, steps=steps)
    if not kept_banked:
        policy.to_npz(npz, score, normalizer=normalizer)
    # ACTUAL forward distance the chosen policy travels (HONEST meters — NOT the gait-reward number, which a UI
    # must not label as "forward {N}" since a body that merely stays upright shows a small-positive reward).
    try:
        _nw = (policy.obs_mean, policy.obs_std) if getattr(policy, "obs_mean", None) is not None else normalizer
        _fr = (recipe_rollout_morph(train_robot, policy, steps=eval_steps, normalizer=_nw,
                                    adaptive=bool(getattr(policy, "adaptive_gains", False))) if recipe
               else rollout_morph(train_robot, policy, steps=eval_steps))
        forward_m = round(float(_fr.get("forward", 0.0)), 3)
    except Exception:  # noqa: BLE001
        forward_m = None
    return {"policy": policy, "npz_path": npz, "score": float(score), "baseline": float(baseline),
            "warm_started": warm is not None, "species": sp, "feature_dim": fd, "generations": generations,
            "seeds": int(max(1, seeds)), "deploy_quality": round(float(best_q), 4), "recipe": bool(recipe),
            "adaptive": bool(adaptive), "kept_banked": bool(kept_banked), "forward_m": forward_m}


def banked_policy_for(gene, *, models_dir: str = "models"):
    """Load the banked MorphPolicy for ``gene``'s species if one exists AND its feature_dim matches this body
    (so the SAME policy that walks this species drives it again — the Theme-1/2 reuse). Returns the policy or
    None. A morphology-conditioned policy is body-agnostic, so a species' banked weights drive every build of
    that species without retraining."""
    from pathlib import Path

    sp = (getattr(gene, "species", None) or getattr(gene, "robot_class", None) or "imported")
    path = Path(models_dir) / ("learned_" + str(sp).replace("/", "_") + ".npz")
    if not path.exists():
        return None
    try:
        import mujoco

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import MorphPolicy
        pol = MorphPolicy.from_npz(str(path))
        fd = encode_robot(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))).feature_dim
        return pol if pol.feature_dim == fd else None
    except Exception:  # noqa: BLE001 - a missing/incompatible bank just means "no learned policy yet"
        return None


def locomotion_episode(gene, *, horizon: int = 2400, models_dir: str = "models", learn: bool = False,
                       generations: int = 30, pop: int = 24, steps: int = 300, progress=None,
                       adaptive="auto") -> dict:
    """Run the BEST available locomotion controller for ``gene`` and return forward-travel stats in the shape
    the scripted episode uses (``{distance_m, forward_m, upright, status, source}``). Prefers a LEARNED
    morphology-conditioned MorphPolicy — banked for this species, or trained on demand when ``learn=True`` —
    which generalises where the scripted trot is brittle (the trot drives several composed bodies BACKWARD).
    Always returns whichever controller travelled FURTHER FORWARD, so the learned path never regresses below
    the scripted baseline. CPU MuJoCo."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.locomotion_controller import run_locomotion_episode

    scripted = run_locomotion_episode(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene)), horizon=horizon)
    scripted = {**scripted, "source": "scripted",
                "forward_m": float(scripted.get("forward_m", scripted.get("distance_m", 0.0)))}

    policy = banked_policy_for(gene, models_dir=models_dir)
    if policy is None and learn:
        # learn-on-demand uses the anti-collapse RECIPE (the control that actually walks) by default
        policy = learn_locomotion(gene, generations=generations, pop=pop, steps=steps, models_dir=models_dir,
                                  progress=progress, recipe=True, adaptive=adaptive)["policy"]
    learned = None
    if policy is not None:
        steps_h = min(int(horizon), 2400)
        # A RECIPE/CPG policy must be driven under the SAME PD + CPG-prior rollout the bank/render/"Learn to move"
        # use (recipe_rollout_morph) — else the learned RESIDUAL runs WITHOUT its gait prior and collapses. Gate on
        # the CPG flag too: a GPU-trained policy carries cpg but no obs_mean (the GPU trainer banks no normalizer),
        # and was being misrouted to the no-CPG rollout_morph -> the same walk that survives in the bank read as fallen.
        if getattr(policy, "obs_mean", None) is not None or getattr(policy, "cpg", None) is not None:
            from virturoid.services.morph_policy import recipe_rollout_morph
            r = recipe_rollout_morph(gene, policy, steps=steps_h)
            upright = bool(r.get("survived", False)) or float(r.get("height_ratio", 0.0)) > 0.6
        else:
            from virturoid.services.morph_policy import rollout_morph
            r = rollout_morph(gene, policy, steps=steps_h)
            upright = bool(r.get("upright", False))
        fwd = float(r.get("forward", 0.0))
        learned = {"distance_m": abs(fwd), "forward_m": fwd, "upright": upright,
                   "height_ratio": r.get("height_ratio"), "status": "learned", "source": "learned"}

    if learned is not None and learned["forward_m"] >= scripted["forward_m"]:
        return learned
    return scripted


def rollout_view(robot, policy, *, steps: int = 360, frame_every: int = 4) -> tuple[dict, str]:
    """Roll the (learned) policy out on ``robot`` (a RobotGene or an imported MJCF string) recording
    per-frame geom poses, so the desktop viewport can REPLAY the learned motion. Returns ``(view, xml)``."""
    import mujoco
    import numpy as np

    from virturoid.services.morph_graph import encode_robot

    if isinstance(robot, str):
        xml = robot
    else:                                                   # replay the body standing on its feet, not ejected
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        xml = compile_gene_to_mjcf(robot, include_floor=True, spawn_z=standing_spawn_z(robot, meshed=False))
    model = mujoco.MjModel.from_xml_string(xml); model.opt.iterations = 20
    data = mujoco.MjData(model); mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    graph = encode_robot(model)
    bq = graph.base_qadr
    # A RECIPE policy (it carries a banked obs normalizer) was trained under position-PD-to-default control +
    # obs normalization — replaying it with the legacy torque-residual would NOT reproduce its gait, so drive
    # it the same way it learned. Legacy policies (obs_mean is None) replay on the residual path exactly as before.
    recipe = getattr(policy, "obs_mean", None) is not None
    if recipe:
        qadr = np.asarray(graph.qadr, dtype=int); vadr = np.asarray(graph.vadr, dtype=int)
        act_u = np.asarray(graph.act_u, dtype=int); clamps = np.asarray(graph.clamps, dtype=float)
        q_def = np.array([float(data.qpos[a]) for a in qadr])
        # Replay with the SAME per-joint gains the policy was trained under (adaptive -> inertia-scaled), else the
        # scalar recipe — otherwise the viewport gait wouldn't match what was learned (the train/replay invariant).
        if getattr(policy, "adaptive_gains", False):
            from virturoid.services.morph_policy import recipe_gains
            kp_v, kd_v, as_v = recipe_gains(model, graph)
        else:                                                # scalar recipe — same source-of-truth constants as training
            from virturoid.services.morph_policy import _ASCALE, _KD_REF, _KP_REF
            kp_v = np.full(graph.n_tokens, _KP_REF); kd_v = np.full(graph.n_tokens, _KD_REF)
            as_v = np.full(graph.n_tokens, _ASCALE)
        # Replay under the SAME trot-CPG prior the policy was trained with (else the learned residual drives a
        # body with no stepping rhythm and the viewport gait won't match what was learned).
        _cpg = getattr(policy, "cpg", None)
        cpg_on = False; res_scale = 1.0; cpg_freq = 0.0; cpg_amp = cpg_phase = None
        if _cpg:
            from virturoid.services.morph_policy import _trot_cpg_tokens
            cpg_amp, cpg_phase, _gate = _trot_cpg_tokens(model, graph, _cpg)
            if _gate:
                cpg_on = True; cpg_freq = float(_cpg["freq"]); res_scale = float(_cpg.get("residual_scale", 0.3))
        _dt = float(model.opt.timestep); _two_pi = 2.0 * np.pi

    def snap():
        fr = []
        for g in range(model.ngeom):
            x = data.geom_xpos[g]; q = np.zeros(4)
            mujoco.mju_mat2Quat(q, data.geom_xmat[g])
            fr.append([float(x[0]), float(x[1]), float(x[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        return fr

    frames = [snap()]
    p0 = float(data.qpos[bq]) if bq >= 0 else 0.0
    for t in range(steps):
        obs = graph.observe(model, data)
        if recipe:
            a = policy.act((obs - policy.obs_mean) / policy.obs_std)
            cphase = _two_pi * cpg_freq * t * _dt if cpg_on else 0.0
            for k in range(graph.n_tokens):
                cpg_off = float(cpg_amp[k] * np.sin(cphase + cpg_phase[k])) if cpg_on else 0.0
                tgt = q_def[k] + cpg_off + res_scale * as_v[k] * float(np.tanh(a[k]))
                tau = kp_v[k] * (tgt - float(data.qpos[qadr[k]])) - kd_v[k] * float(data.qvel[vadr[k]])
                data.ctrl[act_u[k]] = float(np.clip(tau, -clamps[k], clamps[k]))
        else:
            graph.apply(model, data, policy.act(obs), alpha=0.6)
        mujoco.mj_step(model, data)
        if t % frame_every == 0:
            frames.append(snap())
        if not np.all(np.isfinite(data.qpos)):
            break
    travelled = (float(data.qpos[bq]) - p0) if bq >= 0 else 0.0
    view = {"frames": frames, "scenes": [], "scene_index": 0,
            "outcome": {"status": "success" if travelled > 0.05 else "learned",
                        "placed_count": 0, "block_count": 0}}
    return view, xml
