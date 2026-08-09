"""The autonomous, LLM-authored-reward training loop (agentic platform plan WS-R / R1).

This is the piece that makes "ask in NLP, get a trained policy with an intelligent reward" REAL instead of a
mock. It composes three existing parts into one closed loop:

  ``reward_dsl.propose_rewards``  — an LLM (the customer's own, via BYOK) writes candidate reward EXPRESSIONS
                                     over a whitelisted, AST-sandboxed feature vocabulary (no raw ``exec``).
  ``gait_search.search_gait(reward_fn=...)`` — each candidate reward STEERS a real CEM gait search on the body.
  ``reward_dsl.select_reward``     — candidates rank by a code-owned TRUSTED success (``gait_quality.classify``),
                                     and a reward that games (high return, low credibility) is flagged + dropped.

The reward is optimized; success is never the reward's to define — the Eureka/DrEureka failure modes, closed.

**What "training" means here, exactly.** The default (CPU) path optimizes a 5-scalar CPG gait parameterization
by CEM — a real physics search steered by the LLM's reward, but not a neural policy and not PPO. Passing
``train_backend="gpu"`` hands the SELECTED expression to MJX PPO on the GPU box via
:func:`train_selected_reward_on_gpu`, which is the call site ``gpu_trainer.train_gene_on_gpu(reward_expr=…)``
and ``scripts/mjx_morph_attention.py --reward-expr`` were written for and did not have: before it,
``reward_expr=`` had ZERO callers repo-wide, so the plan's "DSL rewards steer MJX PPO" was a trainer half with
no caller. GPU is remote, slow and optional, so it is opt-in per call and never implicit — a CPU run says so.
"""

from __future__ import annotations

from virturoid.services.gait_search import evaluate_gait, search_gait
from virturoid.services.reward_dsl import propose_rewards, select_reward
from virturoid.services.trained_controller import APPLY_MODES as _APPLY_MODES


def _steered_rollout_fn(gene, *, generations: int, pop: int, steps: int, seed: int):
    """Build the ``rollout_fn(compiled_reward) -> (trusted_success, reward_return)`` that ``select_reward`` needs.

    It runs a SHORT reward-steered CEM search and reports the winning gait's TRUSTED success (classify-based:
    forward distance only when the gait is credible, else 0) alongside the reward's own return. The two are
    computed independently, which is what lets ``select_reward`` catch a reward that racked up return without
    producing a credible walk."""
    def rollout_fn(compiled_reward):
        res = search_gait(gene, generations=generations, pop=pop, steps=steps, seed=seed,
                          reward_fn=compiled_reward)
        trusted = float(res.best_forward) if res.best_credible else 0.0
        return trusted, float(res.best_fitness)     # best_fitness == the reward's return under a reward_fn
    return rollout_fn


def compile_authored_rewards(exprs) -> tuple[list, str]:
    """``(candidates, "")`` for AGENT-AUTHORED reward expressions, or ``([], why_not)``.

    The server ``instructions`` told every connected agent that ``train_reward`` is where "you author a
    reward-as-code objective". Until 2026-08-08 you could not: the tool took ``task`` (English) only, there was
    no ``reward`` parameter, and ``reward_dsl.REWARD_FEATURES`` — the 10-term vocabulary the whole DSL is built
    on — was exposed by no tool at all. This is the parameter that makes the claim true. Expressions go through
    exactly the same ``compile_reward`` AST allowlist an LLM-proposed one does (no attributes, no calls outside
    the whitelisted math, bounded exponents), so "authored by the customer" buys no extra trust; a rejection
    comes back naming the feature vocabulary, because an agent that cannot see the vocabulary cannot fix its
    expression.
    """
    from virturoid.services.reward_dsl import _SAFE_FUNCS, REWARD_FEATURES, RewardCandidate, compile_reward
    if isinstance(exprs, str):
        exprs = [exprs]
    if not isinstance(exprs, (list, tuple)) or not exprs:
        return [], "reward must be an expression string, or a list of them"
    out = []
    for i, raw in enumerate(exprs):
        expr = str(raw or "").strip()
        if not expr:
            return [], f"reward[{i}] is empty"
        try:
            out.append(RewardCandidate(name=f"authored_{i}", expr=expr, compiled=compile_reward(expr)))
        except Exception as exc:  # noqa: BLE001 - a syntax error is the agent's to fix; hand back the vocabulary
            return [], (f"reward[{i}] {expr!r} did not compile: {exc}. A reward is ONE arithmetic expression "
                        f"over these per-step features: {', '.join(REWARD_FEATURES)}; callable functions are "
                        f"{', '.join(sorted(_SAFE_FUNCS))}. No attributes, no imports, no subscripting.")
    return out, ""


def _one_round(gene, task, *, llm, reflection, n_rewards, screen_generations, screen_pop,
               final_generations, final_pop, steps, seed, seed_candidates=None, authored=None):
    """One propose → screen → select → train pass. Returns the trained result + the verdict + selection stats.
    ``seed_candidates`` (R6 flywheel hints: rewards proven on morphology-nearby bodies) are prepended to the
    proposed pool and compete in the same honest screen — a hint that doesn't win on THIS body is dropped.

    ``authored`` (the agent's OWN reward expressions) REPLACES the proposal step: the customer asked for their
    objective, not for four guesses at it. A single authored expression also skips the screen — screening exists
    to CHOOSE between candidates, and there is nothing to choose — so it steers the final search directly and
    is still judged by the same un-gameable ``classify`` verdict. Several authored expressions compete in the
    ordinary screen, gaming detector included."""
    if authored:
        if len(authored) == 1:
            best = authored[0]
            final = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed,
                                reward_fn=best.compiled)
            v = evaluate_gait(gene, final.best_params, steps=steps)
            return {"reward_source": "authored", "reward_name": best.name, "reward_expr": best.expr,
                    "final": final, "v": v, "n_gamed": 0, "n_candidates": 1,
                    "ranked": [{"name": best.name, "expr": best.expr, "trusted_success": None,
                                "reward_return": None, "gamed": False,
                                "note": "the only candidate — nothing to select between, so it steered the "
                                        "final search directly and the verdict below judges the RESULT"}]}
        cands = list(authored)
    else:
        cands = propose_rewards(task, n=n_rewards, llm=llm, reflection=reflection)
        if seed_candidates:
            have = {c.expr for c in cands}
            cands = [c for c in seed_candidates if c.expr not in have] + cands
    rollout_fn = _steered_rollout_fn(gene, generations=screen_generations, pop=screen_pop, steps=steps, seed=seed)
    sel = select_reward(cands, rollout_fn)
    best = sel.get("best")
    ranked = [{"name": c.name, "expr": c.expr, "trusted_success": round(c.trusted_success, 4),
               "reward_return": round(c.reward_return, 4), "gamed": c.gamed} for c in sel.get("ranked", [])]
    if best is None:                                         # every candidate gamed/failed -> honest default
        res = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed)
        v = evaluate_gait(gene, res.best_params, steps=steps)
        return {"reward_source": "default_fitness_fallback", "reward_name": None, "reward_expr": None,
                "final": res, "v": v, "ranked": ranked, "n_gamed": sel.get("n_gamed", 0), "n_candidates": len(cands)}
    final = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed,
                        reward_fn=best.compiled)
    v = evaluate_gait(gene, final.best_params, steps=steps)  # classify verdict, reward-independent
    _src = ("authored" if best.name.startswith("authored") else
            "llm" if best.name.startswith("llm") else "template")
    return {"reward_source": _src, "reward_name": best.name,
            "reward_expr": best.expr, "final": final, "v": v, "ranked": ranked,
            "n_gamed": sel.get("n_gamed", 0), "n_candidates": len(cands)}


def train_selected_reward_on_gpu(gene, reward_expr: str, *, out_path: str | None = None, iters: int = 200,
                                 envs: int = 512, progress=None, gpu_timeout: float = 20.0) -> dict:
    """R1's MISSING CALL SITE: hand the SELECTED reward EXPRESSION to MJX PPO on the GPU box.

    Everything under this function already existed and was tested. ``gpu_trainer.train_gene_on_gpu`` accepts
    ``reward_expr=`` and turns it into ``--reward-expr`` (``gpu_trainer.py:403``); ``mjx_morph_attention.py``
    compiles it with ``reward_dsl.compile_reward_batched(xp=jax.numpy)`` and REPLACES the shaped legged_gym
    reward with it (``:423``). What never existed was a caller — ``reward_expr=`` had zero call sites repo-wide,
    so an LLM-authored reward was proposed, AST-sandboxed, screened and gaming-checked, and then only ever
    reached the CPU CEM search over 5 CPG scalars.

    Never raises: a missing box, an unreachable box or a red parity gate all return an honest report and leave
    the CPU result standing. Returns ``{attempted, trained, policy, note, …}``.
    """
    report = {"attempted": True, "trained": False, "policy": None, "backend": "mjx_ppo_gpu",
              "reward_expr": reward_expr or None, "iters": int(iters), "envs": int(envs)}
    if not str(reward_expr or "").strip():
        return {**report, "attempted": False,
                "note": "no reward expression was selected (the loop fell back to the default un-gameable "
                        "fitness), so there is nothing for the DSL to steer PPO with"}
    try:
        from virturoid.services.gpu_trainer import default_training_recipe, gpu_available, train_gene_on_gpu
    except Exception as exc:  # noqa: BLE001 - GPU support absent -> the CPU result stands, honestly labelled
        return {**report, "note": f"GPU trainer unavailable: {type(exc).__name__}: {exc}"}
    if not gpu_available(timeout=gpu_timeout):
        return {**report, "note": "the GPU box did not answer over Tailscale — this reward steered the CPU gait "
                                  "search only; it did NOT steer PPO"}
    try:
        import re
        from pathlib import Path
        from virturoid.services.agent_tools import safe_build_path
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(getattr(gene, "id", "") or "")).strip("_")[:60] or "reward_loop"
        out = (Path(out_path) if out_path else
               safe_build_path(None, "agent_builds") / slug / "reward_policy.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        recipe = default_training_recipe(gene)          # per-body recipe (cpg / adaptive / deploy-gap deltas)
        npz = train_gene_on_gpu(gene, out_path=str(out), iters=int(iters), envs=int(envs), progress=progress,
                                reward_expr=str(reward_expr), **recipe)
    except Exception as exc:  # noqa: BLE001
        return {**report, "note": f"GPU training failed: {type(exc).__name__}: {exc}"}
    if not npz:
        return {**report, "note": "the GPU trainer returned no policy (parity gate red, launch failure or "
                                  "timeout) — the CPU gait result stands and PPO was NOT steered"}
    return {**report, "trained": True, "policy": str(npz), "recipe": recipe,
            "note": "MJX PPO ran with the selected DSL expression REPLACING the shaped legged_gym reward "
                    "(mjx_morph_attention --reward-expr); the policy npz loads on the same CPU MorphPolicy path"}


def _round_is_better(a, b) -> bool:
    """Prefer a credible round; among equal credibility, more forward travel."""
    if a["v"]["credible"] != b["v"]["credible"]:
        return bool(a["v"]["credible"])
    return float(a["v"]["forward"]) > float(b["v"]["forward"])


def run_intelligent_reward_loop(gene, task: str = "walk forward", *, llm=None, n_rewards: int = 4,
                                screen_generations: int = 3, screen_pop: int = 10,
                                final_generations: int = 8, final_pop: int = 24, steps: int = 800,
                                seed: int = 0, iterations: int = 1, bank: bool = True, db=None,
                                train_backend: str = "cpu", gpu_iters: int = 200, gpu_envs: int = 512,
                                progress=None, authored=None) -> dict:
    """Author → screen → select → train → verify → REFLECT → (re-propose), with zero hand-written reward code.

    1. ``propose_rewards`` writes ``n_rewards`` candidate reward expressions (LLM if given, else templates).
    2. each candidate cheaply STEERS a short search; ``select_reward`` ranks by trusted success + flags gaming.
    3. the winning (non-gamed) reward drives a FINAL, longer search.
    4. the result is judged by the SAME un-gameable ``classify`` verdict ``verify_robot`` uses.
    5. an R2 REFLECTION payload names which reward term saturated / whether forward_vel stayed flat; if
       ``iterations > 1`` and the gait isn't yet credible, that reflection feeds the next round's proposal
       (the Eureka feedback loop) — otherwise it's returned as diagnostics. The best round across iterations is
       kept, and a credible result is banked to the flywheel keyed by this body's morphology.
    6. ``train_backend="gpu"`` (or ``"auto"``) then hands the SELECTED expression to MJX PPO on the GPU box —
       see :func:`train_selected_reward_on_gpu`. ``"cpu"`` (the default) does not, and the report says so
       rather than letting "trained a policy" stand for a 5-scalar CEM search.

    Returns a compact, honest report. Never raises — a total failure returns ``ok: False`` with a reason."""
    try:
        from virturoid.services.reward_reflection import build_reflection_payload, format_reflection_for_proposer

        # R6: seed the search with reward expressions proven on morphology-nearest banked bodies (the moat, through
        # the reward loop). These compete in the SAME honest screen as the fresh proposals — a hint that loses on
        # this body is dropped, so recall accelerates without ever forcing a stale reward.
        seed_candidates, reward_hints = [], []
        if authored:
            # The agent stated its objective. A flywheel hint is a GUESS at an objective, so it does not get to
            # compete with one — recall is skipped entirely and the report says the reward is theirs.
            reward_hints = []
        else:
            try:
                from virturoid.services.gait_flywheel import recall_reward_hints
                from virturoid.services.memory_db import MemoryDB
                from virturoid.services.reward_dsl import RewardCandidate, compile_reward
                _rdb = db or MemoryDB()
                reward_hints = recall_reward_hints(_rdb, gene, k=6)
                for i, h in enumerate(reward_hints):
                    try:
                        seed_candidates.append(RewardCandidate(name=f"flywheel_{i}", expr=h["reward_expr"],
                                                               compiled=compile_reward(h["reward_expr"])))
                    except Exception:  # noqa: BLE001 - a malformed banked expr is skipped, never fatal
                        pass
                if db is None:
                    _rdb.close()
            except Exception:  # noqa: BLE001 - no bank yet -> cold-start from templates/LLM
                pass

        best_round, iter_log, reflection_text = None, [], ""
        for it in range(max(1, iterations)):
            r = _one_round(gene, task, llm=llm, reflection=reflection_text, n_rewards=n_rewards,
                           screen_generations=screen_generations, screen_pop=screen_pop,
                           final_generations=final_generations, final_pop=final_pop, steps=steps, seed=seed,
                           seed_candidates=(seed_candidates if it == 0 else None), authored=authored)
            r["reflection"] = build_reflection_payload(gene, r["final"].best_params, r["reward_expr"], steps=steps)
            v = r["v"]
            iter_log.append({"iteration": it, "reward_expr": r["reward_expr"], "verdict": v["verdict"],
                             "credible": v["credible"], "forward_m": round(v["forward"], 3)})
            if best_round is None or _round_is_better(r, best_round):
                best_round = r
            if v["credible"]:
                break                                        # a credible walk — stop; no need to iterate further
            reflection_text = format_reflection_for_proposer(r["reflection"])   # feed the correction forward

        r, v = best_round, best_round["v"]
        out = {"ok": True, "reward_source": r["reward_source"], "reward_name": r.get("reward_name"),
               "reward_expr": r["reward_expr"], "n_candidates": r["n_candidates"], "n_gamed": r["n_gamed"],
               "ranked": r["ranked"], "verdict": v["verdict"], "credible": v["credible"],
               "forward_m": round(v["forward"], 3), "height_ratio": round(v.get("height_ratio", 0.0), 3),
               "gait_params": r["final"].best_params, "reflection": r["reflection"],
               "iterations_run": len(iter_log), "iteration_log": iter_log,
               "reward_hints_recalled": len(reward_hints),
               "seeded_from_flywheel": bool(seed_candidates) and (r.get("reward_name") or "").startswith("flywheel")}

        if r["reward_source"] == "default_fitness_fallback":
            out["note"] = "no reward beat the honesty gate; trained on the default un-gameable fitness"

        # WHAT WAS ACTUALLY OPTIMIZED. Say it in the report rather than letting "train_reward" imply PPO: the
        # CPU path is CEM over 5 CPG scalars (gait_search), which is a real physics search and is not a policy.
        out["optimizer"] = {
            "cpu": "CEM over the 5-scalar CPG gait parameterization (gait_search.search_gait), steered by the "
                   "selected reward expression and judged by the un-gameable classify() verdict",
            "steered_ppo": False,
        }
        backend = str(train_backend or "cpu").lower()
        if backend in ("gpu", "auto"):
            gpu = train_selected_reward_on_gpu(gene, r["reward_expr"] or "", iters=gpu_iters, envs=gpu_envs,
                                               progress=progress)
            out["gpu_training"] = gpu
            out["optimizer"]["steered_ppo"] = bool(gpu.get("trained"))
            if gpu.get("trained"):
                out["policy_npz"] = gpu["policy"]
            elif backend == "gpu":
                out["warnings"] = [*out.get("warnings", []), f"GPU training did not run: {gpu.get('note')}"]
        else:
            out["gpu_training"] = {"attempted": False, "trained": False,
                                   "note": "train_backend='cpu': the reward steered the CPU gait search only. "
                                           "Pass train_backend='gpu' to make this same expression steer MJX PPO."}

        if bank and v["credible"]:
            try:
                from virturoid.services.gait_flywheel import bank_gait, margin_sentence, robustness_margin
                from virturoid.services.memory_db import MemoryDB
                _db = db or MemoryDB()
                # THE ERROR BAR, MEASURED HERE, because this door can afford it and because "credible" alone has
                # shipped both a controller and one lucky float under the same word. The loop above already spent
                # n_rewards screening searches plus a final one — hundreds of rollouts — so the 4-12 rollout
                # fragility ladder at the SAME horizon the verdict was taken at is a few percent on top.
                # ``per_param=False``: the per-axis breakdown costs 3-5x and does not enter the bank decision.
                rob = robustness_margin(gene, r["final"].best_params, steps=steps, per_param=False)
                out["robustness_rel"] = rob["robustness_rel"]
                out["robustness_note"] = margin_sentence(rob)
                # A FRAGILE POINT IS STILL BANKED, stamped ``fragility_v1_fragile``: the row's value here is the
                # reward EXPRESSION that certified it (R6 recall), which is a fact about the reward and not about
                # this operating point's sturdiness. The stamp keeps it out of gated-only gait mining.
                bank_gait(_db, gene, r["final"], reward_expr=r["reward_expr"] or "",   # bank gait + its reward (R6)
                          robustness=rob, door="reward_loop")
                out["banked"] = True
                if db is None:
                    _db.close()
            except Exception:  # noqa: BLE001 - banking is value-add; a miss never fails the loop
                out["banked"] = False
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------------------------------------
# The NLP-facing agent tool: "train my robot to <task>" -> an LLM-authored reward drives a verified training run.
def _train_reward(args: dict) -> dict:
    from virturoid.services import session_state as S
    rid = args.get("robot_id")
    if not rid:
        return {"ok": False, "error": "robot_id is required (a held robot to train)"}
    gene = S.get_robot(rid)
    if gene is None:
        return {"ok": False, "error": f"no held robot '{rid}'; create_robot / submit_design / ingest_project first"}
    # the customer's OWN LLM authors the rewards when a backend is configured (BYOK); else the heuristic templates.
    llm = None
    try:
        from virturoid.services.llm_backend import get_llm
        llm = get_llm()
    except Exception:  # noqa: BLE001 - LLM-off is a first-class path; templates author the rewards
        llm = None
    backend = str(args.get("train_backend") or "cpu").lower()
    if backend not in ("cpu", "gpu", "auto"):
        return {"ok": False, "error": f"train_backend must be 'cpu', 'gpu' or 'auto' (got {backend!r})"}
    # THE AGENT'S OWN REWARD-AS-CODE OBJECTIVE, which the server instructions have always promised and the tool
    # never accepted. Compiled through the same AST allowlist as an LLM's; a bad expression is rejected BEFORE
    # any physics runs, with the feature vocabulary attached so the next attempt can be right.
    authored = None
    if args.get("reward") is not None:
        authored, why = compile_authored_rewards(args["reward"])
        if why:
            return {"ok": False, "error": why}
    out = run_intelligent_reward_loop(
        gene, task=str(args.get("task") or "walk forward"), llm=llm, authored=authored,
        n_rewards=int(args.get("n_rewards", 4)), iterations=int(args.get("iterations", 1)),
        final_generations=int(args.get("generations", 8)), final_pop=int(args.get("pop", 24)),
        steps=int(args.get("steps", 800)), seed=int(args.get("seed", 0)),
        train_backend=backend, gpu_iters=int(args.get("gpu_iters", 200)),
        gpu_envs=int(args.get("gpu_envs", 512)),
        db=None)
    out["reward_authored_by"] = ("you (the calling agent), via the `reward` parameter" if authored else
                                 "llm" if llm is not None else "templates (no LLM backend configured)")
    if out.get("ok"):
        out["applied_to_robot"] = _land_on_robot(rid, out, args, door="train_reward")
    return out


def _land_on_robot(rid: str, out: dict, args: dict, *, door: str) -> dict:
    """CLOSE THE LOOP: put the trained controller on the held robot so the next ``verify_robot`` measures IT.

    This tool read the gene with ``S.get_robot`` and never wrote one back — see ``trained_controller`` for the
    full account and for why "apply by default, credible-gated, undoable" is the contract.

    TWO ARTIFACTS, TWO CHANNELS, AND THE SECOND ONE WAS A CLAIM RATHER THAN A CALL. The CPU arm produces five CPG
    scalars, which land on the gene above. The GPU arm produces a neural policy ``.npz``, which cannot: its
    deployment channel is the POLICY bank that ``verify_robot`` consults (``ai_native_tools._learned_gait_attempt``
    -> ``policy_flywheel.recall_morph_policy``). This function used to SAY exactly that — "it deploys through the
    POLICY bank (verify recalls it)" — and call nothing: no agent-reachable path anywhere had ever called
    ``bank_morph_policy`` (the repo's only caller was ``desktop.py:480``), so recall could never find the file and
    the agent was told its GPU training had landed when it had not. It now goes through
    ``policy_flywheel.land_gpu_policy``, the same helper ``train_held``'s GPU arm uses, whose gate is the bank's
    own credible-deployed-rollout screen ON THIS BODY — so an unbankable policy comes back as an honest refusal
    naming its verdict instead of a sentence about where it would have gone.
    """
    from virturoid.services import session_state as S
    from virturoid.services.policy_flywheel import land_gpu_policy
    from virturoid.services.trained_controller import apply_trained_gait
    mode = str(args.get("apply") or "auto").lower()
    ev = {"task": str(args.get("task") or "walk forward"), "reward_expr": out.get("reward_expr"),
          "reward_source": out.get("reward_source"), "forward_m": out.get("forward_m"),
          "robustness_rel": out.get("robustness_rel")}
    rep = apply_trained_gait(rid, out.get("gait_params") or {}, door=door, apply=mode,
                             credible=bool(out.get("credible")), verdict=str(out.get("verdict") or ""),
                             evidence=ev)
    gpu = out.get("gpu_training") or {}
    if gpu.get("trained"):
        rep["policy_npz"] = gpu.get("policy")
        rep["policy"] = land_gpu_policy(rid, S.get_robot(rid), gpu.get("policy"), apply=mode, door=door)
        rep["policy_note"] = ("MJX PPO also produced a neural policy. It does not ride on the five CPG scalars "
                              "above — `policy` says whether it was BANKED (and so deployable by verify_robot as "
                              "gait_source 'learned_policy'), or refused and why")
    return rep


def _generate_fusion(args: dict) -> dict:
    """NLP tool: 'set up sensor fusion for my robot' -> a deployable EKF/AHRS/odometry stack from its BOM."""
    from virturoid.services import session_state as S
    rid = args.get("robot_id")
    if not rid:
        return {"ok": False, "error": "robot_id is required (a held robot)"}
    gene = S.get_robot(rid)
    if gene is None:
        return {"ok": False, "error": f"no held robot '{rid}'; create_robot / submit_design / ingest_project first"}
    try:
        from virturoid.services.sensor_fusion_compiler import compile_sensor_fusion
        out = compile_sensor_fusion(gene, task=str(args.get("task") or ""))
        out.pop("_files_content", None)                      # the manifest, not the raw file bodies
        return {"ok": True, **out}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _generate_control_scripts(args: dict) -> dict:
    """NLP tool: 'write the control scripts for my robot' -> the operational .py inventory, each validated."""
    from virturoid.services import session_state as S
    rid = args.get("robot_id")
    if not rid:
        return {"ok": False, "error": "robot_id is required (a held robot)"}
    gene = S.get_robot(rid)
    if gene is None:
        return {"ok": False, "error": f"no held robot '{rid}'; create_robot / submit_design / ingest_project first"}
    try:
        import tempfile
        from pathlib import Path
        from virturoid.services.control_script_compiler import compile_control_scripts, validate_scripts
        out = compile_control_scripts(gene, task=str(args.get("task") or ""))
        # validate in a scratch dir so the caller gets an honest compile+dry-run verdict, not just source
        d = Path(tempfile.mkdtemp()) / "scripts"
        d.mkdir(parents=True, exist_ok=True)
        for rel, content in out["files"].items():
            (d / rel).write_text(content, encoding="utf-8")
        report = validate_scripts(d)
        return {"ok": True, "manifest": out["manifest"], "validation": report}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


REWARD_LOOP_TOOLS = {
    "generate_control_scripts": {
        "description": "Write the operational control-script inventory a held robot needs to actually run -- an "
                       "observation assembler that mirrors the policy's training obs, a safety filter that clamps "
                       "every joint command to the PEAK TORQUE of the real actuator the BOM sized for it, a "
                       "safety state machine (estop/stand/active/fall-damping), a watchdog, a teleop stub, and a "
                       "joint calibration routine. Every generated .py is compile-checked AND dry-run before it "
                       "ships; the result includes that honest pass/fail verdict. No human writes the glue.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string"},
            "task": {"type": "string", "description": "the deployment task (affects the obs layout)"}}},
        "handler": _generate_control_scripts, "heavy": False,
    },
    "generate_fusion": {
        "description": "Compile a deployable SENSOR-FUSION stack (state estimation) for a held robot from its "
                       "bill of materials -- a robot_localization EKF, an IMU orientation filter, and a wheel/leg "
                       "odometry source, referencing EXACTLY the sensors the robot has, on the links they mount "
                       "to. Picks the estimator by what the robot IS: a wheeled base gets a 2-D planar filter, a "
                       "legged body gets contact-aided leg-odometry + full-3-D AHRS, a fixed arm gets none (it "
                       "doesn't localize) -- and it discloses any unobservable state. No human writes an EKF YAML.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string"},
            "task": {"type": "string", "description": "the deployment task (affects the sensor suite)"}}},
        "handler": _generate_fusion, "heavy": False,
    },
    "train_reward": {
        "description": "Optimize a held robot's controller from an NLP task ('walk forward fast', 'carry load "
                       "over rough ground') OR from a reward-as-code objective YOU write (pass `reward`): the "
                       "LLM authors candidate reward "
                       "functions over a safe, AST-sandboxed feature vocabulary, each STEERS a real physics "
                       "search, and the winner is chosen by an un-gameable success metric (a reward that games "
                       "is dropped). THE RESULT LANDS ON THE ROBOT: the winning gait is committed to the held "
                       "gene (undo with edit_robot op:'undo'), so the next verify_robot measures the controller "
                       "you just trained and reports gait_source 'tuned_for_this_body::train_reward' -- pass "
                       "apply:'never' for a dry run that only returns the parameters. BE PRECISE ABOUT WHAT IS "
                       "TRAINED: by default (train_backend='cpu') the "
                       "reward steers a CEM search over the 5-scalar CPG gait parameterization -- a real search "
                       "on real physics, NOT a neural policy and NOT PPO. Set train_backend='gpu' and the SAME "
                       "expression additionally steers MJX PPO on the GPU box (it replaces the shaped reward "
                       "outright) and you get a policy .npz back; if the box is unreachable the report says so "
                       "and the CPU result stands. THE POLICY LANDS TOO, on its own channel: a neural policy "
                       "cannot ride on five CPG scalars, so it is re-rolled on THIS body and banked as a "
                       "deployable MorphPolicy skill when that rollout is a credible walk -- verify_robot then "
                       "reports gait_source 'learned_policy'. `applied_to_robot.policy` says whether it banked "
                       "or was refused and with what verdict; that one gate is the bank's and apply:'always' "
                       "does not override it. Returns the chosen reward, the resulting gait, its honest "
                       "verdict, `optimizer.steered_ppo`, and banks a credible result to the flywheel. Uses "
                       "your own LLM subscription when configured; heuristic templates otherwise.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string"},
            "task": {"type": "string", "description": "the goal in plain language"},
            "reward": {"description": "YOUR reward-as-code objective: one arithmetic expression (or a list of "
                                      "them, which then compete in the same anti-gaming screen) over the "
                                      "whitelisted per-step features. Supplying it REPLACES the LLM/template "
                                      "proposal step -- your objective is used, not guessed at. Omit for the "
                                      "autonomous loop.",
                       "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
            "apply": {"type": "string", "enum": list(_APPLY_MODES), "default": "auto",
                      "description": "where the trained controller goes. 'auto' (default) commits it to the "
                                     "held robot when the run's own un-gameable verdict is a credible walk; "
                                     "'never' returns it as an artifact and touches nothing (apply it later "
                                     "with apply_gait); 'always' commits it regardless and says so."},
            "n_rewards": {"type": "integer", "description": "how many reward candidates to author (default 4)"},
            "iterations": {"type": "integer", "description": "reflect-and-re-propose rounds; >1 runs the Eureka "
                           "feedback loop, stopping early once a credible gait is found (default 1)"},
            "train_backend": {"type": "string", "enum": ["cpu", "gpu", "auto"], "description": "'cpu' (default) "
                              "steers the CPG gait search only; 'gpu'/'auto' also steer MJX PPO on the GPU box "
                              "with the selected expression (remote, minutes, needs VIRTUROID_GPU_SSH)"},
            "gpu_iters": {"type": "integer", "description": "PPO iterations when train_backend is gpu/auto (200)"},
            "gpu_envs": {"type": "integer", "description": "parallel MJX envs when train_backend is gpu/auto (512)"},
            "generations": {"type": "integer"}, "pop": {"type": "integer"}}},
        "handler": _train_reward, "heavy": True,
    },
}
