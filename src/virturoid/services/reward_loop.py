"""The autonomous, LLM-authored-reward training loop (agentic platform plan WS-R / R1).

This is the piece that makes "ask in NLP, get a trained policy with an intelligent reward" REAL instead of a
mock. It composes three existing parts into one closed loop:

  ``reward_dsl.propose_rewards``  — an LLM (the customer's own, via BYOK) writes candidate reward EXPRESSIONS
                                     over a whitelisted, AST-sandboxed feature vocabulary (no raw ``exec``).
  ``gait_search.search_gait(reward_fn=...)`` — each candidate reward STEERS a real CEM gait search on the body.
  ``reward_dsl.select_reward``     — candidates rank by a code-owned TRUSTED success (``gait_quality.classify``),
                                     and a reward that games (high return, low credibility) is flagged + dropped.

The reward is optimized; success is never the reward's to define — the Eureka/DrEureka failure modes, closed.
Pure CPU, LLM-optional (heuristic templates backfill), deterministically testable. The MJX/GPU port reuses the
same feature extractor + reward compilation; only the inner ``search_gait`` swaps for the MJX trainer.
"""

from __future__ import annotations

from virturoid.services.gait_search import evaluate_gait, search_gait
from virturoid.services.reward_dsl import propose_rewards, select_reward


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


def run_intelligent_reward_loop(gene, task: str = "walk forward", *, llm=None, n_rewards: int = 4,
                                screen_generations: int = 3, screen_pop: int = 10,
                                final_generations: int = 8, final_pop: int = 24, steps: int = 800,
                                seed: int = 0, bank: bool = True, db=None) -> dict:
    """Author → screen → select → train → verify, with zero hand-written reward code.

    1. ``propose_rewards`` writes ``n_rewards`` candidate reward expressions (LLM if given, else templates).
    2. each candidate cheaply STEERS a short search; ``select_reward`` ranks by trusted success + flags gaming.
    3. the winning (non-gamed) reward drives a FINAL, longer search.
    4. the result is judged by the SAME un-gameable ``classify`` verdict ``verify_robot`` uses, then optionally
       banked to the flywheel keyed by this body's morphology.

    Returns a compact, honest report: the chosen reward, the trained gait + its verdict, and how many candidates
    were rejected for gaming. Never raises — a total failure returns ``ok: False`` with a reason."""
    try:
        cands = propose_rewards(task, n=n_rewards, llm=llm)
        rollout_fn = _steered_rollout_fn(gene, generations=screen_generations, pop=screen_pop,
                                         steps=steps, seed=seed)
        sel = select_reward(cands, rollout_fn)
        best = sel.get("best")
        ranked = [{"name": c.name, "expr": c.expr, "trusted_success": round(c.trusted_success, 4),
                   "reward_return": round(c.reward_return, 4), "gamed": c.gamed}
                  for c in sel.get("ranked", [])]

        if best is None:                                     # every candidate gamed or failed -> honest default
            res = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed)
            v = evaluate_gait(gene, res.best_params, steps=steps)
            return {"ok": True, "reward_source": "default_fitness_fallback", "reward_expr": None,
                    "n_candidates": len(cands), "n_gamed": sel.get("n_gamed", 0), "ranked": ranked,
                    "verdict": v["verdict"], "credible": v["credible"], "forward_m": round(v["forward"], 3),
                    "gait_params": res.best_params,
                    "note": "no reward beat the honesty gate; trained on the default un-gameable fitness"}

        final = search_gait(gene, generations=final_generations, pop=final_pop, steps=steps, seed=seed,
                            reward_fn=best.compiled)
        v = evaluate_gait(gene, final.best_params, steps=steps)   # classify verdict, reward-independent
        out = {"ok": True, "reward_source": ("llm" if best.name.startswith("llm") else "template"),
               "reward_name": best.name, "reward_expr": best.expr,
               "n_candidates": len(cands), "n_gamed": sel.get("n_gamed", 0), "ranked": ranked,
               "verdict": v["verdict"], "credible": v["credible"], "forward_m": round(v["forward"], 3),
               "height_ratio": round(v["height_ratio"], 3), "gait_params": final.best_params}

        if bank and v["credible"]:
            try:
                from virturoid.services.gait_flywheel import bank_gait
                from virturoid.services.memory_db import MemoryDB
                _db = db or MemoryDB()
                bank_gait(_db, gene, final)                  # bank the verified gait keyed by morphology
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
    out = run_intelligent_reward_loop(
        gene, task=str(args.get("task") or "walk forward"), llm=llm,
        n_rewards=int(args.get("n_rewards", 4)),
        final_generations=int(args.get("generations", 8)), final_pop=int(args.get("pop", 24)),
        steps=int(args.get("steps", 800)), seed=int(args.get("seed", 0)),
        db=None)
    out["reward_authored_by"] = "llm" if llm is not None else "templates (no LLM backend configured)"
    return out


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
        "description": "Train a control policy for a held robot from an NLP task ('walk forward fast', 'carry "
                       "load over rough ground') with NO hand-written reward: the LLM authors candidate reward "
                       "functions over a safe feature vocabulary, each STEERS a real search, and the winner is "
                       "chosen by an un-gameable success metric (a reward that games is dropped). Returns the "
                       "chosen reward, the trained gait, its honest verdict, and banks a credible result to the "
                       "flywheel. Uses your own LLM subscription when configured; heuristic templates otherwise.",
        "parameters": {"type": "object", "required": ["robot_id"], "properties": {
            "robot_id": {"type": "string"},
            "task": {"type": "string", "description": "the goal in plain language"},
            "n_rewards": {"type": "integer", "description": "how many reward candidates to author (default 4)"},
            "generations": {"type": "integer"}, "pop": {"type": "integer"}}},
        "handler": _train_reward, "heavy": True,
    },
}
