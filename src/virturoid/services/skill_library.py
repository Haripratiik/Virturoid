"""Skill Library — the cross-USER, cross-MORPHOLOGY, cross-TASK TRAINING flywheel (the moat).

The vision: the FIRST time anyone trains a skill (e.g. User 1's humanoid learning to WALK) it pays the full
cold-training cost, with no prior knowledge. The agent then COMMITS THAT SKILL TO MEMORY as transferable
knowledge. When a LATER user asks for a different robot that needs the SAME skill (User 2's quadruped that
should walk), the agent RECALLS the banked skill and ADAPTS it to the new body with a tiny budget — so the
second robot trains far faster than the first, and every robot after that compounds on the accumulated
knowledge. This works across DIFFERENT bodies because ``MorphPolicy`` is morphology-agnostic (one attention
policy drives any token count at a fixed ``feature_dim``), and across USERS because the bank is a SHARED,
persistent ``MemoryDB``.

This module is the single high-level entry — ``acquire_skill(gene, "walk")`` — that makes recall→adapt→
bank-back the DEFAULT path for any robot+task, instead of cold-training every body from scratch. It is built
ON TOP of the existing pieces (``policy_flywheel`` recall/bank, ``MemoryDB.skills``, ``learn_locomotion``),
not a duplicate. Relates to [[virturoid-flywheel-vision]], [[general-codesign-flywheel]], [[morph-policy-theme1]].
"""

from __future__ import annotations

from pathlib import Path

# Human-facing skill name -> (task_type used by the bank, the structural kinds it transfers across).
# Locomotion skills share ONE transferable MorphPolicy across every legged body; mobile bases share another.
_SKILLS: dict[str, tuple[str, frozenset]] = {
    "walk": ("locomotion", frozenset({"legged"})),
    "trot": ("locomotion", frozenset({"legged"})),
    "run": ("locomotion", frozenset({"legged"})),
    "crawl": ("locomotion", frozenset({"legged"})),
    "gallop": ("locomotion", frozenset({"legged"})),
    "drive": ("locomotion", frozenset({"mobile"})),
    "grasp": ("grasp", frozenset({"manipulator"})),
    "pick_up": ("grasp", frozenset({"manipulator"})),
    "lift": ("grasp", frozenset({"manipulator"})),
    "pick_place": ("pick_place", frozenset({"manipulator"})),
    "sort": ("pick_place", frozenset({"manipulator"})),
}

# How the skill is acquired by training regime (generations, pop, steps, restarts):
#   COLD = no prior knowledge, train hard from scratch (the first user pays this).
#   WARM = recalled a transferable policy, only ADAPT it to this body (few generations -> fast).
_COLD = (50, 40, 800, 3)
_WARM = (15, 40, 800, 1)


def skill_of(name: str):
    """Resolve a human skill name to ``(task_type, kinds)`` or ``None`` if unknown."""
    return _SKILLS.get((name or "").strip().lower())


def deploy_quality(gene, policy, *, steps: int = 2400) -> float:
    """The deployment-aligned skill quality on THIS body: the dense gait reward over a full episode (the
    track_smooth metric — forward speed while upright, smooth). Higher = a better, more usable skill. A RECIPE
    policy is rolled out under the SAME PD-to-default control it learned — scoring it on the legacy torque-
    residual path would understate a gait it never trained for.

    "Is this a RECIPE policy?" comes from the artifact's EXPLICIT marker (``recipe_control``, meta[11]), NOT from
    the incidental presence of a banked obs normalizer. This scores BANKED SKILLS: it is the number ``acquire_skill``
    compares against ``target`` to decide REUSE-vs-retrain, and the ``success_rate`` persisted by ``db.record_skill``
    — i.e. what the cross-user flywheel RANKS and RECALLS for every later body. The GPU trainer banks no normalizer
    at all, so inferring from ``obs_mean`` scored EVERY marked artifact under a controller it never trained with and
    then wrote that wrong score into the bank (task #257, 4th and last site). UNMARKED policies (pre-meta[11]
    artifacts, in-memory policies) keep this site's own legacy inference VERBATIM — obs_mean only, never cpg.

    NORMALIZATION stays a separate question from the control law, and needs no flag here: ``recipe_rollout_morph``
    already picks the normalizer up from the policy itself and feeds RAW observations when there is none, so a
    marked artifact with ``obs_mean is None`` routes to the recipe branch without dividing by an absent normalizer."""
    recipe = getattr(policy, "recipe_control", None)
    if recipe is None:
        recipe = getattr(policy, "obs_mean", None) is not None
    if recipe:
        from virturoid.services.morph_policy import recipe_rollout_morph
        r = recipe_rollout_morph(gene, policy, steps=steps)
    else:
        from virturoid.services.morph_policy import rollout_morph
        r = rollout_morph(gene, policy, steps=steps)
    return float(r.get("gait", -1.0)) if r.get("finite") else -1.0


def recall_skill(gene, skill: str, *, db):
    """Recall the best banked transferable policy for ``skill`` on ``gene``'s structural kind (cross-
    morphology). Returns ``(policy, banked_meta)`` or ``(None, None)``. The policy transfers as long as its
    feature_dim matches this body (constant across morphologies)."""
    resolved = skill_of(skill)
    if resolved is None:
        return None, None
    task_type, _ = resolved
    from virturoid.services.morph_policy import MorphPolicy
    from virturoid.services.task_matched_eval import robot_kind
    meta = db.recall_skill(robot_kind(gene), task_type)
    if not meta or not meta.get("params_path") or not Path(meta["params_path"]).exists():
        return None, None
    try:
        pol = MorphPolicy.from_npz(meta["params_path"])
    except Exception:  # noqa: BLE001 - corrupt/missing -> cold-start
        return None, None
    if not pol.accepts_feature_dim(_feature_dim_for(gene)):    # incompatible body encoding -> can't transfer
        return None, None
    return pol, meta


def _feature_dim_for(gene) -> int:
    import mujoco
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morph_graph import encode_robot
    from virturoid.services.morph_policy import robot_mjcf
    xml = robot_mjcf(gene) if not isinstance(gene, str) else gene
    return int(encode_robot(mujoco.MjModel.from_xml_string(xml)).feature_dim)


def acquire_skill(gene, skill: str = "walk", *, db, models_dir: str = "models", target: float = 0.45,
                  cold=_COLD, warm=_WARM, workers: int | None = None, progress=None) -> dict:
    """Acquire ``skill`` for ``gene`` via the cross-user flywheel — the ONE structural entry point.

    1. RECALL the best banked transferable policy for this skill's kind (works across morphologies).
    2. If it already clears ``target`` on THIS body -> REUSE it, no training (the cheapest path).
    3. Else if recalled -> WARM-ADAPT it with a SMALL budget (it already knows the skill; just fit this body).
    4. Else -> COLD-TRAIN from scratch with a big budget (the first robot ever; expensive).
    5. BANK the result back (keep-best) so the NEXT robot/user compounds on it.

    Returns provenance: ``{skill, task_type, kind, mode, reused, warm_started, quality_before, quality,
    target, reached, generations, seconds, skill_id}``. ``mode`` ∈ reused | warm-adapted | cold-trained.
    Currently wired for the locomotion MorphPolicy (legged/mobile); manipulation skills resolve + recall
    through the SAME bank but need a learned grasp policy banked (the GPU residual path) to warm-adapt."""
    import time

    resolved = skill_of(skill)
    if resolved is None:
        raise ValueError(f"unknown skill {skill!r}; known: {sorted(_SKILLS)}")
    task_type, _kinds = resolved
    from virturoid.services.learn_locomotion import learn_locomotion
    from virturoid.services.task_matched_eval import robot_kind
    kind = robot_kind(gene)

    def say(m):
        if progress:
            progress(m)

    t0 = time.time()

    if task_type == "grasp":
        # GRASP joins the LEARNED-POLICY flywheel (POC proven): a per-token MorphPolicy RESIDUAL on the scripted
        # grasp base transfers across manipulators (8-DOF 0.25->0.75 while a 6-DOF held 1.0). Recall -> reuse if
        # it clears target on THIS body / warm-adapt the recalled residual / cold-train; bank back (keep-best).
        from virturoid.services.grasp_skill import (bank_grasp_residual, evaluate_morph_grasp,
                                                    recall_grasp_residual, train_grasp_residual)
        recalled, rmeta = recall_grasp_residual(gene, db)
        rs = (rmeta or {}).get("residual_scale", 0.12)
        fs = (rmeta or {}).get("fin_residual_scale", 0.04)
        q_before = (evaluate_morph_grasp(gene, recalled, residual_scale=rs, fin_residual_scale=fs)["success_rate"]
                    if recalled is not None else None)
        gens = 0
        if recalled is not None and (q_before or 0) >= target:
            mode, policy, quality = "reused", recalled, q_before
            say(f"banked grasp residual already clears target ({q_before:+.2f} >= {target}) — REUSING")
        else:
            g, p = (warm[0], warm[1]) if recalled is not None else (cold[0], cold[1])
            gens = g * p
            mode = "warm-adapted" if recalled is not None else "cold-trained"
            say(f"{mode} grasp residual: {g}gen x {p}" + (" (warm from banked)" if recalled is not None else " (no memory — first time)"))
            policy, _hist = train_grasp_residual([gene], generations=g, pop=p, warm_start=recalled,
                                                 residual_scale=rs, fin_residual_scale=fs, workers=workers or 1)
            quality = evaluate_morph_grasp(gene, policy, residual_scale=rs, fin_residual_scale=fs)["success_rate"]
            if recalled is not None and (q_before or 0) > quality:    # never regress below the recalled residual
                policy, quality, mode = recalled, q_before, "reused"
        bank_grasp_residual(policy, gene, db, models_dir=models_dir, residual_scale=rs, fin_residual_scale=fs,
                            success_rate=quality)
        say(f"banked grasp residual for {kind} (success={quality:+.2f}); next manipulator compounds on it")
        return {"skill": skill, "task_type": task_type, "kind": kind, "mode": mode,
                "reused": mode == "reused", "warm_started": recalled is not None,
                "quality_before": round(q_before, 3) if q_before is not None else None,
                "quality": round(float(quality), 3), "target": target, "reached": quality >= target,
                "generations": gens, "seconds": round(time.time() - t0, 1),
                "skill_id": f"morph_{kind}_grasp"}

    if task_type != "locomotion":
        # OTHER manipulation (reach/pick_place): CONFIG-mode — a general scripted algorithm (CEM-IK) + per-body
        # tuning (research showed a learned reach does NOT beat scripted IK: 100% vs 25% within 12 cm). Recall the
        # recipe/config through skill_flywheel. (Grasp above + locomotion below use the learned-policy flywheel.)
        from virturoid.services.skill_flywheel import warm_start_for
        ws = warm_start_for(db, kind, task_type, species=getattr(gene, "species", None))
        if ws:
            say(f"recalled a banked '{skill}' manipulation skill in {ws['warm_start']} mode")
        return {"skill": skill, "task_type": task_type, "kind": kind,
                "mode": (f"recalled-{ws['warm_start']}" if ws else "no-banked-skill"),
                "reused": bool(ws), "warm_started": bool(ws),
                "quality_before": (ws or {}).get("success_rate"), "quality": (ws or {}).get("success_rate"),
                "target": target, "reached": bool(ws) and ((ws or {}).get("success_rate") or 0) >= target,
                "generations": 0, "seconds": round(time.time() - t0, 1), "skill_id": (ws or {}).get("skill_id"),
                "note": ("manipulation skill reused (config=scripted-IK recipe + gains/region)" if ws else
                         "no banked skill yet — reach trains via cem_reach (config-mode); both bank here")}

    # --- locomotion: the working cross-morphology LEARNED-POLICY flywheel ----------------------------------
    recalled, meta = recall_skill(gene, skill, db=db)
    q_before = deploy_quality(gene, recalled) if recalled is not None else None
    if recalled is not None:
        say(f"recalled a banked '{skill}' policy (from {meta.get('species') or meta.get('robot_class')}, "
            f"q={q_before:+.3f}) — transferring to this {kind} body")
    if recalled is not None and (q_before or -1) >= target:
        mode, policy, quality, gens = "reused", recalled, q_before, 0
        say(f"banked skill already clears target ({q_before:+.3f} >= {target}) — REUSING, no training")
    else:
        g, p, s, seeds = (warm if recalled is not None else cold)
        mode = "warm-adapted" if recalled is not None else "cold-trained"
        say(f"{mode}: {g}gen × {p} × {seeds} restart(s)" + (" (warm from banked skill)" if recalled is not None else " (no memory yet — first time, expensive)"))
        res = learn_locomotion(gene, generations=g, pop=p, steps=s, seeds=seeds, warm_start=recalled,
                               models_dir=models_dir, workers=workers, progress=progress, recipe=True)
        policy, quality, gens = res["policy"], deploy_quality(gene, res["policy"]), g * seeds

    # BANK back (keep-best). Write a UNIQUE npz so a worse later run never clobbers the banked best file.
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    npz = str(Path(models_dir) / f"skill_{task_type}_{kind}__{getattr(gene, 'id', 'imported')}.npz")
    policy.to_npz(npz, quality)
    # BRIDGE to the per-species eval path: locomotion_episode/banked_policy_for read ``learned_<species>.npz``,
    # so write that too — then the body's own skill is used by the production skill the moment it's acquired.
    sp = getattr(gene, "species", None) or getattr(gene, "robot_class", None) or "imported"
    policy.to_npz(str(Path(models_dir) / ("learned_" + str(sp).replace("/", "_") + ".npz")), quality)
    sid = f"morph_{kind}_{task_type}"
    db.record_skill(sid, kind, task_type, success_rate=float(quality), params_path=npz,
                    species=getattr(gene, "species", None), gene_id=getattr(gene, "id", None),
                    obs_dim=int(policy.feature_dim), act_dim=int(policy.hidden),
                    notes=f"{skill} skill via {mode}; deploy_quality={quality:.3f}")
    reached = quality >= target
    say(f"banked '{skill}' for {kind} (q={quality:+.3f}); next {kind} build compounds on it")
    return {"skill": skill, "task_type": task_type, "kind": kind, "mode": mode,
            "reused": mode == "reused", "warm_started": recalled is not None,
            "quality_before": round(q_before, 3) if q_before is not None else None,
            "quality": round(float(quality), 3), "target": target, "reached": reached,
            "generations": gens, "seconds": round(time.time() - t0, 1), "skill_id": sid}
