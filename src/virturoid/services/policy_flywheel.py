"""Policy flywheel (Theme 2): bank trained MorphPolicies and reuse them across builds (RoboCat-style).

A policy trained on the GPU (``mjx_morph_attention.py``) transfers across morphologies (the attention
arch). This banks it into the shared ``MemoryDB`` keyed by the robot's STRUCTURAL kind (``robot_kind``:
legged / mobile / manipulator) + task — NOT the specific class string — so a *new* legged build (quad,
hexapod, spider…) recalls the SAME transferable policy and starts with learned control instead of from
scratch. The body design flywheel ([[general-codesign-flywheel]]) improves bodies; this reuses brains.
"""

from __future__ import annotations


def bank_morph_policy(npz_path: str, gene, db, *, task_type: str = "locomotion",
                      skill_id: str | None = None) -> dict:
    """Evaluate a trained MorphPolicy on ``gene`` and bank it as a reusable skill keyed by morphology
    kind + task. Returns ``{skill_id, forward, upright, params_path}``."""
    from virturoid.services.morph_policy import MorphPolicy, rollout_morph
    from virturoid.services.task_matched_eval import robot_kind

    pol = MorphPolicy.from_npz(npz_path)
    r = rollout_morph(gene, pol, steps=600)
    kind = robot_kind(gene)
    sid = skill_id or f"morph_{kind}_{task_type}"
    db.record_skill(sid, kind, task_type, success_rate=float(r["forward"]), params_path=str(npz_path),
                    species=getattr(gene, "species", None), gene_id=gene.id,
                    obs_dim=int(pol.feature_dim), act_dim=int(pol.hidden),
                    notes=f"attention MorphPolicy (transfers across morphologies); fwd={r['forward']}")
    return {"skill_id": sid, "forward": float(r["forward"]), "upright": bool(r["upright"]),
            "params_path": str(npz_path)}


def recall_morph_policy(gene, db, *, task_type: str = "locomotion"):
    """Retrieve + load the best banked MorphPolicy for this morphology kind + task (warm-start / reuse).
    Returns a loaded ``MorphPolicy`` or ``None`` if nothing is banked. The policy transfers, so a hexapod
    can reuse a quadruped-banked legged policy."""
    from virturoid.services.morph_policy import MorphPolicy
    from virturoid.services.task_matched_eval import robot_kind

    sk = db.recall_skill(robot_kind(gene), task_type)
    if not sk or not sk.get("params_path"):
        return None
    try:
        return MorphPolicy.from_npz(sk["params_path"])
    except Exception:  # noqa: BLE001 - a missing/corrupt params file just means cold-start
        return None


def _feature_dim_for(gene) -> int:
    """The morphology-agnostic per-token feature width for ``gene`` (constant across bodies)."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morph_graph import encode_robot
    return int(encode_robot(mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))).feature_dim)


def transfer_train_morph(genes, db, *, task_type: str = "locomotion", params_path: str,
                         feature_dim: int | None = None, generations: int = 12, pop: int = 16,
                         steps: int = 220, seed: int = 0, train_fn=None, recall_fn=None,
                         bank_fn=None) -> dict:
    """End-to-end CPU params-mode flywheel: recall the best banked MorphPolicy for this morphology
    kind + task, **warm-start ES from it** (cold-start if nothing banked), train across ``genes``, then
    bank the result so the NEXT related build warm-starts from this one. This is "the next robot learns
    faster than the last" demonstrated on CPU with no GPU — the params branch of the flywheel that the
    GPU-trained ``recall_morph_policy`` path also feeds (startup plan §13 memory, §32 training).

    Returns ``{policy, history, warm_started, skill_id, params_path, baseline, final}``. ``train_fn`` /
    ``recall_fn`` / ``bank_fn`` are injectable so the orchestration is unit-testable without slow physics.
    """
    if train_fn is None:
        from virturoid.services.morph_trainer import train_morph_es as train_fn  # noqa: N806
    if recall_fn is None:
        recall_fn = recall_morph_policy
    if bank_fn is None:
        bank_fn = bank_morph_policy

    if feature_dim is None:
        feature_dim = _feature_dim_for(genes[0])

    init_policy = recall_fn(genes[0], db, task_type=task_type)   # the flywheel READ
    policy, history = train_fn(genes, feature_dim=feature_dim, generations=generations, pop=pop,
                               steps=steps, seed=seed, init_policy=init_policy)
    policy.to_npz(params_path, score=float(history[-1]) if history else 0.0)
    banked = bank_fn(str(params_path), genes[0], db, task_type=task_type)   # the flywheel WRITE
    return {
        "policy": policy,
        "history": history,
        "warm_started": init_policy is not None,
        "skill_id": banked.get("skill_id"),
        "params_path": str(params_path),
        "baseline": float(history[0]) if history else 0.0,
        # The ES is elitist (train_morph_es returns the BEST-seen policy, not the last generation's), so the
        # banked policy's fitness is max(history), not history[-1] — report that to avoid understating it.
        "final": float(max(history)) if history else 0.0,
    }
