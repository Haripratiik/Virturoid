"""Transfer mega-sweep (breakthrough plan WS2, the cheapest compounding lever ~20%) — pick the banked policy
that best TRANSFERS to a NEW body, to warm-start its training from.

Live evidence that motivated this (2026-07-02): a hexapod trained from scratch converges BACKWARD, but the
banked FORWARD-quad residual, deployed zero-shot on the hexapod, walks FORWARD (+0.098 CPU, +0.258 fwd_vel at
GPU iter 0) -- because the morphology-agnostic MorphPolicy carries "how to push forward" as a transferable
attention residual. So before training a new body, roll every banked policy on it (a ~1-2 s CPU rollout each,
NO training) and warm-start from the one that already goes forward. This turns a from-scratch backward-basin run
into a fine-tune of an already-forward gait -- and it is the flywheel's transfer moat made real for locomotion.
"""

from __future__ import annotations

from virturoid.services.install_paths import policy_bank_dir


def evaluate_transfer(gene, npz_path, *, steps: int = 600, decimation: int | None = None) -> dict:
    """Zero-shot: roll a banked policy on ``gene`` with NO training -> its transferred locomotion metrics.
    ``decimation`` = the CONTROL rate to deploy at (plan v2 T1.1); None -> the policy's banked ``decimation``
    (so a 50 Hz-trained checkpoint is scored at 50 Hz, matching how it was trained)."""
    from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
    pol = MorphPolicy.from_npz(npz_path)
    dec = int(decimation if decimation is not None else getattr(pol, "decimation", 1) or 1)
    r = recipe_rollout_morph(gene, pol, steps=steps, decimation=dec)
    return {"npz": npz_path, "forward": float(r.get("forward", 0.0)), "cadence": float(r.get("cadence", 0.0)),
            "upright_frac": r.get("upright_frac"), "survived": bool(r.get("survived")),
            "feature_dim": int(pol.feature_dim), "decimation": dec}


def best_checkpoint_by_deploy(gene, npz_paths, *, steps: int = 900, decimation: int | None = None):
    """Plan v2 T0.1: DEPLOY-SIM checkpoint selection. Roll each training checkpoint of ONE body on the deploy sim
    (CPU) at its control rate and return ``(best_npz, ranked)`` by forward+upright — NEVER select on train-sim
    reward (the divergence curve: MJX reward rises while CPU deploy can flip sign). Reuses ``best_transfer_seed``
    (same rank-by-forward-survived logic); here the 'candidates' are checkpoints of the same policy."""
    return best_transfer_seed(gene, list(npz_paths), steps=steps, require_survived=True, min_forward=-1e9,
                              evaluate=lambda g, c: evaluate_transfer(g, c, steps=steps, decimation=decimation))


def best_transfer_seed(gene, candidates, *, steps: int = 600, require_survived: bool = True,
                       min_forward: float = 0.02, evaluate=None):
    """Roll each banked policy in ``candidates`` (paths) zero-shot on the NEW ``gene`` and return
    ``(best_npz, ranked)``: the seed to warm-start from = the highest FORWARD (and, if ``require_survived``,
    upright) transfer above ``min_forward``, or ``None`` if none transfers forward (-> train from scratch).
    ``ranked`` is every candidate scored, best-first (for the honest report). Only policies whose architecture
    matches the target body's graph (same morphology-agnostic ``feature_dim``) are considered. ``evaluate`` is
    injectable for testing; the default runs the real zero-shot rollout."""
    _eval = evaluate or (lambda g, c: evaluate_transfer(g, c, steps=steps))
    try:
        from virturoid.services.morph_graph import encode_robot
        from virturoid.services.morph_policy import compiled_model, robot_mjcf
        target_fd = int(encode_robot(compiled_model(robot_mjcf(gene))).feature_dim)
    except Exception:  # noqa: BLE001 - if we can't compile the target, don't gate on feature_dim
        target_fd = None

    ranked = []
    for c in candidates:
        try:
            r = _eval(gene, c)
        except Exception:  # noqa: BLE001 - a broken/absent checkpoint just doesn't compete
            continue
        if target_fd is not None and r.get("feature_dim") not in (None, target_fd):
            continue                                          # architecture mismatch -> can't warm-start from it
        ranked.append(r)

    ranked.sort(key=lambda r: (bool(r.get("survived")), float(r.get("forward", 0.0))), reverse=True)
    eligible = [r for r in ranked
                if float(r.get("forward", 0.0)) > min_forward and (r.get("survived") or not require_survived)]
    return (eligible[0]["npz"] if eligible else None), ranked


def gather_banked_policies(models_dir: str = "models", *, extra=None, recursive: bool = True) -> list[str]:
    """Collect candidate banked-policy npz paths from ``models_dir`` (recursively by default) plus any ``extra``
    paths, deduped with extras first. The candidate pool for a transfer sweep."""
    from pathlib import Path
    p = policy_bank_dir(models_dir)
    found = [str(x) for x in (p.rglob("*.npz") if recursive else p.glob("*.npz"))] if p.exists() else []
    return list(dict.fromkeys(list(extra or []) + found))


def transfer_policy_for(gene, *, models_dir: str = "models", candidates=None, steps: int = 600,
                        min_forward: float = 0.02, max_candidates: int = 24):
    """Cross-embodiment warm-start selection (generalizes ``learn_locomotion.banked_policy_for`` beyond same
    species): sweep the banked pool and return ``(policy, npz, ranked)`` for the best FORWARD transfer to
    ``gene``, or ``(None, None, ranked)`` if none transfers forward (-> train from scratch).

    ``max_candidates`` BOUNDS the sweep: screening runs one rollout PER candidate, so an ever-growing banked
    pool (``build/models`` accumulates a checkpoint per training iteration) would make this scale without limit
    and eventually look like a hang. We keep the most RECENT ``max_candidates`` by mtime -- the newest banked
    skills are the most relevant warm-starts -- so screening time is bounded regardless of how much has piled up."""
    cands = candidates if candidates is not None else gather_banked_policies(models_dir)
    if max_candidates and len(cands) > max_candidates:
        import os
        cands = sorted(cands, key=lambda p: (os.path.getmtime(p) if os.path.exists(p) else 0.0),
                       reverse=True)[:max_candidates]
    best, ranked = best_transfer_seed(gene, cands, steps=steps, min_forward=min_forward)
    if best is None:
        return None, None, ranked
    from virturoid.services.morph_policy import MorphPolicy
    return MorphPolicy.from_npz(best), best, ranked
