"""Sim-to-sim transfer measurement — the design-stage "Evaluating Bits, not Atoms" instrument (plan §4.2).

We can't yet measure sim-to-REAL transfer (no hardware), but we CAN measure the next-best honest proxy: does a
candidate's score under NOMINAL dynamics predict its score under PERTURBED dynamics (domain randomization)? If
the ranking holds, the evaluation is trustworthy; if it inverts, the nominal sim is lying about which design /
policy is better. This is the SIMPLER methodology (Pearson r + MMRV) applied sim-to-sim, so every skill can carry
an honest transfer number instead of an unverified sim success rate.

Two metrics, per SIMPLER (arXiv:2405.05941):
- Pearson r        — linear correlation between nominal and perturbed scores (1 = perfectly predictive).
- MMRV             — Mean Maximum Rank Violation: the average worst-case performance you'd lose by trusting the
                     nominal ranking. 0 = the nominal ranking never misleads; higher = more sim2real risk.

The metrics are pure functions (unit-tested on synthetic data); ``run_sim2sim`` drives any pair of eval callables,
and ``locomotion_sim2sim`` binds it to the real recipe rollout under nominal vs randomized dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass


def pearson_r(a: list, b: list) -> float:
    """Pearson correlation between two score lists. 0.0 for degenerate input (constant or <2 points)."""
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    return cov / (va ** 0.5 * vb ** 0.5)


def mmrv(nominal: list, perturbed: list) -> float:
    """Mean Maximum Rank Violation. For each candidate i, find the worst case where the NOMINAL sim says i is
    at-least-as-good as j (``nominal[i] >= nominal[j]``) but PERTURBED reality disagrees (``perturbed[j] >
    perturbed[i]``), and record that performance gap ``perturbed[j] - perturbed[i]``; average the per-candidate
    worst gaps. 0.0 means the nominal ranking is never contradicted (safe to trust). Units = the score's units."""
    n = len(nominal)
    if n < 2 or n != len(perturbed):
        return 0.0
    total = 0.0
    for i in range(n):
        worst = 0.0
        for j in range(n):
            if nominal[i] >= nominal[j] and perturbed[j] > perturbed[i]:
                worst = max(worst, perturbed[j] - perturbed[i])
        total += worst
    return total / n


def _verdict(r: float, mmrv_val: float, score_span: float) -> str:
    """A plain reading of transfer quality. ``score_span`` (max-min of perturbed scores) normalizes MMRV."""
    rel = mmrv_val / score_span if score_span > 1e-9 else 0.0
    if r >= 0.8 and rel <= 0.1:
        return "strong — the nominal sim reliably predicts perturbed performance"
    if r >= 0.5 and rel <= 0.25:
        return "moderate — nominal ranking mostly holds; verify the top candidates under perturbation"
    return "weak — nominal scores do NOT predict perturbed performance; do not trust nominal ranking alone"


@dataclass
class Sim2SimResult:
    labels: list
    nominal: list
    perturbed: list
    pearson_r: float
    mmrv: float
    verdict: str

    def to_dict(self) -> dict:
        return {"n": len(self.labels),
                "pearson_r": round(self.pearson_r, 4), "mmrv": round(self.mmrv, 4), "verdict": self.verdict,
                "candidates": [{"label": l, "nominal": round(a, 4), "perturbed": round(b, 4)}
                               for l, a, b in zip(self.labels, self.nominal, self.perturbed)]}


def transfer_report(nominal: list, perturbed: list, labels: list | None = None) -> Sim2SimResult:
    """Compute the sim2sim transfer report from paired nominal/perturbed scores."""
    labels = labels or [f"cand_{i}" for i in range(len(nominal))]
    span = (max(perturbed) - min(perturbed)) if perturbed else 0.0
    r, mv = pearson_r(nominal, perturbed), mmrv(nominal, perturbed)
    return Sim2SimResult(labels, list(nominal), list(perturbed), r, mv, _verdict(r, mv, span))


def run_sim2sim(candidates: list, eval_nominal, eval_perturbed, *, label_fn=None) -> Sim2SimResult:
    """Score each candidate under a nominal and a perturbed evaluator, then report transfer. ``eval_nominal`` /
    ``eval_perturbed`` map a candidate -> float score; ``label_fn`` maps a candidate -> str (default: index)."""
    labels = [(_safe_label(label_fn, c, i)) for i, c in enumerate(candidates)]
    nominal = [float(eval_nominal(c)) for c in candidates]
    perturbed = [float(eval_perturbed(c)) for c in candidates]
    return transfer_report(nominal, perturbed, labels)


def _safe_label(label_fn, c, i: int) -> str:
    if label_fn is None:
        return f"cand_{i}"
    try:
        return str(label_fn(c))
    except Exception:  # noqa: BLE001
        return f"cand_{i}"


def locomotion_sim2sim(gene, candidates, *, seed: int = 0, gain: float = 0.15, mass: float = 0.1,
                       damping: float = 0.2, friction: float = 0.3, steps: int = 600) -> Sim2SimResult:
    """Real sim-to-sim for locomotion. Each candidate is a ``(label, MorphPolicy)`` pair (e.g. competing
    checkpoints or body scales); score forward travel under NOMINAL dynamics vs under ONE randomized-dynamics
    realization (the SAME perturbation applied to every candidate, so their ranking is comparable), and report
    whether the nominal ranking predicts the perturbed one. Best-effort: needs MuJoCo + ``morph_policy``."""
    import numpy as np

    from virturoid.services.morph_policy import recipe_rollout_morph
    rng = np.random.default_rng(seed)
    g = 1.0 + gain * (2 * rng.random() - 1)
    m = 1.0 + mass * (2 * rng.random() - 1)
    d = 1.0 + damping * rng.random()
    f = 1.0 + friction * (2 * rng.random() - 1)

    def _perturb(model):                                     # the fixed DR realization shared across candidates
        model.actuator_gainprm[:, 0] *= g
        model.body_mass[1:] *= m
        model.body_inertia[1:] *= m
        model.dof_damping[:] *= d
        model.geom_friction[:, 0] *= f

    labels = [c[0] for c in candidates]
    nominal = [float(recipe_rollout_morph(gene, c[1], steps=steps, seed=seed).get("forward", 0.0))
               for c in candidates]
    perturbed = [float(recipe_rollout_morph(gene, c[1], steps=steps, seed=seed, model_perturb=_perturb).get("forward", 0.0))
                 for c in candidates]
    return transfer_report(nominal, perturbed, labels)
