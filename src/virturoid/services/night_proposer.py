"""Night-shift task-proposal policy (breakthrough plan v2 §5.2) — the Dreamer-mode brain that decides WHICH
(body, task) candidates to work on next, so the autonomous loop keeps DISCOVERING instead of grinding one body.
(Distinct from ``task_proposer.py``, which compiles ONE prompt into a verifiable TaskSpec.)

Three arms (stream-4 synthesis; POET / Enhanced-POET / OMNI-EPIC / Absolute-Zero):
  * QD-FRONTIER MUTATION (~55%): perturb a recently-improved QD-archive elite (cheap, reliably grows ANNECS-V).
  * TRANSFER MEGA-SWEEP  (~30%): pair an elite's champion with ANOTHER task/body (eval-only compute, the
    cheapest compounding lever — our ``transfer_seed`` makes it nearly free).
  * LLM EXPLORER          (~15%): a language model proposes a genuinely NEW task family (expensive, §4 spend-gated).
Anneal: if the multi-night ANNECS-V slope flattens, shift weight from mutation to the LLM explorer.

LEARNABILITY BAND (Absolute Zero): a proposal earns a full search only if it's neither trivial nor impossible
for the current solver — accept iff a cheap probe's solve-rate ∈ [0.30, 0.70]; propose-reward = ``0 if rbar in
{0,1} else 1-rbar``. Arms + probe are INJECTABLE callables so this unit-tests with no physics/LLM; production
binds them to the QD archive, the transfer pool, and the spend-guarded LLM, and feeds the survivors to
``night_shift.run_night_shift``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def learnability_reward(solve_rate: float) -> float:
    """Absolute-Zero propose reward: 0 for a trivially-solved (rbar=1) or impossible (rbar=0) task, else 1-rbar
    (peaks on the hard-but-doable frontier)."""
    r = float(solve_rate)
    if r <= 0.0 or r >= 1.0:
        return 0.0
    return 1.0 - r


def in_learnability_band(solve_rate: float, *, lo: float = 0.30, hi: float = 0.70) -> bool:
    """Accept a proposal iff the probe solve-rate is in the learnable band (not too easy, not too hard)."""
    return lo <= float(solve_rate) <= hi


@dataclass
class NightProposer:
    """Draw candidate ``{id, gene, task, arm, ...}`` dicts from the three arms in the given ratios. ``mutate`` /
    ``transfer`` / ``explore`` are callables ``() -> candidate|None`` (injected; production binds them to the QD
    archive, the transfer pool, and the spend-guarded LLM). RNG is seeded -> deterministic draws for tests. Arms
    that are None (e.g. LLM off) are skipped and their weight renormalizes over the wired arms."""
    mutate: object = None
    transfer: object = None
    explore: object = None
    ratios: tuple = (0.55, 0.30, 0.15)               # (mutate, transfer, explore)
    seed: int = 0
    _rng: object = field(default=None, repr=False)

    def __post_init__(self):
        import random
        self._rng = random.Random(self.seed)

    def _wired(self):
        arms = [("mutate", self.mutate), ("transfer", self.transfer), ("explore", self.explore)]
        return [(name, fn, w) for (name, fn), w in zip(arms, self.ratios) if fn is not None]

    def _pick_arm(self) -> str:
        avail = self._wired()
        if not avail:
            return "none"
        total = sum(w for _, _, w in avail) or 1.0
        x = self._rng.random() * total
        acc = 0.0
        for name, _fn, w in avail:
            acc += w
            if x <= acc:
                return name
        return avail[-1][0]

    def propose(self, n: int) -> list:
        """Propose up to ``n`` candidates, tagging each with the arm that produced it. Arms returning None are
        skipped (empty archive / LLM off) — the loop just gets fewer candidates, never crashes."""
        fns = {"mutate": self.mutate, "transfer": self.transfer, "explore": self.explore}
        out, attempts = [], 0
        while len(out) < n and attempts < n * 5:
            attempts += 1
            arm = self._pick_arm()
            if arm == "none":
                break
            try:
                cand = fns[arm]()
            except Exception:  # noqa: BLE001 - a flaky arm must not kill the night; skip this draw
                cand = None
            if cand is not None:
                cand = dict(cand)
                cand.setdefault("arm", arm)
                out.append(cand)
        return out


def filter_learnable(candidates: list, probe_solve_rate, *, lo: float = 0.30, hi: float = 0.70) -> list:
    """Keep only candidates whose ``probe_solve_rate(cand) -> float`` lands in the learnability band, annotating
    each with its ``learnability`` reward. A probe error drops the candidate (fail-closed: never assume learnable)."""
    kept = []
    for c in candidates:
        try:
            rate = float(probe_solve_rate(c))
        except Exception:  # noqa: BLE001 - a failed probe -> not learnable
            continue
        if in_learnability_band(rate, lo=lo, hi=hi):
            c = dict(c)
            c["probe_solve_rate"] = rate
            c["learnability"] = learnability_reward(rate)
            kept.append(c)
    return kept
