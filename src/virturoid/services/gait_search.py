"""Learned, DEPLOYABLE gait via CEM over the crawl-gait controller — the breakthrough that closes the sim-to-sim gap.

MJX-PPO policies converge in the training sim but collapse on CPU deploy (a decimation/contact/gain sim-to-sim
gap that has resisted fixes). This module takes the opposite, robust route: it LEARNS control by optimizing the
parameters of the controller that *already deploys* — the statically-stable ``crawl_gait_rollout`` wave gait —
so the result is CPU-deployable BY CONSTRUCTION (train == deploy, same sim, same controller). The learner is
cross-entropy method (CEM): sample gaits, keep the elite few by a fitness that rewards forward travel ONLY when
the body stays upright and survives (un-gameable — a fast fall-forward scores negative), refit, repeat.

This is honest learned control: the physics is the judge, the verdict is forward+upright+survived, and the banked
policy is the actual deploy controller. Deterministic for a given seed; CPU MuJoCo; parallel across cores.

NOT quad-specific: ``crawl_gait_rollout`` is the one wave-gait engine for ANY leg count, so the same search learns
deployable walks across morphologies — measured upright+surviving walks for quadruped (1.85 m), hexapod (1.03 m),
and octopod (0.71 m); a 14-leg centipede stays upright but its many-leg propulsion is weak (honest frontier).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# search bounds per parameter (physically sensible ranges for a wide-stance quad crawl).
#
# `duty` WAS SEARCHED HERE AND IS NOW REMOVED (2026-08-01, task #265). It was never applied:
# ``crawl_gait_rollout`` derives its live swing fraction (``lift_duty``) STRUCTURALLY from the static-stability
# margin and ignores ``duty`` unless the caller passes ``live_duty=True``, which no caller here does. Measured on
# the grounded authored dog, duty 0.12 / 0.25 / 0.42 all return forward = +0.345000, identical to six decimals.
#
# It was removed rather than left in place because a dead search dimension is not merely wasted budget — it
# MANUFACTURES FALSE EVIDENCE downstream. CEM elite selection shrinks a coordinate's variance every generation
# whether or not it has signal, and fastest where it has none, because the elites are then an unbiased random
# subset. So the 86 banked walks carried duty at mean 0.2544 / stdev 0.0117 against this [0.12, 0.42] range (a
# uniform draw would be 0.270 / 0.087 — 7x wider), and ``gait_hints.mine_gait_hints`` mined that collapse and
# showed the operator "nearby walkers cluster duty near 0.25 (65 robots)" as the TIGHTEST-clustered of all six
# parameters. The flywheel was presenting its purest noise as its strongest evidence.
#
# TO REVIVE IT: thread ``live_duty=True`` into ``evaluate_gait`` -> ``_eval_batch`` -> ``search_gait`` so the CEM
# optimizes a knob that is live DURING the search, then re-bank from scratch under a NEW controller key. Never
# mix pre/post records, and never just turn ``live_duty`` on at deploy against the existing bank: measured at the
# op-point ``fit_gait_for_body`` adopts, that costs the grounded dog +2.069 m CREDIBLE WALK -> +1.431 m and the
# grounded cat +1.333 m CREDIBLE WALK -> +0.356 m FELL by ROLL-OVER.
#
# The 86 stored rows are NOT rewritten. The field is inert once nothing reads it, and a destructive migration
# buys nothing; ``recall_gait`` returns banked dicts verbatim, so an old row keeps its (ignored) duty key.
PARAM_NAMES = ("freq", "hip_amp", "knee_amp", "kp", "kd")
_LO = {"freq": 0.8, "hip_amp": 0.4, "knee_amp": 0.5, "kp": 24.0, "kd": 1.0}
_HI = {"freq": 3.2, "hip_amp": 1.5, "knee_amp": 1.5, "kp": 240.0, "kd": 14.0}


@dataclass
class GaitSearchResult:
    best_params: dict
    best_fitness: float
    best_forward: float
    best_height_ratio: float
    best_survived: bool
    baseline_forward: float
    # was the best a CREDIBLE walk (real stepping, level body) or merely a SLIDE/rearing lurch the fitness discounts?
    # Banking gates on this so the flywheel holds only real walks — a surviving slide covers distance but is not a gait.
    best_credible: bool = False
    history: list = field(default_factory=list)   # per-generation best fitness
    prior_transfer_forward: float | None = None   # zero-shot forward of a warm-start prior on THIS body (R3 screen)
    n_evals: int = 0                              # candidates drawn; baseline/deploy checks are separate
    stopped_reason: str = "generation_limit"
    # --- the error bar the search itself measured, when it was asked to search robustly (``robust_rel``) ---
    best_robust_frac: str | None = None   # "k/n" perturbed copies of the winner that still walked
    best_robust_credible: bool = False    # every perturbed copy walked -> a controller, not one lucky float
    robust_rel: float | None = None       # the relative perturbation each candidate was scored under
    n_rollouts: int = 0                   # PHYSICS ROLLOUTS actually run — the honest cost, >= n_evals

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params, "best_fitness": round(self.best_fitness, 4),
            "best_forward": round(self.best_forward, 4), "best_height_ratio": round(self.best_height_ratio, 3),
            "best_survived": self.best_survived, "best_credible": self.best_credible,
            "baseline_forward": round(self.baseline_forward, 4),
            "n_evals": self.n_evals, "stopped_reason": self.stopped_reason,
            # A VERDICT WITHOUT ITS ERROR BAR IS NOT A VERDICT. ``best_credible`` prints the same two words for a
            # controller and for one lucky float; these three say which it was, at the cost the search paid.
            "best_robust_frac": self.best_robust_frac, "best_robust_credible": self.best_robust_credible,
            "robust_rel": self.robust_rel, "n_rollouts": self.n_rollouts,
            # THE SAME SIGN DEFECT AS `forward_vel`, one layer up. This was
            #     round(abs(best_forward) / abs(baseline_forward), 2) if baseline_forward else None
            # and `forward` is a SIGNED displacement, so a learned gait that walked BACKWARD was reported as a
            # multiple of forward progress: best -0.491 m against a baseline of +0.039 m printed "12.59x
            # improvement". Two abs() calls cannot tell "went 12x further" from "went 12x further the wrong way".
            # Now: a signed ratio, defined only against a baseline that actually went forward. A backward result
            # reads NEGATIVE, and a baseline that was itself not forward makes the ratio meaningless rather than
            # impressive -> None. Both raw numbers stay in this dict, so nothing is hidden by the None.
            "improvement_x": (round(self.best_forward / self.baseline_forward, 2)
                              if self.baseline_forward > 0 else None),
            "history": [round(h, 4) for h in self.history],
        }


def _clip(vec: dict) -> dict:
    return {k: float(min(max(vec[k], _LO[k]), _HI[k])) for k in PARAM_NAMES}


# The crawl rollout emits episode-summary scalars; map the ones it measures to the reward DSL's per-episode
# feature vocabulary. HONESTY: only the six marked "real" come from the rollout; the four marked "unavailable"
# default to 0.0 (the DSL substitutes missing features with 0.0), so a reward TERM over them contributes nothing
# rather than a fabricated value. Enriching foot_clearance/energy/action_smooth needs per-step torque/foot data
# from the rollout (a follow-on); the six real features are enough for a reward to genuinely STEER the search.
def reward_features_from_rollout(r: dict) -> dict:
    """A ``REWARD_FEATURES`` dict computed from a crawl-gait rollout, for an LLM-authored reward to score."""
    return {
        # `speed` IS ALREADY SIGNED at every producer -- `morph_policy` computes it as `forward / n / dt` with
        # `forward = qpos[base_x] - x0`, an honest signed displacement, in BOTH rollouts that emit the key
        # (crawl_gait_rollout:1415, recipe_rollout_morph:990); the degenerate no-leg returns emit a literal 0.0.
        # So NEVER re-apply the sign of `forward` on top of it. This line used to do exactly that:
        #
        #     float(r.get("speed", 0.0)) * (1.0 if float(r.get("forward", 0.0)) >= 0 else -1.0)
        #
        # (-1) x (-1) = +1, so a body that walked BACKWARD reported a POSITIVE forward velocity. Measured on the
        # real Menagerie Go2 ingested through `ingest_project`: forward -0.049 m, speed -0.031 m/s, feature
        # +0.031. That is not a cosmetic reporting bug -- it INVERTS the objective. Every shipped reward template
        # that targets this feature (`reward_dsl._TEMPLATES`: velocity_track, progress_upright, smooth_march,
        # clearance_gait) then pays FULL reward for walking backward: velocity_track scored a -0.409 m/s Go2
        # 0.9997 of its maximum. The trusted screen does NOT rescue this. `gait_quality.classify` requires
        # `forward >= 0.3`, so a backward walk is never CREDIBLE and `_steered_rollout_fn` scores it
        # trusted_success = 0.0 -- but when the WHOLE field reads the same inverted feature, every candidate
        # scores 0.0, `select_reward`'s gaming test `trusted_success < median(0.0)` is false for all of them, and
        # the loop reports `gamed: false` while having spent its entire budget optimizing away from success.
        # Forward walks are unaffected either way (the old multiplier was +1 for them), so this fix changes
        # nothing except the backward half of the axis, which is the half that was wrong.
        "forward_vel": float(r.get("speed", 0.0)),                 # real (signed; +x is forward, -x is backward)
        "upright": 2.0 * float(r.get("upright_frac", 0.0)) - 1.0,   # real: [0,1] frac -> [-1,1] alignment proxy
        "height_ratio": float(r.get("height_ratio", 0.0)),         # real
        "contact_frac": float(r.get("support_frac", 0.0)),         # real
        # M6 (2026-07-24 audit): `alive` is the DOCUMENTED 0/1 survival flag (reward_dsl: "1.0 while upright,
        # 0.0 after a fall"), NOT the raw step count. The rollout reports `alive` as a STEP COUNT (steps survived
        # before a fall, e.g. 500), so passing it through scaled every `*alive` reward term by ~500x and let a
        # barely-moving body score huge. Derive the flag from `survived` (the honest all-steps-upright signal);
        # partial survival is already carried by `upright` and `height_ratio`.
        "alive": 1.0 if r.get("survived") else 0.0,  # real: 0/1 survival flag, never the step count

        "slip": abs(float(r.get("lateral", 0.0))),                 # real proxy (lateral drift = slide cost)
        "foot_clearance": 0.0,   # unavailable from the summary rollout (needs per-step foot z) -> neutral
        "energy": 0.0,           # unavailable (needs per-step torque*qvel) -> neutral
        "action_smooth": 0.0,    # unavailable (needs per-step action deltas) -> neutral
        "dist_to_goal": 0.0,     # task-specific; not a locomotion-rollout output -> neutral
    }


def evaluate_gait(gene, params: dict, *, steps: int = 1200, reward_fn=None) -> dict:
    """Roll out the crawl gait with ``params`` and return {fitness, forward, height_ratio, survived, ...}.

    Default fitness is UN-GAMEABLE: forward travel counts only when the body stays upright (height_ratio >= 0.6)
    and survives; a fall scores negative so it can never win.

    ``reward_fn`` (a compiled ``reward_dsl`` expression, ``fn(features)->float``) makes an LLM-AUTHORED reward
    STEER the search: when given, ``fitness`` becomes the reward's return over this rollout's features. The
    trusted-success signals (``credible``/``verdict`` from ``gait_quality.classify``) are computed the SAME way
    regardless and reported SEPARATELY, so ``select_reward`` can rank by trusted success and flag a reward that
    games (high return, low credibility) — the reward is optimized, success is never the reward's to define.

    ``steps`` IS PART OF THE VERDICT, not a cost knob. At and above ``gait_quality._SETTLE_MIN_STEPS`` the
    rollout can show whether the body was still walking at the end, and ``credible`` then means "walks" rather
    than "had gone a long way by the time we stopped looking". Below it the settling gate is off and this
    returns exactly what it always did. A caller whose result DECIDES something (adopt / bank / report to a
    customer) must therefore pass a horizon long enough to be judged at — see ``gait_flywheel.fit_gait_for_body``
    for why the search horizon and the judging horizon have to be the SAME number.
    """
    from virturoid.services.gait_quality import classify, settling
    from virturoid.services.morph_policy import crawl_gait_rollout
    p = _clip(params)
    r = crawl_gait_rollout(gene, steps=steps, freq=p["freq"], hip_amp=p["hip_amp"], knee_amp=p["knee_amp"],
                           kp=p["kp"], kd=p["kd"], record_qpos=True)
    # HOW MUCH PHYSICS THIS ACTUALLY BOUGHT. ``horizon_steps`` is what was ASKED for; this is what was
    # INTEGRATED before the body collapsed (or 0 when the rollout never ran a step at all). A caller that
    # declines a body "at the 6000-step horizon" is entitled to know whether any rollout ever reached it —
    # measured on the live inchworm, ``crawl_gait_rollout(steps=6000)`` returns in 0.3 s having integrated ~30
    # steps, because the leg-crawl controller has no leg to drive on a limbless spine, and 379 of those were
    # reported to a customer as a search that failed (docs/body_vs_controller_ruling.md D1).
    #
    # ``"steps" in r`` IS THE "any physics at all" TEST, and it is not a guess: ``crawl_gait_rollout``'s real
    # rollout return always carries ``steps``, while its degenerate no-graph early return (no base joint / no
    # actuated tokens) carries none — and that one claims ``alive == steps`` and ``survived: True`` for a body
    # that was never stepped. Reading ``alive`` off it would report a full horizon of physics that did not happen.
    integrated = (int(r.get("alive", 0)) if "steps" in r else 0)
    if not r.get("finite", True):
        return {"fitness": -10.0, "forward": 0.0, "height_ratio": 0.0, "survived": False,
                "cadence": 0.0, "support_frac": 0.0, "credible": False, "verdict": "non-finite",
                "reward_return": -10.0, "horizon_steps": int(steps), "rates": {}, "holds_rate": None,
                "steps_integrated": integrated}
    fwd = float(r.get("forward", 0.0))
    hr = float(r.get("height_ratio", 0.0))
    cad = float(r.get("cadence", 0.0))
    sup = float(r.get("support_frac", 0.0))
    survived = bool(r.get("survived"))
    verdict = classify(r)
    credible = verdict.startswith("CREDIBLE")
    if not survived or hr < 0.5:
        default_fitness = hr - 1.2                            # fell -> negative
    else:
        default_fitness = fwd * (1.0 if credible else 0.3)   # signed forward; non-credible discounted
    reward_return = default_fitness
    if reward_fn is not None:
        try:
            reward_return = float(reward_fn(reward_features_from_rollout(r)))
        except Exception:  # noqa: BLE001 - a bad reward scores -inf-ish, never crashes the search
            reward_return = -1e6
    fitness = reward_return if reward_fn is not None else default_fitness
    # DISCLOSE THE HORIZON AND THE RATE PROFILE. A bare `forward` is a net displacement, and a net displacement
    # cannot distinguish walking from drift-before-falling (task #267) — whoever reads this result is entitled to
    # see how long it was measured for and whether the body was still going at the end. `rates` is {} and
    # `holds_rate` is None for a horizon too short to say, which is honest rather than a default of True.
    s = settling(r)
    return {"fitness": fitness, "forward": fwd, "height_ratio": hr, "survived": survived,
            "cadence": cad, "support_frac": sup, "credible": credible, "verdict": verdict,
            "reward_return": reward_return, "horizon_steps": int(steps),
            "rates": ({int(k): round(v, 4) for k, v in s["rates"].items()} if s else {}),
            "holds_rate": (bool(s["holds_rate"]) if s else None),
            "steps_integrated": integrated}


def perturbed_params(params: dict, rel: float, rng, keys=PARAM_NAMES) -> dict:
    """ONE jointly-perturbed copy of an operating point: every searched parameter independently scaled by
    ``1 + U(-rel, rel)``.

    This is the single definition of "a nearby operating point" in the codebase, and it is shared deliberately:
    ``gait_flywheel.robustness_margin`` MEASURES an adopted point with it and ``search_gait`` OPTIMISES against
    it, so the thing being reported and the thing being selected for are the same thing. Two copies of this
    distribution would let the search quietly optimise a slightly different neighbourhood than the one the error
    bar describes.

    Relative (not absolute) because the parameters differ by two orders of magnitude — ``kd`` lives in [1, 14] and
    ``kp`` in [24, 240], so one absolute epsilon would be a rounding error on one axis and a redesign on another.
    Relative also matches how a hardware tolerance is quoted (+/- x% of a gain), which is what the margin is
    ultimately claiming something about.
    """
    return {**params, **{k: float(params[k]) * (1.0 + rng.uniform(-rel, rel))
                         for k in keys if k in params}}


def evaluate_gait_robust(gene, params: dict, *, steps: int = 1200, reward_fn=None,
                         rel: float = 0.01, n: int = 2, seed: int = 0) -> dict:
    """``evaluate_gait``, but scored by the candidate's NEIGHBOURHOOD instead of its single luckiest rollout.

    Returns the nominal result dict with five extra keys: ``robust_fitness`` (the MEAN fitness over the nominal
    point and its ``n`` perturbed copies), ``robust_frac`` ("k/n" copies that still walked), ``robust_rank``
    (that fraction as a number, or -1.0 for a candidate that did not walk at all), ``robust_credible`` (every
    copy walked) and ``n_rollouts`` (what this cost).

    WHY THE MEAN AND NOT THE WORST. Both were tried. A single fall scores ``height_ratio - 1.2`` — around -0.7 —
    while a non-credible SLIDE that merely survives scores ``0.3 * forward >= 0``, so a worst-case rule ranks a
    knife-edge walk BELOW a body that never lifts a foot, and the CEM mean then migrates out of the walking
    region entirely. The mean discounts fragility hard while staying a smooth signal, and the ORDERING never
    rests on it alone: ``_robust_key`` puts ``robust_rank`` first, so the mean only breaks ties between
    candidates whose neighbourhoods held up equally often.

    SCREENED, so this is nearly free where it does not matter: a candidate that does not walk at its OWN
    parameters cannot be robust, and spending ``n`` more rollouts to confirm that takes budget from the search.
    Only a nominally credible candidate pays for probes — measured on the cat, 2 of 97 candidates.

    ``seed`` fixes the perturbation draws. A caller that will later REPORT a margin must measure it with a
    DIFFERENT seed than it searched under, or the error bar is quoted on the same draws it was optimised
    against; ``gait_flywheel`` holds one out for exactly that reason.
    """
    import random

    base = evaluate_gait(gene, params, steps=steps, reward_fn=reward_fn)
    base["robust_rel"] = float(rel)
    base["n_rollouts"] = 1
    if n < 1 or not (bool(base.get("credible")) and bool(base.get("survived"))):
        base["robust_fitness"] = float(base["fitness"])
        base["robust_frac"] = None
        base["robust_credible"] = False
        base["robust_rank"] = -1.0            # did not walk at its own numbers -> below every walker
        return base
    rng = random.Random(seed)
    total = float(base["fitness"])
    ok = 0
    for _ in range(int(n)):
        r = evaluate_gait(gene, perturbed_params(params, rel, rng), steps=steps, reward_fn=reward_fn)
        total += float(r["fitness"])
        ok += int(bool(r.get("credible")) and bool(r.get("survived")))
    base["n_rollouts"] = 1 + int(n)
    base["robust_fitness"] = total / float(1 + int(n))
    base["robust_frac"] = f"{ok}/{int(n)}"
    base["robust_credible"] = ok == int(n)
    base["robust_rank"] = float(ok) / float(n)
    return base


def _robust_key(r: dict) -> tuple:
    """Order candidates: HOW MUCH OF THE NEIGHBOURHOOD WALKS first, distance only inside that.

    ``robust_rank`` is the FRACTION of perturbed copies that walked (-1.0 for a candidate that did not walk at
    its own numbers, so it is below every walker). Lexicographic rather than a weighted scalar because the two
    questions are not commensurable — no amount of extra travel makes an operating point that no perturbed copy
    of itself can repeat into a deployable one.

    IT IS A FRACTION AND NOT A BOOLEAN, and that was MEASURED, not chosen. The first version keyed on
    "every copy walked", so every non-robust candidate tied at 0 and the mean fitness broke the tie — which let
    a NON-CREDIBLE SLIDE (fitness ``0.3 * forward``, always >= 0) outrank a fragile but real WALK (mean of one
    good rollout and two falls, ~0.8). On the grounded authored cat that flipped the fit from "adopts a fragile
    walk, flagged and unbanked" to "adopts nothing and ships the default, which falls" — a strictly worse
    outcome dressed as a stricter one. Grading by fraction keeps every walker above every non-walker while still
    preferring the wider basin, which is the whole point.

    With robust scoring off, ``robust_rank`` is absent, every candidate ties at 0.0 and this degenerates to
    today's fitness ordering exactly.
    """
    return (float(r.get("robust_rank", 0.0)), float(r.get("robust_fitness", r.get("fitness", -1e9))))


def search_gait(gene, *, generations: int = 8, pop: int = 24, elite_frac: float = 0.3,
                steps: int = 1000, seed: int = 0, workers: int | None = None,
                warm_start: dict | None = None, progress=None, reward_fn=None,
                max_evals: int | None = None, stop_on_credible: bool = False,
                robust_rel: float | None = None, robust_n: int = 2,
                hold_exploration: bool = True) -> GaitSearchResult:
    """CEM over the crawl-gait parameters. Returns the best DEPLOYABLE gait found for ``gene``.

    ``warm_start`` (a prior gait's params, e.g. recalled from the flywheel for a STRUCTURALLY-SIMILAR body) seeds
    the CEM mean and narrows the initial spread, so the search EXPLOITS the specific prior instead of cold-starting
    — this is the flywheel compounding: a learned gait for one quadruped accelerates the next quadruped's search.

    ``robust_rel`` TURNS THIS INTO A ROBUSTNESS-AWARE SEARCH, and it is the answer to task #267. Off (the
    default), every candidate is one rollout and the winner is whichever single draw scored highest — which on
    the grounded authored cat is a point where a **2.4e-5 relative change in step frequency flips CREDIBLE WALK
    to FELL by ROLL-OVER**. Optimising a scalar computed from one deterministic rollout selects for exactly that:
    the highest point in the landscape is systematically a spike, because a spike is what a maximum of a rough
    function looks like. On, each candidate is scored by its NEIGHBOURHOOD at relative size ``robust_rel``
    (``evaluate_gait_robust``), candidates are ranked robust-first (``_robust_key``), and ``stop_on_credible``
    stops only for a candidate whose perturbed copies ALSO walk — so an early lucky draw no longer ends the
    search. Same instinct as the morphology domain randomisation already in the trainer, pointed at the gait
    parameters instead of at the body.

    ``hold_exploration`` (robust mode only) stops the CEM from NARROWING while nothing has walked yet — see the
    block at the elite update. Robust ranking fixes WHICH point is chosen; this is about whether the search can
    reach a good one at all, which is the constraint task #267 measured as binding.

    THE SEED AXIS is covered where it actually lives. ``crawl_gait_rollout`` is deterministic in
    ``(gene, params)`` — there is no rollout seed to perturb — so the only stochastic axis is the SEARCH draw,
    and that is handled one level up by ``gait_flywheel.fit_gait_for_body``'s decorrelated ``seed_restarts``
    (measured: the authored dog walks on seeds 0/3/4 and falls on 1/2). Claiming a seed-robustness probe inside
    this function would be measuring the same number twice.
    """
    import numpy as np

    if generations < 1 or pop < 1:
        raise ValueError("generations and pop must both be >= 1")
    budget = generations * pop if max_evals is None else int(max_evals)
    if budget < 1:
        raise ValueError("max_evals must be >= 1")

    rng = np.random.default_rng(seed)
    # Exploration stays BROAD whether or not there is a prior — a prior must never straitjacket the search
    # (measured: seeding the mean + tightening std around a prior gait caused NEGATIVE transfer). Instead the
    # prior is INJECTED as one guaranteed candidate each generation: if it transfers it wins immediately and CEM
    # exploits it (compounding); if it is a bad transfer it is simply one of `pop` candidates and does no harm.
    mean = np.array([(_LO[k] + _HI[k]) / 2.0 for k in PARAM_NAMES])
    std = np.array([(_HI[k] - _LO[k]) / 4.0 for k in PARAM_NAMES])
    prior_vec = None
    prior_transfer = None
    if warm_start:
        prior_vec = np.array([float(warm_start.get(k, mean[i])) for i, k in enumerate(PARAM_NAMES)])
    n_elite = max(2, int(pop * elite_frac))

    def as_params(vec):
        return {k: float(vec[i]) for i, k in enumerate(PARAM_NAMES)}

    # baseline = the SHIPPED crawl-gait defaults (crawl_gait_rollout's own defaults), so "improvement" is honest
    # (learned vs the default controller), not vs an arbitrary center-of-bounds point.
    baseline = evaluate_gait(gene, {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0,
                                    "kp": 32.0, "kd": 1.5}, steps=steps, reward_fn=reward_fn)
    if prior_vec is not None:                                  # transfer-screen the prior ONCE (dossier R3)
        prior_transfer = evaluate_gait(gene, _clip(as_params(prior_vec)), steps=steps, reward_fn=reward_fn)
    # ``robust_rank`` -2.0 puts the empty seed below even a candidate that did not walk (-1.0), so the FIRST
    # result always replaces it — otherwise a generation in which nothing walked would leave ``best_params`` at
    # the untested centre of the bounds.
    best = {"fitness": -1e9, "robust_fitness": -1e9, "robust_rank": -2.0}
    best_params = as_params(mean)
    history: list[float] = []
    n_evals = 0
    n_rollouts = 1 + (1 if prior_transfer is not None else 0)   # the baseline (and the prior screen) are rollouts too
    stopped_reason = "max_evals" if max_evals is not None else "generation_limit"
    robust = robust_rel is not None and float(robust_rel) > 0.0
    #: the perturbation draws THE SEARCH optimises against. Derived from the search seed so a restart re-draws
    #: them, and deliberately far from the seed ``gait_flywheel`` reports its margin under — see
    #: ``evaluate_gait_robust``: an error bar measured on the draws it was fitted to is not an error bar.
    probe_seed = int(seed) * 7919 + 104729

    def _score(params):
        if not robust:
            return evaluate_gait(gene, params, steps=steps, reward_fn=reward_fn)
        return evaluate_gait_robust(gene, params, steps=steps, reward_fn=reward_fn,
                                    rel=float(robust_rel), n=int(robust_n), seed=probe_seed)

    #: the semantic stop signal. Without robust scoring it is "the horizon called this credible", which is what
    #: shipped a point that a 2.4e-5 change in one parameter flips to a roll-over; with it, a candidate must also
    #: survive being perturbed before it is allowed to end the search.
    #:
    #: ``stopped_reason`` KEEPS THE STRING ``credible_walk`` either way — the reason really is "a credible walk
    #: was found", the robust bar only makes that harder to clear — because callers compare it literally
    #: (``agent_design_tools`` prints ``stopped_reason == 'credible_walk'``) and a renamed constant would turn a
    #: STRONGER result into a report of no early stop at all. What changed is carried by ``best_robust_frac`` /
    #: ``best_robust_credible``, which say which bar was cleared.
    _stop_key = "robust_credible" if robust else "credible"
    _stop_name = "credible_walk"

    for g in range(generations):
        remaining = budget - n_evals
        if remaining <= 0:
            break
        batch_n = min(pop, remaining)
        samples = rng.normal(mean, std, size=(batch_n, len(PARAM_NAMES)))
        if prior_vec is not None:
            samples[0] = prior_vec                             # inject the prior as one guaranteed candidate
        params_list = [_clip(as_params(s)) for s in samples]
        # Credibility is a semantic stop signal, not a scalar fitness threshold. Evaluate serially in this mode so
        # the first verified walk actually saves rollouts; pre-launching a population would only pretend to stop.
        if stop_on_credible:
            results = []
            for params in params_list:
                result = _score(params)
                results.append(result)
                n_evals += 1
                n_rollouts += int(result.get("n_rollouts", 1))
                if bool(result.get(_stop_key)):
                    stopped_reason = _stop_name
                    break
        elif robust:
            # Serial WITHIN a candidate (its probes depend on its own nominal result), parallel ACROSS them —
            # otherwise turning robust scoring on would quietly cancel a caller's ``workers``, which is the one
            # setting that pays for the extra rollouts.
            results = _eval_batch(gene, params_list, steps, workers, reward_fn=reward_fn,
                                  robust=(float(robust_rel), int(robust_n), probe_seed))
            n_evals += len(results)
            n_rollouts += sum(int(r.get("n_rollouts", 1)) for r in results)
        else:
            results = _eval_batch(gene, params_list, steps, workers, reward_fn=reward_fn)
            n_evals += len(results)
            n_rollouts += len(results)
        samples = samples[:len(results)]
        params_list = params_list[:len(results)]
        if robust:
            order = sorted(range(len(results)), key=lambda i: _robust_key(results[i]), reverse=True)
        else:
            order = list(np.argsort(np.array([r["fitness"] for r in results]))[::-1])
        elite = samples[[int(i) for i in order[:min(len(results), n_elite)]]]
        mean = elite.mean(axis=0)
        elite_std = elite.std(axis=0) + 1e-3                    # keep exploration alive
        if hold_exploration and robust and not any(bool(r.get("credible")) for r in results):
            # NOTHING IN THIS GENERATION WALKED, so the elites are the best of a set of falls and their spread
            # carries no information about where a walk is. CEM shrinks the sampling width every generation
            # regardless — and FASTEST where there is no signal, because unranked elites are then an unbiased
            # random subset. This module has already been burned by exactly that: the dead ``duty`` coordinate
            # (see PARAM_NAMES) collapsed to a 7x-too-narrow band and the flywheel mined the collapse as its
            # strongest evidence. Collapsing the WHOLE distribution around noise is the same error with a worse
            # consequence — it is how the search stops being able to REACH a robust point at all, which task
            # #267 measures as the binding constraint (a full 96-eval budget on the grounded authored cat turns
            # up 2 credible candidates of 97 at seed 0 and none at two other seeds, while a point that is 8/8
            # robust and 30% faster exists in the same box).
            #
            # So: keep moving the mean (fall height still carries a little signal) but DO NOT narrow until
            # something walks. Until then this is honest random search over the region, which at a 2% hit rate
            # beats a confident collapse into the wrong basin.
            std = np.maximum(elite_std, std)
        else:
            std = elite_std
        stop_indices = [i for i, r in enumerate(results) if bool(r.get(_stop_key))]
        gbest_i = (max(stop_indices, key=lambda i: _robust_key(results[i]))
                   if stopped_reason == _stop_name and stop_indices else int(order[0]))
        if _robust_key(results[gbest_i]) > _robust_key(best):
            best = results[gbest_i]
            best_params = params_list[gbest_i]
        history.append(float(best["fitness"]))
        if progress:
            progress(f"gen {g + 1}/{generations}: best fitness {best['fitness']:+.3f} "
                     f"(forward {best['forward']:+.3f} m, hr {best['height_ratio']:.2f}, "
                     f"survived {best['survived']}"
                     + (f", neighbours {best.get('robust_frac')}" if robust else "") + ")")
        if stopped_reason == _stop_name:
            break

    return GaitSearchResult(
        best_params=best_params, best_fitness=float(best["fitness"]), best_forward=float(best["forward"]),
        best_height_ratio=float(best["height_ratio"]), best_survived=bool(best["survived"]),
        best_credible=bool(best.get("credible", False)),
        baseline_forward=float(baseline["forward"]), history=history,
        prior_transfer_forward=(float(prior_transfer["forward"]) if prior_transfer is not None else None),
        n_evals=n_evals, stopped_reason=stopped_reason,
        best_robust_frac=best.get("robust_frac"), best_robust_credible=bool(best.get("robust_credible", False)),
        robust_rel=(float(robust_rel) if robust else None), n_rollouts=n_rollouts)


def _eval_batch(gene, params_list, steps, workers, reward_fn=None, robust=None):
    """Evaluate a population, in parallel across processes when possible (falls back to serial).

    A compiled ``reward_fn`` (a code object closing over ``eval``) is NOT picklable, so a reward-steered search
    runs SERIAL — correct and safe; the reward loop uses modest pop/generations so the cost is bounded.

    ``robust`` is ``(rel, n, seed)`` when each candidate is to be scored across its neighbourhood. Every element
    is a plain number, so the robust path parallelises exactly like the plain one — deliberately, because
    robustness-aware search is the mode that most needs a caller's ``workers``."""
    def _one(p):
        if robust is None:
            return evaluate_gait(gene, p, steps=steps, reward_fn=reward_fn)
        rel, n, seed = robust
        return evaluate_gait_robust(gene, p, steps=steps, reward_fn=reward_fn, rel=rel, n=n, seed=seed)

    if reward_fn is not None or workers is None or workers <= 1:
        return [_one(p) for p in params_list]
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_worker, [(gene, p, steps, robust) for p in params_list]))
    except Exception:  # noqa: BLE001 - multiprocessing/pickling issues -> serial
        return [_one(p) for p in params_list]


def _worker(args):
    gene, params, steps, robust = args
    if robust is None:
        return evaluate_gait(gene, params, steps=steps)
    rel, n, seed = robust
    return evaluate_gait_robust(gene, params, steps=steps, rel=rel, n=n, seed=seed)
