"""ONE LAUNCH that answers #255 for a BIPED: does learned control find balance on two legs?

    PYTHONPATH=src python scripts/gpu_biped_balance_probe.py

Everything below is pre-decided so the run needs no judgement at launch time. It is staged, not run:
the GPU box was offline when it was written (`gpu_available()` False, `tailscale status` "offline, last
seen 4d ago"), so this script exists to be executed unchanged the moment the box is back.

-------------------------------------------------------------------------------------------------
THE QUESTION
-------------------------------------------------------------------------------------------------
#206 established that the scripted crawl gait CANNOT balance two legs structurally: `select_duty`
returns beta=0.5 at n_legs<=2, so exactly one leg swings at every instant and the engine's premise
("the rest hold the centre of mass") is unsatisfiable, and the PD target is phase-scheduled with no
balance term at all. #255 established that GPU PPO learns forward locomotion on a composed QUADRUPED.
This run asks whether that transfers to two legs.

The CPU arm of this question was already run and ANSWERED IN PART, and its numbers are the NULLS this
script re-measures locally before spending any GPU time, so the two probes are directly comparable and
the GPU verdict lands against the identical bar. MEASURED on CPU (12 independent OpenAI-ES seeds,
pop 24, 2540 generations, 63,754 training episodes, ~30 min wall on 8 cores, survival-only fitness):

    2/12 seeds reached 6000/6000 at height_ratio 0.995 -- upright, both feet planted, roll <=0.7 deg
    6/12 seeds beat the 1607-step do-nothing null; 6/12 never did
    the ablations broke it: const-action 822 +/- 326 and trunk-blind 638 +/- 191 against full
      2393 +/- 1733 (paired t = +3.10 and +3.62, n=12) -- so it is FEEDBACK, not a pose
    and both solved seeds have cadence 0.00 and |forward| <= 0.006 m: THEY STAND, THEY DO NOT WALK

So the open question this GPU run inherits is NOT "can two legs be balanced at all" -- CPU already
showed yes, by feedback. It is whether a full RL trainer with a forward-velocity objective can hold
that balance WHILE MOVING, which is what CPU ES never produced and what #255 demonstrated on a quad.
Read the verdict block at the bottom of this docstring with that in mind: beating 1607 is no longer
the interesting bar; beating it with cadence > 0 and real forward travel is.

-------------------------------------------------------------------------------------------------
WHY THESE NUMBERS
-------------------------------------------------------------------------------------------------
--ep-len 3000 (6.0 s of physics; 750 control steps at decimation 4)
    #258 root-caused a whole class of bad GPU verdicts: the training objective never OBSERVED the
    fall, because the episode window (1.0 s) was SHORTER than the time-to-fall (1.4 s), so
    alive_bonus was identical for every checkpoint and the reward carried zero information about
    falling. On this biped the nulls are: scripted crawl falls at 0.97 s, do-nothing PD hold falls
    at 3.21 s. An episode must therefore be comfortably longer than 3.21 s or the objective is blind
    to the only thing being asked about. 6.0 s is 1.9x the do-nothing null. It is also exactly the
    horizon the CPU ES trained on, and CPU deploy is scored at 12.0 s in both — same ratio, so
    "trained short, judged long" is the same test on both engines.
    #256 separately measured that LENGTHENING the horizon is not itself the lever (delta ~-8.5
    steps on the quad at 4 s). 6.0 s is chosen to make the fall visible, not to fix training.

--envs 192
    #239's headline ("MJX collapses at POWER-OF-TWO env counts; use N=192") was REFUTED by a
    controlled sweep: 64 and 512 are among the FASTEST and 128/256 among the slowest, and non-power
    N=96 is 3x worse than N=128. THE ENV COUNT WAS NEVER THE CAUSE. The cause was our own emitter --
    `gene_compiler` emitted `cone="elliptic"` with NO iteration caps, so MuJoCo defaulted to Newton
    `iterations=100, ls_iterations=50`, and the jitted MJX kernel cannot early-exit on convergence,
    so every env paid the full nominal count every step. Capping removed the cliff entirely and the
    measured optimum was N=1024 at 39,618 env-steps/s (N>=4096 OOMs at 16.18 GiB).
    HALF of that fix has since landed: `gene_compiler._PHYSICS_OPTION` now emits `iterations="20"`.
    The other half deliberately did NOT land -- `ls_iterations="8"` substitutes a customer's body and
    `cone="pyramidal"` costs a shipped verdict, both measured, see the comment block at
    src/virturoid/services/gene_compiler.py:104-137. So throughput here sits BETWEEN the 102-183
    env-steps/s collapse and the 39,618 ceiling, and is not known in advance.
    192 is kept as the DEFAULT only because it is the one env count measured to hold under real
    training load (#205), i.e. it is the conservative choice, NOT the fast one. If the --smoke
    preflight shows headroom, raise it: `--envs 1024` is the measured optimum under full caps and
    there is no env-count reason to avoid it. Do not read 192 as a tuned value.

--iters 200
    200 x 192 x 3000 = 115.2M physics steps, ~4x the 26.9M of #256's 4-second retest. Sized to be
    one bounded launch, not a campaign.

recipe: taken from `gpu_trainer.default_training_recipe(gene)`, NOT hand-rolled here, so this run
    trains under whatever the product's single source of truth says today. On a 2-leg body that
    resolves to cpg=False / phase_obs=False (the >=3-leg gate; a humanoid measured 0.34 m stable
    WITHOUT the trot-CPG vs 0.16 m with it) plus dr / contact_dr / sphere_feet / real_actuator /
    decimation 4. #255 separately measured the CPG prior COSTS ~40 iterations on a composed quad,
    so its absence here is a help, not a handicap.

-------------------------------------------------------------------------------------------------
WHICH CHECKPOINT IT KEEPS  -- the load-bearing part
-------------------------------------------------------------------------------------------------
#256 measured that training RISES THEN COLLAPSES: on the composed quad the it30 checkpoint survived
1813/2000 steps while the FINAL managed 755, so any figure quoted from a final checkpoint understates
the body ~2.4x. Checkpoint selection is the lever.

So this script trains with --keep-checkpoints, fetches every numbered checkpoint, and scores ALL of
them -- plus the final -- on CPU DEPLOY (`recipe_rollout_morph`, 6000 physics steps, the same rollout
and the same horizon the CPU probe and the crawl baseline were measured on). It keeps the checkpoint
with the longest CPU-deploy survival, ties broken by forward travel, and it PRINTS BOTH the kept one
and the final one, with their ratio, so the understatement is visible in the log rather than inferred.

Selection is on CPU deploy, never on MJX train reward: the MJX->CPU divergence is measured, and #257
found that non-CPG artifacts were being scored under a controller they were never trained with.
`recipe_rollout_morph` adopts each artifact's own banked decimation / action_lpf / sphere_feet /
adaptive-gains / recipe-control marker, so every checkpoint is deployed as it was trained.

-------------------------------------------------------------------------------------------------
WHAT COUNTS AS AN ANSWER
-------------------------------------------------------------------------------------------------
Survival is the proxy, and it is only meaningful against BOTH nulls, because on this body the
do-nothing null BEATS the scripted gait:

    scripted crawl        487 / 6000 steps   (0.97 s)   -- the #206 controller
    do-nothing PD hold   1607 / 6000 steps   (3.21 s)   -- zero learned offset, stand there
    random MorphPolicy    726 +/- 64         (n=8)      -- the untrained-policy distribution

  beating 487 but not 1607  -> NOT an answer. Standing still already does that.
  beating 1607             -> the learned class found something the body does not do passively.
  and then the ABLATIONS decide what that something is (run automatically, same code path as the
  CPU probe): replace the policy with a CONSTANT action equal to its own mean action -> if survival
  holds, it found a POSE, not a feedback law. Freeze the 9 global observation channels (base height,
  world-up . body-up, base linear+angular velocity) at their t=0 values -> if survival holds, whatever
  it learned is not read off the trunk.

A biped that survives by standing still is a real result and must be reported as one, not as walking:
`forward_m` and `cadence` are printed for every kept checkpoint precisely so that cannot be blurred.
That is exactly what CPU ES produced, so a GPU run that only reproduces it has added nothing.

A fourth number worth having, and cheap: run `robustness.py` (scratchpad) or
`morph_policy.recipe_robustness` on the kept checkpoint. On CPU the do-nothing PD hold scores
1446 +/- 49 under per-trial actuator-gain / mass / damping / friction randomization with 0/10 reaching
the horizon, while the better of the two solved ES policies scored 3570 +/- 2581 with 5/10 reaching
it -- so the learned balance is not a knife-edge fitted to one exact model, and neither is it robust.

-------------------------------------------------------------------------------------------------
COST
-------------------------------------------------------------------------------------------------
115.2M physics steps. At the 39,618 env-steps/s ceiling measured under full solver caps that is
~48 min; at a pessimistic 20k/s it is ~96 min. Neither is known for the shipped
`iterations=20`/elliptic contract -- MEASURE IT FIRST:

    PYTHONPATH=src python scripts/gpu_biped_balance_probe.py --smoke     # 6 iters, ~2-4 min

and multiply. --timeout defaults to 7200 s and the remote trainer is killed on deadline (it pins
~9 GB of GPU memory if left detached). Add the parity gate (a short MJX run, enforced not advisory)
and the ~1-2 min one-off XLA compile.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

EVAL_STEPS = 6000            # 12.0 s -- the horizon #206 quoted the crawl baseline on
N_GLOBAL = 9                 # trailing global-state channels in MorphGraph.observe (lines 59-68)


def build_biped():
    """The body under test. Deterministic, no LLM: the composer's own biped recipe."""
    from virturoid.services.morphology_composer import compose_from_spec, morphology_from_requirements
    return compose_from_spec(morphology_from_requirements(
        0.6, 0.2, prompt="a two-legged walking robot", robot_class="biped"))


class _Shim:
    """Duck-types MorphPolicy for recipe_rollout_morph so the two ablations reuse the deploy path exactly."""

    def __init__(self, inner, mode):
        self._inner, self._mode, self._const, self._glob0 = inner, mode, None, None
        self.actions = []

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def accepts_feature_dim(self, fd):
        return self._inner.accepts_feature_dim(fd)

    def adapt_observation(self, obs):
        return self._inner.adapt_observation(obs)

    def act(self, obs, ranges=None, cmd=None, hop=None):
        import numpy as np
        if self._mode == "const":
            return np.asarray(self._const, dtype=float)
        obs = np.asarray(obs, dtype=float)
        if self._mode == "blind":
            if self._glob0 is None:
                self._glob0 = obs[:, -N_GLOBAL:].copy()
            obs = obs.copy()
            obs[:, -N_GLOBAL:] = self._glob0
        a = self._inner.act(obs, ranges=ranges, cmd=cmd, hop=hop)
        self.actions.append(np.asarray(a, dtype=float).copy())
        return a


def cpu_deploy(gene, policy, *, steps=EVAL_STEPS, mode="full", const=None):
    from virturoid.services.morph_policy import recipe_rollout_morph
    sh = _Shim(policy, mode)
    sh._const = const
    r = recipe_rollout_morph(gene, sh, steps=steps)
    return r, sh


def measure_nulls(gene) -> dict:
    """The two nulls the GPU verdict is judged against, re-measured on THIS machine, before spending GPU."""
    import numpy as np
    from virturoid.services.morph_policy import (MorphPolicy, compiled_model, crawl_gait_rollout,
                                                 recipe_rollout_morph, robot_mjcf)
    from virturoid.services.morph_graph import encode_robot
    fd = int(encode_robot(compiled_model(robot_mjcf(gene), solver_iterations=20)).feature_dim)
    zero = MorphPolicy(fd, seed=0)
    zero.set_params(np.zeros_like(zero.get_params()))
    crawl = crawl_gait_rollout(gene, steps=EVAL_STEPS)
    hold = recipe_rollout_morph(gene, zero, steps=EVAL_STEPS)
    rnd = [int(recipe_rollout_morph(gene, MorphPolicy(fd, seed=s), steps=EVAL_STEPS)["alive"]) for s in range(8)]
    a = np.asarray(rnd, float)
    return {"crawl_alive": int(crawl["alive"]), "crawl_forward_m": float(crawl["forward"]),
            "hold_alive": int(hold["alive"]), "hold_forward_m": float(hold["forward"]),
            "random_mean": float(a.mean()), "random_sd": float(a.std(ddof=1)), "random_n": len(rnd),
            "feature_dim": fd}


def score_checkpoints(gene, paths) -> list[dict]:
    from virturoid.services.morph_policy import MorphPolicy
    rows = []
    for p in paths:
        try:
            pol = MorphPolicy.from_npz(str(p))
        except Exception as e:  # noqa: BLE001 - a truncated fetch is not a candidate, and says so
            rows.append({"ckpt": Path(p).name, "error": f"{type(e).__name__}: {e}"})
            continue
        r, _ = cpu_deploy(gene, pol)
        rows.append({"ckpt": Path(p).name, "path": str(p), "alive": int(r["alive"]),
                     "survived": bool(r["survived"]), "forward_m": float(r["forward"]),
                     "height_ratio": float(r["height_ratio"]), "cadence": float(r["cadence"]),
                     "upright_frac": float(r["upright_frac"]), "support_frac": float(r["support_frac"])})
        print(f"  {rows[-1]['ckpt']:<28} alive={rows[-1]['alive']:>5}/{EVAL_STEPS} "
              f"fwd={rows[-1]['forward_m']:+.3f}m cadence={rows[-1]['cadence']:.2f} "
              f"upright={rows[-1]['upright_frac']:.3f}", flush=True)
    return rows


def ablate(gene, npz_path) -> dict:
    """POSE-or-FEEDBACK. Same two ablations the CPU probe ran, so the answers are comparable."""
    import numpy as np
    from virturoid.services.morph_policy import MorphPolicy
    pol = MorphPolicy.from_npz(str(npz_path))
    r_full, sh = cpu_deploy(gene, pol, mode="full")
    acts = np.asarray(sh.actions) if sh.actions else np.zeros((1, 1))
    r_const, _ = cpu_deploy(gene, pol, mode="const", const=acts.mean(axis=0))
    r_blind, _ = cpu_deploy(gene, pol, mode="blind")
    return {"full_alive": int(r_full["alive"]), "const_alive": int(r_const["alive"]),
            "blind_alive": int(r_blind["alive"]),
            "full_forward_m": float(r_full["forward"]), "full_cadence": float(r_full["cadence"]),
            "action_std_over_time": float(np.mean(np.std(acts, axis=0)))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--envs", type=int, default=192, help="see the --envs note in the module docstring: "
                    "192 is the CONSERVATIVE measured-to-hold count, not a tuned optimum")
    ap.add_argument("--ep-len", type=int, default=3000, help="PHYSICS steps per episode (control horizon is "
                    "ep_len//decimation); 3000 = 6.0 s = 1.9x the do-nothing null, so the fall is in-distribution")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--out", default="build/models/biped_balance_probe.npz")
    ap.add_argument("--report", default="build/biped_balance_probe.json")
    ap.add_argument("--smoke", action="store_true", help="6 iterations only: measure throughput, then stop. "
                    "Multiply the reported wall clock by iters/6 to size the real launch.")
    ap.add_argument("--skip-gpu", action="store_true", help="measure the nulls and re-score whatever checkpoints "
                    "are already on disk next to --out; spends no GPU")
    args = ap.parse_args(argv)

    t0 = time.time()
    gene = build_biped()
    print(f"body: {gene.species} class={gene.robot_class} segments={len(gene.segments)}", flush=True)
    print("measuring the nulls on CPU (before spending any GPU)…", flush=True)
    nulls = measure_nulls(gene)
    print(f"  scripted crawl     {nulls['crawl_alive']:>5}/{EVAL_STEPS}  fwd={nulls['crawl_forward_m']:+.3f} m",
          flush=True)
    print(f"  do-nothing PD hold {nulls['hold_alive']:>5}/{EVAL_STEPS}  fwd={nulls['hold_forward_m']:+.3f} m"
          "   <-- THE BAR", flush=True)
    print(f"  random policy      {nulls['random_mean']:>7.1f} +/- {nulls['random_sd']:.1f} (n={nulls['random_n']})",
          flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"body": gene.species, "nulls": nulls, "eval_steps": EVAL_STEPS,
              "config": {"iters": args.iters, "envs": args.envs, "ep_len": args.ep_len,
                         "physics_steps": args.iters * args.envs * args.ep_len}}

    if not args.skip_gpu:
        from virturoid.services.gpu_trainer import default_training_recipe, gpu_available, train_gene_on_gpu
        if not gpu_available():
            print("\nGPU BOX UNREACHABLE. Nothing was spent. Re-run this exact command when it is up; "
                  "or use --skip-gpu to re-score checkpoints already on disk.", flush=True)
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            report["result"] = "gpu_unavailable"
            Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
            return 2
        recipe = default_training_recipe(gene)
        report["recipe"] = {k: v for k, v in recipe.items()}
        print(f"\nrecipe (from gpu_trainer.default_training_recipe, not hand-set here): {recipe}", flush=True)
        iters = 6 if args.smoke else args.iters
        print(f"launching ONE run: iters={iters} envs={args.envs} ep_len={args.ep_len} "
              f"({iters * args.envs * args.ep_len / 1e6:.1f}M physics steps)", flush=True)
        got = train_gene_on_gpu(gene, out_path=str(out), iters=iters, envs=args.envs,
                                ep_len=args.ep_len, timeout=args.timeout, keep_checkpoints=True,
                                progress=lambda m: print(f"  [gpu] {m}", flush=True), **recipe)
        report["gpu_wall_s"] = round(time.time() - t0, 1)
        if got is None:
            print("\nTRAINING RETURNED NOTHING (launch failed, parity gate red, or deadline). "
                  "See ~/app_train.log on the box. Nothing to score.", flush=True)
            report["result"] = "no_artifact"
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
            return 1
        if args.smoke:
            print(f"\nSMOKE DONE in {report['gpu_wall_s']:.0f}s for {iters} iters. "
                  f"Projected for --iters {args.iters}: "
                  f"~{report['gpu_wall_s'] * args.iters / iters / 60:.0f} min.", flush=True)
            return 0

    stem = str(out)[:-4]
    cands = sorted(Path(out).parent.glob(Path(stem).name + "_it*.npz"),
                   key=lambda p: int("".join(ch for ch in p.stem.split("_it")[-1] if ch.isdigit()) or 0))
    if out.exists():
        cands.append(out)
    if not cands:
        print("no checkpoints on disk to score.", flush=True)
        return 1
    print(f"\nscoring {len(cands)} checkpoints on CPU DEPLOY ({EVAL_STEPS} steps) — never on MJX train reward:",
          flush=True)
    rows = score_checkpoints(gene, cands)
    report["checkpoints"] = rows
    ok = [r for r in rows if "alive" in r]
    if not ok:
        print("every checkpoint failed to load.", flush=True)
        return 1
    best = max(ok, key=lambda r: (r["alive"], r["forward_m"]))
    final = next((r for r in ok if r["ckpt"] == out.name), ok[-1])
    report["kept"] = best
    report["final"] = final
    ratio = best["alive"] / max(1, final["alive"])
    print(f"\nKEEPING {best['ckpt']}: alive={best['alive']}/{EVAL_STEPS} fwd={best['forward_m']:+.3f} m", flush=True)
    print(f"FINAL   {final['ckpt']}: alive={final['alive']}/{EVAL_STEPS} fwd={final['forward_m']:+.3f} m"
          f"   -> the final understates this body {ratio:.2f}x  (#256 measured 2.4x on the quad)", flush=True)

    print("\nablations on the kept checkpoint (POSE or FEEDBACK):", flush=True)
    abl = ablate(gene, best["path"])
    report["ablations"] = abl
    print(f"  full={abl['full_alive']}  const-action={abl['const_alive']}  trunk-blind={abl['blind_alive']}"
          f"   (action sd over time {abl['action_std_over_time']:.4f})", flush=True)

    bar = nulls["hold_alive"]
    if best["alive"] <= nulls["crawl_alive"]:
        verdict = "NO — learned control does not even beat the scripted crawl on this body."
    elif best["alive"] <= bar:
        verdict = (f"NO — beats the crawl ({nulls['crawl_alive']}) but NOT the do-nothing null ({bar}). "
                   "Standing still already does this; nothing was learned about balance.")
    elif abl["const_alive"] >= 0.9 * abl["full_alive"] and abl["blind_alive"] >= 0.9 * abl["full_alive"]:
        verdict = (f"PARTIAL — survives {best['alive']} vs the {bar} null, but a CONSTANT action and a "
                   "TRUNK-BLIND policy do as well: it found a standing POSE, not a balance feedback law.")
    elif best["forward_m"] < 0.10 or best["cadence"] < 0.5:
        verdict = (f"REPRODUCES CPU, ADDS NOTHING — survives {best['alive']} by feedback (const "
                   f"{abl['const_alive']}, trunk-blind {abl['blind_alive']}), but forward travel is "
                   f"{best['forward_m']:+.3f} m at cadence {best['cadence']:.2f}: it STANDS. CPU ES already "
                   "reached this in ~30 min with no GPU. The open question is unmoved.")
    else:
        verdict = (f"YES, AND IT MOVES — survives {best['alive']} vs the {bar} do-nothing null, the ablations "
                   f"break it (const {abl['const_alive']}, trunk-blind {abl['blind_alive']}), AND it travels "
                   f"{best['forward_m']:+.3f} m at cadence {best['cadence']:.2f}. This is the thing CPU ES "
                   "never produced: balance held WHILE moving, on two legs.")
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}", flush=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"report -> {Path(args.report).resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
