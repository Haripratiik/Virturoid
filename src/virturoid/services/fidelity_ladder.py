"""Fidelity ladder (breakthrough plan WS1/H4, the "fidelity ladder IS the architecture" lever) — compose a CHEAP
screen with an EXPENSIVE hi-fi rung so the design-search harness spends the scarce resource (GPU-hours) only on
candidates that survive a fast CPU screen.

The audit measured the payoff: ranking ~10 designs by full training is ~10 h, but screen-then-route is ~25 min
because most candidates die cheaply. The screen (a ~1-2 s CPU rollout) rejects bodies that fall or are wildly
unstable; only survivors are ROUTED UP to the hi-fi rung (a GPU residual-training burst that actually produces a
forward walk -- the CPU rung is non-propulsive on its own, so it can screen but not certify locomotion).

This module is the pure, injectable COMPOSER (testable with stubs, no GPU) plus the real GPU hi-fi closure. It
plugs straight into design_search.run_design_search as its ``evaluate``.
"""

from __future__ import annotations


def make_laddered_evaluate(screen_evaluate, hifi_evaluate, *, promote=None, on_rung=None):
    """Return an ``evaluate(spec) -> result`` that runs ``screen_evaluate`` first and only calls the expensive
    ``hifi_evaluate`` when ``promote(screen_result)`` is True. The result is tagged with the ``rung`` that
    produced it (``"screen"`` for a cheap reject, ``"hifi"`` for a promoted candidate) and carries the screen
    summary under ``screen`` so the harness's diagnosis/selection sees why it was (not) promoted.

    ``promote`` default: survived the screen (don't spend GPU on a body that falls on the cheap rung).
    ``on_rung(rung, spec, result)`` is an optional progress hook (stream which rung each candidate reached)."""
    def _promote_default(r: dict) -> bool:
        return bool(r.get("survived"))

    gate = promote or _promote_default

    def evaluate(spec):
        try:                                                    # M6 chaos-safety: an un-compilable/un-rolloutable
            s = dict(screen_evaluate(spec))                     #   body is a CHEAP REJECT, never a crash that aborts
        except Exception as e:  # noqa: BLE001                  #   the whole search.
            return {"survived": False, "rung": "screen", "promoted": False,
                    "note": f"screen_error: {type(e).__name__}"}
        s["rung"] = "screen"
        if on_rung is not None:
            on_rung("screen", spec, s)
        if not gate(s):
            s["promoted"] = False
            return s                                            # cheap reject -> the expensive rung is never touched
        try:                                                    # M6: a GPU crash on the expensive rung degrades to the
            h = dict(hifi_evaluate(spec))                       #   screen verdict (not promoted), never a crash.
        except Exception as e:  # noqa: BLE001
            s["promoted"] = False
            s["note"] = f"hifi_error: {type(e).__name__}"
            return s
        h["rung"] = "hifi"
        h["promoted"] = True
        h.setdefault("screen", {k: s.get(k) for k in ("forward", "cadence", "upright_frac", "survived")})
        if on_rung is not None:
            on_rung("hifi", spec, h)
        return h

    return evaluate


def make_gpu_locomotion_hifi(gene, *, iters: int = 60, envs: int = 2048, out_dir: str | None = None,
                             verify_steps: int = 900, base_reward_weights: dict | None = None,
                             init_npz: str | None = None, decimation: int = 1, action_lpf: float = 0.0,
                             sphere_feet: bool = False, contact_dr: bool = False,
                             train_fn=None, verify_fn=None):
    """The REAL hi-fi rung: train a residual on the GPU for ``gene`` under a spec's edits, fetch the policy, and
    VERIFY it with an independent CPU rollout -> result dict ``{forward,cadence,upright_frac,survived,npz,trained}``.
    ``spec`` edits map to trainer flags: ``cpg`` params (calf_phase/freq) + ``reward`` params (prog_w/fwd_gate_w/…).
    ``init_npz`` WARM-STARTS training from a banked policy (the transfer-seed from ``transfer_seed.best_transfer_seed``
    -- the measured fix for the from-scratch backward basin). ``train_fn``/``verify_fn`` are injectable for testing;
    defaults use the real gpu_trainer + recipe_rollout_morph. Returns ``survived=False`` (an honest floor) if the
    GPU is unavailable or training fails -- the caller's gate then simply won't promote it, and the search
    continues on cheaper candidates."""
    import os
    import tempfile

    _train = train_fn or _default_gpu_train
    _verify = verify_fn or _default_cpu_verify
    outd = out_dir or tempfile.gettempdir()

    def hifi(spec):
        p = (spec or {}).get("params") or {}
        cpg = {k: float(p[k]) for k in ("calf_phase", "freq", "thigh_amp", "calf_amp") if k in p}
        reward_weights = dict(base_reward_weights or {})
        for k in ("prog_w", "clear_w", "swing_w", "alt_w", "air_w", "back_w", "fwd_gate_w"):
            if k in p:
                reward_weights[k] = float(p[k])
        seed = p.get("init_npz", init_npz)                    # a spec can override the warm-start seed per candidate
        dec = int(p.get("decimation", decimation))            # plan v2 T1.1/T1.2/T1.4/T1.5 (deploy-gap): a spec can
        lpf = float(p.get("action_lpf", action_lpf))          #   tune the control rate, action filter, and the two
        sf = bool(p.get("sphere_feet", sphere_feet))          #   contact-model fixes (sphere feet + contact DR) per
        cdr = bool(p.get("contact_dr", contact_dr))           #   candidate; all baked into train AND verify.
        npz = os.path.join(outd, f"hifi_{abs(hash(repr(sorted(p.items())))) % 10_000_000}.npz")
        try:                                                    # M6 chaos-safety: a trainer CRASH/TIMEOUT (CUDA OOM,
            trained = _train(gene, out_path=npz, iters=iters, envs=envs, cpg=cpg, reward_weights=reward_weights,
                             init_npz=seed, decimation=dec, action_lpf=lpf, sphere_feet=sf, contact_dr=cdr)
        except Exception as e:  # noqa: BLE001                  #   box down, ssh drop) degrades to the honest floor,
            return {"forward": 0.0, "cadence": 0.0, "upright_frac": 0.0, "survived": False, "trained": False,
                    "npz": None, "note": f"train_error: {type(e).__name__}"}   # so the night never banks garbage.
        if not trained:
            return {"forward": 0.0, "cadence": 0.0, "upright_frac": 0.0, "survived": False, "trained": False,
                    "npz": None, "note": "gpu_unavailable_or_train_failed"}
        # deploy == train: sphere_feet rides in the banked policy meta[8] (verify auto-adopts); contact_dr is a
        # TRAIN-only robustness aug (no deploy-side counterpart) so it is not threaded into verify. ``seed`` (the
        # warm-start) is passed so the verify can DEPLOY-SELECT over {seed, checkpoints, final} -- the flywheel keeps
        # the best of "what we had vs. what we trained" (a degrading re-train still banks the good warm/checkpoint).
        try:                                                    # M6 chaos-safety: a CORRUPT/un-loadable checkpoint
            r = _verify(gene, trained, steps=verify_steps, decimation=dec, action_lpf=lpf, sphere_feet=sf, seed=seed)
        except Exception as e:  # noqa: BLE001                  #   must not be banked -- honest survived=False floor.
            return {"forward": 0.0, "cadence": 0.0, "upright_frac": 0.0, "survived": False, "trained": True,
                    "npz": None, "note": f"verify_error: {type(e).__name__}"}
        r["trained"] = True
        r["npz"] = r.get("selected_npz") or trained            # bank the DEPLOY-SELECTED policy, not the final
        return r

    return hifi


def _default_gpu_train(gene, *, out_path, iters, envs, cpg, reward_weights, init_npz=None,
                       decimation=1, action_lpf=0.0, sphere_feet=False, contact_dr=False):
    from virturoid.services.gpu_trainer import train_gene_on_gpu
    calf = cpg.get("calf_phase")
    freq = cpg.get("freq")
    return train_gene_on_gpu(gene, out_path=out_path, iters=iters, envs=envs, cpg=bool(cpg),
                             calf_phase=calf, cpg_freq=freq, reward_weights=reward_weights or None,
                             init_npz=init_npz, decimation=decimation, action_lpf=action_lpf,
                             sphere_feet=sphere_feet, contact_dr=contact_dr, keep_checkpoints=True)


def _default_cpu_verify(gene, npz_path, *, steps, decimation=1, action_lpf=0.0, sphere_feet=False, seed=None):
    import glob
    import os
    from virturoid.services.morph_policy import MorphPolicy, recipe_rollout_morph
    from virturoid.services.transfer_seed import best_checkpoint_by_deploy
    # DEPLOY-SELECT (plan v2 T0.1) over {warm-start seed, numbered checkpoints, final}: never trust the FINAL by
    # train reward -- the measured MJX->CPU divergence means an earlier checkpoint (or the warm-start itself) can
    # deploy far better. Fetched checkpoints live at <stem>_it{N}.npz beside npz_path.
    cands = [npz_path] + sorted(glob.glob(npz_path[:-4] + "_it*.npz"))
    if seed and os.path.isfile(seed):
        cands.append(seed)
    cands = [c for c in dict.fromkeys(cands) if os.path.isfile(c)]
    best = npz_path
    if len(cands) > 1:
        try:
            best, _ranked = best_checkpoint_by_deploy(gene, cands, steps=steps, decimation=decimation)
        except Exception:  # noqa: BLE001 - selection is best-effort; fall back to the final
            best = npz_path
    best = best or npz_path
    pol = MorphPolicy.from_npz(best)
    r = recipe_rollout_morph(gene, pol, steps=steps, decimation=decimation, action_lpf=action_lpf, sphere_feet=sphere_feet)
    return {"forward": float(r.get("forward", 0.0)), "cadence": float(r.get("cadence", 0.0)),
            "upright_frac": r.get("upright_frac"), "survived": bool(r.get("survived")), "selected_npz": best}
