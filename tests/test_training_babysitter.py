"""R4 (agentic platform plan WS-R): the training babysitter must read a remote MJX-PPO log and KILL a run that
has gone bad within one eval cadence -- NaN blow-up, policy divergence, a crash -- and SURFACE (not hide) the
softer reward-hacking / stall alarms. The decision logic is a pure function of the log text, so the kill rule is
verifiable offline with synthetic logs; only the live pkill needs the GPU box.
"""
from __future__ import annotations

from virturoid.services.training_babysitter import assess_training_health, parse_train_log


def _log(rows):
    return "\n".join(f"  iter {it:>4}  ep_reward={r:8.2f}  fwd_vel={v:+.3f}  lr=3.0e-04 kl={k:.4f}  (48s)"
                     for it, r, v, k in rows)


def test_parses_the_trainers_real_iter_line():
    rows = parse_train_log(_log([(10, 12.3, 0.41, 0.012), (20, 15.0, 0.55, 0.018)]))
    assert len(rows) == 2
    assert rows[1]["iter"] == 20 and abs(rows[1]["fwd_vel"] - 0.55) < 1e-6 and abs(rows[1]["kl"] - 0.018) < 1e-6


def test_healthy_run_is_not_killed():
    h = assess_training_health(_log([(i, i * 0.5, 0.05 * i, 0.02) for i in range(1, 16)]))
    assert h["status"] == "healthy" and h["kill"] is False


def test_nan_blowup_is_killed():
    log = _log([(10, 12.0, 0.3, 0.02)]) + "\n  iter   20  ep_reward=     nan  fwd_vel=  nan  lr=3.0e-04 kl=nan  (48s)"
    h = assess_training_health(log)
    assert h["kill"] is True and "non-finite" in h["reason"]


def test_policy_divergence_is_killed():
    """approx_kl sustained above the trust region == PPO off the rails -> kill."""
    log = _log([(30, 5.0, 0.1, 0.9), (31, 4.0, 0.08, 1.2), (32, 3.0, 0.05, 1.5)])
    h = assess_training_health(log, kl_ceiling=0.5)
    assert h["kill"] is True and "diverged" in h["reason"]


def test_a_traceback_is_killed():
    h = assess_training_health("  iter 10 ep_reward=1 fwd_vel=0.1 kl=0.02\nTraceback (most recent call last):\n"
                               "RuntimeError: CUDA out of memory")
    assert h["kill"] is True and "crashed" in h["reason"]


def test_reward_hacking_is_surfaced_as_a_warning_not_a_kill():
    """ep_reward climbing while fwd_vel stays flat is the hacking signature -- surface it, but it is not a hard
    kill (the reward may still be salvageable; a human/agent should look at the render)."""
    rows = [(10 + i, 5.0 + i * 1.0, 0.01, 0.02) for i in range(0, 8)]   # reward +7, fwd_vel pinned ~0
    h = assess_training_health(_log(rows))
    assert h["kill"] is False and h["status"] == "warn" and "hacking" in h["reason"]


def test_a_stall_at_zero_is_surfaced():
    rows = [(20 + i, 2.0, 0.0, 0.02) for i in range(0, 14)]              # no forward progress, past warm-up
    h = assess_training_health(_log(rows))
    assert h["status"] == "warn" and "stall" in h["reason"].lower()


def test_no_iterations_yet_is_starting_not_killed():
    h = assess_training_health("compiling on the GPU...\nXLA kernel build...")
    assert h["status"] == "starting" and h["kill"] is False


def test_early_iterations_are_not_prematurely_alarmed():
    """Before warm-up, a low fwd_vel must not trip the stall/hacking alarm (the run is still compiling its way
    up); only a hard failure kills early."""
    h = assess_training_health(_log([(i, 1.0 + i, 0.0, 0.02) for i in range(1, 6)]))
    assert h["kill"] is False and h["status"] in ("healthy", "starting")
