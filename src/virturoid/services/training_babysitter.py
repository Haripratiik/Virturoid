"""The training babysitter (agentic platform plan WS-R / R4): watch a remote MJX-PPO run and KILL it the moment
it has gone bad — never leave a human staring at TensorBoard.

A slow GPU run that has already diverged (NaN loss, the policy blown past its trust region, a reward being farmed
without any forward progress, a collapsed stall at zero) is wasted metal and a wasted deadline. The babysitter
reads the trainer's own per-iteration log — ``iter N  ep_reward=..  fwd_vel=..  kl=..`` — and returns a health
verdict with a kill decision and a NAMED reason, so ``gpu_trainer`` can ``pkill`` the remote process early and
report *why* instead of babysitting it by hand.

The failure modes it catches (all from metrics the trainer already prints):
  * **non-finite** — any NaN/Inf in ep_reward / fwd_vel / kl (the classic blow-up);
  * **policy divergence** — approx_kl sustained far above its trust region (PPO off the rails);
  * **crash** — a Traceback/Error in the log;
  * **reward-hacking alarm** — ep_reward climbing while fwd_vel stays flat/negative: the reward is being gamed
    without moving (the reward-vs-progress divergence Eureka/DrEureka warn about), surfaced not hidden;
  * **collapse/stall** — no forward progress for a long window past warm-up.

The decision logic is a PURE function of the log text — unit-tested on synthetic logs, no GPU, no SSH — so the
kill rule is verifiable offline; only the live ``pkill`` needs the box.
"""

from __future__ import annotations

import math
import re

# The trainer's per-iter line, as of task #258's instrumentation:
#   "  iter  120  ep_reward=   12.34  fwd_vel=+0.412  ep_len=125.0/125  sigma=0.497*  lr=3.0e-04 kl=0.0123  (48s)"
# ep_len/sigma sit BETWEEN fwd_vel and kl, which the `.*?kl=` hop already tolerates — the three fields this
# regex captures are unchanged. (ep_len and sigma are diagnostics for a human reading the log; if the babysitter
# should ever act on a truncated-episode signal, capture ep_len here rather than re-deriving it.)
_ITER_RE = re.compile(
    r"iter\s+(\d+)\s+ep_reward=\s*([-+]?[\d.]+|[-+]?nan|[-+]?inf)\s+fwd_vel=\s*([-+]?[\d.]+|[-+]?nan|[-+]?inf)"
    r"(?:.*?kl=\s*([-+]?[\d.]+|[-+]?nan|[-+]?inf))?", re.I)


def _f(tok: str | None) -> float:
    if tok is None:
        return float("nan")
    t = tok.lower().lstrip("+")
    try:
        return float(t)
    except ValueError:
        return float("nan")


def parse_train_log(text: str) -> list[dict]:
    """Every per-iteration metric row the trainer printed, in order: ``{iter, ep_reward, fwd_vel, kl}``."""
    out = []
    for m in _ITER_RE.finditer(text or ""):
        out.append({"iter": int(m.group(1)), "ep_reward": _f(m.group(2)),
                    "fwd_vel": _f(m.group(3)), "kl": _f(m.group(4))})
    return out


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def assess_training_health(text: str, *, kl_ceiling: float = 0.5, warmup_iters: int = 10,
                           stall_window: int = 12, hack_window: int = 6) -> dict:
    """Verdict on a training log. Returns ``{status, kill, reason, latest_iter, n_iters}`` where ``status`` is
    ``starting`` | ``healthy`` | ``warn`` | ``kill`` and ``kill`` is True only for a hard, unrecoverable failure
    (non-finite / divergence / crash) — the run should be stopped within one eval cadence."""
    text = text or ""
    if re.search(r"\bTraceback\b|^\s*\w*Error:", text, re.M):
        first = next((ln.strip() for ln in text.splitlines() if "Error" in ln or "Traceback" in ln), "crash")
        return {"status": "kill", "kill": True, "reason": f"trainer crashed: {first[:120]}",
                "latest_iter": None, "n_iters": 0}

    iters = parse_train_log(text)
    if not iters:
        return {"status": "starting", "kill": False, "reason": "no iterations logged yet",
                "latest_iter": None, "n_iters": 0}

    last = iters[-1]
    latest = last["iter"]
    # kl may legitimately be absent (parsed as nan); it is ep_reward/fwd_vel going non-finite that is the blow-up.
    if not _finite(last["ep_reward"]) or not _finite(last["fwd_vel"]):
        return {"status": "kill", "kill": True, "reason": f"non-finite metrics at iter {latest} "
                f"(ep_reward={last['ep_reward']}, fwd_vel={last['fwd_vel']}) — the run has blown up",
                "latest_iter": latest, "n_iters": len(iters)}

    recent_kl = [it["kl"] for it in iters[-3:] if _finite(it["kl"])]
    if len(recent_kl) >= 2 and all(k > kl_ceiling for k in recent_kl):
        return {"status": "kill", "kill": True, "reason": f"policy diverged: approx_kl={recent_kl[-1]:.3f} "
                f"sustained above the {kl_ceiling} trust region for {len(recent_kl)} evals",
                "latest_iter": latest, "n_iters": len(iters)}

    # soft alarms (surface, don't hard-kill) — only meaningful past warm-up
    if latest >= warmup_iters and len(iters) >= hack_window:
        win = iters[-hack_window:]
        dr = win[-1]["ep_reward"] - win[0]["ep_reward"]
        dv = win[-1]["fwd_vel"] - win[0]["fwd_vel"]
        if dr > 0.5 and dv <= 0.02 and win[-1]["fwd_vel"] < 0.15:
            return {"status": "warn", "kill": False, "reason": f"reward-hacking alarm: ep_reward +{dr:.2f} over "
                    f"{hack_window} iters while fwd_vel stayed flat ({win[-1]['fwd_vel']:+.3f}) — the reward is "
                    f"climbing without forward progress; check the render, the reward may be gamed",
                    "latest_iter": latest, "n_iters": len(iters)}

    if latest >= warmup_iters and len(iters) >= stall_window:
        win = iters[-stall_window:]
        if all(abs(it["fwd_vel"]) < 0.05 for it in win):
            return {"status": "warn", "kill": False, "reason": f"stalled: fwd_vel ~0 for {stall_window} iters past "
                    f"warm-up (latest {win[-1]['fwd_vel']:+.3f}) — the policy is not learning to move",
                    "latest_iter": latest, "n_iters": len(iters)}

    return {"status": "healthy", "kill": False, "reason": f"iter {latest}: ep_reward={last['ep_reward']:.2f} "
            f"fwd_vel={last['fwd_vel']:+.3f}", "latest_iter": latest, "n_iters": len(iters)}
