"""PolicyImporter Tier P2 — sandboxed policy execution (Input Ingestion plan, Phase 4 + safety contract).

P0 (metadata) and P1 (static AST parse) never run user code. P2 is the first tier that DOES — so it runs under
the same fail-closed isolation as generated detector code (``code_sandbox``): an AST gate (pure math/numpy only —
torch/onnx/os/sys are rejected, which correctly routes those to the deferred P3 native-adapter tier), then a fresh
``python -I -S -E`` subprocess with a stripped env (no API keys), a wall-clock timeout, and a scratch cwd.

It implements the plan's "minimum useful acceptance test": given one simulated observation, the imported policy
returns an action, the action maps to the known actuator count, and its values are finite and within safety
limits — or a structured rejection. Stepping a full sim for N frames is the next tier (needs the imported model);
this is the single-step contract that must pass first.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile

from virturoid.schemas.policy_import import PolicyTier
from virturoid.services.code_sandbox import SandboxError, validate_ast

# P2 allows pure computation + numpy (a black-box numpy controller). torch/jax/onnx are intentionally NOT here:
# they load arbitrary serialized code, so they belong to a vetted P3 native adapter, not this generic sandbox.
POLICY_ALLOWED_IMPORTS = frozenset({"math", "json", "statistics", "numpy", "np"})

_POLICY_RUNNER = '''\
import json, sys
{src}
_obs = json.loads(sys.stdin.read() or "null")
_fn = {entrypoint}
_a = _fn(_obs)
if hasattr(_a, "tolist"):
    _a = _a.tolist()
if not isinstance(_a, (list, tuple)):
    _a = [_a]
print(json.dumps({{"action": [float(x) for x in _a]}}), flush=True)
'''


def _within_limits(action: list[float], safety_limits: dict | None) -> tuple[bool, list[str]]:
    """Check each action against per-key or global min/max limits (ros2_control style). Missing limits -> pass."""
    if not safety_limits:
        return True, []
    warnings: list[str] = []
    ok = True
    # accept either {"global": {"min":..,"max":..}} or per-index/{joint/iface: {min,max}} — check any numeric bounds.
    bounds = []
    for val in safety_limits.values():
        if isinstance(val, dict) and ("min" in val or "max" in val):
            try:
                lo = float(val["min"]) if val.get("min") is not None else -math.inf
                hi = float(val["max"]) if val.get("max") is not None else math.inf
                bounds.append((lo, hi))
            except (TypeError, ValueError):
                continue
    if not bounds:
        return True, []
    for i, a in enumerate(action):
        lo, hi = bounds[i] if i < len(bounds) else bounds[-1]
        if not (lo <= a <= hi):
            ok = False
            warnings.append(f"action[{i}]={a:.4f} outside [{lo}, {hi}]")
    return ok, warnings


def sandbox_policy_step(source: str, *, entrypoint: str, observation, action_dim: int | None = None,
                        safety_limits: dict | None = None, timeout: float = 10.0) -> dict:
    """Run ``entrypoint(observation)`` once in an isolated subprocess; validate the action. Never executes in-process.

    Returns ``{ran, action, action_len, action_dim_ok, finite, within_limits, validation_status, warnings, reason}``.
    ``validation_status='rejected'`` (fail-closed) on AST rejection, timeout, crash, malformed/non-finite output,
    an action-dim mismatch, or a safety-limit violation; ``'sandbox_passed'`` only when every check holds.
    """
    result = {
        "ran": False, "action": None, "action_len": 0, "action_dim_ok": None, "finite": None,
        "within_limits": None, "validation_status": "rejected", "warnings": [], "reason": "",
        "tier": PolicyTier.P2_SANDBOX.value,
    }
    if not entrypoint:
        result["reason"] = "no entrypoint to invoke (run P1 static parse first)."
        return result
    try:
        validate_ast(source, allowed_imports=POLICY_ALLOWED_IMPORTS)
    except SandboxError as exc:
        result["reason"] = f"AST gate rejected the controller: {exc}"
        result["warnings"].append("torch/onnx/jax controllers are rejected here — use a P3 native adapter.")
        return result

    keep = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {k: v for k, v in os.environ.items() if k in keep}
    with tempfile.TemporaryDirectory() as scratch:
        runner = os.path.join(scratch, "_prunner.py")
        with open(runner, "w", encoding="utf-8") as handle:
            handle.write(_POLICY_RUNNER.format(src=source, entrypoint=entrypoint))
        try:
            # -I (ignore PYTHONPATH + user site, don't prepend cwd) + -E, but NOT -S: a real policy needs numpy
            # from system site-packages. The AST allowlist -- not -S -- is what restricts imports to math/numpy,
            # so dropping -S only makes numpy importable, nothing dangerous (import os/torch is already rejected).
            proc = subprocess.run([sys.executable, "-I", "-E", runner],
                                  input=json.dumps(observation), capture_output=True, text=True,
                                  timeout=timeout, cwd=scratch, env=env)
        except subprocess.TimeoutExpired:
            result["reason"] = f"policy timed out after {timeout}s (possible infinite loop)."
            return result
    if proc.returncode != 0:
        result["reason"] = f"policy crashed (exit {proc.returncode}): {(proc.stderr or '')[-300:]}"
        return result
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        result["reason"] = "policy produced no output."
        return result
    try:
        action = json.loads(lines[-1]).get("action")
    except (json.JSONDecodeError, AttributeError):
        result["reason"] = f"policy output was not the expected JSON action: {lines[-1][:200]!r}"
        return result
    if not isinstance(action, list):
        result["reason"] = "policy did not return an action vector."
        return result

    result["ran"] = True
    result["action"] = action
    result["action_len"] = len(action)
    result["finite"] = all(math.isfinite(x) for x in action)
    result["action_dim_ok"] = (action_dim is None) or (len(action) == action_dim)
    within, warns = _within_limits(action, safety_limits)
    result["within_limits"] = within
    result["warnings"].extend(warns)

    if not result["finite"]:
        result["reason"] = "action contains non-finite values."
    elif action_dim is not None and not result["action_dim_ok"]:
        result["reason"] = f"policy returned {len(action)} actions but the model has {action_dim} actuators."
    elif not within:
        result["reason"] = "action violates the declared safety limits."
    else:
        result["validation_status"] = "sandbox_passed"
        result["reason"] = "one-step action is dimension-correct, finite, and within limits."
    return result
