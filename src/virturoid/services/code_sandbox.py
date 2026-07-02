"""Sandboxed execution of LLM-generated evaluation code (breakthrough plan v2 §4.4 / gaps G2+G9).

The generality unlock: for an arbitrary task prompt the LLM can WRITE a ``detect(episode) -> {"ok": bool,
"score": float}`` success-detector (beyond the fixed metric enum), but generated code must never run trusted.
Reference systems are the cautionary tale — Eureka runs reward code in-process; FunSearch ships no sandbox and
says one is critical. Consensus (pysandbox post-mortem): in-process Python restriction is broken by design; the
unit of isolation is an OS PROCESS. This module is the portable core of the two-tier recipe from research
stream 3:

  1. AST gate      — parse; reject imports outside an allowlist, dangerous builtins, and dunder/introspection.
  2. process       — run the code in a fresh ``python -I -S -E`` subprocess (isolated: no site, no env-inherited
                     PYTHON*, no user site) with a STRIPPED env (API keys removed — env inheritance is the #1
                     exfil channel), a wall-clock TIMEOUT, and a fresh scratch cwd.
  3. contract      — the harness owns the entrypoint (the code defines ONE function); result is the LAST stdout
                     line as JSON, byte-capped; score is finite-checked and clamped.
  4. calibration   — before trusting a detector, replay it on known-PASS and known-FAIL fixture episodes; a
                     detector that can't separate them is REJECTED (fail-CLOSED — the honesty gate applied to
                     generated code, consistent with the Product Truth Ledger doctrine).

Windows note: OS-level memory/CPU caps want a Job Object (pywin32) and true no-network/no-filesystem wants the
WSL ``docker --network=none --read-only`` tier; those are optional hardening layered on this portable core (which
already stops loops, floods, stray imports, and key exfil for self-generated, non-adversarial code).
"""

from __future__ import annotations

import ast

# import allowlist for generated evaluation code: pure computation only (no io/os/sys/socket/subprocess).
_ALLOWED_IMPORTS = frozenset({"math", "json", "statistics", "numpy", "np"})
_BANNED_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "globals", "vars",
                           "locals", "getattr", "setattr", "delattr", "breakpoint", "memoryview"})
_BANNED_ATTRS = frozenset({"__subclasses__", "__globals__", "__builtins__", "__mro__", "__code__", "__class__",
                           "__bases__", "__dict__", "__getattribute__", "__reduce__", "__reduce_ex__"})


class SandboxError(RuntimeError):
    """Generated code was rejected (AST gate) or failed to run safely (timeout / crash / bad contract)."""


def validate_ast(src: str, *, allowed_imports=_ALLOWED_IMPORTS) -> None:
    """Static gate: raise :class:`SandboxError` if ``src`` imports outside the allowlist, calls a banned builtin,
    or touches an introspection dunder. Accident-and-casual-exfil guard layered under process isolation."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SandboxError(f"syntax error in generated code: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed_imports:
                    raise SandboxError(f"disallowed import: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed_imports:
                raise SandboxError(f"disallowed import-from: {node.module}")
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise SandboxError(f"disallowed name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            raise SandboxError(f"disallowed attribute: {node.attr}")


_RUNNER = '''\
import json, sys
{src}
_ep = json.loads(sys.stdin.read() or "{{}}")
_r = detect(_ep)
print(json.dumps(_r), flush=True)
'''


def run_detector(src: str, episode: dict, *, timeout: float = 10.0, allowed_imports=_ALLOWED_IMPORTS,
                 max_output: int = 65536) -> dict:
    """Validate + run a generated ``detect(episode)`` in an isolated subprocess; return the validated result
    ``{"ok": bool, "score": float}``. Raises :class:`SandboxError` on rejection, timeout, crash, or a malformed/
    non-finite result. The subprocess runs ``python -I -S -E`` with a MINIMAL env (no API keys) in a fresh cwd."""
    import json
    import math
    import os
    import subprocess
    import sys
    import tempfile

    validate_ast(src, allowed_imports=allowed_imports)
    # minimal env: keep only PATH/SystemRoot (Windows needs SystemRoot for the interpreter); DROP all API keys.
    keep = ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP", "LANG", "LC_ALL")
    env = {k: v for k, v in os.environ.items() if k in keep}
    with tempfile.TemporaryDirectory() as scratch:
        runner = os.path.join(scratch, "_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER.format(src=src))
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "-E", runner],
                               input=json.dumps(episode), capture_output=True, text=True,
                               timeout=timeout, cwd=scratch, env=env)
        except subprocess.TimeoutExpired as e:
            raise SandboxError(f"detector timed out after {timeout}s (likely an infinite loop)") from e
    if p.returncode != 0:
        raise SandboxError(f"detector crashed (exit {p.returncode}): {(p.stderr or '')[-400:]}")
    out = (p.stdout or "")[-max_output:].strip().splitlines()
    if not out:
        raise SandboxError("detector produced no output")
    try:
        r = json.loads(out[-1])
    except json.JSONDecodeError as e:
        raise SandboxError(f"detector output was not JSON: {out[-1][:200]!r}") from e
    if not isinstance(r, dict) or "ok" not in r:
        raise SandboxError(f"detector result missing 'ok': {r!r}")
    score = r.get("score", 1.0 if r.get("ok") else 0.0)
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise SandboxError(f"detector score not finite: {score!r}")
    return {"ok": bool(r["ok"]), "score": float(score)}


_DETECTOR_SYSTEM = (
    "You write a Python success-detector for a robot task. Output ONLY JSON {code, rationale}. `code` MUST "
    "define exactly one function `def detect(ep):` that takes an episode-metrics dict and returns "
    "{'ok': bool, 'score': float}. Use ONLY the math/json/statistics/numpy modules; no I/O, no imports of os/sys; "
    "no introspection dunders. Judge success strictly from the metrics; ok=True only when the task is achieved.")
_DETECTOR_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "rationale": {"type": "string"}},
    "required": ["code"], "additionalProperties": False,
}


def generate_detector(task_prompt: str, pass_fixtures: list, fail_fixtures: list, llm, *,
                      timeout: float = 10.0, max_tokens: int = 900) -> dict:
    """G2/§4.4 GENERATION half: ask ``llm`` to WRITE a ``detect(ep)`` for ``task_prompt``, then GATE it through
    the sandbox's fail-closed calibration on known pass/fail fixtures. Returns ``{code, trusted, reason,
    calibration}`` -- ``code`` is used ONLY if ``trusted`` (it separated the fixtures + ran safely). ``llm`` None
    or any failure -> ``trusted=False`` (never trust an ungenerated/unvalidated detector)."""
    if llm is None:
        return {"code": None, "trusted": False, "reason": "no llm backend", "calibration": None}
    try:
        user = (f"Task: {task_prompt}\nExample metrics that MUST score ok=True:\n{pass_fixtures}\n"
                f"Example metrics that MUST score ok=False:\n{fail_fixtures}\nWrite detect(ep).")
        out = llm.complete_json(_DETECTOR_SYSTEM, user, _DETECTOR_SCHEMA, max_tokens=max_tokens)
        code = (out or {}).get("code")
        if not code:
            return {"code": None, "trusted": False, "reason": "llm returned no code", "calibration": None}
        cal = calibrate_detector(code, pass_fixtures, fail_fixtures, timeout=timeout)
        return {"code": code if cal["trusted"] else None, "trusted": cal["trusted"],
                "reason": cal["reason"], "calibration": cal}
    except Exception as e:  # noqa: BLE001 - any LLM/sandbox failure -> not trusted (fail-closed)
        return {"code": None, "trusted": False, "reason": f"generation error: {e}", "calibration": None}


def calibrate_detector(src: str, pass_fixtures: list, fail_fixtures: list, *, timeout: float = 10.0) -> dict:
    """FAIL-CLOSED calibration gate (§4.4): a generated detector is trusted ONLY if it classifies EVERY known-PASS
    fixture as ok=True and EVERY known-FAIL fixture as ok=False. Returns ``{trusted, reason, pass_ok, fail_ok}``.
    Any sandbox error on a fixture -> not trusted (never assume a broken detector is safe)."""
    try:
        pass_ok = all(run_detector(src, ep, timeout=timeout).get("ok") is True for ep in pass_fixtures)
        fail_ok = all(run_detector(src, ep, timeout=timeout).get("ok") is False for ep in fail_fixtures)
    except SandboxError as e:
        return {"trusted": False, "reason": f"sandbox error during calibration: {e}",
                "pass_ok": False, "fail_ok": False}
    trusted = bool(pass_ok and fail_ok)
    reason = "separates known pass/fail" if trusted else (
        "misclassified a known-pass" if not pass_ok else "misclassified a known-fail")
    return {"trusted": trusted, "reason": reason, "pass_ok": pass_ok, "fail_ok": fail_ok}
