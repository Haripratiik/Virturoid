"""Connectivity + structured-output check for Virturoid's LLM backends.

Run this after wiring up a local Nemotron (Ollama/vLLM/NIM) or Claude to confirm
the model is reachable AND returns schema-valid JSON before you flip an agent on.

    # check whatever VIRTUROID_LLM_BACKEND points at
    python -m virturoid.llm_check

    # check a specific local endpoint without touching your shell env
    python -m virturoid.llm_check --backend local \
        --url http://100.x.y.z:11434/v1 --model nemotron-3-nano

Exit code 0 means the round-trip (prompt -> JSON -> schema check) succeeded.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# A tiny schema the model must satisfy. Mirrors how the real agents are scored:
# strict shape, enum constraint, no extra keys.
_PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "reachable": {"type": "boolean"},
        "robot_class": {"type": "string", "enum": ["manipulator", "mobile_base"]},
        "note": {"type": "string"},
    },
    "required": ["reachable", "robot_class"],
    "additionalProperties": False,
}

_PROBE_SYSTEM = (
    "You are a connectivity probe for a robotics platform. Reply with JSON only: "
    "reachable=true, robot_class is 'manipulator' for a tabletop arm, and a short note."
)
_PROBE_USER = "A tabletop arm that sorts blocks. Confirm you can return structured JSON."


def _validate(obj: dict) -> list[str]:
    issues: list[str] = []
    if not isinstance(obj, dict):
        return [f"response is {type(obj).__name__}, not an object"]
    for key in _PROBE_SCHEMA["required"]:
        if key not in obj:
            issues.append(f"missing required key {key!r}")
    rc = obj.get("robot_class")
    if rc is not None and rc not in _PROBE_SCHEMA["properties"]["robot_class"]["enum"]:
        issues.append(f"robot_class {rc!r} not in enum")
    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check a Virturoid LLM backend end to end.")
    ap.add_argument("--backend", help="override VIRTUROID_LLM_BACKEND (claude|local|mock)")
    ap.add_argument("--url", help="override VIRTUROID_LOCAL_LLM_URL (local backend)")
    ap.add_argument("--model", help="override the model id")
    ap.add_argument("--format", dest="fmt", help="local format mode: json_object|json_schema|text")
    ap.add_argument("--role", default="designer", help="agent role to resolve the backend for")
    args = ap.parse_args(argv)

    if args.backend:
        os.environ["VIRTUROID_LLM_BACKEND"] = args.backend
    if args.url:
        os.environ["VIRTUROID_LOCAL_LLM_URL"] = args.url
    if args.fmt:
        os.environ["VIRTUROID_LOCAL_LLM_FORMAT"] = args.fmt
    from virturoid.services.llm_client import _load_local_env, get_llm

    if args.model:
        os.environ[_model_env(os.environ.get("VIRTUROID_LLM_BACKEND", ""))] = args.model
    _load_local_env()  # fill from project-local .env (real env / CLI overrides win)

    backend = os.environ.get("VIRTUROID_LLM_BACKEND", "off").lower()
    print(f"backend          : {backend}")
    if backend == "local":
        print(f"url              : {os.environ.get('VIRTUROID_LOCAL_LLM_URL', 'http://localhost:8000/v1')}")
        print(f"model            : {os.environ.get('VIRTUROID_LOCAL_LLM_MODEL', 'nvidia/nemotron')}")
        print(f"format_mode      : {os.environ.get('VIRTUROID_LOCAL_LLM_FORMAT', 'json_object')}")
    elif backend == "claude":
        print(f"model            : {os.environ.get('VIRTUROID_CLAUDE_MODEL', 'claude-opus-4-8')}")
    elif backend in ("openai", "chatgpt", "gpt"):
        print(f"model            : {os.environ.get('VIRTUROID_OPENAI_MODEL', 'gpt-4o-mini')}")

    llm = get_llm(args.role)
    if llm is None:
        print("\nNo backend configured (VIRTUROID_LLM_BACKEND=off). The agents run in "
              "deterministic offline mode. Set the env vars to enable an LLM.")
        return 0

    print(f"client           : {getattr(llm, 'name', '?')}\n")
    print("sending probe ...")
    t0 = time.time()
    try:
        out = llm.complete_json(_PROBE_SYSTEM, _PROBE_USER, _PROBE_SCHEMA, max_tokens=256)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        _hint(backend, exc)
        return 1
    dt = time.time() - t0

    print(f"response ({dt:.1f}s): {out}")
    issues = _validate(out)
    if issues:
        print("\nReachable, but the JSON did not satisfy the schema:")
        for i in issues:
            print(f"  - {i}")
        if backend == "local":
            print("  Try --format json_schema (vLLM/NIM) or a stronger model.")
        return 1
    print("\nOK: backend reachable and returns schema-valid JSON.")
    return 0


def _hint(backend: str, exc: Exception) -> None:
    msg = str(exc).lower()
    if backend == "local":
        if "connection" in msg or "refused" in msg or "timed out" in msg or "max retries" in msg:
            print("  Hint: cannot reach the endpoint. Confirm Ollama is running "
                  "(`ollama serve`) and the URL/Tailscale IP + port are correct, and ends in /v1.")
        elif "404" in msg or "not found" in msg:
            print("  Hint: model not found. Run `ollama pull <model>` and check --model.")
        elif "response_format" in msg or "json_schema" in msg or "400" in msg:
            print("  Hint: this server rejects json_schema. Use --format json_object (the default).")
    elif backend == "claude":
        if "api_key" in msg or "authentication" in msg or "401" in msg:
            print("  Hint: set ANTHROPIC_API_KEY.")
    elif backend in ("openai", "chatgpt", "gpt"):
        if "api_key" in msg or "authentication" in msg or "401" in msg or "invalid_api_key" in msg:
            print("  Hint: paste a real key into OPENAI_API_KEY in the project .env file "
                  "(get one at https://platform.openai.com/api-keys).")
        elif "module" in msg or "openai" in msg:
            print("  Hint: pip install openai")


def _model_env(backend: str) -> str:
    backend = (backend or "").lower()
    if backend == "claude":
        return "VIRTUROID_CLAUDE_MODEL"
    if backend in ("openai", "chatgpt", "gpt"):
        return "VIRTUROID_OPENAI_MODEL"
    return "VIRTUROID_LOCAL_LLM_MODEL"


if __name__ == "__main__":
    sys.exit(main())
