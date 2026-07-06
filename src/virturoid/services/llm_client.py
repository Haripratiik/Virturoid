"""Provider-agnostic LLM interface for Virturoid's AI agents (plan §8, §16.2).

The design layer never trusts a raw model output: every agent returns structured
JSON that is validated against a schema and then against physics/feasibility
gates. This module is the swappable client behind that contract:

- ``MockLLM``  — deterministic, offline; used by tests and as a safe default.
- ``ClaudeLLM`` — Anthropic ``claude-opus-4-8`` for hard reasoning (robot design,
  failure analysis, task-graph synthesis). Structured JSON via output_config,
  prompt caching on the frozen system prompt, adaptive thinking. Lazy import.
- ``OpenAILLM`` — a cheap ChatGPT model (default ``gpt-4o-mini``) as a budget
  alternative to Claude for the reasoning agents. Lazy import.
- ``LocalLLM`` — a local NVIDIA Nemotron-class model behind an OpenAI-compatible
  endpoint (vLLM / NIM / Ollama) for high-volume structured tasks. Lazy import.

Routing is per agent role (env ``VIRTUROID_LLM_BACKEND``); when no backend is
configured the agents fall back to the deterministic services, so the product
runs fully offline with zero API keys.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Protocol

# ---------------------------------------------------------------------------
# Internal-LLM spend ledger (agent_first_plan.md G-D). The product's pitch is
# ZERO internal token spend -- a connected frontier agent is the brain. That
# claim must be MEASURED, not asserted. This process-global ledger records, per
# role: how many internal completions actually fired (``calls`` + est. tokens),
# and how many were BLOCKED by the no-internal-LLM switch (proof the deterministic
# fallback carried the load). ``VIRTUROID_NO_INTERNAL_LLM=1`` forces ``get_llm``
# to return None for every role, so zero-spend becomes structural, and the ledger
# shows blocked>0 / calls==0. Exposed to the agent via the ``llm_spend`` tool.
_LEDGER_LOCK = threading.RLock()
_SPEND_LEDGER: dict[str, dict] = {}


def _ledger_entry(role: str) -> dict:
    return _SPEND_LEDGER.setdefault(
        role, {"calls": 0, "blocked": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0})


def record_blocked(role: str) -> None:
    """A role asked for an internal LLM while the no-internal-LLM switch was on -> denied (fallback ran)."""
    with _LEDGER_LOCK:
        _ledger_entry(role)["blocked"] += 1


def record_call(role: str, tokens_in: int = 0, tokens_out: int = 0, usd: float = 0.0) -> None:
    """An internal completion actually fired for ``role`` -> our-token spend."""
    with _LEDGER_LOCK:
        e = _ledger_entry(role)
        e["calls"] += 1
        e["tokens_in"] += int(tokens_in)
        e["tokens_out"] += int(tokens_out)
        e["usd"] += float(usd)


def spend_snapshot() -> dict:
    """The ledger + rolled-up totals. ``internal_calls`` == 0 with the loop complete IS the zero-spend proof."""
    with _LEDGER_LOCK:
        roles = {r: dict(e) for r, e in _SPEND_LEDGER.items()}
    totals = {"internal_calls": sum(e["calls"] for e in roles.values()),
              "blocked": sum(e["blocked"] for e in roles.values()),
              "tokens_in": sum(e["tokens_in"] for e in roles.values()),
              "tokens_out": sum(e["tokens_out"] for e in roles.values()),
              "usd": round(sum(e["usd"] for e in roles.values()), 4)}
    return {"no_internal_llm": os.environ.get("VIRTUROID_NO_INTERNAL_LLM") == "1",
            "backend": os.environ.get("VIRTUROID_LLM_BACKEND", "off").lower(),
            "roles": roles, "totals": totals}


def reset_spend_ledger() -> None:
    with _LEDGER_LOCK:
        _SPEND_LEDGER.clear()


class _LedgerClient:
    """Transparent wrapper that records every internal completion to the process ledger (G-D), then delegates.
    Forwards ``name`` and any other attribute to the wrapped backend, so callers see no difference."""

    def __init__(self, client, role: str) -> None:
        self._client = client
        self._role = role
        self.name = getattr(client, "name", "?")

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        out = self._client.complete_json(system, user, schema, max_tokens=max_tokens,
                                          reasoning_effort=reasoning_effort)
        ti = max(1, (len(system or "") + len(user or "")) // 4)
        to = max(1, len(json.dumps(out, default=str)) // 4)
        record_call(self._role, ti, to)
        return out

    def __getattr__(self, item):
        return getattr(self._client, item)


def unwrap_llm(client):
    """Return the concrete backend behind the G-D ledger wrapper (or the client unchanged). For callers that
    need the real backend type (routing tests, backend-specific config)."""
    return getattr(client, "_client", client)

# Repo root = parents of src/virturoid/services/llm_client.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Project-local config files, in precedence order (first found wins per key).
# These hold the OpenAI/Anthropic keys + model choices for THIS project only, so
# nothing has to be set device-wide. Real environment variables still take
# precedence over the file (so CI / one-off overrides keep working).
_LOCAL_ENV_FILES = (".env", "virturoid.local.env")
_loaded_local_env = False


def _load_local_env() -> None:
    """Populate os.environ from a project-local .env file (does not override real env).

    Looks for ``.env`` / ``virturoid.local.env`` in the repo root and the current
    working directory. Lines are ``KEY=VALUE``; ``#`` comments and blanks ignored;
    surrounding quotes stripped. Idempotent — only runs once per process.
    """
    global _loaded_local_env
    if _loaded_local_env:
        return
    _loaded_local_env = True
    if os.environ.get("VIRTUROID_NO_LOCAL_ENV") == "1":
        return  # tests / CI opt out of reading the project .env for hermeticity
    searched: list[Path] = []
    for base in (_REPO_ROOT, Path.cwd()):
        for name in _LOCAL_ENV_FILES:
            p = base / name
            if p in searched:
                continue
            searched.append(p)
            if not p.is_file():
                continue
            try:
                for raw in p.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:  # real env wins
                        os.environ[key] = value
            except Exception:  # noqa: BLE001 - a malformed local file must never crash a build
                continue


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        """Return a JSON object conforming to ``schema`` (validation is the caller's job). ``reasoning_effort``
        ('minimal'|'low'|'medium'|'high') hints reasoning models to spend less hidden chain on easy tasks;
        backends that don't support it ignore it."""


class MockLLM:
    """Deterministic offline backend. ``responder`` maps (system, user, schema) -> dict."""

    name = "mock"

    def __init__(self, responder=None, fixed: dict | None = None) -> None:
        self._responder = responder
        self._fixed = fixed

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        if self._responder is not None:
            return self._responder(system, user, schema)
        if self._fixed is not None:
            return dict(self._fixed)
        # Last resort: emit a schema-shaped zero object so callers can still validate.
        return {k: _zero_for(v) for k, v in (schema.get("properties") or {}).items()}


class ClaudeLLM:
    """Anthropic Claude backend (default model claude-opus-4-8)."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None) -> None:
        self.model = model
        self._api_key = api_key

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        import anthropic  # lazy: only needed when this backend is used

        client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},  # hard reasoning for design/failure agents
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "{}")
        return json.loads(text)


class OpenAILLM:
    """OpenAI ChatGPT backend (default the cheap ``gpt-4o-mini``) for hard reasoning.

    A budget alternative to Claude for the reasoning agents (designer, failure
    analyst, task graph). Uses the official ``openai`` SDK (lazy import) and the
    portable ``json_object`` response format with the schema injected into the
    prompt + defensive parsing — so it works across every chat model and we keep
    one validation contract. Needs ``OPENAI_API_KEY``; model via
    ``VIRTUROID_OPENAI_MODEL``.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 fallback_model: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        # On a NON-recoverable rate limit (e.g. gpt-5.2's 50 requests/DAY tier cap), retry once on this
        # cheaper, higher-throughput model so the product keeps its LLM intelligence instead of dropping all
        # the way back to the keyword builder. None = no fallback (raise).
        self._fallback_model = fallback_model

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        from openai import OpenAI  # lazy: only needed when this backend is used

        import time

        client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        sys_with_schema = (
            f"{system}\n\nReturn ONLY a single JSON object matching this JSON Schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema)}"
        )
        # Model-aware params: gpt-5 / o-series are REASONING models — they use `max_completion_tokens`
        # (and the budget must include hidden reasoning tokens) and only accept the default temperature.
        # Classic chat models (gpt-4o/4.1) use `max_tokens` + temperature=0.
        is_reasoning = self.model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
        msgs = [{"role": "system", "content": sys_with_schema}, {"role": "user", "content": user}]
        kwargs = {"model": self.model, "response_format": {"type": "json_object"}, "messages": msgs}
        if is_reasoning:
            # Budget = 4x the caller's signal (with a small floor), NOT a flat 8000 — that floor silently
            # granted a tiny structured decision (e.g. the 8-field body-plan intent, max_tokens=600) an
            # 8000-token hidden-reasoning ceiling, burning latency + cost. Honor the caller's intent.
            kwargs["max_completion_tokens"] = max(max_tokens * 4, 2000)
            # reasoning_effort lets a simple structured task (intent/plan) ask for a SHORT hidden chain —
            # the single biggest latency/cost lever on a reasoning model for an easy decision.
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = 0
        # Retry TRANSIENT rate-limits (per-minute / overloaded) with backoff — otherwise a brief 429 silently
        # drops callers back to their deterministic fallback, defeating the LLM-general path. A non-recoverable
        # DAILY cap (e.g. gpt-5.2 "requests per day"/RPD) won't clear in seconds, so don't burn retries on it —
        # go straight to the fallback model (or raise).
        last = None
        for attempt in range(4):
            try:
                response = client.chat.completions.create(**kwargs)
                return _parse_json_object(response.choices[0].message.content)
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc).lower()
                is_rl = "429" in msg or "rate limit" in msg or "overloaded" in msg or "rate_limit" in msg
                daily = "per day" in msg or "rpd" in msg or "requests per day" in msg
                if is_rl and not daily and attempt < 3:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                if is_rl and self._fallback_model and self._fallback_model != self.model:
                    # Keep LLM intelligence under a tier cap: same prompt on the cheaper high-throughput model.
                    return OpenAILLM(self._fallback_model, api_key=self._api_key).complete_json(
                        system, user, schema, max_tokens=max_tokens)
                raise
        if last is not None:
            raise last
        return {}


class LocalLLM:
    """Local Nemotron-class model behind an OpenAI-compatible endpoint (vLLM/NIM/Ollama).

    Structured-output handling differs by server, so the JSON-constraint mode is
    selectable (``VIRTUROID_LOCAL_LLM_FORMAT``):

    - ``json_schema`` — strict grammar-constrained decoding. vLLM and NVIDIA NIM
      accept OpenAI's ``response_format: {type: json_schema}``.
    - ``json_object`` — valid-JSON constraint only (no schema). This is what
      **Ollama's** ``/v1/chat/completions`` supports; the schema is injected into
      the prompt so the model still produces the right shape. **This is the
      default**, because it works on every server including Ollama.
    - ``text`` — no constraint; rely entirely on the prompt + robust parsing.

    Regardless of mode, the schema is always described in-prompt and the response
    is parsed defensively (markdown fences stripped, first JSON object extracted),
    so a model that ignores the constraint still round-trips when it can.
    """

    name = "local"

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", format_mode: str = "json_object") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.format_mode = format_mode

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        import requests  # lazy

        # Always describe the target schema in-prompt: it's the only signal in
        # json_object/text modes, and a harmless reinforcement in json_schema mode.
        sys_with_schema = (
            f"{system}\n\nReturn ONLY a single JSON object matching this JSON Schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema)}"
        )
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": sys_with_schema},
                {"role": "user", "content": user},
            ],
        }
        if self.format_mode == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        elif self.format_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        # "text" mode: send no response_format at all.

        # (connect, read) timeout: a DEAD/unreachable host (e.g. a stale local-LLM IP in .env) fails the CONNECT
        # in ~3s and degrades to the deterministic offline path, instead of freezing the app for the full 120s
        # read window — which on a demo machine reads as a hang/crash.
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(3.05, 120),
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_json_object(content)


# Roles split by how hard the reasoning is. High-volume specialist roles run on a cheap
# fast model; the rest are reasoning roles that get the stronger (and pricier) model.
# This keeps cost down without dumbing down the agents that actually need to think.
# "morphology"/"planner" (the build hot-path) are here: gpt-4.1-mini designs diverse bodies (verified:
# an 8-leg/24-DOF spider) at a fraction of gpt-4.1's cost and with far higher rate limits — gpt-4o-mini
# was too weak (fell back to the fixed template). Hard-reasoning roles keep the stronger model.
_FAST_ROLES = {"part_recommender", "scene", "parser", "species"}
# Body/plan DESIGN drives the whole product's fidelity — it must use a STRONG model, never the cheap
# fast model (gpt-4o-mini was too weak: degenerate/low-part bodies, silent template fallback).
_BUILD_ROLES = {"morphology", "planner"}


def _openai_model_for_role(role: str) -> str:
    """Pick the OpenAI model for a role. Body/plan design (``_BUILD_ROLES``) gets the STRONG model
    (``VIRTUROID_OPENAI_BUILD_MODEL`` → else the reasoning model → else gpt-4.1-mini) so robots are
    high-fidelity; high-volume specialists get the cheap fast model; everything else the reasoning model."""
    reasoning_model = os.environ.get("VIRTUROID_OPENAI_MODEL", "gpt-4o-mini")
    fast_model = os.environ.get("VIRTUROID_OPENAI_FAST_MODEL", "gpt-4.1-mini")
    if role in _BUILD_ROLES:
        return os.environ.get("VIRTUROID_OPENAI_BUILD_MODEL") or reasoning_model or "gpt-4.1-mini"
    return fast_model if role in _FAST_ROLES else reasoning_model


# LOCAL (Ollama/vLLM) cheap-for-breadth / strong-for-depth routing (plan v3 M2/WS7b, AlphaEvolve Flash+Pro
# pattern). The search operators dominate volume: Proposer/Critic are high-fan-out breadth (cheap model), the
# Diagnostician/codegen are the rare deep calls (strong model). Back-compat: if the FAST/BUILD envs are unset,
# EVERY role uses the single VIRTUROID_LOCAL_LLM_MODEL (today's behavior, unchanged).
_LOCAL_FAST_ROLES = _FAST_ROLES | {"proposer", "critic"}
_LOCAL_STRONG_ROLES = _BUILD_ROLES | {"diagnostician", "codegen", "designer"}


def _local_model_for_role(role: str) -> str:
    """Pick the LOCAL model for a role: a cheap high-throughput model for breadth roles
    (``VIRTUROID_LOCAL_LLM_FAST_MODEL``), a strong model for deep roles (``VIRTUROID_LOCAL_LLM_BUILD_MODEL``),
    else the single ``VIRTUROID_LOCAL_LLM_MODEL``. Unset overrides -> the main model for all roles (back-compat)."""
    main = os.environ.get("VIRTUROID_LOCAL_LLM_MODEL", "nvidia/nemotron")
    fast = os.environ.get("VIRTUROID_LOCAL_LLM_FAST_MODEL")
    strong = os.environ.get("VIRTUROID_LOCAL_LLM_BUILD_MODEL")
    if fast and role in _LOCAL_FAST_ROLES:
        return fast
    if strong and role in _LOCAL_STRONG_ROLES:
        return strong
    return main


def get_llm(role: str = "designer") -> LLMClient | None:
    """Resolve a backend from environment, or None to use the deterministic fallback.

    VIRTUROID_LLM_BACKEND: "claude" | "openai" | "local" | "mock" | "off" (default "off").
    VIRTUROID_LOCAL_LLM_URL / VIRTUROID_LOCAL_LLM_MODEL configure the local model.
    VIRTUROID_OPENAI_MODEL configures the ChatGPT model for reasoning agents
    (default gpt-4o-mini); VIRTUROID_OPENAI_FAST_MODEL (optional) routes high-volume
    agents (part recommender, scene, parser, species) to a cheaper model.

    Values may live in a project-local ``.env`` file (gitignored) instead of the
    device environment — see ``_load_local_env``.
    """
    _load_local_env()
    # G-D: the provable zero-internal-token switch. When set, NO role ever gets a backend -- the connected
    # frontier agent (or the deterministic fallback) does everything -- and the ledger records the denial.
    if os.environ.get("VIRTUROID_NO_INTERNAL_LLM") == "1":
        record_blocked(role)
        return None
    backend = os.environ.get("VIRTUROID_LLM_BACKEND", "off").lower()
    if backend in ("off", ""):
        return None
    client: LLMClient | None
    if backend == "mock":
        client = MockLLM()
    elif backend == "claude":
        client = ClaudeLLM(model=os.environ.get("VIRTUROID_CLAUDE_MODEL", "claude-opus-4-8"))
    elif backend in ("openai", "chatgpt", "gpt"):
        model = _openai_model_for_role(role)
        # The strong build model (gpt-5.2) can have a low tier cap (e.g. 50 requests/day); fall back to the
        # cheap high-throughput model on a daily 429 so a busy product keeps LLM-designed bodies. No fallback
        # when the role already runs on the fast model (it would just be itself).
        fast = os.environ.get("VIRTUROID_OPENAI_FAST_MODEL", "gpt-4.1-mini")
        client = OpenAILLM(model=model, fallback_model=(fast if model != fast else None))
    elif backend == "local":
        url = os.environ.get("VIRTUROID_LOCAL_LLM_URL", "http://localhost:8000/v1")
        model = _local_model_for_role(role)                    # role-aware breadth/depth routing (M2/WS7b)
        fmt = os.environ.get("VIRTUROID_LOCAL_LLM_FORMAT", "json_object").lower()
        client = LocalLLM(base_url=url, model=model, format_mode=fmt)
    else:
        return None
    return _LedgerClient(client, role)                         # G-D: count every real internal completion


class SpendCapExceeded(RuntimeError):
    """Raised by SpendGuard when a per-run cap is hit. Design-search operators already treat any LLM exception as
    'no backend' and fall back to the heuristic (fail-OPEN), so a breached cap degrades gracefully to LLM-free."""


class SpendGuard:
    """Wrap an ``LLMClient`` with per-run HARD CAPS (plan v2 §4.2, layer 1: fail-open-to-heuristic). Tracks calls
    and ESTIMATED tokens/cost per role; when ``max_calls`` or ``max_usd`` would be exceeded, ``complete_json``
    raises :class:`SpendCapExceeded` BEFORE the call, so the caller falls back to the heuristic. ``price_in``/
    ``price_out`` are USD per 1M tokens (see the stream-3 pricing table); token counts are a ~len/4 estimate,
    good enough for a budget ceiling. ``on_spend(report)`` streams the running total into the run artifact.

    This is the transparent drop-in wrapper — ``SpendGuard(get_llm(role), max_calls=..., max_usd=...)`` — so the
    operators (Proposer/Critic/Diagnostician) need no changes; the guard IS an LLMClient."""

    def __init__(self, client, *, max_calls: int | None = None, max_usd: float | None = None,
                 price_in: float = 0.0, price_out: float = 0.0, role: str = "designer", on_spend=None) -> None:
        self.client = client
        self.name = f"guarded:{getattr(client, 'name', '?')}"
        self.max_calls, self.max_usd = max_calls, max_usd
        self.price_in, self.price_out, self.role, self.on_spend = price_in, price_out, role, on_spend
        self.calls = self.tokens_in = self.tokens_out = 0
        self.usd = 0.0

    @staticmethod
    def _est_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)

    def report(self) -> dict:
        return {"role": self.role, "calls": self.calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "usd": round(self.usd, 4),
                "max_calls": self.max_calls, "max_usd": self.max_usd}

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int = 2048,
                      reasoning_effort: str | None = None) -> dict:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise SpendCapExceeded(f"call cap {self.max_calls} reached for role {self.role!r}")
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise SpendCapExceeded(f"spend cap ${self.max_usd} reached for role {self.role!r} (spent ${self.usd:.4f})")
        self.calls += 1
        ti = self._est_tokens(system) + self._est_tokens(user)
        self.tokens_in += ti
        out = self.client.complete_json(system, user, schema, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
        to = self._est_tokens(json.dumps(out, default=str))
        self.tokens_out += to
        self.usd += ti / 1e6 * self.price_in + to / 1e6 * self.price_out
        if self.on_spend is not None:
            self.on_spend(self.report())
        return out


# Per-role call caps (plan v2 §4.2 layer-1). A diverse Proposer batch is cheap + high-volume; the Critic screens
# every candidate; the Diagnostician is the expensive strong-tier call, kept rare; codegen (detectors/reward) is
# rarest. Tuned to keep a run's LLM spend in the single-dollar range (stream-3 pricing). Fail-open when hit.
PER_ROLE_CALL_CAPS = {"designer": 40, "proposer": 40, "critic": 80, "diagnostician": 10, "codegen": 6}


def make_routed_llm(role: str = "designer", *, max_calls: int | None = None, max_usd: float | None = None,
                    on_spend=None):
    """Resolve the env-configured backend for ``role`` (``get_llm`` already routes roles to models -- cheap for
    high-volume roles, strong for reasoning) and wrap it in a :class:`SpendGuard` with the per-role call cap
    (plan v2 §4.2). Returns ``None`` when no backend is configured (``VIRTUROID_LLM_BACKEND=off`` -> the caller
    falls back to the heuristic). This is the one call the harness uses so caps ALWAYS apply."""
    client = get_llm(role)
    if client is None:
        return None
    cap = max_calls if max_calls is not None else PER_ROLE_CALL_CAPS.get(role, 40)
    return SpendGuard(client, max_calls=cap, max_usd=max_usd, role=role, on_spend=on_spend)


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object out of a model response, tolerating fences/prose.

    Local models don't always honor a JSON constraint perfectly. We try a direct
    parse first, then strip ```json fences, then fall back to the first balanced
    ``{...}`` span. Raises ValueError if nothing parseable is found.
    """
    s = (text or "").strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    if s.startswith("```"):
        inner = s.split("```", 2)
        if len(inner) >= 2:
            body = inner[1]
            if body.lower().startswith("json"):
                body = body[4:]
            try:
                return json.loads(body.strip())
            except json.JSONDecodeError:
                pass
    start = s.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Local model did not return parseable JSON: {text[:200]!r}")


def _zero_for(prop: dict):
    t = prop.get("type")
    if t == "number" or t == "integer":
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return ""
