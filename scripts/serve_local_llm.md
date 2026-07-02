# Local LLM live-fire runbook (plan v3 M2 / WS7a)

Drive the agentic harness (Proposer / Critic / Diagnostician / detector-codegen) against a **local** model over
an OpenAI-compatible endpoint — **zero external spend, no API key**. This unblocks WS7 (live-fire) and M1
(the Claude+MCP baseline) without a paid key. FunSearch's finding that program search is "not too sensitive to
the exact LLM" means a mid local code model is enough for the search roles.

The wiring is proven hermetically by `tests/test_live_fire_local.py` (an in-process OpenAI stub). This runbook
is the same wiring against a *real* served model on the GPU box.

## Option A — Ollama (simplest)

```bash
# on the GPU box
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b        # ~5 GB Q4; fits a 12 GB GPU. 14b if VRAM allows.
ollama serve                        # exposes http://localhost:11434/v1 (OpenAI-compatible)
```

```bash
# point the harness at it (project-local .env or real env; env wins over .env)
VIRTUROID_LLM_BACKEND=local
VIRTUROID_LOCAL_LLM_URL=http://localhost:11434/v1
VIRTUROID_LOCAL_LLM_MODEL=qwen2.5-coder:7b
VIRTUROID_LOCAL_LLM_FORMAT=json_object      # Ollama supports json_object (default)
# optional cheap-breadth / strong-depth routing (M2a): breadth roles use FAST, deep roles use BUILD
VIRTUROID_LOCAL_LLM_FAST_MODEL=qwen2.5-coder:7b
VIRTUROID_LOCAL_LLM_BUILD_MODEL=qwen2.5-coder:14b
```

## Option B — vLLM (higher throughput, strict schema)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8000
```

```bash
VIRTUROID_LLM_BACKEND=local
VIRTUROID_LOCAL_LLM_URL=http://localhost:8000/v1
VIRTUROID_LOCAL_LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
VIRTUROID_LOCAL_LLM_FORMAT=json_schema       # vLLM supports grammar-constrained json_schema
```

## Run the live-fire

```bash
# Engineer search driven by the local model (falls back to the heuristic proposer if the server is down)
python -c "from virturoid.services.engineer_mode import run_engineer_search; ..."   # see scripts/run_demo.py

# or the head-to-head with Arm 0 (Claude+MCP baseline, M1) once a model is served:
python -m virturoid.run_benchmark --with-baseline
```

## Safety notes (WS7c)

* Every LLM-generated `detect(ep)` / reward snippet runs in `code_sandbox.run_detector`: an AST allowlist
  (`math/json/statistics/numpy` only — so no `os`/`socket`/network), a `python -I -S -E` subprocess with a
  **stripped env** (API keys removed — env inheritance is the #1 exfil channel), a wall-clock timeout, and a
  fresh scratch cwd. A detector is trusted only if it passes the fail-CLOSED calibration gate.
* The local server should bind `localhost` only. Do not expose the endpoint publicly.
* Never place a key in these env vars — the local backend needs none (`api_key=not-needed`).
