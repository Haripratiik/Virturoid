from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from virturoid.services.autonomous_builder import build_robot_package_from_prompt


DEFAULT_PROMPT = "Build a tabletop robot arm that sorts red and blue blocks."

# Static frontend assets live outside the Python package (repo-root /webui).
# parents[0] = src/virturoid, parents[1] = src, parents[2] = repo root.
WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"

# --- Assistant (local-model) configuration -----------------------------------
# The assistant is a thin, swappable layer in the frontend server. It defaults
# to a local Ollama runtime but is structured so other providers can be added
# later via VIRTUROID_ASSISTANT_PROVIDER without touching the build pipeline.
ASSISTANT_PROVIDER = os.environ.get("VIRTUROID_ASSISTANT_PROVIDER", "ollama")
ASSISTANT_MODEL = os.environ.get("VIRTUROID_ASSISTANT_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

_ASSISTANT_SYSTEM_PROMPT = (
    "You are Virturoid Studio's robotics build assistant. You help design, build, "
    "inspect, and improve simulated robots through a deterministic build pipeline. "
    "When the user wants to create or modify a robot, reply with ONLY a single JSON "
    "object on its own line of the form "
    '{"action":"build","prompt":"<full natural-language build request>",'
    '"sensor":null|"rgbd_camera"|"lidar","payload_kg":null|number,'
    '"reach_m":null|number,"train":false|true}. '
    "For anything else (questions, explanations, advice on building or training), "
    "reply normally in concise, technical prose. Never invent file paths or results."
)


def _resolve_build_root(build_root: Path) -> Path:
    """If the chosen build root has no built packages but the curated demo set (build/ui_verify) does, use the
    demo set -- so `python -m virturoid.ui_server` shows the demo instead of an empty studio (the demo foot-gun:
    the auto-demo references package IDs that only exist under build/ui_verify). Once the user builds into their
    own root, that root has packages and is used as-is."""
    def _has_packages(d: Path) -> bool:
        return d.exists() and any((c / "robot").exists() or (c / "simulation").exists()
                                  for c in d.iterdir() if c.is_dir())
    if not _has_packages(build_root):
        demo = Path("build") / "ui_verify"
        try:
            if demo.resolve() != build_root.resolve() and _has_packages(demo):
                print(f"[ui] build root {build_root} has no packages; serving the demo set at {demo}")
                return demo
        except OSError:
            pass
    return build_root


def package_honesty_summary(child) -> dict | None:
    """Compact honesty signals for the package list, read from a build's honest reports (§4.1/§4.8A/§4.8E):
    the BOM<->sim mass-fidelity ratio + flag count, and spec-compliance (did the build honor a detailed prompt?).
    Returns None when neither report exists. Best-effort -- never raises."""
    def _rd(p):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    out: dict = {}
    fid = _rd(Path(child) / "reports" / "bom_sim_fidelity.json")
    if fid:
        out["mass_fidelity_ratio"] = fid.get("mass_fidelity_ratio")
        out["fidelity_flags"] = len(fid.get("flags") or [])
    comp = _rd(Path(child) / "reports" / "spec_compliance.json")
    if comp:
        out["spec_all_honored"] = comp.get("all_honored")
        out["spec_constraints"] = len(comp.get("constraints") or [])
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Virturoid build console.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local server.")
    parser.add_argument("--port", type=int, default=8765, help="Port for the local server.")
    parser.add_argument(
        "--build-root",
        default=str(Path("build") / "ui_workbench"),
        help="Directory where builds triggered from the console are written.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve in the browser instead of opening the native desktop window.",
    )
    args = parser.parse_args()
    build_root = _resolve_build_root(Path(args.build_root))

    if args.web:
        _run_web(args.host, args.port, build_root)
        return

    try:
        run_desktop(args.host, args.port, build_root)
    except ImportError:
        print("pywebview is not installed; falling back to browser mode.")
        print("Install the desktop runtime with:  pip install pywebview")
        _run_web(args.host, args.port, build_root)


def _run_web(host: str, port: int, build_root: Path) -> None:
    server = create_server(host, port, build_root)
    print(f"Virturoid console running at http://{host}:{port}")
    print(f"Build output root: {build_root.resolve()}")
    server.serve_forever()


def run_desktop(host: str, port: int, build_root: Path) -> None:
    """Open the console as a native desktop window backed by the local server.

    The HTTP server runs on a daemon thread; pywebview owns the main thread.
    Raises ImportError when pywebview is unavailable so callers can fall back.
    """
    import threading

    import webview  # noqa: PLC0415  (optional desktop dependency)

    server = create_server(host, port, build_root)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    api = _DesktopApi()
    window = webview.create_window(
        "Virturoid \u00b7 Build Console",
        f"http://{host}:{port}/",
        js_api=api,
        width=1440,
        height=920,
        min_size=(1100, 720),
        background_color="#0c0f12",
        frameless=True,
        easy_drag=False,
    )
    api.bind(webview, window)
    webview.start()


class _DesktopApi:
    """Window-control bridge exposed to the frontend as window.pywebview.api."""

    def __init__(self) -> None:
        self._webview = None
        self._window = None
        self._maximized = False

    def bind(self, webview_module, window) -> None:
        self._webview = webview_module
        self._window = window

    def minimize(self) -> None:
        if self._window is not None:
            self._window.minimize()

    def toggle_maximize(self) -> bool:
        window = self._window
        if window is None:
            return False
        if self._maximized:
            for name in ("restore", "toggle_fullscreen"):
                action = getattr(window, name, None)
                if callable(action):
                    action()
                    break
            self._maximized = False
        else:
            for name in ("maximize", "toggle_fullscreen"):
                action = getattr(window, name, None)
                if callable(action):
                    action()
                    break
            self._maximized = True
        return self._maximized

    def close(self) -> None:
        if self._window is not None:
            self._window.destroy()


def create_server(host: str, port: int, build_root: Path) -> ThreadingHTTPServer:
    build_root = Path(build_root)

    class VirturoidRequestHandler(_Handler):
        root = build_root

    return ThreadingHTTPServer((host, port), VirturoidRequestHandler)


def build_ui_html() -> str:
    """Return the Virturoid Studio shell.

    The heavy UI (styles + dock engine + panel modules) is served as static
    assets from /app/* (the repo-root webui/ folder). This shell wires the
    window chrome, workspace tabs, and the static build form (kept in a
    <template> so the Properties panel can adopt it).

    The substrings 'Virturoid Local Build Workbench', id="buildForm",
    /api/build, and 'Train controller when supported' are intentionally present
    so the build form works and so server-shell tests pass.
    """
    return """<!doctype html>
<html lang="en" data-theme="mission">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Virturoid Local Build Workbench</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" />
  <link rel="stylesheet" href="/app/styles.css" />
  <script type="importmap">
    {
      "imports": {
        "three": "/app/vendor/three/build/three.module.js",
        "three/addons/": "/app/vendor/three/examples/jsm/",
        "three/examples/jsm/": "/app/vendor/three/examples/jsm/",
        "urdf-loader": "/app/vendor/urdf-loader/src/URDFLoader.js"
      }
    }
  </script>
</head>
<body>
  <div class="bg-grid" aria-hidden="true"></div>
  <div class="bg-glow" aria-hidden="true"></div>
  <div class="bg-scan" aria-hidden="true"></div>

  <header class="titlebar">
    <div class="tb-brand pywebview-drag-region">
      <span class="logo-mark" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2 3 7v10l9 5 9-5V7z" />
          <path d="M12 7v5l4 2" />
        </svg>
      </span>
      <span class="tb-name">VIRTUROID</span>
      <span class="tb-sub">studio</span>
    </div>

    <nav class="workspaces" id="workspaces" aria-label="Workspaces">
      <button type="button" class="ws-tab active" data-workspace="build">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="m3 7 9 5 9-5"/><path d="M12 22V12"/></svg>
        <span>Build</span>
      </button>
      <button type="button" class="ws-tab" data-workspace="memory">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2.4"/><circle cx="5" cy="6" r="1.8"/><circle cx="19" cy="6" r="1.8"/><circle cx="6" cy="19" r="1.8"/><circle cx="18" cy="18" r="1.8"/><path d="M10 11 6.5 7.2M14 11l3.3-3.6M10.6 13.6 7.2 17.6M13.6 13.4l3.1 3.2"/></svg>
        <span>Memory</span>
      </button>
      <button type="button" class="ws-tab" data-workspace="analysis">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m7 14 3-4 3 2 4-6"/></svg>
        <span>Analysis</span>
      </button>
    </nav>

    <div class="tb-spacer pywebview-drag-region"></div>

    <div class="tb-right">
      <div class="pkg-switch">
        <select id="packageSelect" class="pkg-select" aria-label="Active robot"></select>
        <span id="pkgBadge" class="pkg-badge muted">No package</span>
      </div>
      <div id="assistantStatus" class="model-chip offline" title="Assistant model status">
        <span class="dot"></span><span id="assistantStatusText">model: checking</span>
      </div>
      <div class="tb-controls" id="winControls" hidden>
        <button type="button" class="tb-btn" id="winMin" aria-label="Minimize" title="Minimize">
          <svg viewBox="0 0 12 12" width="11" height="11" stroke="currentColor" stroke-width="1.3"><line x1="2" y1="6" x2="10" y2="6"/></svg>
        </button>
        <button type="button" class="tb-btn" id="winMax" aria-label="Maximize" title="Maximize">
          <svg viewBox="0 0 12 12" width="11" height="11" fill="none" stroke="currentColor" stroke-width="1.3"><rect x="2.4" y="2.4" width="7.2" height="7.2" rx="1"/></svg>
        </button>
        <button type="button" class="tb-btn tb-close" id="winClose" aria-label="Close" title="Close">
          <svg viewBox="0 0 12 12" width="11" height="11" stroke="currentColor" stroke-width="1.3"><line x1="2.6" y1="2.6" x2="9.4" y2="9.4"/><line x1="9.4" y1="2.6" x2="2.6" y2="9.4"/></svg>
        </button>
      </div>
    </div>
  </header>

  <div id="studio" class="studio"></div>

  <footer class="statusbar">
    <div class="sb-group">
      <span class="sb-item"><span class="sb-k">robot</span><span class="sb-v" id="sbRobot">&mdash;</span></span>
      <span class="sb-sep"></span>
      <span class="sb-item"><span class="sb-k">class</span><span class="sb-v" id="sbClass">&mdash;</span></span>
      <span class="sb-sep"></span>
      <span class="sb-item"><span class="sb-k">scenes</span><span class="sb-v" id="sbScenes">&mdash;</span></span>
      <span class="sb-sep"></span>
      <span class="sb-item" id="sbValid"></span>
    </div>
    <div class="sb-group">
      <span class="sb-hint">drag tabs to rearrange &middot; drag dividers to resize</span>
      <button type="button" class="sb-btn" id="resetLayout" title="Reset this workspace layout">Reset layout</button>
    </div>
  </footer>

  <!-- Static build form: kept in a template so the Properties panel can adopt it
       (and so the build pipeline works without JS). -->
  <template id="tpl-properties">
    <form id="buildForm" data-endpoint="/api/build" class="build-form">
      <label for="prompt">Robot task prompt</label>
      <textarea id="prompt" name="prompt">Build a tabletop robot arm that sorts red and blue blocks into matching bins.</textarea>
      <div class="row">
        <div>
          <label for="sensor">Sensor</label>
          <select id="sensor" name="sensor">
            <option value="">Auto</option>
            <option value="rgbd_camera">RGB-D camera</option>
            <option value="lidar">LiDAR</option>
          </select>
        </div>
        <div>
          <label for="outputName">Output name</label>
          <input id="outputName" name="outputName" placeholder="auto-generated" />
        </div>
      </div>
      <div class="row">
        <div>
          <label for="payloadKg">Payload kg</label>
          <input id="payloadKg" name="payloadKg" type="number" step="0.05" min="0" placeholder="auto" />
        </div>
        <div>
          <label for="reachM">Reach m</label>
          <input id="reachM" name="reachM" type="number" step="0.05" min="0" placeholder="auto" />
        </div>
      </div>
      <label class="check"><input id="train" name="train" type="checkbox" /> <span>Train controller when supported</span></label>
      <div class="actions">
        <button id="submitButton" type="submit" class="button primary">Compile package</button>
        <button type="button" class="button" id="armExample">Arm</button>
        <button type="button" class="button" id="mobileExample">Mobile</button>
      </div>
      <div id="buildResult" class="build-result"></div>
    </form>
  </template>

  <script>
    // Native window chrome wiring (only active inside the pywebview desktop window).
    (function () {
      function bindControls() {
        var controls = document.getElementById("winControls");
        if (!window.pywebview || !window.pywebview.api || !controls) return;
        controls.hidden = false;
        var min = document.getElementById("winMin");
        var max = document.getElementById("winMax");
        var close = document.getElementById("winClose");
        if (min) min.addEventListener("click", function () { window.pywebview.api.minimize(); });
        if (max) max.addEventListener("click", function () { window.pywebview.api.toggle_maximize(); });
        if (close) close.addEventListener("click", function () { window.pywebview.api.close(); });
      }
      window.addEventListener("pywebviewready", bindControls);
      if (window.pywebview && window.pywebview.api) bindControls();
    })();
  </script>
  <script type="module">
    try {
      await import("/app/main.js");
    } catch (error) {
      console.error(error);
      var host = document.getElementById("studio");
      if (host) {
        host.innerHTML = '<div class="boot-error">Frontend failed to load: ' + String(error.message || error) + '</div>';
      }
    }
  </script>
</body>
</html>
"""


def run_build_from_payload(payload: dict, build_root: Path) -> dict:
    prompt = str(payload.get("prompt") or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
    output_name = _safe_output_name(str(payload.get("output_name") or _slugify(prompt)))
    output_dir = Path(build_root) / output_name
    result = build_robot_package_from_prompt(
        prompt=prompt,
        output_dir=output_dir,
        payload_kg=_optional_float(payload.get("payload_kg")),
        reach_m=_optional_float(payload.get("reach_m")),
        sensor=payload.get("sensor") or None,
        train=bool(payload.get("train")),
    )
    return {
        "result": result.to_dict(),
        "package_url_prefix": f"/package/{output_name}",
        "workbench_url": f"/package/{output_name}/reports/workbench.html",
        "contract_url": f"/package/{output_name}/reports/robot_package_contract.json",
    }


# --- Assistant layer ---------------------------------------------------------


def assistant_status() -> dict:
    """Report whether a local model backend is reachable for the assistant."""
    info = {"provider": ASSISTANT_PROVIDER, "model": ASSISTANT_MODEL, "online": False, "models": []}
    if ASSISTANT_PROVIDER == "ollama":
        try:
            with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as response:
                data = json.loads(response.read().decode("utf-8"))
            info["models"] = [m.get("name") for m in data.get("models", []) if m.get("name")]
            info["online"] = True
            info["model_available"] = any(
                str(name).split(":")[0] == ASSISTANT_MODEL.split(":")[0] for name in info["models"]
            )
        except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
            info["online"] = False
    return info


def _ollama_chat(messages: list[dict], model: str) -> str | None:
    """Call a local Ollama chat model. Returns text or None if unavailable."""
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        return (data.get("message") or {}).get("content")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def _assistant_complete(messages: list[dict]) -> str | None:
    """Provider-agnostic completion. Add new providers here without touching callers."""
    if ASSISTANT_PROVIDER == "ollama":
        return _ollama_chat(messages, ASSISTANT_MODEL)
    return None


def _extract_build_action(text: str) -> dict | None:
    """Pull a {"action":"build",...} JSON object out of a model reply."""
    if not text or '"action"' not in text:
        return None
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("action") == "build":
            return obj
    return None


_BUILD_INTENT = re.compile(
    r"\b(build|create|make|design|generate|assemble|spawn)\b.*\b(robot|arm|manipulator|gripper|"
    r"mobile|base|rover|drone|bot|machine)\b",
    re.IGNORECASE,
)


def _fallback_build_action(text: str) -> dict | None:
    """Heuristic build detection used when no local model is available."""
    if text and _BUILD_INTENT.search(text):
        return {"action": "build", "prompt": text.strip(), "train": False}
    return None


def run_assistant_turn(payload: dict, build_root: Path) -> dict:
    """Run one assistant turn: chat, and optionally drive a real build.

    Returns {role, content, action, build?}. When a local model is offline the
    assistant still detects build intent heuristically so it remains useful.
    """
    messages = payload.get("messages") or []
    user_text = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            user_text = str(message.get("content") or "")
            break

    status = assistant_status()
    model_messages = [{"role": "system", "content": _ASSISTANT_SYSTEM_PROMPT}, *(
        {"role": m.get("role", "user"), "content": str(m.get("content") or "")}
        for m in messages
        if m.get("role") in {"user", "assistant"}
    )]

    raw_reply = _assistant_complete(model_messages) if status.get("online") else None
    used_model = raw_reply is not None

    action = _extract_build_action(raw_reply or "") if used_model else None
    if action is None:
        action = _fallback_build_action(user_text)

    if action is not None:
        build_payload = {
            "prompt": action.get("prompt") or user_text,
            "sensor": action.get("sensor"),
            "payload_kg": action.get("payload_kg"),
            "reach_m": action.get("reach_m"),
            "train": bool(action.get("train")),
        }
        try:
            build = run_build_from_payload(build_payload, build_root)
        except Exception as exc:  # noqa: BLE001
            return {
                "role": "assistant",
                "content": f"I tried to build that but the pipeline raised an error:\n\n{exc}",
                "action": "error",
                "model_used": used_model,
            }
        summary = _summarize_build(build, used_model)
        return {
            "role": "assistant",
            "content": summary,
            "action": "build",
            "build": build,
            "model_used": used_model,
        }

    if used_model and raw_reply:
        return {"role": "assistant", "content": raw_reply.strip(), "action": "chat", "model_used": True}

    # No model and no build intent: a helpful, honest local reply.
    hint = (
        "A local model isn't running, so I'm in deterministic mode. I can still build robots: "
        "tell me what to build (e.g. \"build a tabletop arm that sorts blocks\"). "
        "To enable full conversation, start Ollama and pull a model "
        f"(current target: {ASSISTANT_MODEL})."
    )
    return {"role": "assistant", "content": hint, "action": "chat", "model_used": False}


def _summarize_build(build: dict, used_model: bool) -> str:
    result = build.get("result", {})
    name = str(build.get("package_url_prefix", "")).split("/")[-1]
    valid = "valid" if result.get("package_valid") else "not yet valid"
    readiness = result.get("readiness") or {}
    score = readiness.get("score")
    lines = [
        f"Built **{name}** \u2014 a {result.get('selected_robot_class', 'robot')} "
        f"({result.get('selected_species', 'unknown species')}) for the "
        f"{result.get('task_type', 'task')} task. The package is {valid}.",
    ]
    if score is not None:
        lines.append(f"Readiness score: {score}%.")
    lines.append("I've loaded it into the viewport and updated the inspector.")
    return "\n\n".join(lines)


class _Handler(BaseHTTPRequestHandler):
    root: Path = Path("build") / "ui_workbench"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(build_ui_html(), "text/html; charset=utf-8")
            return
        if parsed.path == "/app" or parsed.path.startswith("/app/"):
            self._send_webui_file(parsed.path)
            return
        if parsed.path == "/api/packages":
            self._send_json(self._list_packages())
            return
        if parsed.path == "/api/scorecard":
            self._send_json(self._scorecard(parsed))
            return
        if parsed.path == "/api/flywheel":
            self._send_json(self._flywheel())
            return
        if parsed.path == "/api/design_brain":
            self._send_json(self._design_brain())
            return
        if parsed.path == "/api/tools":                        # agentic tool surface: DISCOVER the tools
            from virturoid.services.agent_tools import tool_specs
            self._send_json({"tools": tool_specs()})
            return
        if parsed.path == "/api/episode":
            self._send_episode(parsed)
            return
        if parsed.path == "/api/assistant/status":
            self._send_json(assistant_status())
            return
        if parsed.path.startswith("/package/"):
            self._send_package_file(parsed.path)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/build":
            self._handle_json_post(lambda payload: run_build_from_payload(payload, self.root))
            return
        if parsed.path == "/api/assistant/chat":
            self._handle_json_post(lambda payload: run_assistant_turn(payload, self.root))
            return
        if parsed.path == "/api/tool":                         # agentic tool surface: INVOKE a tool by name
            self._handle_json_post(self._agent_tool_call)
            return
        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def _agent_tool_call(self, payload) -> dict:
        """Dispatch an agentic tool: ``{tool, args}`` -> ``{ok, tool, result|error}`` (localhost tool surface
        so an external agent / MCP bridge can drive the platform over HTTP; see services/agent_tools)."""
        from virturoid.services.agent_tools import call_tool
        return call_tool(payload.get("tool", ""), payload.get("args") or {})

    def _handle_json_post(self, handler) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            result = handler(payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(result)

    def _scorecard(self, parsed) -> dict:
        """The unified Honesty Scorecard for a package (§4.1) — computed on demand so it works even for packages
        built before the scorecard was persisted. Reads the build's honest reports (readiness + spec-compliance +
        fidelity + sim2sim)."""
        from urllib.parse import parse_qs
        name = (parse_qs(parsed.query).get("package") or [""])[0]
        if not name:
            return {"rows": [], "n_claims": 0, "headline": "no package selected"}
        pkg = self.root / _safe_output_name(name)
        if not pkg.exists():
            return {"rows": [], "n_claims": 0, "headline": "package not found"}
        try:
            from virturoid.services.honesty_scorecard import scorecard_from_package
            return scorecard_from_package(pkg)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "rows": [], "n_claims": 0, "headline": "scorecard unavailable"}

    def _flywheel(self) -> dict:
        """The flywheel COMPOUNDING CURVE (the moat KPI) for the build set — the per-cycle series written by
        scripts/run_flywheel_for_demo.py at the build root. Empty (with a hint) until that script runs."""
        p = self.root / "flywheel_compounding.json"
        if not p.exists():
            return {"series": [], "n_cycles": 0, "compounding": False,
                    "headline": "run scripts/run_flywheel_for_demo.py to populate the compounding curve"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"series": [], "n_cycles": 0, "compounding": False, "headline": f"unreadable: {exc}"}

    def _design_brain(self) -> dict:
        """The Design Brain panel (the moat MEASURED): MAP-Elites coverage/QD-score + provenance compounding
        from the build set's shared memory. Best-effort over candidate memory dirs; zeros + a hint until real
        builds have populated the archive/provenance (autonomous_build's keystone writes them)."""
        try:
            from virturoid.services.design_brain import design_brain_summary
        except Exception as exc:  # noqa: BLE001
            return {"archive_coverage": 0, "provenance_edges": 0, "headline": f"unavailable: {exc}"}
        for cand in (self.root / "memory", self.root, self.root.parent / "memory"):
            s = design_brain_summary(cand)
            if s.get("archive_coverage") or s.get("provenance_edges"):
                return s
        return design_brain_summary(self.root / "memory")   # zeros + headline

    def _send_package_file(self, path: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            self._send_json({"error": "Missing package file path."}, status=HTTPStatus.NOT_FOUND)
            return
        package_name = _safe_output_name(parts[1])
        relative = Path(*parts[2:]) if len(parts) > 2 else Path("reports") / "workbench.html"
        target = (self.root / package_name / relative).resolve()
        root = (self.root / package_name).resolve()
        if root not in [target, *target.parents] or not target.exists() or target.is_dir():
            self._send_json({"error": "Package file not found."}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(target.read_bytes(), _content_type(target))

    def _send_webui_file(self, path: str) -> None:
        relative = unquote(path[len("/app/"):]) if path.startswith("/app/") else ""
        parts = [part for part in relative.split("/") if part and part != ".."]
        target = (WEBUI_DIR / Path(*parts)).resolve() if parts else WEBUI_DIR.resolve()
        root = WEBUI_DIR.resolve()
        if root not in [target, *target.parents] or not target.exists() or target.is_dir():
            self._send_json({"error": "Asset not found."}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(target.read_bytes(), _content_type(target))

    def _send_episode(self, parsed) -> None:
        """Run one real episode for a built package and return geom metadata + per-frame world poses, so the
        viewport can animate the actual learned walk / grasp (faithful to physics, no in-browser kinematics)."""
        from urllib.parse import parse_qs

        from virturoid.services.viewer_sim import simulate_episode_for_viewer

        q = parse_qs(parsed.query)
        package_name = _safe_output_name((q.get("package") or [""])[0])
        try:
            scene_index = int((q.get("scene") or ["0"])[0])
        except ValueError:
            scene_index = 0
        pkg = (self.root / package_name).resolve()
        if not package_name or self.root.resolve() not in [pkg, *pkg.parents] or not pkg.is_dir():
            self._send_json({"error": "Unknown package."}, status=HTTPStatus.NOT_FOUND)
            return
        try:
            view = simulate_episode_for_viewer(pkg, scene_index=scene_index)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"Episode replay failed: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json(view)

    def _list_packages(self) -> dict:
        root = Path(self.root).resolve()
        packages: list[dict] = []
        if root.exists():
            for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
                if not child.is_dir():
                    continue
                has_urdf = (child / "robot" / "robot.urdf").exists()
                has_scenes = (child / "simulation" / "mujoco" / "compiled_scene_index.json").exists()
                if not (has_urdf or has_scenes):  # list packages that can render (URDF) OR replay (scenes)
                    continue
                scene_count = 0
                index_path = child / "simulation" / "mujoco" / "compiled_scene_index.json"
                if index_path.exists():
                    try:
                        scene_count = len(json.loads(index_path.read_text(encoding="utf-8")).get("scenes", []))
                    except (json.JSONDecodeError, OSError):
                        scene_count = 0
                valid: bool | None = None
                robot_class: str | None = None
                species: str | None = None
                contract_path = child / "reports" / "robot_package_contract.json"
                if contract_path.exists():
                    try:
                        contract = json.loads(contract_path.read_text(encoding="utf-8"))
                        valid = bool(contract.get("ok"))
                        robot_class = contract.get("robot_class")
                        species = contract.get("species")
                    except (json.JSONDecodeError, OSError):
                        valid = None
                # Real morphology facts from the actual model -- the accurate species + actuated DOF, so the
                # memory shows what was really built (not a hardcoded placeholder species/DOF).
                dof: int | None = None
                genome_path = child / "robot" / "robot_genome.json"
                if genome_path.exists():
                    try:
                        genome = json.loads(genome_path.read_text(encoding="utf-8"))
                        species = species or genome.get("species")
                        joints = genome.get("joints") or []
                        dof = sum(1 for j in joints if isinstance(j, dict)
                                  and str(j.get("joint_type", "")).lower() not in ("fixed", "weld", ""))
                    except (json.JSONDecodeError, OSError):
                        pass
                spec_summary = None
                spec_path = child / "reports" / "spec_sheet.json"
                if spec_path.exists():
                    try:
                        sp = json.loads(spec_path.read_text(encoding="utf-8"))
                        spec_summary = {
                            "summary": sp.get("summary"),
                            "mass_kg": (sp.get("physical") or {}).get("mass_kg"),
                            "cost_usd": (sp.get("power_and_cost") or {}).get("est_parts_cost_usd"),
                            "power_w": (sp.get("power_and_cost") or {}).get("est_power_draw_w"),
                            "success": (sp.get("performance") or {}).get("success_rate"),
                            "task": (sp.get("performance") or {}).get("task"),
                        }
                    except (json.JSONDecodeError, OSError):
                        spec_summary = None
                honesty = package_honesty_summary(child)        # §4.1/§4.8A/§4.8E honest signals
                packages.append(
                    {
                        "id": child.name,
                        "scene_count": scene_count,
                        "has_meshes": (child / "cad" / "mesh" / "visual").exists(),
                        "valid": valid,
                        "robot_class": robot_class,
                        "species": species,
                        "dof": dof,
                        "spec": spec_summary,
                        "honesty": honesty,
                    }
                )
        return {"build_root": str(root), "packages": packages}

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_text(json.dumps(payload, indent=2), "application/json; charset=utf-8", status=status)

    def _send_text(self, text: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(text.encode("utf-8"), content_type, status=status)

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def _optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _safe_output_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._-")
    return cleaned[:80] or "virturoid_build"


def _slugify(value: str) -> str:
    return _safe_output_name(value.lower())[:48]


def _content_type(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".json":
        return "application/json; charset=utf-8"
    if path.suffix == ".xml" or path.suffix == ".urdf":
        return "application/xml; charset=utf-8"
    if path.suffix in {".js", ".mjs"}:
        return "text/javascript; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    if path.suffix == ".svg":
        return "image/svg+xml"
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    if path.suffix == ".stl":
        return "application/octet-stream"
    return "text/plain; charset=utf-8"


if __name__ == "__main__":
    main()
