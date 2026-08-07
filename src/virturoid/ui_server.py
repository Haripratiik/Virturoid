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

from virturoid.services.package_status import has_robot_package, package_status

DEFAULT_PROMPT = "Build a tabletop robot arm that sorts red and blue blocks."

# Static frontend assets live outside the Python package (repo-root /webui).
# parents[0] = src/virturoid, parents[1] = src, parents[2] = repo root.
WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"

# Virturoid Studio (the new React UI) is served ADDITIVELY at /studio from the
# Vite build output. The legacy UI at / is untouched; both share the same API.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

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
    parser.add_argument(
        "--ui",
        choices=("legacy", "studio"),
        default=os.environ.get("VIRTUROID_UI", "legacy"),
        help="Which frontend to open: the legacy console (/) or Virturoid Studio (/studio/). "
             "Applies to BOTH the native desktop window and --web (which prints the matching URL). "
             "Default legacy until Studio reaches parity.",
    )
    args = parser.parse_args()
    build_root = _resolve_build_root(Path(args.build_root))

    if args.ui == "studio" and not (FRONTEND_DIST / "index.html").exists():
        print("Virturoid Studio is not built yet. Build it with:")
        print("  cd frontend && npm install && npm run build")
        print("Falling back to the legacy console.")
        args.ui = "legacy"

    if args.web:
        _run_web(args.host, args.port, build_root, ui=args.ui)
        return

    try:
        run_desktop(args.host, args.port, build_root, ui=args.ui)
    except ImportError as exc:
        # pywebview is an OPTIONAL extra (pyproject `[desktop]`); it is imported lazily so the web
        # path never needs it. Name the extra, not a bare `pip install pywebview`, so the fix
        # matches how the rest of the project is installed.
        print(f"The native desktop window needs pywebview, which is not installed ({exc}).")
        print('Install it with:  pip install -e ".[desktop]"     (or: pip install pywebview)')
        print("Falling back to browser mode.")
        _run_web(args.host, args.port, build_root, ui=args.ui)


def _run_web(host: str, port: int, build_root: Path, ui: str = "legacy") -> None:
    """Serve in the browser. ``ui`` selects which URL is printed -- Studio lives at /studio/, and
    printing only the root sent every reader of the README's headline command to the LEGACY console
    (which also brands itself 'Virturoid Studio', so the mistake was invisible)."""
    server = create_server(host, port, build_root)
    if ui == "studio":
        print(f"Virturoid Studio running at http://{host}:{port}/studio/")
        print(f"Legacy build console at     http://{host}:{port}/")
    else:
        print(f"Virturoid build console running at http://{host}:{port}/")
        print(f"Virturoid Studio (React app) at    http://{host}:{port}/studio/")
    print(f"Build output root: {build_root.resolve()}")
    server.serve_forever()


def run_desktop(host: str, port: int, build_root: Path, ui: str = "legacy") -> None:
    """Open the console as a NATIVE desktop window (pywebview / WebView2) backed by
    the local server \u2014 no browser, no tabs; the app owns its own window chrome.

    The HTTP server runs on a daemon thread; pywebview owns the main thread.
    Raises ImportError when pywebview is unavailable so callers can fall back.
    ``ui`` picks the frontend the window opens: "legacy" (/) or "studio" (/studio).
    """
    import threading

    import webview  # noqa: PLC0415  (optional desktop dependency)

    server = create_server(host, port, build_root)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    studio = ui == "studio"
    api = _DesktopApi()
    window = webview.create_window(
        "Virturoid \u00b7 Studio" if studio else "Virturoid \u00b7 Build Console",
        f"http://{host}:{port}/studio" if studio else f"http://{host}:{port}/",
        js_api=api,
        width=1440,
        height=920,
        min_size=(1100, 720),
        background_color="#1a1815" if studio else "#0c0f12",
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


def build_agent_sessions_html() -> str:
    """C1-C3 (plan_v5 W-C): the 'watch the agent build' viewer. A self-contained page that live-polls
    /api/sessions and, per held robot, shows its fresh render + summary from /api/sessions/<id>. Because
    sessions are file-backed (G-B), a robot authored by a SEPARATE stdio MCP client (Claude Code/Codex)
    appears here in the running app — the app is the viewer for the external agent's work."""
    return """<!doctype html><html><head><meta charset="utf-8"><title>Virturoid - Agent Sessions</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d1d9;font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:18px 24px;border-bottom:1px solid #21262d;display:flex;align-items:baseline;gap:14px}
h1{margin:0;font-size:18px;letter-spacing:.3px;color:#e6edf3}
.sub{color:#7d8590;font-size:12px}
.dot{width:8px;height:8px;border-radius:50%;background:#3fb950;display:inline-block;margin-right:6px;animation:p 2s infinite}
@keyframes p{50%{opacity:.35}}
main{display:grid;grid-template-columns:300px 1fr;gap:0;height:calc(100vh - 61px)}
#list{border-right:1px solid #21262d;overflow:auto}
.row{padding:12px 18px;border-bottom:1px solid #161b22;cursor:pointer}
.row:hover{background:#161b22}
.row.sel{background:#1f2937;border-left:3px solid #58a6ff;padding-left:15px}
.rid{color:#58a6ff;font-size:12px}
.rlabel{color:#e6edf3;margin-top:2px}
.rprompt{color:#7d8590;font-size:11px;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#detail{padding:24px;overflow:auto}
#detail img{max-width:560px;width:100%;border:1px solid #21262d;border-radius:8px;background:#010409}
table{border-collapse:collapse;margin-top:16px;font-size:13px}
td{padding:4px 16px 4px 0;border-bottom:1px solid #161b22}
td:first-child{color:#7d8590}
.empty{color:#7d8590;padding:40px;text-align:center}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;background:#1f6feb33;color:#58a6ff;font-size:11px}
</style></head><body>
<header><h1>Virturoid</h1><span class="sub"><span class="dot"></span>Agent Sessions - live view of what the connected agent is holding</span></header>
<main>
  <div id="list"><div class="empty">no sessions yet<br><span style="font-size:11px">connect an agent (submit_design / create_robot) and it appears here</span></div></div>
  <div id="detail"><div class="empty">select a robot</div></div>
</main>
<script>
let sel=null, robots=[];
async function poll(){
  try{
    const r=await fetch('/api/sessions'); const j=await r.json(); robots=j.robots||[];
    const list=document.getElementById('list');
    if(!robots.length){list.innerHTML='<div class="empty">no sessions yet<br><span style="font-size:11px">connect an agent and it appears here</span></div>';return;}
    list.innerHTML=robots.map(x=>`<div class="row ${x.robot_id===sel?'sel':''}" onclick="pick('${x.robot_id}')">`+
      `<div class="rid">${x.robot_id}</div><div class="rlabel">${x.robot_class||'?'} <span class="badge">${x.label||''}</span></div>`+
      `<div class="rprompt">${(x.prompt||'').replace(/</g,'&lt;')}</div></div>`).join('');
    if(!sel && robots.length){pick(robots[0].robot_id);}
  }catch(e){}
}
async function pick(id){
  sel=id; poll();
  const d=document.getElementById('detail'); d.innerHTML='<div class="empty">rendering '+id+'...</div>';
  try{
    const r=await fetch('/api/sessions/'+id); const j=await r.json(); const s=j.summary||{};
    const rows=[['class',s.robot_class],['kind',s.kind],['segments',s.n_segments],['dof',s.dof],
      ['appendages',JSON.stringify(s.appendages||{})],['height (m)',s.standing_height_m],
      ['mass (kg)',s.total_mass_kg],['material',s.material],['end effector',s.end_effector],
      ['undo depth',(j.meta||{}).undo_depth]];
    d.innerHTML=(j.render_url?`<img src="${j.render_url}?t=${Date.now()}">`:'<div class="empty">no render</div>')+
      '<table>'+rows.filter(x=>x[1]!=null&&x[1]!=='').map(x=>`<tr><td>${x[0]}</td><td>${x[1]}</td></tr>`).join('')+'</table>';
  }catch(e){d.innerHTML='<div class="empty">could not load '+id+'</div>';}
}
poll(); setInterval(poll, 2500);
</script></body></html>"""


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
  <!-- No external font CDN: this is a LOCAL-FIRST offline console (2026-07-24 audit). Loading Google Fonts
       hung/flashed with no network and leaked a request off-box; styles.css already declares system fallbacks
       ("Segoe UI"/system-ui/Consolas) so the console renders cleanly with zero external dependencies. -->
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
    # The Studio build action is AI-first: autonomous_build calls the LLM planner
    # and grounded morphology path, then physics validates the result.  It must
    # never silently fall back to the older template package builder.
    from virturoid.services.autonomous_build import autonomous_build

    report = autonomous_build(
        prompt=prompt,
        output_dir=output_dir,
        train=bool(payload.get("train")),
        memory_dir=Path(build_root) / "memory",
    )
    result = report.to_dict()
    # Preserve the legacy console's small result contract while it shares the
    # same endpoint with Studio.
    result.update({
        "selected_robot_class": report.robot_class,
        "selected_species": report.species,
        "package_valid": report.feasible,
        "readiness": {},
        "output_dir": str(output_dir),
        "output_name": output_name,
    })
    has_package = bool(report.feasible and output_dir.exists())
    return {
        "result": result,
        "package_url_prefix": f"/package/{output_name}" if has_package else None,
        "workbench_url": f"/package/{output_name}/reports/workbench.html" if has_package else None,
        "contract_url": f"/package/{output_name}/reports/robot_package_contract.json" if has_package else None,
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


def _assistant_complete(messages: list[dict], model: str = ASSISTANT_MODEL) -> str | None:
    """Provider-agnostic completion. Add new providers here without touching callers."""
    if ASSISTANT_PROVIDER == "ollama":
        return _ollama_chat(messages, model)
    return None


def _selected_assistant_model(payload: dict, status: dict) -> str:
    """Choose an installed local model for one request without mutating global configuration."""
    requested = str(payload.get("model") or "").strip()
    available = {str(name) for name in status.get("models") or []}
    # Accept only a tag returned by the local runtime. This prevents a client from asking Ollama to run an
    # arbitrary model; clients without the new field retain the configured default.
    return requested if requested and requested in available else ASSISTANT_MODEL


def _assistant_model_available(model: str, status: dict) -> bool:
    """Match Ollama's optional ``:tag`` suffix the same way the status endpoint does."""
    return any(str(name).split(":")[0] == model.split(":")[0] for name in status.get("models") or [])


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


# --- Build-intent detection (no local model) ---------------------------------
# THE MATCHING RULE, in three layers. The old rule required an IMPERATIVE VERB BEFORE THE NOUN
# ("build a quadruped robot"), so the phrasing the input placeholder itself invites — a bare NOUN
# PHRASE, "a quadruped robot that walks" — fell through to the offline hint and the primary
# interaction dead-ended. A robot is a THING, so naming the thing is a legitimate request for it.
#
#   1. ASKING (wins over everything): the message opens with an interrogative or a meta verb
#      ("what is a quadruped?", "explain the gripper", "is this arm valid?"). Never a build --
#      these ask ABOUT robots rather than FOR one.
#   2. IMPERATIVE: a build verb anywhere before a robot noun ("build a quadruped robot",
#      "can you make me an arm?"). A build even when phrased as a polite question, because the
#      verb states the intent.
#   3. NOUN PHRASE: no verb needed -- the message NAMES a robot or a robot morphology
#      ("a quadruped robot that walks", "six-legged walker, 3 kg") and is not itself a question.
#      A trailing "?" disqualifies this layer, which is what keeps "a quadruped?" out.
_ROBOT_NOUN = (
    r"robots?|arms?|manipulators?|grippers?|end[ -]effectors?|mobile bases?|rovers?|drones?|"
    r"quadcopters?|bots?|machines?|quadrupeds?|bipeds?|hexapods?|octopods?|humanoids?|"
    r"walkers?|crawlers?|cobots?|exoskeletons?|gantr(?:y|ies)"
)
_BUILD_VERB = r"build|create|make|design|generate|assemble|spawn|construct|prototype"

# Layer 1 -- opens by ASKING. Interrogatives + meta verbs + greetings, never a dispatchable build.
_ASKING = re.compile(
    r"^\W*(?:what|why|how|when|where|which|who|whose|is|are|was|were|do|does|did|am|has|have|had|"
    r"explain|tell|show|list|compare|describe|define|summari[sz]e|help|hi|hello|hey|thanks|thank)\b",
    re.IGNORECASE,
)
# Layer 2 -- an imperative build verb standing before a robot noun.
_BUILD_IMPERATIVE = re.compile(rf"\b(?:{_BUILD_VERB})\b.*?\b(?:{_ROBOT_NOUN})\b", re.IGNORECASE | re.DOTALL)
# Layer 3 -- the message names a robot at all.
_ROBOT_MENTION = re.compile(rf"\b(?:{_ROBOT_NOUN})\b", re.IGNORECASE)

# Kept as the documented name for the imperative form (external callers/tests import it).
_BUILD_INTENT = _BUILD_IMPERATIVE


def _fallback_build_action(text: str) -> dict | None:
    """Heuristic build detection used when no local model is available. See the rule above."""
    body = (text or "").strip()
    if not body:
        return None
    if _ASKING.match(body):                                  # 1. asking ABOUT robots -> chat
        return None
    if _BUILD_IMPERATIVE.search(body):                       # 2. "build/make/design ... a robot"
        return {"action": "build", "prompt": body, "train": False}
    if body.endswith("?"):                                   # a question with no build verb -> chat
        return None
    if _ROBOT_MENTION.search(body):                          # 3. a bare noun phrase naming a robot
        return {"action": "build", "prompt": body, "train": False}
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

    # AI-native design-edit route (docs/ai_native_plan.md P3): when the client has a robot/scene OPEN
    # (sends robot_id/scene_id), the assistant makes an INCREMENTAL localized edit instead of a full rebuild
    # ("make it taller" lengthens legs on the held gene). Additive: absent those ids, the build flow is unchanged.
    robot_id, scene_id = payload.get("robot_id"), payload.get("scene_id")
    if robot_id or scene_id:
        try:
            from virturoid.services.assistant_core import handle_turn
            turn = handle_turn(user_text, robot_id=robot_id, scene_id=scene_id)
            return {"role": "assistant", "content": turn["reply"], "action": None, "turn": turn}
        except Exception as exc:  # noqa: BLE001 - never crash the chat; fall through to the build assistant
            return {"role": "assistant", "content": f"Could not edit: {type(exc).__name__}: {exc}", "action": None}

    status = assistant_status()
    selected_model = _selected_assistant_model(payload, status)
    selected_available = _assistant_model_available(selected_model, status)
    model_messages = [{"role": "system", "content": _ASSISTANT_SYSTEM_PROMPT}, *(
        {"role": m.get("role", "user"), "content": str(m.get("content") or "")}
        for m in messages
        if m.get("role") in {"user", "assistant"}
    )]

    raw_reply = _assistant_complete(model_messages, selected_model) if status.get("online") and selected_available else None
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
        if payload.get("no_build"):
            # Additive, opt-in (the new Studio UI sends no_build=true): return the
            # detected intent instead of blocking this request on a multi-minute
            # build -- the client dispatches it as a cancellable /api/jobs job.
            return {
                "role": "assistant",
                "content": "That's a build — dispatching it as a live job.",
                "action": "build_intent",
                "build_intent": build_payload,
                "model_used": used_model,
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
    # Name the variables that ACTUALLY control this chat assistant. VIRTUROID_LLM_BACKEND /
    # OPENAI_API_KEY (the README's Configuration table) drive the DESIGN pipeline, not this layer --
    # a user who set those exactly as documented was still told to install Ollama, with no way to
    # discover why. Say which knob is which, and say that builds work without any of them.
    hint = (
        f"No chat model is reachable, so I can only answer with the build pipeline itself "
        f"(describe a robot and I will build it -- that path needs no model at all).\n\n"
        f"This chat assistant is configured by VIRTUROID_ASSISTANT_PROVIDER "
        f"(currently {ASSISTANT_PROVIDER!r}), VIRTUROID_ASSISTANT_MODEL (currently {selected_model!r}) "
        f"and, for the ollama provider, OLLAMA_HOST (currently {OLLAMA_HOST}). Start that runtime and "
        f"pull the model to enable free-form chat.\n\n"
        f"Note these are SEPARATE from VIRTUROID_LLM_BACKEND / OPENAI_API_KEY, which choose the "
        f"language model that authors robot anatomy in the build pipeline."
    )
    return {"role": "assistant", "content": hint, "action": "chat", "model_used": False}


def _summarize_build(build: dict, used_model: bool) -> str:
    result = build.get("result", {})
    package_prefix = build.get("package_url_prefix")
    if not package_prefix:
        detail = "; ".join(result.get("notes") or []) or "The LLM plan did not clear grounding."
        return f"No package was generated. {detail}"
    name = str(package_prefix).split("/")[-1]
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
        if parsed.path == "/studio" or parsed.path.startswith("/studio/"):
            self._send_studio_file(parsed.path)
            return
        if parsed.path == "/api/jobs":
            from virturoid.services import job_registry
            self._send_json({"jobs": job_registry.list_jobs()})
            return
        if parsed.path.startswith("/api/jobs/"):
            self._send_job_events(parsed)
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
        if parsed.path == "/api/moat":                         # the verified-morphology memory, gates and all
            self._send_json(self._moat(parsed))
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
        if parsed.path == "/sessions":                         # C1-C3: the 'watch the agent build' viewer page
            self._send_text(build_agent_sessions_html(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/sessions":                     # G-B: the app is the VIEWER of external-agent work
            from virturoid.services import session_state
            self._send_json({"robots": session_state.list_robots()})
            return
        if parsed.path.startswith("/api/sessions/"):
            self._send_session_detail(parsed.path[len("/api/sessions/"):].strip("/"))
            return
        if parsed.path.startswith("/api/agent_render/"):
            self._send_agent_render(parsed.path[len("/api/agent_render/"):].strip("/"))
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
        if parsed.path == "/api/jobs":
            def _create_job(payload: dict) -> dict:
                from virturoid.services import job_registry
                return {"job": job_registry.create(str(payload.get("kind", "")), payload.get("args") or {}, self.root)}
            self._handle_json_post(_create_job)
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            from virturoid.services import job_registry
            job_id = parsed.path[len("/api/jobs/"):-len("/cancel")].strip("/")
            self._send_json({"ok": job_registry.request_cancel(job_id)})
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
        # PARSE first, and report a bad REQUEST as 400 — not 500. A malformed body, a non-numeric
        # Content-Length, or a JSON scalar/array where an object is required is the CALLER's error; returning
        # 500 told an integrator our server had crashed and made their own bugs undebuggable (red-team finding).
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("negative Content-Length")
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            payload = json.loads(raw) if raw.strip() else {}
            if not isinstance(payload, dict):
                raise TypeError(f"body must be a JSON object, got {type(payload).__name__}")
        except Exception as exc:  # noqa: BLE001 - malformed input from the client
            self._send_json({"error": f"bad request: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            result = handler(payload)
        except Exception as exc:  # noqa: BLE001 - a genuine server-side failure
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

    def _moat(self, parsed) -> dict:
        """GET /api/moat[?package=<id>] -> what the verified-morphology memory holds and what recall did.

        The moat was the one part of the product with no surface: `/api/flywheel` reads a JSON file a demo
        script writes, and `/api/design_brain` reports coverage and edge COUNTS. Neither could answer "how many
        banked rows carry a measured error bar" (1 of 101) or "did recalling memory make the robot walk further"
        (on the dominant kind: no -- -0.0336 m over 2163 deploys). Both are read straight from the bank here.

        Memory-dir resolution matches `_design_brain`: the build root's own `memory/` first, then the root, then
        its sibling -- so a demo build set and the developer's workspace each report their own bank.
        """
        from urllib.parse import parse_qs

        from virturoid.services.moat_panel import moat_panel
        name = (parse_qs(parsed.query).get("package") or [""])[0]
        pkg = None
        if name:
            candidate = self.root / _safe_output_name(name)
            pkg = candidate if candidate.exists() else None
        for cand in (self.root / "memory", self.root, self.root.parent / "memory"):
            try:
                out = moat_panel(cand, package_dir=pkg)
            except Exception as exc:  # noqa: BLE001 - a status panel must never take the server down
                return {"error": str(exc), "db_present": False}
            if out.get("db_present"):
                return out
        return moat_panel(self.root / "memory", package_dir=pkg)   # honest empty + the path it looked in

    def _design_brain(self) -> dict:
        """The Design Brain panel (the moat MEASURED): MAP-Elites coverage/QD-score + provenance compounding
        from the build set's shared memory. Best-effort over candidate memory dirs; zeros + a hint until real
        builds have populated the archive/provenance (autonomous_build's keystone writes them)."""
        try:
            from virturoid.services.design_brain import design_brain_summary
            from virturoid.services.flywheel_status import moat_status
        except Exception as exc:  # noqa: BLE001
            return {"archive_coverage": 0, "provenance_edges": 0, "headline": f"unavailable: {exc}"}

        def _with_brain(cand):
            s = design_brain_summary(cand)
            try:                                             # the P1-P3 brain layers (transfer ledger, gated metric,
                s["brain"] = moat_status(cand).get("brain")  # episodes, per-kind compounding) — traceable live numbers
            except Exception:  # noqa: BLE001
                s["brain"] = None
            return s

        for cand in (self.root / "memory", self.root, self.root.parent / "memory"):
            s = _with_brain(cand)
            if s.get("archive_coverage") or s.get("provenance_edges") or (s.get("brain") or {}).get("episodes"):
                return s
        return _with_brain(self.root / "memory")   # zeros + headline

    def _send_session_detail(self, robot_id: str) -> None:
        """GET /api/sessions/<id> -> the held robot's summary + a fresh render URL, so the webapp can
        live-follow whatever the connected agent is currently building/editing (G-B viewer)."""
        from virturoid.services import session_state
        from virturoid.services.ai_native_tools import _render_gene
        gene = session_state.get_robot(robot_id)
        if gene is None:
            self._send_json({"error": "Unknown robot_id."}, status=HTTPStatus.NOT_FOUND)
            return
        from virturoid.services.agent_tools import call_tool
        summary = call_tool("get_robot", {"robot_id": robot_id}).get("result", {})
        meta = session_state.robot_meta(robot_id)
        png = _render_gene(gene, f"session_{robot_id}")        # build/agent_renders/session_<id>.png
        render_url = f"/api/agent_render/{Path(png).name}" if png else None
        self._send_json({"robot_id": robot_id, "summary": summary, "meta": meta, "render_url": render_url})

    def _send_agent_render(self, name: str) -> None:
        from virturoid.services.ai_native_tools import _RENDER_DIR
        target = (_RENDER_DIR / _safe_output_name(name)).resolve()
        root = _RENDER_DIR.resolve()
        if root not in target.parents or not target.exists() or target.is_dir():
            self._send_json({"error": "Render not found."}, status=HTTPStatus.NOT_FOUND)
            return
        self._send_bytes(target.read_bytes(), _content_type(target))

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

    def _send_job_events(self, parsed) -> None:
        """GET /api/jobs/<id>?since=N -> {job, events} (1 Hz polling transport; the
        event shape is SSE-ready so a stream endpoint later is a transport swap)."""
        from urllib.parse import parse_qs

        from virturoid.services import job_registry

        job_id = parsed.path[len("/api/jobs/"):].strip("/")
        try:
            since = int((parse_qs(parsed.query).get("since") or ["0"])[0])
        except ValueError:
            since = 0
        found = job_registry.events_since(job_id, since)
        if found is None:
            self._send_json({"error": "Unknown job."}, status=HTTPStatus.NOT_FOUND)
            return
        job, events = found
        self._send_json({"job": job, "events": events})

    def _send_studio_file(self, path: str) -> None:
        """Serve the built Virturoid Studio SPA from frontend/dist under /studio.
        Hashed /studio/assets/* get immutable caching; everything else falls back
        to index.html (SPA routing) served no-cache so new builds show instantly."""
        if not FRONTEND_DIST.exists():
            self._send_json(
                {"error": "Virturoid Studio is not built. Run: cd frontend && npm install && npm run build"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        relative = unquote(path[len("/studio"):]).lstrip("/")
        parts = [part for part in relative.split("/") if part and part != ".."]
        target = (FRONTEND_DIST / Path(*parts)).resolve() if parts else (FRONTEND_DIST / "index.html").resolve()
        root = FRONTEND_DIST.resolve()
        if root not in [target, *target.parents] or not target.exists() or target.is_dir():
            target = (FRONTEND_DIST / "index.html").resolve()  # SPA fallback
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", _studio_content_type(target))
        self.send_header("Content-Length", str(len(data)))
        if "/assets/" in str(target).replace("\\", "/"):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

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
                if not has_robot_package(child):  # list packages that can render (URDF) OR replay (scenes)
                    continue
                scene_count = 0
                index_path = child / "simulation" / "mujoco" / "compiled_scene_index.json"
                if index_path.exists():
                    try:
                        scene_count = len(json.loads(index_path.read_text(encoding="utf-8")).get("scenes", []))
                    except (json.JSONDecodeError, OSError):
                        scene_count = 0
                # ONE status derivation for every surface (header chip, library card, inspector,
                # status bar, Verify tab). `valid` stays on the wire as the raw contract fact for
                # older clients, but no surface renders it as a generic verdict any more --
                # see services/package_status.py for why they used to contradict each other.
                status = package_status(child)
                valid: bool | None = status["contract_ok"]
                robot_class: str | None = None
                species: str | None = None
                contract_path = child / "reports" / "robot_package_contract.json"
                if contract_path.exists():
                    try:
                        contract = json.loads(contract_path.read_text(encoding="utf-8"))
                        robot_class = contract.get("robot_class")
                        species = contract.get("species")
                    except (json.JSONDecodeError, OSError):
                        pass
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
                        "status": status,
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


def _studio_content_type(path: Path) -> str:
    """MIME types for the Studio bundle (mimetypes + overrides the stdlib misses)."""
    import mimetypes

    overrides = {
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".svg": "image/svg+xml",
        ".wasm": "application/wasm",
        ".urdf": "application/xml; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    if path.suffix in overrides:
        return overrides[path.suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


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
    if path.suffix == ".png":
        return "image/png"
    if path.suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if path.suffix == ".webp":
        return "image/webp"
    if path.suffix == ".gif":
        return "image/gif"
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    if path.suffix == ".stl":
        return "application/octet-stream"
    return "text/plain; charset=utf-8"


if __name__ == "__main__":
    main()
