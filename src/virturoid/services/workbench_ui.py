from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

WORKBENCH_UI_URI = "reports/workbench.html"


def write_workbench_ui(
    package_dir: Path,
    *,
    summary: dict[str, Any],
    artifacts: dict[str, str],
    training: dict | None = None,
) -> Path:
    package_dir = Path(package_dir)
    html = build_workbench_ui(package_dir, summary=summary, artifacts=artifacts, training=training)
    path = package_dir / WORKBENCH_UI_URI
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def build_workbench_ui(
    package_dir: Path,
    *,
    summary: dict[str, Any],
    artifacts: dict[str, str],
    training: dict | None = None,
) -> str:
    package_dir = Path(package_dir)
    robot = _read_json(package_dir / "robot" / "robot_genome.json")
    scene_set = _read_json(package_dir / "simulation" / "scene_set.json")
    compiled_index = _read_json(package_dir / "simulation" / "mujoco" / "compiled_scene_index.json")
    package_contract = _read_json(package_dir / "reports" / "robot_package_contract.json")
    readiness_report = _read_json(package_dir / "reports" / "mvp_readiness_report.json") or {}
    package_validation = _read_json(package_dir / "reports" / "package_validation_report.json")
    policy_metrics = _read_json(package_dir / "training" / "policy_training_metrics.json")

    contract_checks = package_contract.get("artifact_checks", [])
    failing_checks = [item for item in contract_checks if not item.get("exists") or item.get("parse_status") == "fail"]
    scenes = scene_set.get("scenes", [])
    first_scene = scenes[0] if scenes else {}
    prompt = summary.get("prompt", "")
    training_state = training or summary.get("training")
    training_ran = bool(training_state and training_state.get("ran"))

    artifact_rows = "\n".join(
        f"""
        <tr>
          <td>{escape(key)}</td>
          <td><a href="../{escape(uri)}">{escape(uri)}</a></td>
          <td>{_status_pill("available" if (package_dir / uri).exists() else "missing")}</td>
        </tr>
        """
        for key, uri in artifacts.items()
    )
    contract_rows = "\n".join(
        f"""
        <tr>
          <td>{escape(item.get("key", ""))}</td>
          <td>{escape(item.get("uri", ""))}</td>
          <td>{_status_pill("pass" if item.get("exists") else "missing")}</td>
          <td>{_status_pill(item.get("parse_status", "not_checked"))}</td>
          <td>{escape(item.get("detail", ""))}</td>
        </tr>
        """
        for item in contract_checks
    )
    scene_rows = "\n".join(
        f"""
        <tr>
          <td>{escape(scene.get("id", ""))}</td>
          <td>{len(scene.get("objects", []))}</td>
          <td>{escape(str(scene.get("variation_parameters", {}).get("seed", "")))}</td>
          <td>{escape(", ".join(scene.get("requirement_trace", [])[:4]))}</td>
        </tr>
        """
        for scene in scenes
    )
    compiled_rows = "\n".join(
        f"""
        <tr>
          <td>{escape(item.get("purpose", ""))}</td>
          <td>{escape(item.get("scene_id", ""))}</td>
          <td><a href="../{escape(item.get("mujoco_xml", ""))}">{escape(item.get("mujoco_xml", ""))}</a></td>
          <td>{item.get("object_count", 0)}</td>
        </tr>
        """
        for item in compiled_index.get("scenes", [])
    )
    validation_rows = _validation_rows(package_validation)
    readiness_rows = _readiness_rows(readiness_report)
    capabilities = "".join(f"<li>{escape(item)}</li>" for item in package_contract.get("capabilities", []))
    robot_links = "".join(f"<li>{escape(link)}</li>" for link in robot.get("links", []))
    robot_joints = "".join(
        f"<li><span>{escape(joint.get('name', ''))}</span><small>{escape(joint.get('joint_type', ''))}</small></li>"
        for joint in robot.get("joints", [])
    )
    scene_map = _scene_map_svg(first_scene)
    training_panel = _training_panel(training_state, policy_metrics)

    payload = {
        "robotClass": summary.get("selected_robot_class"),
        "packageType": summary.get("package_type"),
        "packageValid": summary.get("package_valid"),
        "sceneCount": len(scenes),
        "compiledSceneCount": compiled_index.get("scene_count", 0),
        "failingContractChecks": len(failing_checks),
        "readinessScore": readiness_report.get("score"),
    }
    payload_json = json.dumps(payload, sort_keys=True)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Virturoid Build Workbench</title>
  <style>
    :root {{
      --bg: #eef2ef;
      --surface: #fbfcf7;
      --surface-2: #f3f6f0;
      --surface-3: #e6ece5;
      --ink: #181b17;
      --muted: #657069;
      --line: #c4ccc5;
      --rail: #1d201f;
      --primary: #b15a16;
      --primary-2: #007c78;
      --accent: #2e63b7;
      --good: #2f7d32;
      --warn: #b15a16;
      --bad: #b8332c;
      --radius: 8px;
      --shadow: 0 16px 40px rgba(23, 30, 25, 0.14);
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        linear-gradient(90deg, rgba(27, 33, 30, 0.055) 1px, transparent 1px),
        linear-gradient(rgba(27, 33, 30, 0.055) 1px, transparent 1px),
        radial-gradient(900px 460px at 70% -14%, rgba(0, 124, 120, 0.11), transparent 58%),
        var(--bg);
      background-size: 26px 26px, 26px 26px, auto, auto;
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
      font-size: 16px;
      line-height: 1.5;
    }}
    a {{ color: var(--primary-2); text-underline-offset: 3px; }}
    .shell {{ min-height: 100dvh; display: grid; grid-template-columns: 280px minmax(0, 1fr); }}
    aside {{
      background: var(--rail);
      color: #f7fbf5;
      padding: 24px;
      border-right: 4px solid var(--primary);
      position: sticky;
      top: 0;
      height: 100dvh;
      overflow: auto;
    }}
    main {{ padding: 24px; }}
    .brand {{ display: grid; gap: 4px; margin-bottom: 28px; }}
    .brand strong {{ font-size: 22px; letter-spacing: 0; }}
    .brand span {{ color: #cbd6d0; font-size: 13px; }}
    .nav {{ display: grid; gap: 8px; }}
    .nav button {{
      width: 100%;
      min-height: 44px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(255,255,255,0.055);
      color: #f7fbf5;
      border-radius: var(--radius);
      text-align: left;
      padding: 10px 12px;
      cursor: pointer;
      font: inherit;
    }}
    .nav button[aria-selected="true"] {{ background: #f7fbf5; color: var(--rail); border-color: #f7fbf5; }}
    .nav button:focus-visible, a:focus-visible {{ outline: 3px solid #f3b23d; outline-offset: 2px; }}
    .side-block {{ margin-top: 26px; padding-top: 18px; border-top: 1px solid rgba(255,255,255,0.16); }}
    .side-block p {{ color: #cbd6d0; margin: 0; font-size: 14px; }}
    .topline {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 48px); line-height: 1.05; letter-spacing: 0; max-width: 900px; }}
    .prompt {{ color: var(--muted); max-width: 900px; margin: 10px 0 0; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .button {{
      min-height: 44px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 10px 14px;
      color: var(--ink);
      background: var(--surface);
      text-decoration: none;
      font-weight: 700;
      cursor: pointer;
    }}
    .button.primary {{ background: var(--primary); color: white; border-color: #7d3d0b; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }}
    .metric, .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }}
    .metric {{ padding: 16px; min-height: 112px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; text-transform: uppercase; font-weight: 800; }}
    .metric strong {{ display: block; font-size: 26px; margin-top: 8px; }}
    .metric small {{ color: var(--muted); }}
    section[role="tabpanel"] {{ display: none; }}
    section[role="tabpanel"].active {{ display: block; }}
    .grid {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr); gap: 16px; }}
    .panel {{ padding: 18px; margin-bottom: 16px; }}
    .panel h2, .panel h3 {{ margin: 0 0 12px; }}
    .split-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    ul.clean {{ margin: 0; padding: 0; list-style: none; display: grid; gap: 8px; }}
    ul.clean li {{ display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--surface-2); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--surface-2); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0; }}
    .status {{ display: inline-flex; align-items: center; min-height: 28px; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 800; border: 1px solid currentColor; }}
    .status.pass, .status.available, .status.valid {{ color: var(--good); background: #dcebdc; }}
    .status.warn, .status.skip, .status.not_checked {{ color: var(--warn); background: #f7e2cf; }}
    .status.fail, .status.missing, .status.invalid {{ color: var(--bad); background: #f2d7d5; }}
    .scene-map {{ width: 100%; aspect-ratio: 16 / 10; border-radius: var(--radius); border: 1px solid var(--line); background: var(--surface-3); overflow: hidden; }}
    .codebox {{ background: #101211; color: #f7fbf5; border-radius: var(--radius); padding: 14px; overflow: auto; font-family: "Cascadia Mono", Consolas, monospace; font-size: 13px; }}
    .filter {{ min-height: 44px; width: min(420px, 100%); border: 1px solid var(--line); border-radius: var(--radius); padding: 10px 12px; font: inherit; }}
    @media (max-width: 900px) {{
      .shell {{ grid-template-columns: 1fr; }}
      aside {{ position: static; height: auto; }}
      .topline {{ display: grid; }}
      .actions {{ justify-content: flex-start; }}
      .metrics, .grid, .split-list {{ grid-template-columns: 1fr; }}
    }}
    @media (prefers-reduced-motion: no-preference) {{
      .panel, .metric {{ transition: transform 180ms ease, box-shadow 180ms ease; }}
      .panel:hover, .metric:hover {{ transform: translateY(-1px); }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <strong>Virturoid</strong>
        <span>Build Workbench</span>
      </div>
      <nav class="nav" aria-label="Workbench sections">
        <button type="button" aria-selected="true" data-tab="overview">Overview</button>
        <button type="button" aria-selected="false" data-tab="robot">Robot</button>
        <button type="button" aria-selected="false" data-tab="simulation">Simulation</button>
        <button type="button" aria-selected="false" data-tab="training">Training</button>
        <button type="button" aria-selected="false" data-tab="artifacts">Artifacts</button>
      </nav>
      <div class="side-block">
        <p>{escape(summary.get("package_type", ""))}</p>
      </div>
    </aside>
    <main>
      <div class="topline">
        <div>
          <h1>{escape(summary.get("selected_species", "Generated robot package"))}</h1>
          <p class="prompt">{escape(prompt)}</p>
        </div>
        <div class="actions">
          <a class="button primary" href="robot_package_contract.json">Open Contract</a>
          <a class="button" href="mvp_readiness_report.json">MVP Readiness</a>
          <a class="button" href="autonomous_build_summary.json">Build Summary</a>
        </div>
      </div>
      <div class="metrics" aria-label="Package metrics">
        <div class="metric"><span>Package</span><strong>{_plain_status(summary.get("package_valid"))}</strong><small>{escape(summary.get("selected_robot_class", ""))}</small></div>
        <div class="metric"><span>Scenes</span><strong>{len(scenes)}</strong><small>{compiled_index.get("scene_count", 0)} compiled</small></div>
        <div class="metric"><span>Artifacts</span><strong>{len(artifacts)}</strong><small>{len(failing_checks)} failing checks</small></div>
        <div class="metric"><span>Readiness</span><strong>{readiness_report.get("score", "NA")}</strong><small>{_readiness_label(readiness_report)}</small></div>
      </div>

      <section id="overview" role="tabpanel" class="active">
        <div class="grid">
          <div class="panel">
            <h2>Package Contract</h2>
            <table>
              <thead><tr><th>Capability</th><th>State</th></tr></thead>
              <tbody>{''.join(f"<tr><td>{escape(item)}</td><td>{_status_pill('available')}</td></tr>" for item in package_contract.get("capabilities", []))}</tbody>
            </table>
          </div>
          <div class="panel">
            <h2>Scene Preview</h2>
            <div class="scene-map" aria-label="First generated scene map">{scene_map}</div>
          </div>
        </div>
        <div class="panel">
          <h2>MVP Readiness</h2>
          <table>
            <thead><tr><th>Gate</th><th>Status</th><th>Required</th><th>Evidence</th><th>Detail</th></tr></thead>
            <tbody>{readiness_rows}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Contract Checks</h2>
          <table>
            <thead><tr><th>Artifact</th><th>URI</th><th>Exists</th><th>Parse</th><th>Detail</th></tr></thead>
            <tbody>{contract_rows}</tbody>
          </table>
        </div>
      </section>

      <section id="robot" role="tabpanel">
        <div class="grid">
          <div class="panel">
            <h2>Robot Genome</h2>
            <div class="split-list">
              <div><h3>Links</h3><ul class="clean">{robot_links}</ul></div>
              <div><h3>Joints</h3><ul class="clean">{robot_joints}</ul></div>
            </div>
          </div>
          <div class="panel">
            <h2>Model Files</h2>
            <ul class="clean">
              <li><span>Genome</span><a href="../robot/robot_genome.json">robot_genome.json</a></li>
              <li><span>URDF</span><a href="../robot/robot.urdf">robot.urdf</a></li>
              <li><span>Primary MJCF</span><a href="../{escape(artifacts.get("mujoco_xml", "simulation/mujoco/mvp_scene.xml"))}">open</a></li>
            </ul>
          </div>
        </div>
      </section>

      <section id="simulation" role="tabpanel">
        <div class="panel">
          <h2>Generated Scenes</h2>
          <table>
            <thead><tr><th>Scene</th><th>Objects</th><th>Seed</th><th>Trace</th></tr></thead>
            <tbody>{scene_rows}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Compiled Simulator Scenes</h2>
          <table>
            <thead><tr><th>Purpose</th><th>Scene</th><th>MJCF</th><th>Objects</th></tr></thead>
            <tbody>{compiled_rows}</tbody>
          </table>
        </div>
      </section>

      <section id="training" role="tabpanel">
        {training_panel}
      </section>

      <section id="artifacts" role="tabpanel">
        <div class="panel">
          <h2>Artifact Browser</h2>
          <input class="filter" type="search" id="artifactFilter" placeholder="Filter artifacts" aria-label="Filter artifacts" />
          <table id="artifactTable">
            <thead><tr><th>Key</th><th>Artifact</th><th>Status</th></tr></thead>
            <tbody>{artifact_rows}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Package Validation</h2>
          <table>
            <thead><tr><th>Check</th><th>Status</th><th>Message</th></tr></thead>
            <tbody>{validation_rows}</tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
  <script>
    const packageState = {payload_json};
    document.querySelectorAll("[data-tab]").forEach((button) => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll("[data-tab]").forEach((item) => item.setAttribute("aria-selected", "false"));
        document.querySelectorAll("[role='tabpanel']").forEach((panel) => panel.classList.remove("active"));
        button.setAttribute("aria-selected", "true");
        document.getElementById(button.dataset.tab).classList.add("active");
      }});
    }});
    const filter = document.getElementById("artifactFilter");
    if (filter) {{
      filter.addEventListener("input", () => {{
        const query = filter.value.toLowerCase();
        document.querySelectorAll("#artifactTable tbody tr").forEach((row) => {{
          row.style.display = row.textContent.toLowerCase().includes(query) ? "" : "none";
        }});
      }});
    }}
  </script>
</body>
</html>
"""


def _validation_rows(report: dict | None) -> str:
    if not report:
        return '<tr><td>Route package validation</td><td><span class="status not_checked">not checked</span></td><td>No route-specific validation report.</td></tr>'
    # `checks` is a list on the arm route but a name->result MAPPING on the gene route -- normalize both.
    # (Slicing a dict here used to throw, and the caller's silent try/except dropped the WHOLE workbench
    # report, so gene-engine builds -- mobile bases, quads -- had no reports/workbench.html.)
    checks = report.get("checks", [])
    if isinstance(checks, dict):
        norm = []
        for k, v in checks.items():
            if isinstance(v, dict):
                norm.append({"name": v.get("name", k), "status": v.get("status", ""), "message": v.get("message", "")})
            elif isinstance(v, str):
                norm.append({"name": k, "status": v, "message": ""})
            else:
                norm.append({"name": k, "status": "pass" if v else "fail", "message": ""})
        checks = norm
    if not isinstance(checks, list):
        checks = []
    rows = [
        f"<tr><td>{escape(str(item.get('name', '')))}</td><td>{_status_pill(str(item.get('status', '')))}</td><td>{escape(str(item.get('message', '')))}</td></tr>"
        for item in checks[:40] if isinstance(item, dict)
    ]
    return "\n".join(rows) or '<tr><td>Route package validation</td><td><span class="status not_checked">not checked</span></td><td>No checks reported.</td></tr>'


def _readiness_rows(report: dict | None) -> str:
    if not report:
        return '<tr><td>MVP readiness</td><td><span class="status not_checked">not checked</span></td><td>true</td><td></td><td>No readiness report found.</td></tr>'
    return "\n".join(
        f"""
        <tr>
          <td>{escape(item.get("label", ""))}</td>
          <td>{_status_pill(item.get("status", ""))}</td>
          <td>{escape(str(item.get("required", "")))}</td>
          <td>{_artifact_link(item.get("evidence_uri"))}</td>
          <td>{escape(item.get("detail", ""))}</td>
        </tr>
        """
        for item in report.get("gates", [])
    )


def _readiness_label(report: dict | None) -> str:
    if not report:
        return "not checked"
    return "ready" if report.get("ready") else "needs work"


def _training_panel(training: dict | None, metrics: dict | None) -> str:
    if training and training.get("ran"):
        rows = [
            ("Policy", training.get("policy_id", "")),
            ("Method", training.get("method", "")),
            ("Initial reward", training.get("initial_reward", "")),
            ("Best reward", training.get("best_reward", "")),
            ("Mean reach distance", training.get("eval_mean_reach_distance_m", "")),
            ("Success rate", training.get("eval_success_rate", "")),
        ]
        metric_rows = "".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in rows)
        return f"""
        <div class="panel">
          <h2>Learned Controller</h2>
          <table><tbody>{metric_rows}</tbody></table>
          <p><a class="button primary" href="../software/controller/controller_bundle.json">Controller Bundle</a>
          <a class="button" href="../software/controller/controller.py">Controller Entrypoint</a></p>
        </div>
        """
    if training and not training.get("ran"):
        return f"""
        <div class="panel">
          <h2>Training Status</h2>
          <p>{escape(training.get("reason", "Training did not run."))}</p>
        </div>
        """
    if metrics:
        return f"""
        <div class="panel">
          <h2>Training Metrics</h2>
          <pre class="codebox">{escape(json.dumps(metrics, indent=2))}</pre>
        </div>
        """
    return """
      <div class="panel">
        <h2>Training Status</h2>
        <p>This package has build and simulation artifacts. Run the generic command with <code>--train</code> to add a learned controller bundle where supported.</p>
      </div>
    """


def _scene_map_svg(scene: dict) -> str:
    objects = scene.get("objects", [])
    shapes = []
    for item in objects:
        x, y = _project(item.get("pose_xyz_rpy", [0, 0, 0, 0, 0, 0]))
        name = item.get("name", "")
        object_type = item.get("object_type", "")
        if object_type in {"container", "zone"}:
            shapes.append(f'<circle cx="{x}" cy="{y}" r="18" fill="#dfeee7" stroke="#0f6b3d" stroke-width="2"><title>{escape(name)}</title></circle>')
        elif object_type in {"obstacle", "conveyor"}:
            shapes.append(f'<rect x="{x - 14}" y="{y - 14}" width="28" height="28" fill="#d8d2c5" stroke="#8a5700" stroke-width="2"><title>{escape(name)}</title></rect>')
        else:
            fill = "#c94747" if "red" in name else "#4169a8" if "blue" in name else "#455a64"
            shapes.append(f'<rect x="{x - 9}" y="{y - 9}" width="18" height="18" fill="{fill}"><title>{escape(name)}</title></rect>')
    return f"""<svg viewBox="0 0 640 400" role="img" aria-label="Top-down scene sketch">
      <rect width="640" height="400" fill="#e8e5dc" />
      <path d="M40 200 H600" stroke="#aab5b9" stroke-width="2" stroke-dasharray="8 8" />
      <path d="M320 40 V360" stroke="#aab5b9" stroke-width="2" stroke-dasharray="8 8" />
      {''.join(shapes)}
    </svg>"""


def _project(pose: list) -> tuple[int, int]:
    x = float(pose[0]) if len(pose) > 0 else 0.0
    y = float(pose[1]) if len(pose) > 1 else 0.0
    return int(110 + x * 260), int(200 - y * 380)


def _status_pill(status: str) -> str:
    label = str(status or "unknown")
    css = label.lower().replace("_", "-")
    if label in {"true", "ok"}:
        css = "pass"
    return f'<span class="status {escape(css)}">{escape(label)}</span>'


def _artifact_link(uri: str | None) -> str:
    if not uri:
        return ""
    return f'<a href="../{escape(uri)}">{escape(uri)}</a>'


def _plain_status(value: Any) -> str:
    if value is True:
        return "Valid"
    if value is False:
        return "Invalid"
    return "Unknown"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
