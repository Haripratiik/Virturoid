import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const fmtPct = (x) => (x == null ? "-" : Math.round(Number(x) * 100) + "%");
const normPath = (path) => String(path || "").replace(/\\/g, "/");
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
));

const EXAMPLES = [
  "a tabletop arm that sorts red and blue blocks into matching bins",
  "a humanoid that picks up boxes and places them on a shelf",
  "a mobile robot that carries parts across the room",
];

const ACTIONS = [
  {
    label: "Evaluate", sub: "physics run", msg: "evaluate it",
    icon: '<path d="m9 11 3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
  },
  {
    label: "Improve", sub: "find fix", msg: "it keeps missing some grasps, make it better",
    icon: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
  },
  {
    label: "Perceive", sub: "camera loop", msg: "run it with the camera",
    icon: '<path d="M23 7l-7 5 7 5V7Z"/><rect x="1" y="5" width="15" height="14" rx="2"/>',
  },
  {
    label: "Train", sub: "policy bundle", msg: "train a policy",
    icon: '<circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.2 4.2l2.8 2.8M17 17l2.8 2.8M1 12h4M19 12h4M4.2 19.8 7 17M17 7l2.8-2.8"/>',
  },
  {
    label: "Export", sub: "package", msg: "export the package",
    icon: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
  },
];

let built = false;
let lastRobot = null;
let jobActive = false;
let scene, camera, renderer, controls, meshGroup;
let frames = [];
let frameIndex = 0;
let playing = false;
let lastTick = 0;
let viewerReady = false;
let progressBlock = null;
let memoryReturnFocus = null;

function addMsg(text, who, isErr = false) {
  const log = $("activity");
  const empty = log.querySelector(".act-empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = `act act-${who}` + (isErr ? " err" : "");
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function renderProgress(events) {
  const log = $("activity");
  if (!progressBlock) {
    progressBlock = document.createElement("div");
    progressBlock.className = "progress-block";
    log.appendChild(progressBlock);
  }
  progressBlock.innerHTML = "";
  events.slice(-7).forEach((event, index, arr) => {
    const row = document.createElement("div");
    row.className = "act-step" + (index === arr.length - 1 ? " live" : "");
    row.innerHTML = `<span class="pdot"></span><span>${esc(event.message || event.stage || "")}</span>`;
    progressBlock.appendChild(row);
  });
  log.scrollTop = log.scrollHeight;
  updateStepper(events);
}

function clearProgress() {
  if (progressBlock) progressBlock.remove();
  progressBlock = null;
}

const PHASE_KEYS = {
  design: ["species", "design", "reward", "translate", "propose"],
  build: ["build", "compil", "assemble"],
  validate: ["valid", "structural", "buildab", "feasib"],
  evaluate: ["evaluat", "redesign", "co-design", "codesign", "tune", "run", "perceiv", "train", "done"],
};

function updateStepper(events) {
  const phases = ["design", "build", "validate", "evaluate"];
  const reached = new Set();
  let isDone = false;
  for (const event of events) {
    const text = `${event.stage || ""} ${event.message || ""}`.toLowerCase();
    if (event.stage === "done") isDone = true;
    for (const phase of phases) {
      if (PHASE_KEYS[phase].some((key) => text.includes(key))) reached.add(phase);
    }
  }
  const lastReached = phases.filter((phase) => reached.has(phase)).pop();
  for (const phase of phases) {
    const item = $("stepper").querySelector(`[data-phase="${phase}"]`);
    item.classList.remove("active", "done");
    if (!reached.has(phase)) continue;
    if (isDone || phase !== lastReached) item.classList.add("done");
    else item.classList.add("active");
  }
  $("pipeline-status").textContent = isDone ? "complete" : (jobActive ? "running" : "idle");
}

function resetStepper() {
  for (const phase of ["design", "build", "validate", "evaluate"]) {
    $("stepper").querySelector(`[data-phase="${phase}"]`).classList.remove("active", "done");
  }
  $("pipeline-status").textContent = "idle";
}

async function sendMessage(message) {
  addMsg(message, "user");
  setBusy(true);
  jobActive = true;
  clearProgress();
  resetStepper();
  let response;
  try {
    response = await (await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    })).json();
  } catch {
    addMsg("Could not reach the server.", "bot", true);
    setBusy(false);
    jobActive = false;
    return;
  }
  if (response.error) {
    addMsg(response.error, "bot", true);
    setBusy(false);
    jobActive = false;
    return;
  }
  pollJob(response.job_id, response.intent);
}

async function pollJob(jobId, intent) {
  const tick = async () => {
    let job;
    try {
      job = await (await fetch(`/api/job/${jobId}`)).json();
    } catch {
      setTimeout(tick, 800);
      return;
    }
    renderProgress(job.events || []);
    if (!job.done) {
      setTimeout(tick, 600);
      return;
    }
    clearProgress();
    if (job.error) addMsg(job.error, "bot", true);
    else if (job.result) addMsg(job.result.message, "bot");
    jobActive = false;
    updateStepper(job.events || []);
    await refreshProject();
    if (!job.error && built && viewerReady && ["build", "iterate", "adjust_and_rebuild", "evaluate"].includes(intent)) {
      loadEpisode();
    }
    setBusy(false);
  };
  tick();
}

function setBusy(value) {
  $("build-btn").disabled = value;
  $("prompt-input").disabled = value;
  $("preview-btn").disabled = value;
  $("build-btn-label").textContent = value ? "Building..." : "Build package";
  document.querySelectorAll(".action-btn").forEach((button) => {
    button.disabled = value || !built;
  });
}

async function refreshProject() {
  let project;
  try {
    project = await (await fetch("/api/project")).json();
  } catch {
    return;
  }
  built = !!project.built;
  const chip = $("project-chip");
  $("actions-card").setAttribute("aria-disabled", built ? "false" : "true");
  document.querySelectorAll(".action-btn").forEach((button) => {
    button.disabled = !built || jobActive;
  });
  $("replay-btn").disabled = !built;
  if (!built) {
    chip.textContent = "No package";
    chip.className = "chip chip-muted";
    return;
  }
  const evaluation = project.evaluation || {};
  const autonomy = project.autonomy || {};
  const rate = evaluation.success_rate ?? autonomy.final_success_rate;
  chip.textContent = `${autonomy.species || "robot"} / ${fmtPct(rate)}`;
  chip.className = "chip " + (autonomy.succeeded ? "chip-ok" : "chip-warn");
  renderResults(project);
}

function renderResults(project) {
  const evaluation = project.evaluation || {};
  const autonomy = project.autonomy || {};
  const design = autonomy.converged_design || {};
  const compute = autonomy.compute || {};
  const rate = evaluation.success_rate ?? autonomy.final_success_rate;
  const body = $("results-body");
  body.innerHTML = "";

  const kpis = document.createElement("div");
  kpis.className = "kpi-row";
  const target = autonomy.target_success_rate ?? 0.8;
  const rateClass = rate != null && rate >= target ? "good" : "warn";
  kpis.innerHTML = `
    <div class="kpi">
      <div class="k-label">Task success</div>
      <div class="k-value ${rateClass}">${fmtPct(rate)}</div>
      <div class="k-sub">${autonomy.succeeded ? "meets target" : "not export-ready"}</div>
    </div>
    <div class="kpi">
      <div class="k-label">Task objects</div>
      <div class="k-value">${evaluation.blocks_placed ?? "-"}<small>/${evaluation.blocks_total ?? "-"}</small></div>
      <div class="k-sub">${esc(evaluation.task || autonomy.task_type || "task")}</div>
    </div>`;
  body.appendChild(kpis);
  body.appendChild(renderEvidenceGates(project));

  if (autonomy.initial_success_rate != null && autonomy.final_success_rate != null &&
      autonomy.final_success_rate !== autonomy.initial_success_rate) {
    body.appendChild(section("Improvement delta", kvHtml([
      ["first build", fmtPct(autonomy.initial_success_rate)],
      ["latest build", fmtPct(autonomy.final_success_rate)],
    ])));
  }

  if (lastRobot) {
    const rows = [
      ["class", lastRobot.robot_class || "-"],
      ["links", (lastRobot.links || []).join(" -> ") || "-"],
      ["end effector", (lastRobot.end_effectors || []).join(", ") || "-"],
    ];
    if ((lastRobot.sensors || []).filter(Boolean).length) {
      rows.push(["sensors", lastRobot.sensors.filter(Boolean).join(", ")]);
    }
    let head = esc(lastRobot.species || lastRobot.name || "robot");
    if (lastRobot.requested_species && lastRobot.requested_species !== lastRobot.species) {
      head += ` <span class="kv-note">(requested ${esc(lastRobot.requested_species)}; built nearest buildable)</span>`;
    }
    body.appendChild(section("Robot genome", `<div class="built-head mono">${head}</div>${kvHtml(rows)}`));
  }

  const notes = (autonomy.notes || []).filter((note) =>
    /redesign|flywheel|buildable|lesson|co-design|reach|nearest/i.test(note)
  );
  if (notes.length) {
    const block = section("AI reasoning", "");
    for (const note of notes.slice(0, 4)) {
      const item = document.createElement("div");
      item.className = "note";
      item.textContent = note;
      block.appendChild(item);
    }
    body.appendChild(block);
  }

  const clusters = evaluation.failure_clusters || [];
  if (clusters.length) {
    const block = section("Failure clusters", "");
    const total = clusters.reduce((sum, cluster) => sum + cluster.count, 0) || 1;
    for (const cluster of clusters) {
      const row = document.createElement("div");
      row.className = "bar-row";
      const width = Math.round(100 * cluster.count / total);
      row.innerHTML = `<span>${esc(cluster.label)}</span><span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span><span class="n">${cluster.count}</span>`;
      block.appendChild(row);
    }
    body.appendChild(block);
  }

  if (Object.keys(design).length) {
    body.appendChild(section("Chosen body parameters", kvHtml(
      Object.entries(design).map(([key, value]) => [
        key.replace(/_/g, " "),
        typeof value === "number" ? value.toFixed(1) : value,
      ])
    )));
  }

  if (compute.physics_executed) {
    const block = section("Physics proof", "");
    const item = document.createElement("div");
    item.className = "provenance";
    item.innerHTML = `<div class="pv-badge"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>MuJoCo run evidence</div>` +
      `<span class="mono">${(compute.physics_steps || 0).toLocaleString()}</span> sim steps / ` +
      `<span class="mono">${(compute.ik_evaluations || 0).toLocaleString()}</span> IK evals / ` +
      `<span class="mono">${compute.models_simulated || 0}</span> models / ` +
      `<span class="mono">${compute.wall_time_seconds || 0}s</span>`;
    block.appendChild(item);
    body.appendChild(block);
  }

  const artifacts = project.artifacts || [];
  if (artifacts.length) {
    const details = document.createElement("details");
    details.className = "disc";
    details.innerHTML = `<summary>Artifact browser (${artifacts.length})</summary>`;
    const list = document.createElement("div");
    list.className = "artifacts";
    let group = "";
    for (const artifact of artifacts.slice(0, 160)) {
      const top = normPath(artifact).split("/")[0];
      if (top !== group) {
        group = top;
        const groupLabel = document.createElement("div");
        groupLabel.className = "art-group";
        groupLabel.textContent = top;
        list.appendChild(groupLabel);
      }
      const link = document.createElement("a");
      link.textContent = artifact;
      link.href = "/api/artifact?path=" + encodeURIComponent(artifact);
      link.target = "_blank";
      list.appendChild(link);
    }
    details.appendChild(list);
    body.appendChild(details);
  }
}

function renderEvidenceGates(project) {
  const artifacts = project.artifacts || [];
  const normalized = artifacts.map(normPath);
  const has = (test) => normalized.some(test);
  const hasExactStep = has((path) => path.startsWith("cad/exact/") && path.endsWith(".step"));
  const hasCompiledSim = has((path) => path.includes("simulation/mujoco/compiled_scene_index.json"));
  const hasPhysics = !!project.evaluation || !!(project.autonomy && project.autonomy.compute && project.autonomy.compute.physics_executed);
  const hasCvContract = has((path) => path.includes("datasets/synthetic_observations") && path.endsWith("annotations.json"));
  const hasRenderedCv = has((path) => /datasets\/synthetic_observations\/.*\.(png|exr|pcd|jpg|jpeg|npy|npz)$/i.test(path));
  const hasController = has((path) => path.includes("software/controller/controller_bundle.json"));
  const hasLedger = has((path) => path.includes("product_readiness_ledger.json"));

  const gates = [
    gate("CAD", hasExactStep ? "warn" : "missing", hasExactStep ? "STEP present; verify exactness" : "no exact CAD evidence"),
    gate("Simulation", hasCompiledSim ? "pass" : "missing", hasCompiledSim ? "compiled MuJoCo scenes" : "not compiled"),
    gate("Physics", hasPhysics ? "pass" : "missing", hasPhysics ? "evaluation evidence present" : "no physics report"),
    gate("CV data", hasRenderedCv ? "pass" : (hasCvContract ? "warn" : "missing"), hasRenderedCv ? "rendered frames present" : (hasCvContract ? "annotation contract only" : "no perception output")),
    gate("Controller", hasController ? "pass" : "warn", hasController ? "controller bundle exported" : "not trained/exported"),
    gate("Ledger", hasLedger ? "pass" : "missing", hasLedger ? "product ledger present" : "legacy readiness only"),
  ];

  const wrap = document.createElement("div");
  wrap.className = "evidence-grid";
  wrap.innerHTML = gates.join("");
  return wrap;
}

function gate(label, status, detail) {
  const chip = status === "pass" ? "pass" : status === "warn" ? "check" : "blocked";
  return `<div class="evidence-gate gate-${status}">
    <div class="gate-top"><strong>${esc(label)}</strong><span class="gate-chip">${chip}</span></div>
    <small>${esc(detail)}</small>
  </div>`;
}

function section(title, inner) {
  const element = document.createElement("div");
  element.className = "section";
  element.innerHTML = `<h3>${esc(title)}</h3>` + (typeof inner === "string" ? inner : "");
  if (typeof inner !== "string") element.appendChild(inner);
  return element;
}

function kvHtml(pairs) {
  return pairs.map(([key, value]) =>
    `<div class="kv"><span class="k">${esc(String(key))}</span><span class="v">${esc(String(value))}</span></div>`
  ).join("");
}

function initThree() {
  const canvas = $("viewer-canvas");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b0a);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
  camera.up.set(0, 0, 1);
  camera.position.set(1.1, -1.1, 0.85);
  controls = new OrbitControls(camera, canvas);
  controls.target.set(0.4, 0, 0.15);
  controls.enableDamping = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(1.5, -1, 2.5);
  scene.add(key);
  const grid = new THREE.GridHelper(4, 32, 0x38423b, 0x1a201c);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  meshGroup = new THREE.Group();
  scene.add(meshGroup);
  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(animate);
  viewerReady = true;
}

function resize() {
  const wrap = $("viewer-canvas").parentElement;
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(height, 1);
  camera.updateProjectionMatrix();
}

function buildMeshes(geoms) {
  while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0]);
  meshGroup.userData.meshes = geoms.map((geom) => {
    let shape;
    const size = geom.size || [0.05, 0.05, 0.05];
    if (geom.type === "box") shape = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    else if (geom.type === "sphere") shape = new THREE.SphereGeometry(size[0], 20, 16);
    else if (geom.type === "cylinder") {
      shape = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 24);
      shape.rotateX(Math.PI / 2);
    } else if (geom.type === "capsule") {
      shape = new THREE.CapsuleGeometry(size[0], 2 * size[1], 6, 16);
      shape.rotateX(Math.PI / 2);
    } else if (geom.type === "plane") {
      shape = new THREE.PlaneGeometry(2.2, 1.6);
    } else {
      shape = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * (size[2] || size[0]));
    }
    const color = geom.rgba || [0.6, 0.6, 0.6, 1];
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      transparent: color[3] < 1,
      opacity: color[3],
      metalness: 0.16,
      roughness: 0.58,
    });
    const mesh = new THREE.Mesh(shape, material);
    meshGroup.add(mesh);
    return mesh;
  });
}

function applyFrame(index) {
  const meshes = meshGroup.userData.meshes || [];
  const frame = frames[index];
  if (!frame) return;
  for (let i = 0; i < meshes.length && i < frame.length; i++) {
    const pose = frame[i];
    meshes[i].position.set(pose[0], pose[1], pose[2]);
    meshes[i].quaternion.set(pose[4], pose[5], pose[6], pose[3]);
  }
  $("scrub").value = index;
  $("frame-label").textContent = `${index + 1} / ${frames.length}`;
}

function animate(time) {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (playing && frames.length && time - lastTick > 55) {
    lastTick = time;
    frameIndex = (frameIndex + 1) % frames.length;
    applyFrame(frameIndex);
    if (frameIndex === frames.length - 1) setPlaying(false);
  }
  if (renderer) renderer.render(scene, camera);
}

const PLAY_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><polygon points="6 4 20 12 6 20 6 4"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>';
function setPlaying(value) {
  playing = value;
  $("play-btn").innerHTML = value ? PAUSE_SVG : PLAY_SVG;
}

async function loadEpisode(sceneIndex = 0) {
  $("viewer-loading").hidden = false;
  $("replay-btn").disabled = true;
  try {
    const data = await (await fetch("/api/viewer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_index: sceneIndex }),
    })).json();
    if (data.error) {
      addMsg("Viewer: " + data.error, "bot", true);
      return;
    }
    buildMeshes(data.geoms);
    frames = data.frames || [];
    frameIndex = 0;
    $("viewer-empty").style.display = frames.length ? "none" : "flex";
    const scrub = $("scrub");
    scrub.max = Math.max(0, frames.length - 1);
    scrub.value = 0;
    scrub.disabled = frames.length === 0;
    $("play-btn").disabled = frames.length === 0;
    if (frames.length) applyFrame(0);
    const outcome = data.outcome || {};
    const chip = $("viewer-outcome");
    chip.innerHTML = `<span class="dot"></span>${esc(outcome.status || "-")} / ${outcome.placed_count ?? 0}/${outcome.block_count ?? 0}`;
    chip.className = "chip " + (outcome.status === "success" ? "chip-ok" : "chip-warn");
    renderSceneSelect(data.scenes || [], data.scene_index ?? 0);
    lastRobot = data.robot;
    try {
      renderResults(await (await fetch("/api/project")).json());
    } catch {
      // The viewer can still function if the project summary refresh fails.
    }
    setPlaying(true);
  } catch {
    addMsg("Could not run the viewer simulation.", "bot", true);
  } finally {
    $("viewer-loading").hidden = true;
    $("replay-btn").disabled = !built;
  }
}

function renderSceneSelect(scenes, current) {
  const select = $("scene-select");
  select.innerHTML = "";
  for (const sceneInfo of scenes) {
    const option = document.createElement("option");
    option.value = sceneInfo.index;
    option.textContent = `Scene ${sceneInfo.index + 1}: ${sceneInfo.name}`;
    if (sceneInfo.index === current) option.selected = true;
    select.appendChild(option);
  }
  select.disabled = scenes.length <= 1;
}

async function previewModel() {
  const prompt = $("prompt-input").value.trim();
  if (!prompt) {
    $("prompt-input").focus();
    return;
  }
  if (!viewerReady) {
    addMsg("3D viewer unavailable in this browser.", "bot", true);
    return;
  }
  $("viewer-loading").hidden = false;
  $("preview-btn").disabled = true;
  addMsg(prompt, "user");
  try {
    const data = await (await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    })).json();
    if (data.error) {
      addMsg("Preview: " + data.error, "bot", true);
      return;
    }
    buildMeshes(data.geoms);
    frames = data.frames || [];
    frameIndex = 0;
    $("viewer-empty").style.display = "none";
    setPlaying(false);
    $("play-btn").disabled = true;
    const scrub = $("scrub");
    scrub.max = 0;
    scrub.value = 0;
    scrub.disabled = true;
    if (frames.length) applyFrame(0);
    $("frame-label").textContent = "model";
    const robot = data.robot || {};
    const chip = $("viewer-outcome");
    chip.innerHTML = `<span class="dot"></span>preview / ${esc(robot.robot_class || "robot")} / ${robot.dof ?? 0} DOF`;
    chip.className = "chip chip-muted";
    lastRobot = robot;
    renderPreviewResults(robot);
    addMsg(`Preview composed: ${robot.robot_class || "robot"} (${robot.dof ?? 0} DOF). Build it to generate evidence.`, "bot");
  } catch {
    addMsg("Could not preview the model.", "bot", true);
  } finally {
    $("viewer-loading").hidden = true;
    $("preview-btn").disabled = false;
  }
}

function renderPreviewResults(robot) {
  const body = $("results-body");
  const rows = [
    ["class", robot.robot_class || "-"],
    ["degrees of freedom", String(robot.dof ?? "-")],
    ["building blocks", (robot.links || []).join(" -> ") || "-"],
    ["end effector", (robot.end_effectors || []).join(", ") || "none"],
    ["kinematic tree", robot.valid ? "valid" : "needs review"],
  ];
  body.innerHTML = "";
  body.appendChild(section("Preview only", `<div class="built-head mono">${esc(robot.species || robot.name || "robot")}</div>${kvHtml(rows)}`));
  const gates = document.createElement("div");
  gates.className = "evidence-grid";
  gates.innerHTML = [
    gate("Geometry", "warn", "composed only"),
    gate("Physics", "missing", "not run"),
    gate("CAD", "missing", "not exported"),
    gate("Ledger", "missing", "not generated"),
  ].join("");
  body.appendChild(gates);
}

async function openMemory() {
  const drawer = $("memory-drawer");
  drawer.hidden = false;
  memoryReturnFocus = document.activeElement;
  drawer.querySelector(".icon-close").focus();
  const body = $("memory-body");
  body.innerHTML = `<div class="act-empty">Loading memory...</div>`;
  let memory;
  try {
    memory = await (await fetch("/api/memory")).json();
  } catch {
    body.innerHTML = `<div class="act-empty">Could not load memory.</div>`;
    return;
  }
  if (!memory.exists) {
    body.innerHTML = `<div class="act-empty">No saved memory yet. Build a few robots and this panel becomes the reuse map.</div>`;
    return;
  }
  const stats = memory.stats || {};
  const statDefs = [
    ["runs", "Builds"], ["species", "Species"], ["lessons", "Lessons"],
    ["designs", "Designs"], ["transfers", "Reuse"], ["failures", "Failures"],
  ];
  body.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "stat-grid";
  for (const [key, label] of statDefs) {
    const item = document.createElement("div");
    item.className = "stat";
    item.innerHTML = `<div class="s-val">${stats[key] ?? 0}</div><div class="s-lab">${label}</div>`;
    grid.appendChild(item);
  }
  body.appendChild(grid);
  if (memory.trainable_rows != null) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = `${memory.trainable_rows} high-success design(s) are eligible for local training export.`;
    body.appendChild(note);
  }
  const recent = memory.recent || [];
  if (recent.length) {
    const block = section("Recent builds", "");
    for (const run of recent) {
      const row = document.createElement("div");
      row.className = "run-row";
      row.innerHTML = `<div class="r-top"><span class="r-prompt">${esc((run.prompt || "").slice(0, 64))}</span>` +
        `<span class="chip ${run.succeeded ? "chip-ok" : "chip-warn"}">${fmtPct(run.success_rate)}</span></div>` +
        `<div class="r-meta">${esc(run.robot_class || "")} / ${esc(run.task_type || "")} / ${esc(run.backend || "offline")}</div>`;
      block.appendChild(row);
    }
    body.appendChild(block);
  }
}

function closeMemory() {
  if ($("memory-drawer").hidden) return;
  $("memory-drawer").hidden = true;
  if (memoryReturnFocus && memoryReturnFocus.focus) memoryReturnFocus.focus();
}

function initExamples() {
  const box = $("examples");
  for (const example of EXAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = example.length > 38 ? example.slice(0, 35) + "..." : example;
    button.title = example;
    button.addEventListener("click", () => {
      $("prompt-input").value = example;
      $("prompt-input").focus();
    });
    box.appendChild(button);
  }
}

function initActions() {
  const grid = $("action-grid");
  for (const action of ACTIONS) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-btn";
    button.disabled = true;
    button.innerHTML = `<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${action.icon}</svg>` +
      `<span>${action.label}<small>${action.sub}</small></span>`;
    button.addEventListener("click", () => sendMessage(action.msg));
    grid.appendChild(button);
  }
}

async function initBackendChip() {
  const chip = $("backend-chip");
  const label = $("backend-label");
  try {
    await (await fetch("/api/project")).json();
    label.textContent = "MuJoCo backend";
    chip.className = "chip chip-ok";
  } catch {
    label.textContent = "offline";
    chip.className = "chip chip-bad";
  }
}

function init() {
  $("activity").innerHTML = `<div class="act-empty">Submit a prompt to start a build. The run stream appears here.</div>`;
  initExamples();
  initActions();
  try {
    initThree();
  } catch (error) {
    const empty = $("viewer-empty");
    empty.style.display = "flex";
    empty.innerHTML = "<p class='empty-sub'>3D viewer unavailable in this browser.</p>";
    console.error("3D init failed:", error);
  }
  initBackendChip();
  refreshProject().then(() => {
    if (built && viewerReady) loadEpisode();
  });

  $("build-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const prompt = $("prompt-input").value.trim();
    if (!prompt) {
      $("prompt-input").focus();
      return;
    }
    sendMessage(prompt);
  });
  $("prompt-input").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      $("build-form").requestSubmit();
    }
  });
  $("preview-btn").addEventListener("click", previewModel);
  $("replay-btn").addEventListener("click", () => loadEpisode(parseInt($("scene-select").value || "0", 10)));
  $("scene-select").addEventListener("change", (event) => loadEpisode(parseInt(event.target.value || "0", 10)));
  $("play-btn").addEventListener("click", () => {
    if (frames.length) setPlaying(!playing);
  });
  $("scrub").addEventListener("input", (event) => {
    setPlaying(false);
    frameIndex = Number(event.target.value);
    applyFrame(frameIndex);
  });
  $("memory-btn").addEventListener("click", openMemory);
  $("memory-drawer").querySelectorAll("[data-close]").forEach((element) => {
    element.addEventListener("click", closeMemory);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMemory();
  });
}

init();
