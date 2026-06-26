import { DockManager } from "./dock.js";
import { getJson, getText, packageUrl } from "./util.js";
import { viewerSection } from "./viewer3d.js";
import { scenesSection } from "./scenes.js";
import { readinessSection } from "./readiness.js";
import { contractSection } from "./contract.js";
import { validationSection } from "./validation.js";
import { artifactsSection } from "./artifacts.js";
import { reportsSection } from "./reports.js";
import { compareSection } from "./compare.js";
import { propertiesPanel } from "./properties.js";
import { assistantPanel } from "./assistant.js";
import { outlinerPanel } from "./outliner.js";
import { consolePanel } from "./console.js";
import { createMemoryPanels, selectMemoryNode, loadMemoryBuilds } from "./memory.js";
import { createAnalysisPanels } from "./analysis.js";
import { runAutoDemo } from "./autodemo.js";

const ICON = {
  viewport: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="m3 7 9 5 9-5"/><path d="M12 22V12"/></svg>',
  assistant: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="13" rx="2"/><path d="M8 21h8M9 10h.01M15 10h.01M9 14h6"/></svg>',
  outliner: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16M7 12h13M10 19h10M4 12h.01M7 19h.01"/></svg>',
  properties: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M5 12H2M22 12h-3"/><circle cx="12" cy="12" r="4"/></svg>',
  console: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M13 15h4"/></svg>',
  scenes: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m12 2 9 5-9 5-9-5z"/><path d="m3 12 9 5 9-5"/></svg>',
  readiness: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21a9 9 0 1 1 9-9"/><path d="M12 12 16 8"/></svg>',
  contract: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6"/><path d="m9 14 2 2 4-4"/></svg>',
  validation: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 5 6v6c0 4 3 6.5 7 9 4-2.5 7-5 7-9V6z"/><path d="m9 12 2 2 4-4"/></svg>',
  artifacts: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m21 8-9-5-9 5v8l9 5 9-5z"/><path d="M12 13v9"/></svg>',
  reports: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M8 13h8M8 17h6"/></svg>',
  graph: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2.4"/><circle cx="5" cy="6" r="1.8"/><circle cx="19" cy="6" r="1.8"/><circle cx="6" cy="19" r="1.8"/></svg>',
  note: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 4h16v12l-4 4H4z"/><path d="M8 9h8M8 13h5"/></svg>',
  list: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',
};

function adapt(id, title, icon, section) {
  return {
    id, title, icon,
    dependsOnPackage: !!section.dependsOnPackage,
    mount(el, ctx) {
      section.render(el, ctx);
      section._mounted = true;
      if (section.dependsOnPackage && section.refresh) section.refresh(ctx);
    },
    onShow(ctx) { if (section.onShow) section.onShow(ctx); },
    onResize(ctx) { if (section.onResize) section.onResize(ctx); },
    refresh(ctx) { if (section.refresh) section.refresh(ctx); },
  };
}

const state = { currentPackage: null, packages: [], workspace: "build" };
const bus = new EventTarget();

const packageSelect = document.getElementById("packageSelect");
const pkgBadge = document.getElementById("pkgBadge");
const assistantChip = document.getElementById("assistantStatus");
const assistantChipText = document.getElementById("assistantStatusText");
const wsTabs = Array.from(document.querySelectorAll(".ws-tab"));
const sb = {
  robot: document.getElementById("sbRobot"),
  class: document.getElementById("sbClass"),
  scenes: document.getElementById("sbScenes"),
  valid: document.getElementById("sbValid"),
};
const resetLayoutBtn = document.getElementById("resetLayout");
const EM_DASH = "\u2014";

const ctx = {
  bus,
  viewerRequest: null,
  assistant: { status: null },
  get package() { return state.currentPackage; },
  get packages() { return state.packages; },
  packageUrl(rel) { return state.currentPackage ? packageUrl(state.currentPackage, rel) : null; },
  packageUrlFor(id, rel) { return packageUrl(id, rel); },
  async fetchJson(rel) { return state.currentPackage ? getJson(packageUrl(state.currentPackage, rel)) : null; },
  async fetchText(rel) { return state.currentPackage ? getText(packageUrl(state.currentPackage, rel)) : null; },
  fetchJsonFor(id, rel) { return getJson(packageUrl(id, rel)); },
  log(message, kind = "info") {
    bus.dispatchEvent(new CustomEvent("log", { detail: { message, kind, ts: Date.now() } }));
  },
  goTo(panelId) {
    if (!dock._findDockWithPanel(panelId)) setWorkspace("build");
    dock.focusPanel(panelId);
  },
  setPackage(id, opts = {}) { return setPackage(id, opts); },
  async refreshPackages() { return loadPackages(); },
  onPackageChange(cb) { bus.addEventListener("package", cb); },
};

const memoryPanels = createMemoryPanels(ICON);
const analysisPanels = createAnalysisPanels(ICON);
// The dock mounts these panel objects by reference, so capture the assistant entry the
// dock actually mounts -- mount() sets _els on THIS object, not the imported singleton.
const assistantEntry = { ...assistantPanel, icon: ICON.assistant };

const panelList = [
  { id: "viewport", title: "Viewport", icon: ICON.viewport, ...adapt("viewport", "Viewport", ICON.viewport, viewerSection) },
  assistantEntry,
  { ...outlinerPanel, icon: ICON.outliner },
  { ...propertiesPanel, icon: ICON.properties },
  { ...consolePanel, icon: ICON.console },
  adapt("scenes", "Scenes", ICON.scenes, scenesSection),
  adapt("readiness", "Readiness", ICON.readiness, readinessSection),
  adapt("contract", "Contract", ICON.contract, contractSection),
  adapt("validation", "Validation", ICON.validation, validationSection),
  adapt("artifacts", "Artifacts", ICON.artifacts, artifactsSection),
  adapt("reports", "Reports", ICON.reports, reportsSection),
  adapt("compare", "Compare", ICON.reports, compareSection),
  ...memoryPanels,
  ...analysisPanels,
];

const dock = new DockManager(document.getElementById("studio"), { panels: panelList, ctx });

const LAYOUTS = {
  build: {
    type: "row",
    children: [
      { type: "dock", id: "left", size: 1.05, panels: ["assistant"], active: "assistant" },
      {
        type: "col", size: 2.5, children: [
          { type: "dock", id: "center-top", size: 2.1, panels: ["viewport"], active: "viewport" },
          { type: "dock", id: "center-bottom", size: 1, panels: ["console", "readiness", "scenes", "contract", "validation", "artifacts", "reports"], active: "console" },
        ],
      },
      {
        type: "col", size: 0.95, children: [
          { type: "dock", id: "right-top", size: 1, panels: ["outliner"], active: "outliner" },
          { type: "dock", id: "right-bottom", size: 1, panels: ["properties"], active: "properties" },
        ],
      },
    ],
  },
  memory: {
    type: "row",
    children: [
      { type: "dock", id: "mem-left", size: 0.85, panels: ["memoryList"], active: "memoryList" },
      { type: "dock", id: "mem-center", size: 2.6, panels: ["memoryGraph"], active: "memoryGraph" },
      { type: "dock", id: "mem-right", size: 1.1, panels: ["memoryNote"], active: "memoryNote" },
    ],
  },
  analysis: {
    type: "col",
    children: [
      {
        type: "row", size: 2, children: [
          { type: "dock", id: "an-left", size: 0.9, panels: ["analysisList"], active: "analysisList" },
          { type: "dock", id: "an-center", size: 2.4, panels: ["analysisCompare"], active: "analysisCompare" },
        ],
      },
      { type: "dock", id: "an-bottom", size: 1, panels: ["analysisDetail"], active: "analysisDetail" },
    ],
  },
};

function setWorkspace(name) {
  if (!LAYOUTS[name]) name = "build";
  state.workspace = name;
  wsTabs.forEach((t) => t.classList.toggle("active", t.dataset.workspace === name));
  dock.setWorkspace(`virturoid.layout.${name}.v2`, LAYOUTS[name]);
}

function updateBadge() {
  const meta = state.currentPackage ? state.packages.find((p) => p.id === state.currentPackage) : null;
  if (!state.currentPackage) {
    pkgBadge.className = "pkg-badge muted";
    pkgBadge.textContent = "No package";
  } else {
    const valid = meta ? meta.valid : null;
    pkgBadge.className = `pkg-badge ${valid === true ? "ok" : valid === false ? "bad" : "muted"}`;
    pkgBadge.textContent = valid === true ? "valid" : valid === false ? "invalid" : "unknown";
  }
  updateStatusBar(meta);
}

function updateStatusBar(meta) {
  if (!sb.robot) return;
  sb.robot.textContent = state.currentPackage || EM_DASH;
  sb.class.textContent = (meta && meta.robot_class) || EM_DASH;
  sb.scenes.textContent = meta && meta.scene_count != null ? String(meta.scene_count) : EM_DASH;
  if (!meta) { sb.valid.className = "sb-item"; sb.valid.textContent = ""; return; }
  const valid = meta.valid;
  sb.valid.className = `sb-item sb-pill ${valid === true ? "ok" : valid === false ? "bad" : "muted"}`;
  sb.valid.textContent = valid === true ? "valid" : valid === false ? "invalid" : "unverified";
}

function setPackage(id, { refreshList = false } = {}) {
  state.currentPackage = id || null;
  if (packageSelect.value !== (id || "")) packageSelect.value = id || "";
  updateBadge();
  bus.dispatchEvent(new CustomEvent("package", { detail: { id: state.currentPackage } }));
  dock.notifyPackageChanged();
  if (id) ctx.log(`Active robot: ${id}`, "info");
  if (refreshList) loadPackages();
}

async function loadPackages() {
  const payload = await getJson("/api/packages");
  state.packages = (payload && payload.packages) || [];
  const previous = state.currentPackage;
  packageSelect.innerHTML = "";

  if (!state.packages.length) {
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "No robots yet";
    packageSelect.appendChild(opt);
    packageSelect.disabled = true;
    updateBadge();
    return;
  }
  packageSelect.disabled = false;
  for (const pkg of state.packages) {
    const opt = document.createElement("option");
    opt.value = pkg.id;
    const mark = pkg.valid === true ? " \u2713" : pkg.valid === false ? " \u2717" : "";
    opt.textContent = `${pkg.id}${mark}`;
    packageSelect.appendChild(opt);
  }
  if (state.packages.some((p) => p.id === previous)) {
    packageSelect.value = previous;
    updateBadge();
  } else {
    setPackage(state.packages[0].id);
  }
}

async function refreshAssistantStatus() {
  const status = await getJson("/api/assistant/status");
  ctx.assistant.status = status;
  const online = status && status.online && status.model_available;
  assistantChip.className = `model-chip ${online ? "online" : "offline"}`;
  if (!status) {
    assistantChipText.textContent = "model: unknown";
  } else if (status.online && status.model_available) {
    assistantChipText.textContent = `model: ${status.model}`;
  } else if (status.online) {
    assistantChipText.textContent = `model: ${status.model} (pull)`;
  } else {
    assistantChipText.textContent = "model: offline";
  }
  bus.dispatchEvent(new CustomEvent("assistant-status", { detail: status }));
}

packageSelect.addEventListener("change", () => setPackage(packageSelect.value || null));
wsTabs.forEach((t) => t.addEventListener("click", () => setWorkspace(t.dataset.workspace)));
if (resetLayoutBtn) resetLayoutBtn.addEventListener("click", () => { dock.resetWorkspace(); ctx.log("Layout reset", "info"); });

async function init() {
  await Promise.all([loadPackages(), refreshAssistantStatus()]);
  setWorkspace("build");
  setInterval(refreshAssistantStatus, 15000);
}

init().then(() => {
  if (new URLSearchParams(location.search).has("demo")) {
    runAutoDemo({
      setWorkspace, setPackage,
      goTo: (id) => ctx.goTo(id),
      viewerSection, assistantPanel: assistantEntry,
      selectMemoryNode, loadMemoryBuilds,
    });
  }
});
