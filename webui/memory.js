import { el, clear, escapeHtml } from "./util.js";
import { NODES, LINKS, GROUPS } from "./species-data.js";

const memBus = new EventTarget();

// Concentric rings give the graph a readable "species tree" hierarchy:
// classes at the core, species around them, capabilities and methods outward.
const RING = { class: 0, species: 150, task: 270, skill: 270, training: 380, concept: 380 };

function seedNodes() {
  const byGroup = {};
  NODES.forEach((n) => { (byGroup[n.group] = byGroup[n.group] || []).push(n); });
  const placed = new Map();
  Object.entries(byGroup).forEach(([group, items]) => {
    const radius = RING[group] ?? 280;
    items.forEach((n, i) => {
      const angle = (i / items.length) * Math.PI * 2 + (group.length * 0.7);
      placed.set(n.id, {
        x: Math.cos(angle) * radius + (Math.random() - 0.5) * 30,
        y: Math.sin(angle) * radius + (Math.random() - 0.5) * 30,
      });
    });
  });
  return NODES.map((n) => ({ ...n, ...placed.get(n.id), vx: 0, vy: 0 }));
}

const state = { selected: null, nodes: seedNodes(), treeMode: true, builds: [] };
const nodeById = new Map(state.nodes.map((n) => [n.id, n]));
const adjacency = new Map(state.nodes.map((n) => [n.id, new Set()]));
LINKS.forEach(([a, b]) => {
  if (adjacency.has(a) && adjacency.has(b)) {
    adjacency.get(a).add(b);
    adjacency.get(b).add(a);
  }
});

// Dynamic class -> built-instance links, rebuilt from /api/packages so each real build is its own tree node.
let dynLinks = [];
function allLinks() { return dynLinks.length ? LINKS.concat(dynLinks) : LINKS; }

// ---- The default view is the ROBOT SPECIES TREE: root -> classes -> species, force-directed into a radial
// tree (organic, like the full graph but species-only). The wider task/skill/training web is the "Full graph"
// toggle. Ring radii differ per mode so tree mode reads as a clean root-out hierarchy. ----
const TREE_GROUPS = new Set(["root", "class", "species", "build"]);
const TREE_RING = { root: 0, class: 185, species: 390, build: 470 };
function ringFor(n) {
  return state.treeMode ? (TREE_RING[n.group] ?? 0) : (RING[n.group] ?? 280);
}

// What shows where. The hardcoded reference species (quadruped.composed.12dof etc.) are NO LONGER drawn --
// they were placeholders with made-up DOF; their notes survive only as each class's build/train tips. The
// tree is root -> class -> your real builds; the full graph is the wider task/skill web (minus per-build
// instances).
function visible(n) {
  if (!n || n.group === "species") return false;
  if (state.treeMode) return n.group === "root" || n.group === "class" || n.group === "build";
  return n.group !== "build";
}
function cleanName(id) {
  const s = String(id || "");
  return s.replace(/^build[_ ](an?|the)[_ ]/i, "").replace(/_/g, " ").trim() || s;
}
// The class's build/train tips live in its (now-hidden) reference-species note; drop the made-up
// "N actuated joints" description line and keep just the Build/Train guidance.
function tipsBody(note) {
  const i = String(note || "").search(/\*\*Build/i);
  return i >= 0 ? note.slice(i) : (note || "");
}

function select(id) {
  state.selected = id;
  memBus.dispatchEvent(new CustomEvent("select", { detail: id }));
}

// ---- Live "studio memory": reflect the robots ACTUALLY built (from /api/packages) onto the species tree,
// so building a robot visibly updates the memory -- the class it maps to gets a build-count badge and its
// note lists the real instances. Refetched whenever the Memory tab is shown. ----
async function loadBuilds() {
  try {
    const data = await fetch("/api/packages").then((r) => r.json());
    state.builds = Array.isArray(data && data.packages) ? data.packages : [];
  } catch (e) {
    state.builds = [];
  }
  syncBuildNodes();
  memBus.dispatchEvent(new CustomEvent("builds"));
}

// Exposed so the self-running demo (autodemo.js) can refresh the tree and highlight a build node.
export { select as selectMemoryNode, loadBuilds as loadMemoryBuilds };

// ---- Flywheel COMPOUNDING CURVE (the moat KPI): banked assets per build cycle, from /api/flywheel
// (populated by scripts/run_flywheel_for_demo.py). Rendered in the Notes panel's default view. ----
let _flywheel = null;
async function loadFlywheel() {
  try { _flywheel = await fetch("/api/flywheel").then((r) => r.json()); } catch (e) { _flywheel = null; }
  memBus.dispatchEvent(new CustomEvent("flywheel"));
}
function appendFlywheelChart(host) {
  const fw = _flywheel;
  if (!fw || !Array.isArray(fw.series) || !fw.series.length) return;
  host.appendChild(el("div", { class: "panel-section-title", text: "Flywheel — banked assets per cycle (the moat)" }));
  const max = Math.max(...fw.series.map((s) => s.banked)) || 1;
  const chart = el("div", { style: "display:flex;flex-direction:column;gap:6px;margin:8px 0;" });
  fw.series.forEach((s) => {
    const w = Math.round((100 * s.banked) / max);
    chart.appendChild(el("div", { style: "display:flex;align-items:center;gap:8px;font-size:12px;" }, [
      el("span", { text: `cycle ${s.cycle}${s.warm_start ? " ⟲ warm" : ""}`, style: "width:96px;opacity:0.75;" }),
      el("div", { style: `height:14px;width:${w}%;min-width:4px;background:#5fc27e;border-radius:3px;` }),
      el("span", { text: String(s.banked), style: "opacity:0.85;" }),
    ]));
  });
  host.appendChild(chart);
  host.appendChild(el("p", { text: fw.headline || "",
    style: `font-size:13px;margin-top:4px;color:${fw.compounding ? "#5fc27e" : "#e1aa3c"};` }));
}

function classNodeForBuild(b) {
  const cls = String((b && b.robot_class) || "").toLowerCase();
  const name = String((b && (b.id || b.name)) || "").toLowerCase();
  if (cls === "manipulator") return "manipulator";
  if (cls === "mobile_base") return "mobile_base";
  if (cls === "humanoid") return "humanoid";
  if (cls === "quadruped") return "quadruped";
  if (cls === "legged") {
    if (name.includes("hex") || name.includes("six")) return "hexapod";
    if (name.includes("oct") || name.includes("eight")) return "octopod";
    return "quadruped";
  }
  return null;
}

function classOf(node) {
  if (node.group === "class") return node.id;
  if (node.group === "species") {
    for (const id of adjacency.get(node.id) || []) {
      const n = nodeById.get(id);
      if (n && n.group === "class") return n.id;
    }
  }
  return null;
}

function buildsForNode(node) {
  const cls = classOf(node);
  if (!cls) return [];
  return state.builds.filter((b) => classNodeForBuild(b) === cls);
}

function buildRow(b) {
  const ok = b.valid === true || b.valid === "valid";
  return el("button", { class: "mem-built-row", type: "button",
                        onClick: () => select("build:" + (b.id || b.name || b.robot_class)) }, [
    el("span", { class: "mem-built-dot", style: `background:${ok ? "#b8ff3a" : "#e8b765"}` }),
    el("span", { class: "mem-built-name", text: cleanName(b.id || b.name || "(unnamed)") }),
    el("span", { class: "mem-built-cls", text: String(b.species || b.robot_class || "") }),
  ]);
}

function appendBuiltSection(host, node) {
  const bs = buildsForNode(node);
  if (!bs.length) return;
  host.appendChild(el("div", { class: "panel-section-title", text: `Built in this studio (${bs.length})` }));
  host.appendChild(el("div", { class: "mem-built" }, bs.map(buildRow)));
}

function appendBuiltSummary(host) {
  const bs = state.builds || [];
  if (!bs.length) return;
  host.appendChild(el("div", { class: "panel-section-title", text: `Built in this studio (${bs.length})` }));
  host.appendChild(el("div", { class: "mem-built" }, bs.map(buildRow)));
}

function rebuildAdjacency() {
  adjacency.clear();
  state.nodes.forEach((n) => adjacency.set(n.id, new Set()));
  allLinks().forEach(([a, b]) => {
    if (adjacency.has(a) && adjacency.has(b)) { adjacency.get(a).add(b); adjacency.get(b).add(a); }
  });
}

// Turn each real build into its OWN tree node hanging off its class -- a class "splits up" into the robots
// actually built (instead of a count badge), and each instance node is clickable for its tips.
function syncBuildNodes() {
  state.nodes = state.nodes.filter((n) => n.group !== "build");
  dynLinks = [];
  (state.builds || []).forEach((b) => {
    const cls = classNodeForBuild(b);
    const c = cls && nodeById.get(cls);
    if (!c) return;
    const id = "build:" + (b.id || b.name || b.robot_class);
    state.nodes.push({
      id, label: cleanName(b.id || b.name || "build"), group: "build", build: b,
      x: c.x + (Math.random() - 0.5) * 60, y: c.y + (Math.random() - 0.5) * 60, vx: 0, vy: 0,
    });
    dynLinks.push([cls, id]);
  });
  nodeById.clear();
  state.nodes.forEach((n) => nodeById.set(n.id, n));
  rebuildAdjacency();
}

function canonicalSpeciesNote(classId) {
  for (const id of (adjacency.get(classId) || [])) {
    const n = nodeById.get(id);
    if (n && n.group === "species" && n.note) return n;
  }
  return null;
}

// A built instance inherits the build + train tips of its class's canonical species (its real specs up top).
function renderBuildNote(host, node) {
  const b = node.build || {};
  const color = (GROUPS.build || {}).color || "#7fdfff";
  host.appendChild(el("div", { class: "mem-note-head" }, [
    el("span", { class: "mem-note-kind", style: `color:${color}`, text: "BUILT ROBOT" }),
    el("h2", { text: node.label }),
  ]));
  const ok = b.valid === true || b.valid === "valid";
  const tags = [b.species || b.robot_class, b.dof != null ? `${b.dof} DOF` : null,
                ok ? "valid" : "not valid", `${b.scene_count || 0} scene(s)`].filter(Boolean);
  host.appendChild(el("div", { class: "mem-tags" }, tags.map((t) => el("span", { class: "mem-tag", text: t }))));
  const real = (b.species || b.robot_class || "robot") + (b.dof != null ? `, ${b.dof} actuated joints` : "");
  host.appendChild(el("div", { class: "markdown mem-note-body",
    html: noteMarkdown(`This SPECIFIC build, from its real model: **${real}**.`) }));
  const cls = classNodeForBuild(b);
  const sp = cls ? canonicalSpeciesNote(cls) : null;
  if (sp && sp.note) {
    host.appendChild(el("div", { class: "panel-section-title", text: "Build + train tips (its type)" }));
    host.appendChild(el("div", { class: "markdown mem-note-body", html: noteMarkdown(tipsBody(sp.note)) }));
  } else {
    host.appendChild(el("div", { class: "empty", text: "No type tips available yet." }));
  }
}

function noteMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let inList = false;
  const inline = (s) => s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
  for (const line of lines) {
    const li = line.match(/^\s*-\s+(.*)$/);
    if (li) {
      if (!inList) { out.push("<ul>"); inList = true; }
      out.push(`<li>${inline(li[1])}</li>`);
    } else {
      if (inList) { out.push("</ul>"); inList = false; }
      if (line.trim()) out.push(`<p>${inline(line)}</p>`);
    }
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

// ---------------- Graph panel ----------------
function makeGraphPanel(icon) {
  let canvas, ctx2d, host, raf;
  const cam = { x: 0, y: 0, scale: 0.7 };
  const camTarget = { x: 0, y: 0, active: false };
  let dragNode = null, dragging = false, panning = false, last = null, moved = 0, hoverId = null;

  function resize() {
    if (!canvas || !host) return;
    const r = host.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, r.width * dpr);
    canvas.height = Math.max(1, r.height * dpr);
    canvas.style.width = r.width + "px";
    canvas.style.height = r.height + "px";
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function step() {
    const nodes = state.nodes.filter(visible);
    const rep = state.treeMode ? 4200 : 2000;  // tree has few nodes -> spread harder so every label fits
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy || 0.01;
        const f = rep / d2;
        const d = Math.sqrt(d2);
        const ux = dx / d, uy = dy / d;
        a.vx += ux * f; a.vy += uy * f;
        b.vx -= ux * f; b.vy -= uy * f;
      }
    }
    allLinks().forEach(([ai, bi]) => {
      const a = nodeById.get(ai), b = nodeById.get(bi);
      if (!a || !b) return;
      if (!visible(a) || !visible(b)) return;
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = (d - 120) * 0.015;
      const ux = dx / d, uy = dy / d;
      a.vx += ux * f; a.vy += uy * f;
      b.vx -= ux * f; b.vy -= uy * f;
    });
    nodes.forEach((n) => {
      // Gently hold each node near its hierarchy ring so the tree stays legible.
      const target = ringFor(n);
      const r = Math.hypot(n.x, n.y) || 0.01;
      const f = (target - r) * 0.004;
      n.vx += (n.x / r) * f;
      n.vy += (n.y / r) * f;
      if (n === dragNode) return;
      n.vx *= 0.85; n.vy *= 0.85;
      n.x += n.vx; n.y += n.vy;
    });
  }

  function toScreen(n, w, h) {
    return { x: w / 2 + (n.x + cam.x) * cam.scale, y: h / 2 + (n.y + cam.y) * cam.scale };
  }

  function draw() {
    if (!ctx2d) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx2d.clearRect(0, 0, w, h);
    const neighbors = state.selected ? adjacency.get(state.selected) : null;

    allLinks().forEach(([ai, bi]) => {
      const a = nodeById.get(ai), b = nodeById.get(bi);
      if (!a || !b) return;
      if (!visible(a) || !visible(b)) return;
      const pa = toScreen(a, w, h), pb = toScreen(b, w, h);
      const active = state.selected && (ai === state.selected || bi === state.selected);
      ctx2d.strokeStyle = active ? "rgba(184,255,58,0.5)" : "rgba(120,140,150,0.16)";
      ctx2d.lineWidth = active ? 1.6 : 1;
      ctx2d.beginPath(); ctx2d.moveTo(pa.x, pa.y); ctx2d.lineTo(pb.x, pb.y); ctx2d.stroke();
    });

    // Pass 1: draw node dots; collect label candidates for de-collision.
    const labelCands = [];
    state.nodes.forEach((n) => {
      if (!visible(n)) return;
      const p = toScreen(n, w, h);
      const g = GROUPS[n.group] || { color: "#8aa0ad" };
      const isSel = n.id === state.selected;
      const dim = state.selected && !isSel && !(neighbors && neighbors.has(n.id));
      const r = (n.group === "root" ? 13 : n.group === "class" ? 10 : n.group === "species" ? 8 : 6) * (isSel ? 1.4 : 1);
      ctx2d.globalAlpha = dim ? 0.22 : 1;
      if (isSel) {
        ctx2d.beginPath(); ctx2d.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
        ctx2d.strokeStyle = g.color; ctx2d.lineWidth = 1.5; ctx2d.stroke();
      }
      ctx2d.beginPath(); ctx2d.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx2d.fillStyle = g.color; ctx2d.fill();
      ctx2d.globalAlpha = 1;
      const pr = isSel ? 0 : n.id === hoverId ? 1 : n.group === "root" ? 2 : n.group === "class" ? 3 : n.group === "species" ? 4 : 5;
      labelCands.push({ n, p, r, isSel, dim, pr });
    });

    // Pass 2: greedy label placement -- draw labels in priority order, skipping any whose box would
    // overlap an already-drawn one. The selected + hovered labels always win. This kills text overlap.
    labelCands.sort((a, b) => a.pr - b.pr);
    const boxes = [];
    ctx2d.textAlign = "center";
    labelCands.forEach(({ n, p, r, isSel, dim }) => {
      ctx2d.font = `${isSel || n.group === "class" || n.group === "root" ? 600 : 400} 12px 'IBM Plex Mono', monospace`;
      const tw = ctx2d.measureText(n.label).width;
      const cx = p.x, cy = p.y + r + 14;
      const box = { x0: cx - tw / 2 - 3, x1: cx + tw / 2 + 3, y0: cy - 11, y1: cy + 4 };
      const always = isSel || n.id === hoverId;
      if (!always && boxes.some((q) => box.x0 < q.x1 && box.x1 > q.x0 && box.y0 < q.y1 && box.y1 > q.y0)) return;
      boxes.push(box);
      ctx2d.globalAlpha = dim ? 0.5 : 1;
      ctx2d.fillStyle = isSel ? "#ffffff" : dim ? "rgba(220,228,233,0.55)" : "#eaf0f4";
      ctx2d.fillText(n.label, cx, cy);
      ctx2d.globalAlpha = 1;
    });
  }

  function loop() {
    step();
    if (camTarget.active) {
      cam.x += (camTarget.x - cam.x) * 0.12;
      cam.y += (camTarget.y - cam.y) * 0.12;
      if (Math.hypot(camTarget.x - cam.x, camTarget.y - cam.y) < 1) camTarget.active = false;
    }
    draw();
    raf = requestAnimationFrame(loop);
  }

  function pick(mx, my) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    let best = null, bestD = 18;
    state.nodes.forEach((n) => {
      if (!visible(n)) return;
      const p = toScreen(n, w, h);
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestD) { bestD = d; best = n; }
    });
    return best;
  }

  return {
    id: "memoryGraph", title: "Knowledge Graph", icon, dependsOnPackage: false,
    mount(elx) {
      clear(elx);
      host = el("div", { class: "mem-graph" });
      canvas = el("canvas", {});
      host.appendChild(canvas);
      const titleSub = el("span", { class: "mem-title-sub",
        text: "robot classes \u2192 species \u00b7 click a species for its build + train tips" });
      const titleMain = el("span", { class: "mem-title-main", text: "Robot Species Tree" });
      host.appendChild(el("div", { class: "mem-title" }, [titleMain, titleSub]));
      // Two-tab segmented control -- a clearer way to switch views than a single arrow button.
      const segTree = el("button", { class: "mem-seg-btn is-active", type: "button", text: "Species tree" });
      const segFull = el("button", { class: "mem-seg-btn", type: "button", text: "Full graph" });
      function setMode(tree) {
        state.treeMode = tree;
        segTree.classList.toggle("is-active", tree);
        segFull.classList.toggle("is-active", !tree);
        titleMain.textContent = tree ? "Robot Species Tree" : "Full knowledge graph";
        titleSub.textContent = tree
          ? "robot classes \u2192 species \u00b7 click a species for its build + train tips"
          : "classes, species, tasks, skills + training methods \u00b7 click any node";
        cam.x = 0; cam.y = 0; cam.scale = tree ? 0.7 : 0.8;
      }
      segTree.addEventListener("click", () => setMode(true));
      segFull.addEventListener("click", () => setMode(false));
      host.appendChild(el("div", { class: "mem-seg" }, [segTree, segFull]));
      host.appendChild(el("div", { class: "mem-graph-hint", text: "drag to pan \u00b7 scroll to zoom" }));
      const legendItems = [
        ["#e8c06b", "Root"],
        ["#b8ff3a", "Class"],
        ["#e3ebf0", "Species"],
        ["#7fb6c2", "Capability"],
        ["#94a2ae", "Method"],
      ];
      const legend = el("div", { class: "mem-legend" },
        legendItems.map(([color, label]) =>
          el("span", { class: "mem-leg" }, [
            el("span", { class: "mem-leg-dot", style: `background:${color}` }),
            el("span", { text: label }),
          ])));
      host.appendChild(legend);
      elx.appendChild(host);
      ctx2d = canvas.getContext("2d");
      resize();

      canvas.addEventListener("pointerdown", (e) => {
        canvas.setPointerCapture(e.pointerId);
        last = { x: e.offsetX, y: e.offsetY }; moved = 0;
        const n = pick(e.offsetX, e.offsetY);
        if (n) { dragNode = n; dragging = true; } else { panning = true; }
      });
      canvas.addEventListener("pointermove", (e) => {
        if (!last) return;
        const dx = e.offsetX - last.x, dy = e.offsetY - last.y;
        moved += Math.abs(dx) + Math.abs(dy);
        if (dragging && dragNode) {
          dragNode.x += dx / cam.scale; dragNode.y += dy / cam.scale;
          dragNode.vx = 0; dragNode.vy = 0;
        } else if (panning) {
          cam.x += dx / cam.scale; cam.y += dy / cam.scale;
        }
        last = { x: e.offsetX, y: e.offsetY };
      });
      canvas.addEventListener("pointerup", (e) => {
        if (moved < 4) {
          const n = pick(e.offsetX, e.offsetY);
          if (n) select(n.id);
        }
        dragNode = null; dragging = false; panning = false; last = null;
      });
      canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.1 : 0.9;
        cam.scale = Math.max(0.3, Math.min(3, cam.scale * factor));
      }, { passive: false });
      canvas.addEventListener("mousemove", (e) => {
        if (dragging || panning) return;
        const hn = pick(e.offsetX, e.offsetY);
        hoverId = hn ? hn.id : null;
        canvas.style.cursor = hn ? "pointer" : "grab";
      });

      // Gently center the camera on the selected node (from graph or index list).
      memBus.addEventListener("select", () => {
        const n = nodeById.get(state.selected);
        if (n) { camTarget.x = -n.x; camTarget.y = -n.y; camTarget.active = true; }
      });
      loop();
      loadBuilds();
    },
    onShow() { resize(); loadBuilds(); },
    onResize() { resize(); },
  };
}

// ---------------- Note panel ----------------
function makeNotePanel(icon) {
  let host;
  function render() {
    if (!host) return;
    clear(host);
    const node = state.selected ? nodeById.get(state.selected) : null;
    if (!node) {
      host.appendChild(el("div", { class: "empty", text: "Select a species in the tree to read its build + training tips." }));
      appendFlywheelChart(host);
      appendBuiltSummary(host);
      return;
    }
    if (node.group === "build") { renderBuildNote(host, node); return; }
    const g = GROUPS[node.group] || {};
    host.appendChild(el("div", { class: "mem-note-head" }, [
      el("span", { class: "mem-note-kind", style: `color:${g.color}`, text: (g.label || node.group).toUpperCase() }),
      el("h2", { text: node.label }),
    ]));
    if (node.tags && node.tags.length) {
      host.appendChild(el("div", { class: "mem-tags" }, node.tags.map((t) => el("span", { class: "mem-tag", text: `#${t}` }))));
    }
    host.appendChild(el("div", { class: "markdown mem-note-body", html: noteMarkdown(node.note || "") }));
    if (node.group === "class") {
      const sp = canonicalSpeciesNote(node.id);
      if (sp && sp.note) {
        host.appendChild(el("div", { class: "panel-section-title", text: "Build + train tips" }));
        host.appendChild(el("div", { class: "markdown mem-note-body", html: noteMarkdown(tipsBody(sp.note)) }));
      }
    }
    appendBuiltSection(host, node);
    const links = Array.from(adjacency.get(node.id) || []);
    if (links.length) {
      host.appendChild(el("div", { class: "panel-section-title", text: "Linked notes" }));
      host.appendChild(el("div", { class: "mem-links" }, links.map((id) => {
        const n = nodeById.get(id);
        return el("button", { class: "mem-link", type: "button", onClick: () => select(id) }, n ? n.label : id);
      })));
    }
  }
  return {
    id: "memoryNote", title: "Notes", icon, dependsOnPackage: false,
    mount(elx) { clear(elx); host = el("div", { class: "mem-note" }); elx.appendChild(host); render(); memBus.addEventListener("select", render); memBus.addEventListener("builds", render); memBus.addEventListener("flywheel", render); loadFlywheel(); },
    onShow() { render(); loadFlywheel(); },
  };
}

// ---------------- List panel ----------------
function makeListPanel(icon) {
  let host, search;
  function render() {
    if (!host) return;
    clear(host);
    const q = (search && search.value || "").toLowerCase();
    const groups = {};
    state.nodes.forEach((n) => {
      if (q && !(`${n.label} ${(n.tags || []).join(" ")}`.toLowerCase().includes(q))) return;
      (groups[n.group] = groups[n.group] || []).push(n);
    });
    Object.keys(GROUPS).forEach((gk) => {
      if (gk === "species") return;       // reference species are hidden -- folded into each class's tips
      const items = groups[gk];
      if (!items || !items.length) return;
      host.appendChild(el("div", { class: "panel-section-title", text: GROUPS[gk].label }));
      items.forEach((n) => {
        host.appendChild(el("button", {
          class: `mem-list-item${n.id === state.selected ? " active" : ""}`, type: "button",
          onClick: () => select(n.id),
        }, [
          el("span", { class: "mem-list-dot", style: `background:${GROUPS[gk].color}` }),
          el("span", { text: n.label }),
        ]));
      });
    });
  }
  return {
    id: "memoryList", title: "Index", icon, dependsOnPackage: false,
    mount(elx) {
      clear(elx);
      const wrap = el("div", { class: "mem-list" });
      search = el("input", { class: "mem-search", placeholder: "Search knowledge..." });
      search.addEventListener("input", render);
      wrap.appendChild(search);
      host = el("div", { class: "mem-list-body" });
      wrap.appendChild(host);
      elx.appendChild(wrap);
      render();
      memBus.addEventListener("select", render);
    },
    onShow() { render(); },
  };
}

export function createMemoryPanels(ICON) {
  return [makeListPanel(ICON.list), makeGraphPanel(ICON.graph), makeNotePanel(ICON.note)];
}
