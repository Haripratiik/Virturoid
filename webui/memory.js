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

const state = { selected: null, nodes: seedNodes(), treeMode: true };
const nodeById = new Map(state.nodes.map((n) => [n.id, n]));
const adjacency = new Map(state.nodes.map((n) => [n.id, new Set()]));
LINKS.forEach(([a, b]) => {
  if (adjacency.has(a) && adjacency.has(b)) {
    adjacency.get(a).add(b);
    adjacency.get(b).add(a);
  }
});

// ---- The default view is the ROBOT SPECIES TREE: root -> classes -> species, force-directed into a radial
// tree (organic, like the full graph but species-only). The wider task/skill/training web is the "Full graph"
// toggle. Ring radii differ per mode so tree mode reads as a clean root-out hierarchy. ----
const TREE_GROUPS = new Set(["root", "class", "species"]);
const TREE_RING = { root: 0, class: 185, species: 390 };
function ringFor(n) {
  return state.treeMode ? (TREE_RING[n.group] ?? 0) : (RING[n.group] ?? 280);
}

function select(id) {
  state.selected = id;
  memBus.dispatchEvent(new CustomEvent("select", { detail: id }));
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
    const nodes = state.treeMode ? state.nodes.filter((n) => TREE_GROUPS.has(n.group)) : state.nodes;
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
    LINKS.forEach(([ai, bi]) => {
      const a = nodeById.get(ai), b = nodeById.get(bi);
      if (!a || !b) return;
      if (state.treeMode && !(TREE_GROUPS.has(a.group) && TREE_GROUPS.has(b.group))) return;
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

    LINKS.forEach(([ai, bi]) => {
      const a = nodeById.get(ai), b = nodeById.get(bi);
      if (!a || !b) return;
      if (state.treeMode && !(TREE_GROUPS.has(a.group) && TREE_GROUPS.has(b.group))) return;
      const pa = toScreen(a, w, h), pb = toScreen(b, w, h);
      const active = state.selected && (ai === state.selected || bi === state.selected);
      ctx2d.strokeStyle = active ? "rgba(184,255,58,0.5)" : "rgba(120,140,150,0.16)";
      ctx2d.lineWidth = active ? 1.6 : 1;
      ctx2d.beginPath(); ctx2d.moveTo(pa.x, pa.y); ctx2d.lineTo(pb.x, pb.y); ctx2d.stroke();
    });

    // Pass 1: draw node dots; collect label candidates for de-collision.
    const labelCands = [];
    state.nodes.forEach((n) => {
      if (state.treeMode && !TREE_GROUPS.has(n.group)) return;
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
      if (state.treeMode && !TREE_GROUPS.has(n.group)) return;
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
    },
    onShow() { resize(); },
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
      return;
    }
    const g = GROUPS[node.group] || {};
    host.appendChild(el("div", { class: "mem-note-head" }, [
      el("span", { class: "mem-note-kind", style: `color:${g.color}`, text: (g.label || node.group).toUpperCase() }),
      el("h2", { text: node.label }),
    ]));
    if (node.tags && node.tags.length) {
      host.appendChild(el("div", { class: "mem-tags" }, node.tags.map((t) => el("span", { class: "mem-tag", text: `#${t}` }))));
    }
    host.appendChild(el("div", { class: "markdown mem-note-body", html: noteMarkdown(node.note || "") }));
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
    mount(elx) { clear(elx); host = el("div", { class: "mem-note" }); elx.appendChild(host); render(); memBus.addEventListener("select", render); },
    onShow() { render(); },
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
