// Lightweight dockable workstation layout engine.
//
// A layout is a tree of nodes:
//   { type: "row"|"col", size, children: [...] }
//   { type: "dock", id, size, panels: [panelId...], active: panelId }
//
// Panels are registered with { id, title, icon, mount(el, ctx), onShow, onResize }.
// Panel content elements are created once and re-attached across re-renders so
// stateful panels (e.g. the 3D viewport) survive layout changes and tab moves.

import { el, clear } from "./util.js";

export class DockManager {
  constructor(root, { panels, ctx }) {
    this.root = root;
    this.ctx = ctx;
    this.panels = new Map(panels.map((p) => [p.id, p]));
    this.contents = new Map(); // panelId -> { el, mounted }
    this.tree = null;
    this.storageKey = null;
    this._onResize = () => this.resizeAll();
    window.addEventListener("resize", this._onResize);
  }

  setWorkspace(storageKey, defaultTree) {
    this.storageKey = storageKey;
    this.tree = this._load() || structuredClone(defaultTree);
    this._defaultTree = defaultTree;
    this.render();
  }

  resetWorkspace() {
    if (!this._defaultTree) return;
    this.tree = structuredClone(this._defaultTree);
    this.save();
    this.render();
  }

  _content(panelId) {
    let entry = this.contents.get(panelId);
    if (!entry) {
      const def = this.panels.get(panelId);
      const node = el("div", { class: "panel-content", dataset: { panel: panelId } });
      entry = { el: node, mounted: false, def };
      this.contents.set(panelId, entry);
    }
    return entry;
  }

  _ensureMounted(panelId) {
    const entry = this._content(panelId);
    if (!entry.mounted && entry.def) {
      try {
        entry.def.mount(entry.el, this.ctx);
      } catch (error) {
        console.error("panel mount failed:", panelId, error);
        entry.el.innerHTML = `<div class="empty error">Panel failed: ${error.message || error}</div>`;
      }
      entry.mounted = true;
    }
    return entry;
  }

  render() {
    clear(this.root);
    // Detach all content nodes first (keep them in memory).
    for (const entry of this.contents.values()) {
      if (entry.el.parentNode) entry.el.parentNode.removeChild(entry.el);
    }
    this.root.appendChild(this._renderNode(this.tree, null, 0));
    requestAnimationFrame(() => this.resizeAll());
  }

  _renderNode(node, parent, index) {
    if (node.type === "dock") return this._renderDock(node);

    const box = el("div", { class: `dock-split dock-${node.type}` });
    node.children.forEach((child, i) => {
      const childEl = this._renderNode(child, node, i);
      childEl.style.flex = `${child.size || 1} 1 0`;
      childEl.classList.add("dock-cell");
      box.appendChild(childEl);
      if (i < node.children.length - 1) {
        box.appendChild(this._splitter(node, i, box));
      }
    });
    return box;
  }

  _renderDock(node) {
    const dock = el("div", { class: "dock", dataset: { dockId: node.id } });
    if (!node.active || !node.panels.includes(node.active)) node.active = node.panels[0];

    const tabs = el("div", { class: "dock-tabs" });
    const body = el("div", { class: "dock-body" });

    node.panels.forEach((panelId) => {
      const def = this.panels.get(panelId);
      if (!def) return;
      const tab = el("div", {
        class: `dock-tab${panelId === node.active ? " active" : ""}`,
        draggable: "true",
        dataset: { panelId, dockId: node.id },
        title: def.title,
      }, [
        def.icon ? el("span", { class: "dock-tab-ic", html: def.icon }) : null,
        el("span", { text: def.title }),
      ]);
      tab.addEventListener("click", () => this._activate(node, panelId));
      tab.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/dock-panel", panelId);
        e.dataTransfer.setData("text/dock-source", node.id);
        e.dataTransfer.effectAllowed = "move";
        tab.classList.add("dragging");
      });
      tab.addEventListener("dragend", () => tab.classList.remove("dragging"));
      tabs.appendChild(tab);
    });

    if (!node.panels.length) {
      body.appendChild(el("div", { class: "dock-empty", text: "Drop a panel tab here" }));
    } else {
      const entry = this._ensureMounted(node.active);
      body.appendChild(entry.el);
      requestAnimationFrame(() => this._show(node.active));
    }

    this._wireDropZone(dock, tabs, body, node);

    dock.appendChild(tabs);
    dock.appendChild(body);
    return dock;
  }

  _wireDropZone(dock, tabs, body, node) {
    const over = (e) => {
      if (e.dataTransfer.types.includes("text/dock-panel")) {
        e.preventDefault();
        dock.classList.add("drop-target");
      }
    };
    const leave = () => dock.classList.remove("drop-target");
    const drop = (e) => {
      e.preventDefault();
      dock.classList.remove("drop-target");
      const panelId = e.dataTransfer.getData("text/dock-panel");
      const sourceId = e.dataTransfer.getData("text/dock-source");
      if (!panelId || sourceId === node.id) return;
      this._movePanel(panelId, sourceId, node.id);
    };
    [tabs, body].forEach((zone) => {
      zone.addEventListener("dragover", over);
      zone.addEventListener("dragleave", leave);
      zone.addEventListener("drop", drop);
    });
  }

  _activate(node, panelId) {
    if (node.active === panelId) return;
    node.active = panelId;
    this.save();
    this.render();
  }

  _movePanel(panelId, sourceId, targetId) {
    const source = this._findDock(sourceId);
    const target = this._findDock(targetId);
    if (!source || !target) return;
    source.panels = source.panels.filter((p) => p !== panelId);
    if (source.active === panelId) source.active = source.panels[0] || null;
    if (!target.panels.includes(panelId)) target.panels.push(panelId);
    target.active = panelId;
    this.save();
    this.render();
  }

  _findDock(id, node = this.tree) {
    if (!node) return null;
    if (node.type === "dock") return node.id === id ? node : null;
    for (const child of node.children || []) {
      const found = this._findDock(id, child);
      if (found) return found;
    }
    return null;
  }

  _splitter(parentNode, index, box) {
    const isRow = parentNode.type === "row";
    const handle = el("div", { class: `dock-splitter ${isRow ? "vertical" : "horizontal"}` });
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      handle.setPointerCapture(e.pointerId);
      const cells = Array.from(box.children).filter((c) => c.classList.contains("dock-cell"));
      const a = cells[index];
      const b = cells[index + 1];
      const nodeA = parentNode.children[index];
      const nodeB = parentNode.children[index + 1];
      const startPos = isRow ? e.clientX : e.clientY;
      const sizeA = isRow ? a.offsetWidth : a.offsetHeight;
      const sizeB = isRow ? b.offsetWidth : b.offsetHeight;
      const totalWeight = (nodeA.size || 1) + (nodeB.size || 1);
      const totalPx = sizeA + sizeB;

      const move = (ev) => {
        const pos = isRow ? ev.clientX : ev.clientY;
        let newA = sizeA + (pos - startPos);
        newA = Math.max(80, Math.min(totalPx - 80, newA));
        const ratio = newA / totalPx;
        nodeA.size = totalWeight * ratio;
        nodeB.size = totalWeight * (1 - ratio);
        a.style.flex = `${nodeA.size} 1 0`;
        b.style.flex = `${nodeB.size} 1 0`;
        this.resizeAll();
      };
      const up = (ev) => {
        handle.releasePointerCapture(ev.pointerId);
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        this.save();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    return handle;
  }

  _show(panelId) {
    const entry = this.contents.get(panelId);
    if (entry && entry.def && entry.def.onShow) {
      try { entry.def.onShow(this.ctx); } catch (e) { console.error(e); }
    }
  }

  resizeAll() {
    for (const [id, entry] of this.contents.entries()) {
      if (entry.el.isConnected && entry.def && entry.def.onResize) {
        try { entry.def.onResize(this.ctx); } catch (e) { console.error(e); }
      }
    }
  }

  // Called by main.js when the active package changes.
  notifyPackageChanged() {
    for (const [id, entry] of this.contents.entries()) {
      if (!entry.mounted || !entry.def) continue;
      if (entry.def.dependsOnPackage && entry.def.refresh) {
        try { entry.def.refresh(this.ctx); } catch (e) { console.error(e); }
      }
    }
  }

  focusPanel(panelId) {
    const dock = this._findDockWithPanel(panelId);
    if (dock) { dock.active = panelId; this.save(); this.render(); }
  }

  _findDockWithPanel(panelId, node = this.tree) {
    if (!node) return null;
    if (node.type === "dock") return node.panels.includes(panelId) ? node : null;
    for (const child of node.children || []) {
      const found = this._findDockWithPanel(panelId, child);
      if (found) return found;
    }
    return null;
  }

  save() {
    if (!this.storageKey) return;
    try { localStorage.setItem(this.storageKey, JSON.stringify(this.tree)); } catch (e) { /* ignore */ }
  }

  _load() {
    if (!this.storageKey) return null;
    try {
      const raw = localStorage.getItem(this.storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }
}
