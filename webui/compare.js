import { el, clear, emptyState } from "./util.js";

// Each row: [label, accessor]. Reads the compact spec the server attaches to every package in /api/packages.
const ROWS = [
  ["Class", (p) => p.robot_class || p.species || "-"],
  ["DOF", (p) => (p.dof != null ? p.dof : "-")],
  ["Mass", (p) => (p.spec && p.spec.mass_kg != null ? `${p.spec.mass_kg} kg` : "-")],
  ["Est. parts cost", (p) => (p.spec && p.spec.cost_usd != null ? `$${Number(p.spec.cost_usd).toLocaleString()}` : "-")],
  ["Est. power", (p) => (p.spec && p.spec.power_w != null ? `${p.spec.power_w} W` : "-")],
  ["Task", (p) => (p.spec && p.spec.task) || "-"],
  ["Task success", (p) => (p.spec && p.spec.success != null ? `${Math.round(p.spec.success * 100)}%` : "-")],
  ["Scenes", (p) => (p.scene_count != null ? p.scene_count : "-")],
];

export const compareSection = {
  _root: null,

  render(panel) {
    clear(panel);
    if (!document.getElementById("compare-style")) {
      const s = document.createElement("style");
      s.id = "compare-style";
      s.textContent = ".compare-table{width:100%;border-collapse:collapse;font-size:13px}" +
        ".compare-table th,.compare-table td{border:1px solid var(--panel-3);padding:6px 10px;text-align:left;white-space:nowrap}" +
        ".compare-table th{background:var(--panel-2);color:var(--muted);font-weight:600}" +
        ".compare-table td:first-child,.compare-table th:first-child{color:var(--muted);font-weight:600}";
      document.head.appendChild(s);
    }
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Compare" }),
      el("p", { text: "Specs side-by-side across every robot you have built." }),
    ]));
    this._root = el("div", { style: "overflow:auto;" });
    panel.appendChild(this._root);
  },

  async refresh(ctx) {
    const root = clear(this._root);
    const pkgs = (ctx.packages || []).filter((p) => p.dof != null || p.spec);
    if (!pkgs.length) {
      root.appendChild(emptyState("Build some robots and they will appear here for side-by-side comparison."));
      return;
    }
    const header = el("tr", {}, [el("th", { text: "Spec" }), ...pkgs.map((p) => el("th", { text: p.id }))]);
    const rows = ROWS.map(([label, get]) =>
      el("tr", {}, [el("td", { text: label }), ...pkgs.map((p) => el("td", { text: String(get(p)) }))]));
    rows.push(el("tr", {}, [
      el("td", { text: "Summary" }),
      ...pkgs.map((p) => el("td", {
        text: (p.spec && p.spec.summary) || "-",
        style: "font-size:12px;opacity:0.8;white-space:normal;min-width:170px;",
      })),
    ]));
    root.appendChild(el("table", { class: "compare-table" }, [
      el("thead", {}, [header]),
      el("tbody", {}, rows),
    ]));
  },
};
