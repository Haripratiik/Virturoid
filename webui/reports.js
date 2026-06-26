import { el, clear, needsPackage, emptyState, escapeHtml } from "./util.js";

function renderMarkdown(md) {
  const lines = escapeHtml(md).split(/\r?\n/);
  const out = [];
  let inCode = false;
  let inList = false;
  let para = [];

  const flushPara = () => {
    if (para.length) {
      out.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
    }
  };
  const closeList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw;
    if (line.trim().startsWith("```")) {
      flushPara();
      closeList();
      if (!inCode) {
        out.push("<pre><code>");
        inCode = true;
      } else {
        out.push("</code></pre>");
        inCode = false;
      }
      continue;
    }
    if (inCode) {
      out.push(line);
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushPara();
      closeList();
      const level = heading[1].length;
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const listItem = line.match(/^\s*[-*]\s+(.*)$/);
    if (listItem) {
      flushPara();
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${inline(listItem[1])}</li>`);
      continue;
    }
    if (line.trim() === "") {
      flushPara();
      closeList();
      continue;
    }
    para.push(line.trim());
  }
  flushPara();
  closeList();
  if (inCode) out.push("</code></pre>");
  return out.join("\n");
}

function inline(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

export const reportsSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Reports" }),
      el("p", { text: "Generated summaries for this package." }),
    ]));
    this._root = el("div", {});
    panel.appendChild(this._root);
  },

  async refresh(ctx) {
    const root = clear(this._root);
    if (!ctx.package) {
      root.appendChild(needsPackage());
      return;
    }
    root.appendChild(emptyState("Loading reports..."));

    const [summary, markdown, ledger, spec] = await Promise.all([
      ctx.fetchJson("reports/autonomous_build_summary.json"),
      ctx.fetchText("reports/mvp_summary.md"),
      ctx.fetchJson("reports/product_readiness_ledger.json"),
      ctx.fetchJson("reports/spec_sheet.json"),
    ]);
    clear(root);

    root.appendChild(el("div", { class: "actions", style: "margin-bottom:16px;" }, [
      el("a", { class: "button", href: ctx.packageUrl("reports/workbench.html"), target: "_blank", rel: "noreferrer" }, "Open package workbench"),
      el("a", { class: "button", href: ctx.packageUrl("reports/index.html"), target: "_blank", rel: "noreferrer" }, "Open HTML report"),
      el("a", { class: "button", href: ctx.packageUrl("reports/autonomous_build_summary.json"), target: "_blank", rel: "noreferrer" }, "Build summary JSON"),
    ]));

    if (spec) {
      const phys = spec.physical || {}, cost = spec.power_and_cost || {}, act = spec.actuation || {}, perf = spec.performance || {};
      const sz = phys.size_m;
      const fmt = (v, suffix = "") => (v === null || v === undefined || v === "" ? "-" : `${v}${suffix}`);
      const rows = [
        ["Degrees of freedom", fmt(spec.dof)],
        ["Mass", fmt(phys.mass_kg, " kg")],
        ["Size (LxWxH)", sz ? `${sz.length_m}x${sz.width_m}x${sz.height_m} m` : "-"],
        ["Peak joint torque", fmt(act.peak_joint_torque_nm, " Nm")],
        ["Est. power draw", fmt(cost.est_power_draw_w, " W")],
        ["Est. parts cost", cost.est_parts_cost_usd != null ? `$${Number(cost.est_parts_cost_usd).toLocaleString()}` : "-"],
        ["Sensors", (spec.sensing || []).join(", ") || "-"],
        ["Task success", perf.success_rate != null ? `${Math.round(perf.success_rate * 100)}%` : "-"],
      ];
      root.appendChild(el("h2", { text: "Spec Sheet", style: "margin:8px 0 4px;" }));
      root.appendChild(el("p", { text: spec.summary || "", style: "font-size:15px;opacity:0.85;margin:0 0 12px;" }));
      root.appendChild(el("div", { class: "metrics" }, rows.map(([k, v]) =>
        el("div", { class: "metric" }, [el("span", { text: k }), el("strong", { text: String(v) })]))));
      root.appendChild(el("div", { class: "actions", style: "margin:12px 0 20px;" }, [
        el("a", { class: "button", href: ctx.packageUrl("reports/spec_sheet.md"), target: "_blank", rel: "noreferrer" }, "Open spec sheet (Markdown)"),
        el("a", { class: "button", href: ctx.packageUrl("reports/deployment_guide.md"), target: "_blank", rel: "noreferrer" }, "Build it for real (deployment guide)"),
      ]));
    }

    if (summary || ledger) {
      const metrics = [];
      if (summary) {
        metrics.push(el("div", { class: "metric" }, [el("span", { text: "Robot class" }), el("strong", { text: summary.selected_robot_class || "-" })]));
        metrics.push(el("div", { class: "metric" }, [el("span", { text: "Task type" }), el("strong", { text: summary.task_type || "-" })]));
        metrics.push(el("div", { class: "metric" }, [el("span", { text: "Package valid" }), el("strong", { text: summary.package_valid ? "Yes" : "No" })]));
      }
      // The Product Readiness Ledger is the export truth source (not the legacy MVP completeness %).
      if (ledger) {
        metrics.push(el("div", { class: "metric" }, [el("span", { text: "Safe to export" }), el("strong", { text: ledger.safe_to_export ? "Yes" : "No" })]));
      } else if (summary && summary.readiness) {
        metrics.push(el("div", { class: "metric" }, [el("span", { text: "Completeness" }), el("strong", { text: `${summary.readiness.score}%` })]));
      }
      root.appendChild(el("div", { class: "metrics" }, metrics));
    }

    if (markdown) {
      root.appendChild(el("div", { class: "markdown", html: renderMarkdown(markdown) }));
    } else {
      root.appendChild(emptyState("No reports/mvp_summary.md in this package."));
    }
  },
};
