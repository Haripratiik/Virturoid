import { el, clear, pill, emptyState } from "./util.js";

const anBus = new EventTarget();
const anState = { selected: null };

function selectBuild(id) {
  anState.selected = id;
  anBus.dispatchEvent(new CustomEvent("select", { detail: id }));
}

function bar(value, max, color) {
  const pct = Math.max(0, Math.min(100, max ? (value / max) * 100 : 0));
  return el("div", { class: "an-bar" }, [
    el("div", { class: "an-bar-fill", style: `width:${pct}%;background:${color || "var(--accent)"}` }),
    el("span", { class: "an-bar-label", text: `${Math.round(value)}${max === 100 ? "%" : ""}` }),
  ]);
}

// ---------------- Compare panel ----------------
function makeComparePanel(icon) {
  let host;
  async function load(ctx) {
    if (!host) return;
    clear(host);
    host.appendChild(el("div", { class: "panel-head" }, [
      el("span", { class: "eyebrow", text: "Analysis / Efficiency" }),
      el("h1", { text: "Build comparison" }),
    ]));
    host.appendChild(el("p", { class: "an-help", html:
      "Each row is one robot the studio has built. <strong>Readiness</strong> is how complete the package is "
      + "(valid model, scenes, contract). <strong>Est. success</strong> is the simulated task success rate from its "
      + "training dry-run. Click a row to open its full breakdown below." }));
    if (!ctx.packages.length) { host.appendChild(emptyState("No builds yet. Build a robot first.")); return; }

    const tableHost = el("div", { class: "table-wrap" });
    host.appendChild(tableHost);
    tableHost.appendChild(el("div", { class: "empty", text: "Loading metrics..." }));

    const rows = await Promise.all(ctx.packages.map(async (pkg) => {
      const [readiness, dry] = await Promise.all([
        ctx.fetchJsonFor(pkg.id, "reports/mvp_readiness_report.json"),
        ctx.fetchJsonFor(pkg.id, "runs/mvp_training/dry_run_result.json"),
      ]);
      return { pkg, readiness, dry };
    }));

    clear(tableHost);
    const table = el("table", {}, [
      el("thead", {}, el("tr", {}, [
        el("th", { text: "Robot" }), el("th", { text: "Class" }),
        el("th", { text: "Readiness" }), el("th", { text: "Est. success" }),
        el("th", { text: "Scenes" }), el("th", { text: "Valid" }),
      ])),
      el("tbody", {}, rows.map(({ pkg, readiness, dry }) => {
        const score = readiness && readiness.score != null ? readiness.score : 0;
        const success = dry && dry.estimated_success_rate != null ? dry.estimated_success_rate * 100 : 0;
        const tr = el("tr", { class: pkg.id === anState.selected ? "an-active" : "" }, [
          el("td", {}, el("span", { class: "mono", text: pkg.id })),
          el("td", { text: pkg.robot_class || "-" }),
          el("td", {}, bar(score, 100, "var(--accent)")),
          el("td", {}, bar(success, 100, dry ? "var(--cyan)" : "var(--line-bright)")),
          el("td", { text: String(pkg.scene_count ?? 0) }),
          el("td", {}, pill(pkg.valid ? "yes" : "no", pkg.valid ? "ok" : "bad")),
        ]);
        tr.addEventListener("click", () => selectBuild(pkg.id));
        return tr;
      })),
    ]);
    tableHost.appendChild(table);
  }
  return {
    id: "analysisCompare", title: "Comparison", icon, dependsOnPackage: true,
    mount(elx, ctx) { clear(elx); host = el("div", { class: "an-compare" }); elx.appendChild(host); load(ctx); },
    refresh(ctx) { load(ctx); },
    onShow(ctx) { /* table persists */ },
  };
}

// ---------------- List panel ----------------
function makeListPanel(icon) {
  let host;
  function render(ctx) {
    if (!host) return;
    clear(host);
    host.appendChild(el("div", { class: "panel-section-title", text: "Builds" }));
    if (!ctx.packages.length) { host.appendChild(el("div", { class: "outliner-hint", text: "No builds yet." })); return; }
    ctx.packages.forEach((pkg) => {
      host.appendChild(el("button", {
        class: `lib-item${pkg.id === anState.selected ? " active" : ""}`, type: "button",
        onClick: () => selectBuild(pkg.id),
      }, [
        el("span", { class: `lib-dot ${pkg.valid === true ? "ok" : pkg.valid === false ? "bad" : "muted"}` }),
        el("span", { class: "lib-name", text: pkg.id }),
      ]));
    });
  }
  return {
    id: "analysisList", title: "Builds", icon, dependsOnPackage: true,
    mount(elx, ctx) { clear(elx); host = el("div", { class: "outliner" }); elx.appendChild(host); render(ctx); anBus.addEventListener("select", () => render(ctx)); },
    refresh(ctx) { render(ctx); },
  };
}

// ---------------- Detail panel ----------------
function makeDetailPanel(icon) {
  let host, ctxRef;

  function kv(label, value) {
    return el("div", { class: "metric" }, [el("span", { text: label }), el("strong", { text: value })]);
  }

  async function render() {
    if (!host) return;
    const ctx = ctxRef;
    const id = anState.selected || (ctx && ctx.package);
    clear(host);
    if (!id) { host.appendChild(emptyState("Select a build to see its analysis.")); return; }

    host.appendChild(el("div", { class: "an-detail-head" }, [el("h2", { text: id })]));
    host.appendChild(el("div", { class: "empty", text: "Loading reports..." }));

    const [dry, improvement, promotion, feedback] = await Promise.all([
      ctx.fetchJsonFor(id, "runs/mvp_training/dry_run_result.json"),
      ctx.fetchJsonFor(id, "reports/improvement_report.json"),
      ctx.fetchJsonFor(id, "reports/promotion_decision.json"),
      ctx.fetchJsonFor(id, "reports/training_feedback.json"),
    ]);

    clear(host);
    host.appendChild(el("div", { class: "an-detail-head" }, [el("h2", { text: id })]));

    if (dry) {
      host.appendChild(el("div", { class: "panel-section-title", text: "Training dry-run" }));
      host.appendChild(el("p", { class: "an-section-note", text: "Simulated rollout results per scene group (no physics engine was run)." }));
      host.appendChild(el("div", { class: "metrics" }, [
        kv("Backend", dry.backend || "-"),
        kv("Est. success", dry.estimated_success_rate != null ? `${Math.round(dry.estimated_success_rate * 100)}%` : "-"),
        kv("Scenes", String(dry.total_scenes ?? "-")),
        kv("Episodes", String(dry.total_planned_episodes ?? "-")),
      ]));
      const groups = dry.group_results || [];
      if (groups.length) {
        host.appendChild(el("div", { class: "table-wrap" }, [el("table", {}, [
          el("thead", {}, el("tr", {}, [el("th", { text: "Group" }), el("th", { text: "Scenes" }), el("th", { text: "Episodes" }), el("th", { text: "Success" })])),
          el("tbody", {}, groups.map((g) => el("tr", {}, [
            el("td", { text: g.purpose }),
            el("td", { text: String(g.scene_count ?? "") }),
            el("td", { text: String(g.planned_episodes ?? "") }),
            el("td", {}, bar((g.simulated_success_rate || 0) * 100, 100, "var(--cyan)")),
          ]))),
        ])]));
      }
    }

    if (improvement) {
      host.appendChild(el("div", { class: "panel-section-title", text: "Improvement" }));
      host.appendChild(el("p", { class: "an-section-note", text: "What a revision pass changed versus the original run." }));
      host.appendChild(el("div", { class: "an-outcome" }, [
        pill(improvement.outcome || "n/a", improvement.outcome === "improved" ? "ok" : "warn"),
        el("span", { class: "mono", text: improvement.comparison_basis || "" }),
      ]));
      const metrics = improvement.metrics || [];
      if (metrics.length) {
        host.appendChild(el("div", { class: "table-wrap" }, [el("table", {}, [
          el("thead", {}, el("tr", {}, [el("th", { text: "Metric" }), el("th", { text: "Before" }), el("th", { text: "After" }), el("th", { text: "\u0394" })])),
          el("tbody", {}, metrics.map((m) => el("tr", {}, [
            el("td", { text: m.name }),
            el("td", { class: "mono", text: String(m.before) }),
            el("td", { class: "mono", text: String(m.after) }),
            el("td", {}, pill((m.delta > 0 ? "+" : "") + m.delta, m.delta > 0 ? "ok" : m.delta < 0 ? "bad" : "skip")),
          ]))),
        ])]));
      }
      if (Array.isArray(improvement.next_actions) && improvement.next_actions.length) {
        host.appendChild(el("ul", { class: "an-actions" }, improvement.next_actions.map((a) => el("li", { text: a }))));
      }
    }

    if (promotion) {
      host.appendChild(el("div", { class: "panel-section-title", text: "Promotion decision" }));
      host.appendChild(el("div", { class: "metrics" },
        Object.entries(promotion)
          .filter(([, v]) => typeof v !== "object")
          .map(([k, v]) => kv(k, String(v)))));
    }
    if (feedback) {
      host.appendChild(el("div", { class: "panel-section-title", text: "Training feedback" }));
      host.appendChild(el("div", { class: "metrics" },
        Object.entries(feedback)
          .filter(([, v]) => typeof v !== "object")
          .map(([k, v]) => kv(k, String(v)))));
    }

    if (!dry && !improvement && !promotion && !feedback) {
      host.appendChild(emptyState("No analysis artifacts for this build (it may be a quick/dry build)."));
    }
  }

  return {
    id: "analysisDetail", title: "Detail", icon, dependsOnPackage: true,
    mount(elx, ctx) { ctxRef = ctx; clear(elx); host = el("div", { class: "an-detail" }); elx.appendChild(host); anBus.addEventListener("select", render); render(); },
    refresh(ctx) { ctxRef = ctx; render(); },
    onShow(ctx) { ctxRef = ctx; },
  };
}

export function createAnalysisPanels(ICON) {
  return [makeListPanel(ICON.list), makeComparePanel(ICON.reports), makeDetailPanel(ICON.note)];
}
