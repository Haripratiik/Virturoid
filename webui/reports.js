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

    const [summary, markdown] = await Promise.all([
      ctx.fetchJson("reports/autonomous_build_summary.json"),
      ctx.fetchText("reports/mvp_summary.md"),
    ]);
    clear(root);

    root.appendChild(el("div", { class: "actions", style: "margin-bottom:16px;" }, [
      el("a", { class: "button", href: ctx.packageUrl("reports/workbench.html"), target: "_blank", rel: "noreferrer" }, "Open package workbench"),
      el("a", { class: "button", href: ctx.packageUrl("reports/index.html"), target: "_blank", rel: "noreferrer" }, "Open HTML report"),
      el("a", { class: "button", href: ctx.packageUrl("reports/autonomous_build_summary.json"), target: "_blank", rel: "noreferrer" }, "Build summary JSON"),
    ]));

    if (summary) {
      root.appendChild(el("div", { class: "metrics" }, [
        el("div", { class: "metric" }, [el("span", { text: "Robot class" }), el("strong", { text: summary.selected_robot_class || "-" })]),
        el("div", { class: "metric" }, [el("span", { text: "Task type" }), el("strong", { text: summary.task_type || "-" })]),
        el("div", { class: "metric" }, [el("span", { text: "Package valid" }), el("strong", { text: summary.package_valid ? "Yes" : "No" })]),
        el("div", { class: "metric" }, [el("span", { text: "Readiness" }), el("strong", { text: summary.readiness ? `${summary.readiness.score}%` : "-" })]),
      ]));
    }

    if (markdown) {
      root.appendChild(el("div", { class: "markdown", html: renderMarkdown(markdown) }));
    } else {
      root.appendChild(emptyState("No reports/mvp_summary.md in this package."));
    }
  },
};
