import { el, clear, pill, needsPackage, emptyState, statusKind } from "./util.js";

export const readinessSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "MVP readiness" }),
      el("p", { text: "Audit of whether the generated package meets the current MVP bar." }),
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
    root.appendChild(emptyState("Loading readiness report..."));

    const report = await ctx.fetchJson("reports/mvp_readiness_report.json");
    clear(root);
    if (!report) {
      root.appendChild(emptyState("No mvp_readiness_report.json in this package."));
      return;
    }

    root.appendChild(
      el("div", { class: "card score" }, [
        el("div", { class: "num", text: `${report.score ?? "-"}%` }),
        el("div", { class: "meta" }, [
          el("div", {}, pill(report.ready ? "ready" : "not ready", report.ready ? "ok" : "bad")),
          el("p", { style: "margin:6px 0 0;", text: report.goal || "" }),
        ]),
      ]),
    );

    const gates = report.gates || [];
    const gateList = el("div", { class: "gate-list" },
      gates.map((gate) =>
        el("div", { class: "gate" }, [
          el("div", {}, [
            pill(gate.status, statusKind(gate.status)),
            el("div", { class: "req", text: gate.required ? "required" : "optional" }),
          ]),
          el("div", {}, [
            el("div", { class: "label", text: gate.label || gate.key }),
            el("div", { class: "detail", text: gate.detail || "" }),
          ]),
        ])));
    root.appendChild(gateList);

    if (Array.isArray(report.next_steps) && report.next_steps.length) {
      root.appendChild(el("h3", { style: "margin-top:18px;", text: "Next steps" }));
      root.appendChild(el("ul", { style: "color:var(--muted);" },
        report.next_steps.map((step) => el("li", { text: step }))));
    }
  },
};
