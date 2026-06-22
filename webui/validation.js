import { el, clear, pill, needsPackage, emptyState, statusKind } from "./util.js";

export const validationSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Package validation" }),
      el("p", { text: "File-level validation: expected artifacts exist and parse." }),
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
    root.appendChild(emptyState("Loading validation report..."));

    const report = await ctx.fetchJson("reports/package_validation_report.json");
    clear(root);
    if (!report) {
      root.appendChild(emptyState("No package_validation_report.json in this package."));
      return;
    }

    const checks = report.checks || [];
    const passed = checks.filter((c) => c.status === "pass").length;
    const failed = checks.length - passed;

    root.appendChild(el("div", { class: "counts" }, [
      pill(report.ok ? "valid" : "invalid", report.ok ? "ok" : "bad"),
      pill(`${passed} passed`, "ok"),
      failed ? pill(`${failed} failed`, "bad") : pill("0 failed", "skip"),
    ]));

    const ordered = [...checks].sort((a, b) => (a.status === "pass") - (b.status === "pass"));
    const rows = ordered.map((check) =>
      el("tr", {}, [
        el("td", {}, el("span", { class: "mono", text: check.name || "" })),
        el("td", {}, pill(check.status || "", statusKind(check.status))),
        el("td", { class: "detail", text: check.message || "" }),
      ]));

    root.appendChild(
      el("div", { class: "table-wrap" }, [
        el("table", {}, [
          el("thead", {}, el("tr", {}, [
            el("th", { text: "Check" }),
            el("th", { text: "Status" }),
            el("th", { text: "Message" }),
          ])),
          el("tbody", {}, rows),
        ]),
      ]),
    );
  },
};
