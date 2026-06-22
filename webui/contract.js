import { el, clear, pill, needsPackage, emptyState, statusKind } from "./util.js";

export const contractSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Package contract" }),
      el("p", { text: "Shared robot package contract: class, capabilities, and artifact checks." }),
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
    root.appendChild(emptyState("Loading contract..."));

    const contract = await ctx.fetchJson("reports/robot_package_contract.json");
    clear(root);
    if (!contract) {
      root.appendChild(emptyState("No robot_package_contract.json in this package."));
      return;
    }

    root.appendChild(el("div", { class: "metrics" }, [
      el("div", { class: "metric" }, [el("span", { text: "Robot class" }), el("strong", { text: contract.robot_class || "-" })]),
      el("div", { class: "metric" }, [el("span", { text: "Species" }), el("strong", { text: contract.species || "-" })]),
      el("div", { class: "metric" }, [el("span", { text: "Package type" }), el("strong", { text: contract.package_type || "-" })]),
      el("div", { class: "metric" }, [el("span", { text: "Contract OK" }), el("strong", {}, pill(contract.ok ? "yes" : "no", contract.ok ? "ok" : "bad"))]),
    ]));

    const caps = contract.capabilities || [];
    if (caps.length) {
      root.appendChild(el("div", { class: "counts" }, caps.map((c) => pill(c, "skip"))));
    }

    const checks = contract.artifact_checks || [];
    const rows = checks.map((check) =>
      el("tr", {}, [
        el("td", { text: check.key || "" }),
        el("td", {}, check.uri
          ? el("a", { href: ctx.packageUrl(check.uri), target: "_blank", rel: "noreferrer", class: "mono", text: check.uri })
          : el("span", { text: "" })),
        el("td", {}, pill(check.exists ? "yes" : "no", check.exists ? "ok" : "bad")),
        el("td", {}, pill(check.parse_status || "n/a", statusKind(check.parse_status))),
        el("td", { class: "detail", text: check.detail || "" }),
      ]));

    root.appendChild(
      el("div", { class: "table-wrap" }, [
        el("table", {}, [
          el("thead", {}, el("tr", {}, [
            el("th", { text: "Key" }),
            el("th", { text: "URI" }),
            el("th", { text: "Exists" }),
            el("th", { text: "Parse" }),
            el("th", { text: "Detail" }),
          ])),
          el("tbody", {}, rows),
        ]),
      ]),
    );
  },
};
