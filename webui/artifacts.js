import { el, clear, needsPackage, emptyState, getJson } from "./util.js";

export const artifactsSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,
  _viewer: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Artifacts" }),
      el("p", { text: "Generated files in this package. Open raw, or preview JSON inline." }),
    ]));
    this._root = el("div", {});
    this._viewer = el("div", { style: "margin-top:16px;" });
    panel.appendChild(this._root);
    panel.appendChild(this._viewer);
  },

  async refresh(ctx) {
    const root = clear(this._root);
    clear(this._viewer);
    if (!ctx.package) {
      root.appendChild(needsPackage());
      return;
    }
    root.appendChild(emptyState("Loading artifacts..."));

    const summary = await ctx.fetchJson("reports/autonomous_build_summary.json");
    clear(root);
    const artifacts = (summary && summary.artifacts) || {};
    const entries = Object.entries(artifacts);
    if (!entries.length) {
      root.appendChild(emptyState("No artifact list found (reports/autonomous_build_summary.json missing)."));
      return;
    }

    const rows = entries.map(([key, uri]) =>
      el("tr", {}, [
        el("td", { text: key }),
        el("td", {}, el("a", { href: ctx.packageUrl(uri), target: "_blank", rel: "noreferrer", class: "mono", text: uri })),
        el("td", {}, String(uri).endsWith(".json")
          ? el("button", { class: "button", type: "button", onClick: () => this.preview(ctx, key, uri) }, "Preview")
          : el("a", { class: "button", href: ctx.packageUrl(uri), target: "_blank", rel: "noreferrer" }, "Open")),
      ]));

    root.appendChild(
      el("div", { class: "table-wrap" }, [
        el("table", {}, [
          el("thead", {}, el("tr", {}, [
            el("th", { text: "Artifact" }),
            el("th", { text: "Path" }),
            el("th", { text: "Action" }),
          ])),
          el("tbody", {}, rows),
        ]),
      ]),
    );
  },

  async preview(ctx, key, uri) {
    clear(this._viewer);
    this._viewer.appendChild(el("h3", { text: `${key} — ${uri}` }));
    const data = await getJson(ctx.packageUrl(uri));
    if (data === null) {
      this._viewer.appendChild(emptyState("Could not load or parse this JSON.", true));
      return;
    }
    this._viewer.appendChild(el("pre", { class: "json-view", text: JSON.stringify(data, null, 2) }));
  },
};
