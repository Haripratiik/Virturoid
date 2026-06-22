import { el, clear, pill, needsPackage, emptyState } from "./util.js";

export const scenesSection = {
  dependsOnPackage: true,
  _mounted: false,
  _root: null,

  render(panel) {
    clear(panel);
    panel.appendChild(el("div", { class: "panel-head" }, [
      el("h1", { text: "Simulation scenes" }),
      el("p", { text: "Compiled MuJoCo scenes generated for this package." }),
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
    root.appendChild(emptyState("Loading scenes..."));

    const index = await ctx.fetchJson("simulation/mujoco/compiled_scene_index.json");
    clear(root);
    const scenes = (index && index.scenes) || [];
    if (!scenes.length) {
      root.appendChild(emptyState("No compiled scenes found for this package."));
      return;
    }

    const byPurpose = scenes.reduce((acc, s) => {
      acc[s.purpose] = (acc[s.purpose] || 0) + 1;
      return acc;
    }, {});
    root.appendChild(el("div", { class: "counts" },
      Object.entries(byPurpose).map(([purpose, count]) => pill(`${purpose}: ${count}`, "skip"))));

    const rows = scenes.map((scene) =>
      el("tr", {}, [
        el("td", { text: scene.purpose }),
        el("td", {}, el("span", { class: "mono", text: scene.scene_id })),
        el("td", { text: String(scene.object_count ?? "") }),
        el("td", {}, [
          el("button", {
            class: "button",
            type: "button",
            onClick: () => {
              ctx.viewerRequest = { mode: "scene", scene: scene.mujoco_xml };
              ctx.goTo("viewport");
            },
          }, "View in 3D"),
          el("a", {
            class: "button",
            href: ctx.packageUrl(scene.mujoco_xml),
            target: "_blank",
            rel: "noreferrer",
            style: "margin-left:8px;",
          }, "XML"),
        ]),
      ]));

    root.appendChild(
      el("div", { class: "table-wrap" }, [
        el("table", {}, [
          el("thead", {}, el("tr", {}, [
            el("th", { text: "Purpose" }),
            el("th", { text: "Scene id" }),
            el("th", { text: "Objects" }),
            el("th", { text: "Actions" }),
          ])),
          el("tbody", {}, rows),
        ]),
      ]),
    );
  },
};
