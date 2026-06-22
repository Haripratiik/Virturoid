import { el, clear, emptyState } from "./util.js";

function parseRobot(urdfText) {
  const doc = new DOMParser().parseFromString(urdfText, "application/xml");
  if (doc.querySelector("parsererror")) return null;
  const links = Array.from(doc.querySelectorAll("robot > link")).map((l) => l.getAttribute("name"));
  const joints = Array.from(doc.querySelectorAll("robot > joint")).map((j) => ({
    name: j.getAttribute("name"),
    type: j.getAttribute("type"),
    parent: (j.querySelector("parent") || {}).getAttribute?.("link"),
    child: (j.querySelector("child") || {}).getAttribute?.("link"),
  }));
  const childToJoint = new Map(joints.map((j) => [j.child, j]));
  const childrenOf = new Map();
  joints.forEach((j) => {
    if (!childrenOf.has(j.parent)) childrenOf.set(j.parent, []);
    childrenOf.get(j.parent).push(j.child);
  });
  const root = links.find((l) => !childToJoint.has(l)) || links[0];
  return { links, joints, childrenOf, childToJoint, root };
}

function renderTree(robot, link, container, depth) {
  const joint = robot.childToJoint.get(link);
  const row = el("div", { class: "tree-row", style: `padding-left:${depth * 14 + 8}px` }, [
    el("span", { class: "tree-ic", text: depth === 0 ? "\u25c9" : "\u2514" }),
    el("span", { class: "tree-link", text: link }),
    joint ? el("span", { class: `tree-joint ${joint.type}`, text: joint.type }) : null,
  ]);
  container.appendChild(row);
  (robot.childrenOf.get(link) || []).forEach((child) => renderTree(robot, child, container, depth + 1));
}

export const outlinerPanel = {
  id: "outliner",
  title: "Outliner",
  dependsOnPackage: true,
  _ctx: null,
  _root: null,

  mount(elx, ctx) {
    this._ctx = ctx;
    clear(elx);
    this._root = el("div", { class: "outliner" });
    elx.appendChild(this._root);
    this.refresh(ctx);
  },

  async refresh(ctx) {
    if (!this._root) return;
    const root = clear(this._root);

    // Library
    root.appendChild(el("div", { class: "panel-section-title", text: "Robot library" }));
    const lib = el("div", { class: "outliner-lib" });
    if (!ctx.packages.length) {
      lib.appendChild(el("div", { class: "outliner-hint", text: "No robots yet. Build one." }));
    } else {
      ctx.packages.forEach((pkg) => {
        const item = el("button", {
          class: `lib-item${pkg.id === ctx.package ? " active" : ""}`,
          type: "button",
          onClick: () => ctx.setPackage(pkg.id),
        }, [
          el("span", { class: `lib-dot ${pkg.valid === true ? "ok" : pkg.valid === false ? "bad" : "muted"}` }),
          el("span", { class: "lib-name", text: pkg.id }),
          el("span", { class: "lib-meta", text: pkg.robot_class || "" }),
        ]);
        lib.appendChild(item);
      });
    }
    root.appendChild(lib);

    if (!ctx.package) return;

    // Robot structure
    root.appendChild(el("div", { class: "panel-section-title", text: "Robot structure" }));
    const treeHost = el("div", { class: "outliner-tree" });
    root.appendChild(treeHost);
    const urdf = await ctx.fetchText("robot/robot.urdf");
    if (!urdf) {
      treeHost.appendChild(el("div", { class: "outliner-hint", text: "No URDF found." }));
    } else {
      const robot = parseRobot(urdf);
      if (!robot) treeHost.appendChild(el("div", { class: "outliner-hint", text: "Could not parse URDF." }));
      else {
        renderTree(robot, robot.root, treeHost, 0);
        treeHost.appendChild(el("div", { class: "outliner-count mono", text: `${robot.links.length} links \u00b7 ${robot.joints.length} joints` }));
      }
    }

    // Scenes
    root.appendChild(el("div", { class: "panel-section-title", text: "Scenes" }));
    const sceneHost = el("div", { class: "outliner-scenes" });
    root.appendChild(sceneHost);
    const index = await ctx.fetchJson("simulation/mujoco/compiled_scene_index.json");
    const scenes = (index && index.scenes) || [];
    if (!scenes.length) {
      sceneHost.appendChild(el("div", { class: "outliner-hint", text: "No scenes." }));
    } else {
      scenes.slice(0, 40).forEach((scene) => {
        sceneHost.appendChild(el("button", {
          class: "scene-item", type: "button",
          title: scene.scene_id,
          onClick: () => { ctx.viewerRequest = { mode: "scene", scene: scene.mujoco_xml }; ctx.goTo("viewport"); },
        }, [
          el("span", { class: "scene-purpose", text: scene.purpose }),
          el("span", { class: "scene-id mono", text: scene.scene_id }),
        ]));
      });
    }
  },
};
