import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";
import { el, clear } from "./util.js";
import { buildSceneFromMuJoCoXml } from "./mujoco_scene.js";

let v = null; // persistent three.js context

function initThree(stage) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e12);
  scene.fog = new THREE.Fog(0x0a0e12, 6, 22);

  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
  camera.position.set(1.1, 0.9, 1.3);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.25, 0.3, 0);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(2.5, 4, 3);
  key.castShadow = true;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xbfd9ff, 0.35);
  fill.position.set(-2, 2, -1);
  scene.add(fill);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(40, 40),
    new THREE.MeshStandardMaterial({ color: 0x10161c, roughness: 0.98, metalness: 0 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.001;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(20, 40, 0x3a8a2e, 0x1c2630);
  grid.material.opacity = 0.35;
  grid.material.transparent = true;
  scene.add(grid);

  const content = new THREE.Group();
  scene.add(content);

  v = { renderer, scene, camera, controls, content, stage, robot: null, sceneRoot: null, loadedPkg: null, loadedMode: null, episode: null };

  function animate(now) {
    requestAnimationFrame(animate);
    controls.update();
    stepEpisode(now || 0);
    renderer.render(scene, camera);
  }
  animate(0);
}

function resize() {
  if (!v) return;
  const w = v.stage.clientWidth;
  const h = v.stage.clientHeight;
  if (!w || !h) return;
  v.camera.aspect = w / h;
  v.camera.updateProjectionMatrix();
  v.renderer.setSize(w, h, false);
}

function clearContent() {
  if (!v) return;
  v.content.clear();
  v.robot = null;
  v.sceneRoot = null;
  v.episode = null;
}

// ---- Episode playback: animate the actual recorded walk / grasp (world geom poses per frame) ----
function episodeGeomMesh(g) {
  const size = g.size || [0.05, 0.05, 0.05];
  const rgba = g.rgba || [0.6, 0.6, 0.62, 1];
  const alpha = rgba[3] === undefined ? 1 : rgba[3];
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
    metalness: 0.1,
    roughness: 0.65,
    transparent: alpha < 1,
    opacity: alpha,
  });
  let geometry;
  switch (g.type) {
    case "box":
      geometry = new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
      break;
    case "sphere":
      geometry = new THREE.SphereGeometry(size[0] || 0.02, 20, 14);
      break;
    case "ellipsoid":
      geometry = new THREE.SphereGeometry(1, 20, 14);
      break;
    case "capsule":
      geometry = new THREE.CapsuleGeometry(size[0] || 0.02, (size[1] || 0.02) * 2, 6, 16);
      geometry.rotateX(Math.PI / 2); // three capsule axis is +Y; MuJoCo capsule axis is +Z
      break;
    case "cylinder":
      geometry = new THREE.CylinderGeometry(size[0] || 0.02, size[0] || 0.02, (size[1] || 0.02) * 2, 22);
      geometry.rotateX(Math.PI / 2);
      break;
    default:
      geometry = new THREE.BoxGeometry((size[0] || 0.025) * 2, (size[1] || 0.025) * 2, (size[2] || 0.025) * 2);
  }
  const mesh = new THREE.Mesh(geometry, material);
  if (g.type === "ellipsoid") mesh.scale.set(size[0] || 0.02, size[1] || 0.02, size[2] || 0.02);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function applyEpisodeFrame(ep, idx) {
  const frame = ep.frames[idx];
  if (!frame) return;
  const n = Math.min(ep.meshes.length, frame.length);
  for (let i = 0; i < n; i += 1) {
    const m = ep.meshes[i];
    const p = frame[i];
    if (!m || !p) continue;
    m.position.set(p[0], p[1], p[2]);
    m.quaternion.set(p[4], p[5], p[6], p[3]); // MuJoCo quat [w,x,y,z] -> three (x,y,z,w)
  }
}

function stepEpisode(now) {
  const ep = v && v.episode;
  if (!ep || !ep.playing || ep.frames.length < 2) return;
  if (!ep.lastTime) ep.lastTime = now;
  if (now - ep.lastTime < ep.frameMs) return;
  ep.lastTime = now;
  ep.idx = (ep.idx + 1) % ep.frames.length;
  applyEpisodeFrame(ep, ep.idx);
  if (ep.onFrame) ep.onFrame(ep.idx);
}

let loadSeq = 0; // bumped on every (re)load; a stale async load aborts instead of double-rendering on top.

async function loadEpisode(refs, ctx) {
  const myLoad = (loadSeq += 1);
  clearContent();
  refs.jointPanel.style.display = "none";
  if (!ctx.package) {
    setStatus(refs, "Build or pick a robot first.", true);
    return;
  }
  setStatus(refs, "Running the episode in physics...");
  let view;
  try {
    const resp = await fetch(`/api/episode?package=${encodeURIComponent(ctx.package)}&scene=0`);
    view = await resp.json();
  } catch (error) {
    setStatus(refs, `Episode request failed: ${error.message || error}`, true);
    return;
  }
  if (myLoad !== loadSeq) return; // a newer load started while we were fetching -- don't add a stale copy
  if (view.error) {
    setStatus(refs, view.error, true);
    return;
  }
  if (!view.frames || !view.frames.length) {
    setStatus(refs, "Episode produced no frames.", true);
    return;
  }

  const group = new THREE.Group();
  group.rotation.x = -Math.PI / 2; // MuJoCo z-up -> three y-up
  const meshes = [];
  (view.geoms || []).forEach((g) => {
    if (g.type === "plane") {
      meshes.push(null); // the viewport already draws a ground
      return;
    }
    const mesh = episodeGeomMesh(g);
    group.add(mesh);
    meshes.push(mesh);
  });
  v.content.add(group);
  v.sceneRoot = group;

  const oc = view.outcome || {};
  const TASK_LABELS = { locomotion: "walk", navigation: "navigate", pick_place_sort: "sort",
                        pick_place_box: "lift", stack: "stack", push: "push" };
  const taskLabel = TASK_LABELS[view.task] || view.task;
  const summary =
    view.task === "locomotion"
      ? `${taskLabel} · ${oc.forward_m ?? "?"} m forward · ${oc.upright ? "upright" : "fell"}`
      : view.task === "navigation"
        ? `${taskLabel} · ${oc.status === "reached" ? "goal reached" : (oc.status || "in progress")}`
        : `${taskLabel} · ${oc.placed_count ?? 0}/${oc.block_count ?? 0} placed`;

  v.episode = { frames: view.frames, meshes, idx: 0, playing: true, frameMs: 1000 / 30, lastTime: 0, onFrame: null };
  applyEpisodeFrame(v.episode, 0);
  fitCamera(group);

  if (refs.playbar) {
    refs.playbar.style.display = "flex";
    refs.scrub.max = String(view.frames.length - 1);
    refs.scrub.value = "0";
    refs.playBtn.textContent = "Pause";
    v.episode.onFrame = (i) => {
      refs.scrub.value = String(i);
    };
  }
  setStatus(refs, `Episode: ${summary} · ${view.frame_count} frames`);
}

let currentObject = null;

function resetView() {
  if (v && currentObject) fitCamera(currentObject);
}

function fitCamera(object) {
  currentObject = object;
  object.updateMatrixWorld(true);
  // Frame the robot + task objects, NOT the ground plane/grid -- fitting the whole scene (incl. the floor)
  // dwarfs the robot and pulls the camera way back, which is why Scene mode looked tiny.
  const box = new THREE.Box3();
  object.traverse((child) => {
    if (child.isMesh && !child.userData.isFloor) box.expandByObject(child);
  });
  if (box.isEmpty()) box.setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.3);
  const dist = maxDim * 2.2;
  v.controls.target.copy(center);
  v.camera.position.set(center.x + dist * 0.8, center.y + dist * 0.6, center.z + dist * 0.8);
  v.controls.update();
}

function setStatus(refs, message, isError = false) {
  refs.status.textContent = message;
  refs.status.classList.toggle("error", isError);
}

function buildJointSliders(refs, robot) {
  clear(refs.jointSliders);
  const joints = Object.entries(robot.joints || {})
    .filter(([, j]) => j.jointType === "revolute" || j.jointType === "continuous")
    .sort(([a], [b]) => a.localeCompare(b));

  if (!joints.length) {
    refs.jointPanel.style.display = "none";
    return;
  }
  refs.jointPanel.style.display = "";
  for (const [name, joint] of joints) {
    const lower = Number.isFinite(joint.limit?.lower) && joint.limit.lower !== 0 ? joint.limit.lower : -3.14;
    const upper = Number.isFinite(joint.limit?.upper) && joint.limit.upper !== 0 ? joint.limit.upper : 3.14;
    const valueEl = el("small", { text: "0.00" });
    const slider = el("input", {
      type: "range",
      min: String(lower),
      max: String(upper),
      step: "0.01",
      value: "0",
    });
    slider.addEventListener("input", () => {
      const angle = Number(slider.value);
      joint.setJointValue(angle);
      valueEl.textContent = angle.toFixed(2);
    });
    refs.jointSliders.appendChild(
      el("div", { class: "joint-row" }, [
        el("div", { class: "top" }, [el("span", { text: name }), valueEl]),
        slider,
      ]),
    );
  }
}

function addFallbackVisuals(robot) {
  let added = 0;
  robot.traverse((child) => {
    if (!child.isURDFLink) return;
    let hasMesh = false;
    child.traverse((sub) => {
      if (sub.isMesh) hasMesh = true;
    });
    if (hasMesh) return;
    const name = String(child.name || "").toLowerCase();
    let geometry;
    if (name.includes("wheel")) {
      geometry = new THREE.CylinderGeometry(0.06, 0.06, 0.04, 24);
      geometry.rotateZ(Math.PI / 2);
    } else if (name.includes("mast") || name.includes("sensor")) {
      geometry = new THREE.BoxGeometry(0.04, 0.04, 0.2);
    } else {
      geometry = new THREE.BoxGeometry(0.26, 0.2, 0.08);
    }
    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({ color: 0x4a5d73, metalness: 0.12, roughness: 0.58 }),
    );
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    child.add(mesh);
    added += 1;
  });
  return added;
}

async function loadRobot(refs, ctx) {
  const myLoad = (loadSeq += 1);
  clearContent();
  setStatus(refs, "Loading URDF...");

  const urdfUrl = ctx.packageUrl("robot/robot.urdf");
  const text = await ctx.fetchText("robot/robot.urdf");
  if (myLoad !== loadSeq) return; // superseded by a newer load
  if (!text) {
    // Legged / engine-built packages have no URDF but do have a replayable episode -- show the motion.
    if (refs.modeSelect) refs.modeSelect.value = "episode";
    await loadEpisode(refs, ctx);
    return;
  }

  const loader = new URDFLoader();
  loader.workingPath = "";
  loader.loadMeshCb = (path, manager, onComplete) => {
    const normalized = String(path).replace(/\\/g, "/").replace(/^(\.\.\/)+/, "");
    const meshUrl = ctx.packageUrl(normalized);
    const ext = normalized.split(".").pop().toLowerCase();
    if (ext === "stl") {
      import("three/addons/loaders/STLLoader.js").then(({ STLLoader }) => {
        new STLLoader(manager).load(
          meshUrl,
          (geometry) => {
            const material = new THREE.MeshStandardMaterial({ color: 0x9aa3ad, metalness: 0.25, roughness: 0.5 });
            onComplete(new THREE.Mesh(geometry, material));
          },
          undefined,
          () => onComplete(null, new Error(`mesh load failed: ${meshUrl}`)),
        );
      });
      return;
    }
    onComplete(null, new Error(`unsupported mesh: ${ext}`));
  };

  let robot;
  try {
    robot = loader.parse(text);
  } catch (error) {
    setStatus(refs, `Failed to parse URDF: ${error.message || error}`, true);
    return;
  }

  robot.rotation.x = -Math.PI / 2; // URDF z-up -> three y-up
  robot.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  // STL meshes load asynchronously; give them a moment then assess.
  setTimeout(() => {
    let meshCount = 0;
    robot.traverse((child) => {
      if (child.isMesh) meshCount += 1;
    });
    if (meshCount === 0) {
      const added = addFallbackVisuals(robot);
      setStatus(refs, `Robot loaded with ${added} placeholder shape(s) (URDF has no meshes). Try Scene mode for full 3D.`);
    } else {
      setStatus(refs, `Robot loaded (${meshCount} mesh part(s)).`);
    }
    fitCamera(robot);
  }, 250);

  if (myLoad !== loadSeq) return; // superseded while parsing
  v.content.add(robot);
  v.robot = robot;
  buildJointSliders(refs, robot);
  fitCamera(robot);
}

async function populateScenes(refs, ctx) {
  const index = await ctx.fetchJson("simulation/mujoco/compiled_scene_index.json");
  const scenes = (index && index.scenes) || [];
  clear(refs.sceneSelect);
  if (!scenes.length) {
    refs.sceneSelect.appendChild(el("option", { value: "", text: "No compiled scenes" }));
    refs.sceneSelect.disabled = true;
    return scenes;
  }
  refs.sceneSelect.disabled = false;
  for (const scene of scenes) {
    refs.sceneSelect.appendChild(
      el("option", { value: scene.mujoco_xml, text: `${scene.purpose} · ${scene.scene_id}` }),
    );
  }
  return scenes;
}

async function loadScene(refs, ctx, xmlPath) {
  const myLoad = (loadSeq += 1);
  clearContent();
  refs.jointPanel.style.display = "none";
  const target = xmlPath || refs.sceneSelect.value;
  if (!target) {
    setStatus(refs, "No scene selected.", true);
    return;
  }
  setStatus(refs, `Loading scene: ${target.split("/").pop()}...`);
  const xmlText = await ctx.fetchText(target);
  if (myLoad !== loadSeq) return; // superseded by a newer load
  if (!xmlText) {
    setStatus(refs, "Could not load scene XML.", true);
    return;
  }
  try {
    const root = buildSceneFromMuJoCoXml(xmlText);
    v.content.add(root);
    v.sceneRoot = root;
    fitCamera(root);
    setStatus(refs, `Scene loaded: ${target.split("/").pop()}`);
  } catch (error) {
    setStatus(refs, error.message || String(error), true);
  }
}

async function reload(refs, ctx) {
  if (!v) return;
  if (!ctx.package) {
    clearContent();
    if (refs.empty) refs.empty.style.display = "grid";
    setStatus(refs, "", false);
    refs.status.style.display = "none";
    return;
  }
  if (refs.empty) refs.empty.style.display = "none";
  refs.status.style.display = "";
  if (refs.playbar) refs.playbar.style.display = "none";
  if (refs.modeSelect.value === "episode" || refs.modeSelect.value === "robot") {
    clear(refs.sceneSelect);                                    // don't leave a stale scene name in the disabled picker
    refs.sceneSelect.appendChild(el("option", { value: "", text: "— n/a in this mode —" }));
    refs.sceneSelect.disabled = true;
    if (refs.modeSelect.value === "episode") {
      await loadEpisode(refs, ctx);
    } else {
      await loadRobot(refs, ctx);
    }
  } else {
    await populateScenes(refs, ctx);
    await loadScene(refs, ctx);
  }
  v.loadedPkg = ctx.package;
  v.loadedMode = refs.modeSelect.value;
}

export const viewerSection = {
  dependsOnPackage: true,
  _mounted: false,
  _refs: null,

  render(panel, ctx) {
    clear(panel);
    const modeSelect = el("select", {}, [
      el("option", { value: "robot", text: "Robot" }),
      el("option", { value: "scene", text: "Scene" }),
      el("option", { value: "episode", text: "Episode (play)" }),
    ]);
    const sceneSelect = el("select", { disabled: true });
    const resetBtn = el("button", { class: "button vbtn", type: "button", title: "Reset view" }, "Reset view");
    const reloadBtn = el("button", { class: "button vbtn", type: "button", title: "Reload" }, "Reload");

    const toolbar = el("div", { class: "viewer-toolbar" }, [
      el("div", { class: "field" }, [el("label", { text: "Mode" }), modeSelect]),
      el("div", { class: "field viewer-scene-field" }, [el("label", { text: "Scene" }), sceneSelect]),
      resetBtn,
      reloadBtn,
    ]);

    const status = el("div", { class: "viewer-status", text: "Initializing..." });
    const empty = el("div", { class: "viewer-empty" }, [
      el("div", { class: "viewer-empty-ic", html: '<svg viewBox="0 0 24 24" width="42" height="42" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="m3 7 9 5 9-5"/><path d="M12 22V12"/></svg>' }),
      el("div", { class: "viewer-empty-title", text: "No robot loaded" }),
      el("div", { class: "viewer-empty-sub", text: "Ask the assistant to build one, or use the Properties panel." }),
    ]);
    empty.style.display = "none";
    const jointSliders = el("div", {});
    const jointPanel = el("div", { class: "joint-panel" }, [el("h3", { text: "Joint angles" }), jointSliders]);
    jointPanel.style.display = "none";
    const playBtn = el("button", { class: "button vbtn", type: "button" }, "Pause");
    const scrub = el("input", { type: "range", min: "0", max: "0", value: "0", step: "1", class: "viewer-scrub" });
    scrub.style.flex = "1";
    const playbar = el("div", { class: "viewer-playbar" }, [playBtn, scrub]);
    playbar.style.cssText = "position:absolute;left:50%;bottom:14px;transform:translateX(-50%);gap:10px;align-items:center;background:rgba(10,16,20,0.85);border:1px solid #1c2630;border-radius:10px;padding:8px 12px;width:min(62%,540px);z-index:6;";
    playbar.style.display = "none";
    const stage = el("div", { class: "viewer-stage" }, [status, empty, jointPanel, playbar]);

    panel.appendChild(el("div", { class: "viewer-wrap" }, [toolbar, stage]));

    const refs = { modeSelect, sceneSelect, reloadBtn, resetBtn, status, empty, jointPanel, jointSliders, stage, playbar, playBtn, scrub };
    this._refs = refs;

    if (!v) initThree(stage);
    else stage.insertBefore(v.renderer.domElement, status);

    modeSelect.addEventListener("change", () => {
      sceneSelect.disabled = modeSelect.value !== "scene";
      reload(refs, ctx);
    });
    sceneSelect.addEventListener("change", () => {
      if (modeSelect.value === "scene") loadScene(refs, ctx);
    });
    reloadBtn.addEventListener("click", () => reload(refs, ctx));
    resetBtn.addEventListener("click", () => resetView());
    playBtn.addEventListener("click", () => {
      if (!v || !v.episode) return;
      v.episode.playing = !v.episode.playing;
      v.episode.lastTime = 0;
      playBtn.textContent = v.episode.playing ? "Pause" : "Play";
    });
    scrub.addEventListener("input", () => {
      if (!v || !v.episode) return;
      v.episode.playing = false;
      playBtn.textContent = "Play";
      v.episode.idx = Number(scrub.value) || 0;
      applyEpisodeFrame(v.episode, v.episode.idx);
    });

    window.addEventListener("resize", () => {
      if (panel.classList.contains("active")) resize();
    });
  },

  refresh(ctx) {
    if (!this._refs || !v) return;
    // Auto-pick a sensible default mode per package.
    const meta = (ctx && ctx.package) || null;
    if (meta) reload(this._refs, ctx);
  },

  onResize() {
    if (v) resize();
  },

  onShow(ctx) {
    // three.js needs a sized stage; resize after the panel is visible.
    requestAnimationFrame(async () => {
      resize();
      if (!v || !this._refs) return;

      const request = ctx.viewerRequest;
      if (request && request.mode === "scene") {
        ctx.viewerRequest = null;
        this._refs.modeSelect.value = "scene";
        this._refs.sceneSelect.disabled = false;
        await populateScenes(this._refs, ctx);
        if (request.scene) this._refs.sceneSelect.value = request.scene;
        await loadScene(this._refs, ctx, request.scene);
        v.loadedPkg = ctx.package;
        v.loadedMode = "scene";
        return;
      }

      if (ctx.package && v.loadedPkg !== ctx.package) {
        reload(this._refs, ctx);
      }
    });
  },
};
