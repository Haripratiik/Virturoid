import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader from "urdf-loader";
import { buildSceneFromMuJoCoXml } from "./mujoco_scene.js";

const state = {
  packages: [],
  selectedPackage: null,
  mode: "robot",
  robot: null,
  sceneRoot: null,
  jointSliders: [],
};

const canvas = document.getElementById("viewport");
const packageSelect = document.getElementById("packageSelect");
const sceneSelect = document.getElementById("sceneSelect");
const modeSelect = document.getElementById("modeSelect");
const statusEl = document.getElementById("status");
const jointPanel = document.getElementById("jointPanel");
const jointSlidersEl = document.getElementById("jointSliders");
const reloadButton = document.getElementById("reloadButton");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9edf0);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
camera.position.set(1.2, 0.9, 1.4);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0.25, 0.35, 0);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
keyLight.position.set(2.5, 4, 3);
keyLight.castShadow = true;
scene.add(keyLight);
scene.add(new THREE.DirectionalLight(0xbfd9ff, 0.35).translateX(-2).translateY(2));

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 8),
  new THREE.MeshStandardMaterial({ color: 0xd8dde2, roughness: 0.95 }),
);
ground.rotation.x = -Math.PI / 2;
ground.position.set(0.4, 0, 0);
ground.receiveShadow = true;
scene.add(ground);

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function packageUrl(packageId, relativePath) {
  const clean = String(relativePath).replace(/\\/g, "/").replace(/^\/+/, "");
  const packagePath = String(packageId)
    .replace(/\\/g, "/")
    .replace(/^\/+/, "")
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  const filePath = clean
    .split("/")
    .map(encodeURIComponent)
    .join("/");
  return `/package/${packagePath}/${filePath}`;
}

function clearRobot() {
  if (state.robot) {
    scene.remove(state.robot);
    state.robot = null;
  }
  jointSlidersEl.innerHTML = "";
  state.jointSliders = [];
  jointPanel.hidden = true;
}

function clearSceneRoot() {
  if (state.sceneRoot) {
    scene.remove(state.sceneRoot);
    state.sceneRoot = null;
  }
}

function fitCameraToObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 0.35);
  const distance = maxDim * 2.2;

  controls.target.copy(center);
  camera.position.set(center.x + distance * 0.75, center.y + distance * 0.55, center.z + distance * 0.75);
  controls.update();
}

function buildJointSliders(robot) {
  jointSlidersEl.innerHTML = "";
  state.jointSliders = [];

  const joints = Object.entries(robot.joints || {})
    .filter(([, joint]) => joint.jointType === "revolute" || joint.jointType === "continuous")
    .sort(([a], [b]) => a.localeCompare(b));

  if (!joints.length) {
    jointPanel.hidden = true;
    return;
  }

  jointPanel.hidden = false;
  for (const [name, joint] of joints) {
    const row = document.createElement("label");
    row.className = "joint-row";

    const title = document.createElement("span");
    title.textContent = name;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.step = "0.01";
    const lower = Number.isFinite(joint.limit?.lower) ? joint.limit.lower : -3.14;
    const upper = Number.isFinite(joint.limit?.upper) ? joint.limit.upper : 3.14;
    slider.min = String(lower);
    slider.max = String(upper);
    slider.value = "0";

    const value = document.createElement("small");
    value.textContent = "0.00";

    slider.addEventListener("input", () => {
      const angle = Number(slider.value);
      joint.setJointValue(angle);
      value.textContent = angle.toFixed(2);
    });

    row.append(title, slider, value);
    jointSlidersEl.appendChild(row);
    state.jointSliders.push({ name, joint, slider });
  }
}

function countMeshes(object) {
  let total = 0;
  object.traverse((child) => {
    if (child.isMesh) total += 1;
  });
  return total;
}

function addFallbackLinkVisuals(robot) {
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
      geometry = new THREE.BoxGeometry(0.3, 0.24, 0.09);
    }

    const mesh = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({ color: 0x4a5d73, metalness: 0.12, roughness: 0.58 }),
    );
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    child.add(mesh);
  });
}

async function packageHasStlMeshes(packageInfo) {
  try {
    const response = await fetch(packageUrl(packageInfo.id, "cad/mesh/visual/cad_arm_base.stl"), { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}

async function loadRobot(packageInfo) {
  clearRobot();
  setStatus("Loading URDF meshes...");

  const loader = new URDFLoader();
  loader.loadMeshCb = (path, manager, onComplete) => {
    const normalized = path.replace(/\\/g, "/");
    const relativeToPackage = normalized.replace(/^(\.\.\/)+/, "");
    const meshUrl = packageUrl(packageInfo.id, relativeToPackage);
    const extension = normalized.split(".").pop().toLowerCase();
    if (extension === "stl") {
      import("three/addons/loaders/STLLoader.js").then(({ STLLoader }) => {
        const stlLoader = new STLLoader(manager);
        stlLoader.load(
          meshUrl,
          (geometry) => {
            const material = new THREE.MeshStandardMaterial({ color: 0x5f6670, metalness: 0.15, roughness: 0.55 });
            onComplete(new THREE.Mesh(geometry, material));
          },
          undefined,
          () => onComplete(null, new Error(`Failed to load mesh: ${meshUrl}`)),
        );
      });
      return;
    }
    onComplete(null, new Error(`Unsupported mesh type: ${extension}`));
  };

  const urdfUrl = packageUrl(packageInfo.id, packageInfo.urdf);
  const robot = await new Promise((resolve, reject) => {
    loader.load(
      urdfUrl,
      resolve,
      undefined,
      reject,
    );
  });

  robot.traverse((child) => {
    if (child.isMesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  const meshCount = countMeshes(robot);
  if (meshCount === 0) {
    addFallbackLinkVisuals(robot);
    setStatus(
      `Robot loaded with placeholder shapes (this package URDF has no STL meshes). Try Simulation scene mode for full 3D.`,
    );
  } else {
    setStatus(`Robot loaded from ${packageInfo.id} (${meshCount} mesh part(s))`);
  }

  scene.add(robot);
  state.robot = robot;
  buildJointSliders(robot);
  fitCameraToObject(robot);
}

async function loadScene(packageInfo, sceneInfo) {
  clearSceneRoot();
  const label = sceneInfo ? sceneInfo.scene_id : "mvp_scene";
  setStatus(`Loading MuJoCo scene: ${label}...`);

  const xmlPath = sceneInfo ? sceneInfo.mujoco_xml : packageInfo.mvp_scene;
  const response = await fetch(packageUrl(packageInfo.id, xmlPath));
  if (!response.ok) {
    throw new Error(`Could not load scene XML (${response.status}).`);
  }

  const xmlText = await response.text();
  const sceneRoot = buildSceneFromMuJoCoXml(xmlText);
  scene.add(sceneRoot);
  state.sceneRoot = sceneRoot;
  fitCameraToObject(sceneRoot);
  setStatus(`Scene loaded: ${label}`);
}

function updateSceneSelectState() {
  const hasScenes = Boolean(state.selectedPackage?.scenes?.length);
  sceneSelect.disabled = state.mode !== "scene" || !hasScenes;
}

function populateSceneSelect(packageInfo) {
  sceneSelect.innerHTML = "";
  const scenes = packageInfo.scenes || [];
  if (!scenes.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "mvp_scene.xml (fallback)";
    sceneSelect.appendChild(option);
    sceneSelect.disabled = true;
    return;
  }

  sceneSelect.disabled = false;
  for (const sceneInfo of scenes) {
    const option = document.createElement("option");
    option.value = sceneInfo.mujoco_xml;
    option.textContent = `${sceneInfo.purpose} · ${sceneInfo.scene_id}`;
    option.dataset.sceneId = sceneInfo.scene_id;
    sceneSelect.appendChild(option);
  }
  updateSceneSelectState();
}

function selectedSceneInfo(packageInfo) {
  const selected = sceneSelect.value;
  if (!selected) return null;
  return (packageInfo.scenes || []).find((item) => item.mujoco_xml === selected) || null;
}

async function reloadView() {
  const packageInfo = state.selectedPackage;
  if (!packageInfo) {
    setStatus("Pick a build package first.", true);
    return;
  }

  try {
    if (state.mode === "robot") {
      clearSceneRoot();
      await loadRobot(packageInfo);
    } else {
      clearRobot();
      await loadScene(packageInfo, selectedSceneInfo(packageInfo));
    }
  } catch (error) {
    console.error(error);
    setStatus(error.message || String(error), true);
  }
}

async function loadPackages() {
  try {
    const response = await fetch("/api/packages");
    if (!response.ok) {
      throw new Error(`Package list request failed (${response.status}). Is viewer/server.py running?`);
    }
    const payload = await response.json();
    state.packages = payload.packages || [];

    packageSelect.innerHTML = "";
    if (!state.packages.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No packages found under build/";
      packageSelect.appendChild(option);
      setStatus("No robot packages found. Run a Virturoid build first.", true);
      return;
    }

    for (const packageInfo of state.packages) {
      const option = document.createElement("option");
      option.value = packageInfo.id;
      option.textContent = `${packageInfo.id} (${packageInfo.scene_count} scenes)`;
      packageSelect.appendChild(option);
    }

    const params = new URLSearchParams(window.location.search);
    const requested = params.get("package");
    const requestedMode = params.get("mode");
    const preferred =
      state.packages.find((item) => item.id === requested) ||
      state.packages.find((item) => item.id === "cv_world_arm") ||
      state.packages[0];
    const match = preferred;
    packageSelect.value = match.id;
    state.selectedPackage = match;
    populateSceneSelect(match);

    if (requestedMode === "robot" || requestedMode === "scene") {
      state.mode = requestedMode;
      modeSelect.value = requestedMode;
    } else if (!(await packageHasStlMeshes(match))) {
      state.mode = "scene";
      modeSelect.value = "scene";
    }
    updateSceneSelectState();
    await reloadView();
  } catch (error) {
    console.error(error);
    setStatus(error.message || String(error), true);
  }
}

packageSelect.addEventListener("change", async () => {
  const packageInfo = state.packages.find((item) => item.id === packageSelect.value);
  state.selectedPackage = packageInfo || null;
  if (packageInfo) populateSceneSelect(packageInfo);
  await reloadView();
});

sceneSelect.addEventListener("change", async () => {
  if (state.mode === "scene") await reloadView();
});

modeSelect.addEventListener("change", async () => {
  state.mode = modeSelect.value;
  updateSceneSelectState();
  await reloadView();
});

reloadButton.addEventListener("click", reloadView);

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener("resize", resize);
resize();
animate();
loadPackages();
