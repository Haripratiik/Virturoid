import * as THREE from "three";

function parseNumbers(text, count) {
  const values = String(text || "")
    .trim()
    .split(/\s+/)
    .map(Number)
    .filter((value) => Number.isFinite(value));
  while (values.length < count) values.push(0);
  return values.slice(0, count);
}

function colorFromMaterial(materials, geomEl) {
  const rgba = geomEl.getAttribute("rgba");
  if (rgba) {
    const [r, g, b] = parseNumbers(rgba, 4);
    return new THREE.Color(r, g, b);
  }
  const materialName = geomEl.getAttribute("material");
  if (materialName && materials[materialName]) {
    return materials[materialName].clone();
  }
  return new THREE.Color(0.6, 0.6, 0.62);
}

function makeMaterial(materials, geomEl) {
  return new THREE.MeshStandardMaterial({
    color: colorFromMaterial(materials, geomEl),
    metalness: 0.08,
    roughness: 0.72,
  });
}

function addGeomMesh(group, geomEl, materials) {
  const type = geomEl.getAttribute("type") || "box";
  const material = makeMaterial(materials, geomEl);
  let mesh = null;

  if (type === "box") {
    const [sx, sy, sz] = parseNumbers(geomEl.getAttribute("size"), 3);
    mesh = new THREE.Mesh(new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2), material);
  } else if (type === "sphere") {
    const [radius] = parseNumbers(geomEl.getAttribute("size"), 1);
    mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 16), material);
  } else if (type === "cylinder") {
    const [radius] = parseNumbers(geomEl.getAttribute("size"), 1);
    const fromto = parseNumbers(geomEl.getAttribute("fromto"), 6);
    const start = new THREE.Vector3(fromto[0], fromto[1], fromto[2]);
    const end = new THREE.Vector3(fromto[3], fromto[4], fromto[5]);
    const length = start.distanceTo(end);
    mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, 24), material);
    const midpoint = start.clone().add(end).multiplyScalar(0.5);
    mesh.position.copy(midpoint);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.clone().sub(start).normalize());
  } else if (type === "plane") {
    const [sx, sy] = parseNumbers(geomEl.getAttribute("size"), 3);
    mesh = new THREE.Mesh(new THREE.PlaneGeometry(sx * 2, sy * 2), material);
    mesh.rotation.x = -Math.PI / 2;
  } else {
    return;
  }

  const pos = parseNumbers(geomEl.getAttribute("pos"), 3);
  if (type !== "cylinder") {
    mesh.position.set(pos[0], pos[1], pos[2]);
  }
  const euler = parseNumbers(geomEl.getAttribute("euler"), 3);
  if (euler.some((value) => value !== 0) && type !== "cylinder") {
    mesh.rotation.set(euler[0], euler[1], euler[2], "ZYX");
  }
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
}

function walkBody(bodyEl, parentGroup, materials) {
  const group = new THREE.Group();
  const pos = parseNumbers(bodyEl.getAttribute("pos"), 3);
  group.position.set(pos[0], pos[1], pos[2]);
  const euler = parseNumbers(bodyEl.getAttribute("euler"), 3);
  if (euler.some((value) => value !== 0)) {
    group.rotation.set(euler[0], euler[1], euler[2], "ZYX");
  }

  bodyEl.querySelectorAll(":scope > geom").forEach((geomEl) => addGeomMesh(group, geomEl, materials));
  bodyEl.querySelectorAll(":scope > body").forEach((child) => walkBody(child, group, materials));
  parentGroup.add(group);
}

export function buildSceneFromMuJoCoXml(xmlText) {
  const doc = new DOMParser().parseFromString(xmlText, "application/xml");
  const parserError = doc.querySelector("parsererror");
  if (parserError) {
    throw new Error("Could not parse MuJoCo XML.");
  }

  const materials = {};
  doc.querySelectorAll("asset > material").forEach((materialEl) => {
    const rgba = parseNumbers(materialEl.getAttribute("rgba"), 4);
    materials[materialEl.getAttribute("name")] = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  });

  const root = new THREE.Group();
  root.name = "mujoco_scene";
  root.rotation.x = -Math.PI / 2;

  const worldbody = doc.querySelector("worldbody");
  if (!worldbody) {
    throw new Error("MuJoCo XML has no <worldbody>.");
  }

  worldbody.querySelectorAll(":scope > geom").forEach((geomEl) => addGeomMesh(root, geomEl, materials));
  worldbody.querySelectorAll(":scope > body").forEach((bodyEl) => walkBody(bodyEl, root, materials));

  const grid = new THREE.GridHelper(4, 32, 0x8a9399, 0xd9dee2);
  grid.rotation.x = Math.PI / 2;
  root.add(grid);

  return root;
}
