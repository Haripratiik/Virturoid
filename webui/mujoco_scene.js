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
  let selfPositioned = false; // fromto geoms set their own position + rotation from the endpoints

  if (type === "box") {
    const [sx, sy, sz] = parseNumbers(geomEl.getAttribute("size"), 3);
    mesh = new THREE.Mesh(new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2), material);
  } else if (type === "sphere") {
    const [radius] = parseNumbers(geomEl.getAttribute("size"), 1);
    mesh = new THREE.Mesh(new THREE.SphereGeometry(radius || 0.02, 24, 16), material);
  } else if (type === "ellipsoid") {
    const sz = parseNumbers(geomEl.getAttribute("size"), 3);
    mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), material);
    mesh.scale.set(sz[0] || 0.02, sz[1] || 0.02, sz[2] || 0.02);
  } else if (type === "cylinder" || type === "capsule") {
    // MuJoCo: either fromto (two endpoints) OR size=(radius, half-length) centred at pos, axis = local +Z.
    // capsule was previously UNHANDLED (fell through to return), so every capsule LINK of an arm vanished.
    const sizes = parseNumbers(geomEl.getAttribute("size"), 2);
    const radius = sizes[0] || 0.02;
    const fromtoAttr = geomEl.getAttribute("fromto");
    if (fromtoAttr) {
      const ft = parseNumbers(fromtoAttr, 6);
      const start = new THREE.Vector3(ft[0], ft[1], ft[2]);
      const end = new THREE.Vector3(ft[3], ft[4], ft[5]);
      const length = start.distanceTo(end) || 0.05;
      const geo = type === "capsule"
        ? new THREE.CapsuleGeometry(radius, length, 8, 16)
        : new THREE.CylinderGeometry(radius, radius, length, 24);
      mesh = new THREE.Mesh(geo, material);
      mesh.position.copy(start.clone().add(end).multiplyScalar(0.5));
      if (length > 0) {
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), end.clone().sub(start).normalize());
      }
      selfPositioned = true;
    } else {
      const halfLen = sizes[1] || 0.05;
      const geo = type === "capsule"
        ? new THREE.CapsuleGeometry(radius, halfLen * 2, 8, 16)
        : new THREE.CylinderGeometry(radius, radius, halfLen * 2, 24);
      geo.rotateX(Math.PI / 2); // three axis +Y -> MuJoCo capsule/cylinder axis +Z
      mesh = new THREE.Mesh(geo, material);
    }
  } else if (type === "plane") {
    const [sx, sy] = parseNumbers(geomEl.getAttribute("size"), 3);
    mesh = new THREE.Mesh(new THREE.PlaneGeometry((sx || 2) * 2, (sy || 2) * 2), material);
    mesh.rotation.x = -Math.PI / 2;
  } else {
    return;
  }

  if (!selfPositioned) {
    const pos = parseNumbers(geomEl.getAttribute("pos"), 3);
    mesh.position.set(pos[0], pos[1], pos[2]);
    const euler = parseNumbers(geomEl.getAttribute("euler"), 3);
    if (euler.some((value) => value !== 0)) {
      mesh.rotation.set(euler[0], euler[1], euler[2], "ZYX");
    }
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
  if (doc.querySelector("parsererror")) {
    throw new Error("Could not parse MuJoCo XML.");
  }

  const materials = {};
  doc.querySelectorAll("asset > material").forEach((materialEl) => {
    const rgba = parseNumbers(materialEl.getAttribute("rgba"), 4);
    materials[materialEl.getAttribute("name")] = new THREE.Color(rgba[0], rgba[1], rgba[2]);
  });

  const root = new THREE.Group();
  root.name = "mujoco_scene";
  // MuJoCo is z-up; rotate into three.js y-up.
  root.rotation.x = -Math.PI / 2;

  const worldbody = doc.querySelector("worldbody");
  if (!worldbody) {
    throw new Error("MuJoCo XML has no <worldbody>.");
  }

  worldbody.querySelectorAll(":scope > geom").forEach((geomEl) => addGeomMesh(root, geomEl, materials));
  worldbody.querySelectorAll(":scope > body").forEach((bodyEl) => walkBody(bodyEl, root, materials));

  const grid = new THREE.GridHelper(4, 32, 0x3a8a2e, 0x1c2630);
  grid.material.opacity = 0.4;
  grid.material.transparent = true;
  grid.rotation.x = Math.PI / 2;
  root.add(grid);

  return root;
}
