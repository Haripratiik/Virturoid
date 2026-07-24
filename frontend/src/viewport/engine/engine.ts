import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";
import { buildSceneFromMuJoCoXml } from "./mujocoScene";
import { packageUrl, tryText } from "@/api/client";
import type { EpisodeView } from "@/api/types";

// Imperative three.js engine — a careful port of webui/viewer3d.js (r160-pinned).
// This module is framework-free; ViewportCanvas.tsx is the only React boundary.
// Debugged conventions preserved exactly:
//   MuJoCo quat [w,x,y,z] -> three (x,y,z,w) · z-up -> y-up root rotations ·
//   capsule +Z axis · fromto endpoint math · floor-excluding camera framing ·
//   loadSeq stale-load guards · 30 fps episode data / 60 fps render decoupling.

export type ViewMode = "robot" | "scene" | "episode";

export interface EngineEvents {
  onStatus?: (message: string, isError: boolean) => void;
  onFrame?: (idx: number, total: number) => void;
  onPlayState?: (playing: boolean) => void;
  onRobot?: (robot: URDFRobot | null) => void;
  onEpisodeMeta?: (summary: string | null) => void;
  onSelect?: (objectName: string | null) => void;
}

interface EpisodeState {
  frames: number[][][];
  meshes: Array<THREE.Object3D | null>;
  idx: number;
  playing: boolean;
  frameMs: number; // episode data rate: 30 fps (render loop stays 60)
  lastTime: number;
}

const TASK_LABELS: Record<string, string> = {
  locomotion: "walk",
  navigation: "navigate",
  pick_place_sort: "sort",
  pick_place_box: "lift",
  stack: "stack",
  push: "push",
};

export class ViewportEngine {
  private renderer: THREE.WebGLRenderer | null = null;
  private scene: THREE.Scene | null = null;
  private camera!: THREE.PerspectiveCamera;
  private controls!: OrbitControls;
  private content!: THREE.Group;
  private grid: THREE.GridHelper | null = null;
  private axes: THREE.AxesHelper | null = null;
  private wireframeOn = false;
  private container: HTMLElement | null = null;
  private episode: EpisodeState | null = null;
  private currentObject: THREE.Object3D | null = null;
  private loadSeq = 0; // bumped on every (re)load; a stale async load aborts
  private raf = 0;
  private _sizeTmp = new THREE.Vector2(); // reused by resize()'s change-guard, no per-frame allocation
  private raycaster = new THREE.Raycaster();
  private highlighted: THREE.Mesh | null = null;
  private highlightPrev: THREE.Color | null = null;
  events: EngineEvents = {};
  robot: URDFRobot | null = null;

  get mounted(): boolean {
    return this.renderer !== null;
  }

  mount(container: HTMLElement): void {
    if (this.renderer) {
      // Re-attach the persistent canvas (viewport never re-initializes across workspace switches).
      if (this.renderer.domElement.parentElement !== container) {
        container.appendChild(this.renderer.domElement);
      }
      this.container = container;
      this.resize();
      return;
    }
    this.container = container;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";
    this.renderer = renderer;

    const scene = new THREE.Scene();
    this.scene = scene;
    scene.background = new THREE.Color(0x18181b); // --color-canvas (cool neutral)
    scene.fog = new THREE.Fog(0x18181b, 6, 22);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
    this.camera.position.set(1.1, 0.9, 1.3);

    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.target.set(0.25, 0.3, 0);
    // Blender-style navigation: left OR middle drag orbits, right drag or
    // Shift+drag pans (OrbitControls treats modifier+rotate as pan), wheel
    // zooms toward the pointer. Clamped so the camera never dives under the
    // floor or gets lost at infinity.
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.ROTATE,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.zoomSpeed = 0.5; // gentle wheel — tabletop robots are small subjects
    this.controls.rotateSpeed = 0.9;
    this.controls.minDistance = 0.05;
    this.controls.maxDistance = 40;
    this.controls.maxPolarAngle = Math.PI * 0.497;

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
      new THREE.MeshStandardMaterial({ color: 0x202024, roughness: 0.98, metalness: 0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.001;
    ground.receiveShadow = true;
    ground.userData.isFloor = true;
    scene.add(ground);

    const grid = new THREE.GridHelper(20, 40, 0x3d3d42, 0x2a2a2e);
    (grid.material as THREE.Material).opacity = 0.35;
    (grid.material as THREE.Material).transparent = true;
    grid.userData.isFloor = true;
    scene.add(grid);
    this.grid = grid;

    const axes = new THREE.AxesHelper(0.5);
    axes.visible = false; // off by default; toggled from the viewport toolbar
    axes.userData.isFloor = true;
    scene.add(axes);
    this.axes = axes;

    this.content = new THREE.Group();
    scene.add(this.content);

    renderer.domElement.addEventListener("pointerdown", this.onPointerDown);

    const animate = (now: number) => {
      this.raf = requestAnimationFrame(animate);
      // M2 (2026-07-24 audit): the GL buffer was freezing at a stale layout size (measured 136x436 while the
      // CSS box was 785x533), so the robot rendered stretched. resize() is now change-guarded (a no-op when the
      // size and DPR are unchanged), so calling it every frame GUARANTEES the buffer tracks the container even
      // if a ResizeObserver event is missed or fires before layout settles — it can never stay frozen.
      this.resize();
      this.controls.update();
      this.stepEpisode(now || 0);
      renderer.render(scene, this.camera);
    };
    animate(0);
    this.resize();
  }

  detach(): void {
    // Keep GL context + scene alive; just release the DOM slot.
    this.container = null;
  }

  dispose(): void {
    if (!this.renderer) return;
    cancelAnimationFrame(this.raf);
    this.renderer.domElement.removeEventListener("pointerdown", this.onPointerDown);
    this.renderer.dispose();
    this.renderer = null;
    this.container = null;
  }

  resize(): void {
    if (!this.renderer || !this.container) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    // Change-guarded so it's safe to call every frame: bail unless the CSS box or the device-pixel-ratio
    // actually changed (a monitor move / browser zoom changes DPR). setSize(w,h,false) keeps the canvas CSS at
    // 100% (the container drives display size) and sizes only the drawing BUFFER to w*h*DPR.
    const pr = Math.min(window.devicePixelRatio || 1, 2);
    const size = this.renderer.getSize(this._sizeTmp);
    if (size.x === w && size.y === h && this.renderer.getPixelRatio() === pr) return;
    if (this.renderer.getPixelRatio() !== pr) this.renderer.setPixelRatio(pr);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  clearContent(): void {
    if (!this.content) return;
    this.content.clear();
    this.robot = null;
    this.episode = null;
    this.highlighted = null;
    this.events.onRobot?.(null);
    this.events.onEpisodeMeta?.(null);
  }

  // ---- Viewport display toggles (frontend-only CAD affordances) ----

  get gridVisible(): boolean {
    return this.grid?.visible ?? true;
  }

  setGridVisible(visible: boolean): void {
    if (this.grid) this.grid.visible = visible;
  }

  get axesVisible(): boolean {
    return this.axes?.visible ?? false;
  }

  setAxesVisible(visible: boolean): void {
    if (this.axes) this.axes.visible = visible;
  }

  get wireframe(): boolean {
    return this.wireframeOn;
  }

  setWireframe(on: boolean): void {
    this.wireframeOn = on;
    this.applyWireframe();
  }

  /** Re-applied after every load so newly added meshes honor the toggle. */
  private applyWireframe(): void {
    this.content?.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const m of mats) {
        if ((m as THREE.MeshStandardMaterial).wireframe !== undefined) {
          (m as THREE.MeshStandardMaterial).wireframe = this.wireframeOn;
        }
      }
    });
  }

  /** Capture the current view as a PNG data URL (renders one fresh frame first). */
  screenshot(): string | null {
    if (!this.renderer || !this.scene) return null;
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }

  // ---- Selection (raycast click -> named ancestor -> Inspector sync) ----

  private onPointerDown = (e: PointerEvent) => {
    if (!this.renderer || !this.container || e.button !== 0) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);
    const hits = this.raycaster.intersectObjects(this.content.children, true);
    const hit = hits.find((h) => (h.object as THREE.Mesh).isMesh && !h.object.userData.isFloor);
    if (!hit) {
      this.setHighlight(null);
      this.events.onSelect?.(null);
      return;
    }
    let node: THREE.Object3D | null = hit.object;
    while (node && !node.name && node.parent !== this.content) node = node.parent;
    this.setHighlight(hit.object as THREE.Mesh);
    this.events.onSelect?.(node?.name || hit.object.name || null);
  };

  private setHighlight(mesh: THREE.Mesh | null): void {
    if (this.highlighted && this.highlightPrev) {
      const mat = this.highlighted.material as THREE.MeshStandardMaterial;
      if (mat?.emissive) mat.emissive.copy(this.highlightPrev);
    }
    this.highlighted = null;
    this.highlightPrev = null;
    if (mesh) {
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat?.emissive) {
        this.highlightPrev = mat.emissive.clone();
        mat.emissive.set(0x5a3a14); // ember-tinted emissive highlight (r160-safe)
        this.highlighted = mesh;
      }
    }
  }

  /** Highlight a link/body by name (Outliner -> viewport sync). */
  highlightByName(name: string | null): void {
    if (!this.content) return;
    if (!name) {
      this.setHighlight(null);
      return;
    }
    const node = this.content.getObjectByName(name);
    let mesh: THREE.Mesh | null = null;
    node?.traverse((c) => {
      if (!mesh && (c as THREE.Mesh).isMesh) mesh = c as THREE.Mesh;
    });
    this.setHighlight(mesh);
  }

  // ---- Camera ----

  frameContent(): void {
    if (this.currentObject) this.fitCamera(this.currentObject);
  }

  private fitCamera(object: THREE.Object3D): void {
    this.currentObject = object;
    object.updateMatrixWorld(true);
    // Frame the robot + task objects, NOT the ground plane/grid — fitting the whole
    // scene (incl. floor) dwarfs the robot and pulls the camera way back.
    const box = new THREE.Box3();
    object.traverse((child) => {
      if ((child as THREE.Mesh).isMesh && !child.userData.isFloor) box.expandByObject(child);
    });
    if (box.isEmpty()) box.setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.3);
    // Clamp so a rogue artifact can never push the camera into the fog.
    const dist = Math.min(maxDim * 2.2, 12);
    this.controls.target.copy(center);
    this.camera.position.set(center.x + dist * 0.8, center.y + dist * 0.6, center.z + dist * 0.8);
    this.controls.update();
  }

  setCameraPreset(preset: "front" | "side" | "top" | "iso"): void {
    if (!this.currentObject) return;
    this.currentObject.updateMatrixWorld(true);
    const box = new THREE.Box3();
    this.currentObject.traverse((child) => {
      if ((child as THREE.Mesh).isMesh && !child.userData.isFloor) box.expandByObject(child);
    });
    if (box.isEmpty()) box.setFromObject(this.currentObject);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const d = Math.max(size.x, size.y, size.z, 0.3) * 2.4;
    const positions: Record<string, [number, number, number]> = {
      front: [center.x, center.y + d * 0.15, center.z + d],
      side: [center.x + d, center.y + d * 0.15, center.z],
      top: [center.x, center.y + d, center.z + 0.001],
      iso: [center.x + d * 0.8, center.y + d * 0.6, center.z + d * 0.8],
    };
    const [x, y, z] = positions[preset];
    this.controls.target.copy(center);
    this.camera.position.set(x, y, z);
    this.controls.update();
  }

  // ---- Status plumbing ----

  private status(message: string, isError = false): void {
    this.events.onStatus?.(message, isError);
  }

  // ---- Robot (URDF) ----

  async loadRobot(pkg: string): Promise<void> {
    const myLoad = ++this.loadSeq;
    this.clearContent();
    this.status("Loading URDF...");

    const text = await tryText(packageUrl(pkg, "robot/robot.urdf"));
    if (myLoad !== this.loadSeq) return; // superseded by a newer load
    if (!text) {
      // Legged / engine-built packages have no URDF but do have a replayable episode.
      await this.loadEpisode(pkg, 0);
      return;
    }

    const loader = new URDFLoader();
    loader.workingPath = "";
    loader.loadMeshCb = (path, manager, onComplete) => {
      const normalized = String(path)
        .replace(/\\/g, "/")
        .replace(/^(\.\.\/)+/, "");
      const meshUrl = packageUrl(pkg, normalized);
      const ext = normalized.split(".").pop()?.toLowerCase();
      if (ext === "stl") {
        void import("three/examples/jsm/loaders/STLLoader.js").then(({ STLLoader }) => {
          new STLLoader(manager).load(
            meshUrl,
            (geometry) => {
              // CAD exporters write STL in MILLIMETERS; URDF origins are in
              // meters. A tabletop robot whose mesh spans >10 "meters" is a
              // unit mismatch, not a building-sized robot — rescale mm→m.
              geometry.computeBoundingBox();
              const bb = geometry.boundingBox!;
              const span = Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z);
              if (span > 10) geometry.scale(0.001, 0.001, 0.001);
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

    let robot: URDFRobot;
    try {
      robot = loader.parse(text);
    } catch (error) {
      this.status(`Failed to parse URDF: ${(error as Error).message || error}`, true);
      return;
    }

    robot.rotation.x = -Math.PI / 2; // URDF z-up -> three y-up
    robot.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    // STL meshes load asynchronously; give them a moment then assess.
    setTimeout(() => {
      if (myLoad !== this.loadSeq) return;
      let meshCount = 0;
      robot.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) meshCount += 1;
      });
      if (meshCount === 0) {
        const added = this.addFallbackVisuals(robot);
        this.status(`Robot loaded with ${added} placeholder shape(s) (URDF has no meshes). Try Scene mode for full 3D.`);
      } else {
        this.status(`Robot loaded (${meshCount} mesh part(s)).`);
      }
      this.fitCamera(robot);
    }, 250);

    if (myLoad !== this.loadSeq) return; // superseded while parsing
    this.content.add(robot);
    this.robot = robot;
    this.applyWireframe();
    this.events.onRobot?.(robot);
    this.fitCamera(robot);
  }

  private addFallbackVisuals(robot: URDFRobot): number {
    let added = 0;
    robot.traverse((child) => {
      if (!(child as unknown as { isURDFLink?: boolean }).isURDFLink) return;
      let hasMesh = false;
      child.traverse((sub) => {
        if ((sub as THREE.Mesh).isMesh) hasMesh = true;
      });
      if (hasMesh) return;
      const name = String(child.name || "").toLowerCase();
      let geometry: THREE.BufferGeometry;
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

  setJointValue(name: string, value: number): boolean {
    const joint = this.robot?.joints?.[name];
    if (!joint) return false;
    joint.setJointValue(value);
    return true;
  }

  // ---- Scene (MuJoCo XML) ----

  async loadScene(pkg: string, xmlRelPath: string): Promise<void> {
    const myLoad = ++this.loadSeq;
    this.clearContent();
    this.status(`Loading scene: ${xmlRelPath.split("/").pop()}...`);
    const xmlText = await tryText(packageUrl(pkg, xmlRelPath));
    if (myLoad !== this.loadSeq) return;
    if (!xmlText) {
      this.status("Could not load scene XML.", true);
      return;
    }
    try {
      const root = buildSceneFromMuJoCoXml(xmlText);
      this.content.add(root);
      this.applyWireframe();
      this.fitCamera(root);
      this.status(`Scene loaded: ${xmlRelPath.split("/").pop()}`);
    } catch (error) {
      this.status((error as Error).message || String(error), true);
    }
  }

  // ---- Episode replay (the real physics run, faithful frames) ----

  private episodeMaterial(g: { rgba?: number[] }): THREE.MeshStandardMaterial {
    const rgba = g.rgba || [0.6, 0.6, 0.62, 1];
    const alpha = rgba[3] === undefined ? 1 : rgba[3];
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      metalness: 0.1,
      roughness: 0.65,
      transparent: alpha < 1,
      opacity: alpha,
    });
  }

  private episodeGeomMesh(g: { type: string; size?: number[]; rgba?: number[] }): THREE.Mesh {
    const size = g.size || [0.05, 0.05, 0.05];
    const material = this.episodeMaterial(g);
    let geometry: THREE.BufferGeometry;
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

  private episodeMeshAsset(pkg: string, g: { mesh_uri?: string; mesh_scale?: number; rgba?: number[] },
                            loadSeq: number): THREE.Group {
    const holder = new THREE.Group();
    if (!g.mesh_uri) return holder;
    const material = this.episodeMaterial(g);
    const url = packageUrl(pkg, g.mesh_uri);
    void import("three/examples/jsm/loaders/STLLoader.js").then(({ STLLoader }) => {
      new STLLoader().load(
        url,
        (geometry) => {
          if (loadSeq !== this.loadSeq) return;
          const scale = g.mesh_scale ?? 0.001; // package STL is millimetres; MuJoCo uses the same scale.
          geometry.scale(scale, scale, scale);
          geometry.computeVertexNormals();
          const mesh = new THREE.Mesh(geometry, material);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          material.wireframe = this.wireframeOn;
          holder.add(mesh);
        },
        undefined,
        () => {
          // The physics episode remains valid if a browser cannot fetch a decorative asset.  Keep the holder so
          // its frame index stays aligned; the hidden collision primitive is deliberately not substituted here.
        },
      );
    });
    return holder;
  }

  private applyEpisodeFrame(idx: number): void {
    const ep = this.episode;
    const frame = ep?.frames[idx];
    if (!ep || !frame) return;
    const n = Math.min(ep.meshes.length, frame.length);
    for (let i = 0; i < n; i += 1) {
      const m = ep.meshes[i];
      const p = frame[i];
      if (!m || !p) continue;
      m.position.set(p[0], p[1], p[2]);
      m.quaternion.set(p[4], p[5], p[6], p[3]); // MuJoCo quat [w,x,y,z] -> three (x,y,z,w)
    }
  }

  private stepEpisode(now: number): void {
    const ep = this.episode;
    if (!ep || !ep.playing || ep.frames.length < 2) return;
    if (!ep.lastTime) ep.lastTime = now;
    if (now - ep.lastTime < ep.frameMs) return;
    ep.lastTime = now;
    ep.idx = (ep.idx + 1) % ep.frames.length;
    this.applyEpisodeFrame(ep.idx);
    this.events.onFrame?.(ep.idx, ep.frames.length);
  }

  async loadEpisode(pkg: string, sceneIndex = 0): Promise<void> {
    const myLoad = ++this.loadSeq;
    this.clearContent();
    this.status("Running the episode in physics...");
    let view: EpisodeView;
    try {
      const resp = await fetch(`/api/episode?package=${encodeURIComponent(pkg)}&scene=${sceneIndex}`);
      view = (await resp.json()) as EpisodeView;
    } catch (error) {
      this.status(`Episode request failed: ${(error as Error).message || error}`, true);
      return;
    }
    if (myLoad !== this.loadSeq) return; // a newer load started while we were fetching
    if (view.error) {
      this.status(view.error, true);
      return;
    }
    if (!view.frames || !view.frames.length) {
      this.status("Episode produced no frames.", true);
      return;
    }

    const group = new THREE.Group();
    group.rotation.x = -Math.PI / 2; // MuJoCo z-up -> three y-up
    const meshes: Array<THREE.Object3D | null> = [];
    (view.geoms || []).forEach((g) => {
      if (g.type === "plane") {
        meshes.push(null); // the viewport already draws a ground
        return;
      }
      const alpha = (g.rgba || [0.6, 0.6, 0.62, 1])[3] ?? 1;
      if (alpha <= 0) {
        meshes.push(null); // primitive collider hidden beneath a detailed visual mesh
        return;
      }
      const mesh = g.type === "mesh" ? this.episodeMeshAsset(pkg, g, myLoad) : this.episodeGeomMesh(g);
      group.add(mesh);
      meshes.push(mesh);
    });
    this.content.add(group);

    const oc = (view.outcome || {}) as Record<string, unknown>;
    const taskLabel = TASK_LABELS[view.task || ""] || view.task || "task";
    const summary =
      view.task === "locomotion"
        ? `${taskLabel} · ${oc.forward_m ?? "?"} m forward · ${oc.upright ? "upright" : "fell"}`
        : view.task === "navigation"
          ? `${taskLabel} · ${oc.status === "reached" ? "goal reached" : String(oc.status || "in progress")}`
          : `${taskLabel} · ${oc.placed_count ?? 0}/${oc.block_count ?? 0} placed`;

    this.applyWireframe();
    this.episode = { frames: view.frames, meshes, idx: 0, playing: true, frameMs: 1000 / 30, lastTime: 0 };
    this.applyEpisodeFrame(0);
    this.fitCamera(group);
    this.events.onPlayState?.(true);
    this.events.onFrame?.(0, view.frames.length);
    this.events.onEpisodeMeta?.(`${summary} · ${view.frame_count} frames`);
    this.status(`Episode: ${summary}`);
  }

  get playing(): boolean {
    return this.episode?.playing ?? false;
  }

  get frameCount(): number {
    return this.episode?.frames.length ?? 0;
  }

  play(): void {
    if (!this.episode) return;
    this.episode.playing = true;
    this.episode.lastTime = 0;
    this.events.onPlayState?.(true);
  }

  pause(): void {
    if (!this.episode) return;
    this.episode.playing = false;
    this.events.onPlayState?.(false);
  }

  togglePlay(): void {
    if (!this.episode) return;
    if (this.episode.playing) this.pause();
    else this.play();
  }

  seek(idx: number): void {
    if (!this.episode) return;
    this.episode.playing = false;
    this.episode.idx = Math.max(0, Math.min(idx, this.episode.frames.length - 1));
    this.applyEpisodeFrame(this.episode.idx);
    this.events.onPlayState?.(false);
    this.events.onFrame?.(this.episode.idx, this.episode.frames.length);
  }

  stepFrame(delta: number): void {
    if (!this.episode) return;
    const n = this.episode.frames.length;
    this.seek((((this.episode.idx + delta) % n) + n) % n);
  }
}

// Module singleton — the engine survives across workspace switches by design
// (the canvas re-attaches; GL context and loaded content are never rebuilt).
export const engine = new ViewportEngine();

// Test hook: automated UI checks (scripts/clickthrough.mjs) drive the mouse
// over the canvas and read camera state through this to prove that orbit,
// pan and zoom really move the camera.
(window as unknown as { __virturoidEngine?: ViewportEngine }).__virturoidEngine = engine;
