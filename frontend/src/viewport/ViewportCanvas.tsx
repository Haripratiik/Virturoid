import { useEffect, useRef, useState } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Box,
  Focus,
  Video,
  Grid3x3,
  Axis3d,
  Camera,
  Maximize,
  CircleHelp,
  Bot,
  LibraryBig,
} from "lucide-react";
import { engine } from "./engine/engine";
import { useViewportStore } from "./viewportState";
import { useAppStore, useSelectionStore, log } from "@/state/app";
import { useSceneIndex } from "@/api/queries";
import { Btn } from "@/components/ui";
import { Tip } from "@/components/Explain";
import type { ViewMode } from "./engine/engine";

// The ONLY React <-> three.js boundary. The engine is a module singleton whose
// canvas re-attaches here; loaded content survives workspace switches.

export function ViewportCanvas({ showSceneStrip = false }: { showSceneStrip?: boolean }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const frameLabelRef = useRef<HTMLSpanElement>(null);
  const scrubRef = useRef<HTMLInputElement>(null);
  const activePackage = useAppStore((s) => s.activePackage);
  const mode = useViewportStore((s) => s.mode);
  const sceneXml = useViewportStore((s) => s.sceneXml);
  const status = useViewportStore((s) => s.status);
  const statusError = useViewportStore((s) => s.statusError);
  const playing = useViewportStore((s) => s.playing);
  const hasEpisode = useViewportStore((s) => s.hasEpisode);
  const frameCount = useViewportStore((s) => s.frameCount);
  const sceneIndex = useSceneIndex(activePackage);

  // Mount / re-attach the engine canvas.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    engine.mount(host);
    const vp = useViewportStore.getState();
    engine.events = {
      onStatus: (m, isErr) => {
        vp.setStatus(m, isErr);
        if (isErr) log("error", m);
      },
      onFrame: (idx, total) => {
        // Transient path: update DOM directly at 30 fps, no React re-render.
        if (frameLabelRef.current) frameLabelRef.current.textContent = `${idx + 1}/${total}`;
        if (scrubRef.current) scrubRef.current.value = String(idx);
      },
      onPlayState: (p) => vp.setPlaying(p),
      onEpisodeMeta: (meta) => vp.setEpisode(meta, meta ? engine.frameCount : 0),
      onSelect: (name) =>
        useSelectionStore.getState().select(name ? { kind: "link", id: name } : null),
    };
    const ro = new ResizeObserver(() => engine.resize());
    ro.observe(host);
    return () => {
      ro.disconnect();
      engine.detach();
    };
  }, []);

  // Load content when package / mode / scene changes.
  useEffect(() => {
    if (!activePackage) {
      engine.clearContent();
      return;
    }
    if (mode === "robot") void engine.loadRobot(activePackage);
    else if (mode === "episode") void engine.loadEpisode(activePackage, 0);
    else if (mode === "scene") {
      const xml = sceneXml ?? sceneIndex.data?.scenes?.[0]?.mujoco_xml;
      if (xml) void engine.loadScene(activePackage, xml);
    }
  }, [activePackage, mode, sceneXml, sceneIndex.data]);

  const setMode = (m: ViewMode) => useViewportStore.getState().setMode(m);
  const scenes = sceneIndex.data?.scenes ?? [];

  // Display toggles (frontend-only; the engine holds the truth).
  const [gridOn, setGridOn] = useState(true);
  const [axesOn, setAxesOn] = useState(false);
  const [wireOn, setWireOn] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  function saveScreenshot() {
    const url = engine.screenshot();
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activePackage ?? "viewport"}.png`;
    a.click();
    log("info", "Saved a viewport screenshot.");
  }

  const toggleBtn = (pressed: boolean) =>
    `flex h-6 items-center gap-1 rounded-ctl px-1.5 text-2xs transition-colors ${
      pressed ? "bg-raised text-primary" : "text-muted hover:text-secondary"
    }`;

  return (
    <div className="flex h-full flex-col overflow-hidden bg-canvas">
      {/* Viewport toolbar — the CAD header row (Blender/Isaac register).
          bg-panel: one level lighter than the tab strip above, so the two
          stacked bars read as tab chrome vs editor header, not a double bar. */}
      <div className="relative flex h-8 shrink-0 items-center gap-1.5 border-b border-hairline bg-panel px-2">
        <div className="flex h-6 overflow-hidden rounded-ctl border border-hairline">
          {(
            [
              { id: "robot", label: "Robot", Icon: Box, hint: "The robot's body on its own — pose it joint by joint from the Inspector" },
              { id: "scene", label: "Scene", Icon: Focus, hint: "The robot inside a physics test environment (table, objects, bins)" },
              { id: "episode", label: "Episode", Icon: Video, hint: "Replay of a real simulated attempt at the task, frame by frame" },
            ] as const
          ).map(({ id, label, Icon, hint }) => (
            <Tip key={id} text={hint}>
              <button
                type="button"
                onClick={() => setMode(id)}
                aria-pressed={mode === id}
                className={`flex h-full items-center gap-1.5 px-2.5 text-2xs transition-colors ${
                  mode === id ? "bg-accent-dim text-accent" : "text-secondary hover:bg-raised/60 hover:text-primary"
                }`}
              >
                <Icon size={12} aria-hidden />
                {label}
              </button>
            </Tip>
          ))}
        </div>
        {mode === "scene" && scenes.length > 0 && (
          <select
            aria-label="Scene"
            className="h-6 max-w-52 rounded-ctl border border-hairline bg-raised px-2 text-2xs text-secondary"
            value={sceneXml ?? scenes[0]?.mujoco_xml ?? ""}
            onChange={(e) => useViewportStore.getState().setSceneXml(e.target.value)}
          >
            {scenes.map((s) => (
              <option key={s.scene_id} value={s.mujoco_xml}>
                {s.purpose} · {s.scene_id.slice(-18)}
              </option>
            ))}
          </select>
        )}

        <span className="flex-1" />

        {/* Camera */}
        <div className="flex h-6 items-center gap-0.5 rounded-ctl border border-hairline px-0.5">
          {(["front", "side", "top", "iso"] as const).map((p) => (
            <Tip key={p} text={`Snap the camera to the ${p === "iso" ? "isometric (3/4)" : p} view`}>
              <button
                type="button"
                onClick={() => engine.setCameraPreset(p)}
                className="rounded-ctl px-1.5 py-0.5 font-mono text-2xs uppercase text-muted hover:bg-raised hover:text-primary"
              >
                {p}
              </button>
            </Tip>
          ))}
          <Tip text="Frame — center and zoom the camera on the robot (F)">
            <button
              type="button"
              aria-label="Frame content"
              onClick={() => engine.frameContent()}
              className="rounded-ctl p-1 text-muted hover:bg-raised hover:text-primary"
            >
              <Maximize size={12} aria-hidden />
            </button>
          </Tip>
        </div>

        <span className="h-4 w-px bg-hairline" aria-hidden />

        {/* Display toggles */}
        <Tip text="Show or hide the floor grid (each square is 0.5 m)">
          <button
            type="button"
            aria-pressed={gridOn}
            aria-label="Toggle grid"
            onClick={() => {
              engine.setGridVisible(!gridOn);
              setGridOn(!gridOn);
            }}
            className={toggleBtn(gridOn)}
          >
            <Grid3x3 size={12} aria-hidden />
          </button>
        </Tip>
        <Tip text="Show or hide the world axes (red X, green Y, blue Z)">
          <button
            type="button"
            aria-pressed={axesOn}
            aria-label="Toggle axes"
            onClick={() => {
              engine.setAxesVisible(!axesOn);
              setAxesOn(!axesOn);
            }}
            className={toggleBtn(axesOn)}
          >
            <Axis3d size={13} aria-hidden />
          </button>
        </Tip>
        <Tip text="Wireframe — see through the robot's surfaces to its structure">
          <button
            type="button"
            aria-pressed={wireOn}
            aria-label="Toggle wireframe"
            onClick={() => {
              engine.setWireframe(!wireOn);
              setWireOn(!wireOn);
            }}
            className={toggleBtn(wireOn)}
          >
            wire
          </button>
        </Tip>
        <Tip text="Save the current view as a PNG image">
          <button
            type="button"
            aria-label="Screenshot"
            onClick={saveScreenshot}
            className={toggleBtn(false)}
          >
            <Camera size={13} aria-hidden />
          </button>
        </Tip>

        <span className="h-4 w-px bg-hairline" aria-hidden />

        <Tip text="How to move around the 3D view">
          <button
            type="button"
            aria-label="Viewport help"
            aria-expanded={helpOpen}
            onClick={() => setHelpOpen((v) => !v)}
            className={toggleBtn(helpOpen)}
          >
            <CircleHelp size={13} aria-hidden />
          </button>
        </Tip>
        {helpOpen && (
          <div className="absolute right-2 top-full z-40 mt-1.5 w-64 rounded-card border border-hairline-strong bg-overlay p-3 shadow-xl">
            <div className="mb-2 text-2xs font-semibold uppercase tracking-wider text-secondary">Navigating the 3D view</div>
            {(
              [
                ["Drag / MMB drag", "orbit around the robot"],
                ["Right-drag / Shift+drag", "pan the view"],
                ["Scroll", "zoom toward the pointer"],
                ["Click a part", "select it — details open in the Inspector"],
                ["F", "frame the robot in view"],
                ["Space", "play / pause an episode"],
                ["← →", "step one frame at a time"],
              ] as const
            ).map(([key, what]) => (
              <div key={key} className="flex items-baseline justify-between gap-3 py-0.5">
                <span className="font-mono text-2xs text-primary">{key}</span>
                <span className="text-right text-2xs text-muted">{what}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="vs-brackets relative min-h-0 flex-1">
      <div ref={hostRef} className="absolute inset-0 [&>canvas]:!h-full [&>canvas]:!w-full" />

      {/* Status line */}
      {status && (
        <div
          className={`absolute left-3 ${hasEpisode && mode === "episode" ? "bottom-14" : "bottom-3"} max-w-[70%] rounded-ctl border border-hairline bg-panel/80 px-2.5 py-1 text-xs backdrop-blur ${
            statusError ? "text-fail" : "text-secondary"
          }`}
          role="status"
        >
          {status}
        </div>
      )}

      {/* Empty state — quiet, points at the two real starting points. */}
      {!activePackage && (
        <div className="vs-dotfield absolute inset-0 flex flex-col items-center justify-center gap-4 p-6">
          <div className="text-center">
            <div className="vs-display text-xl font-semibold text-primary">Nothing on the stage yet</div>
            <div className="mx-auto mt-1.5 max-w-96 text-xs leading-relaxed text-muted">
              Describe a robot to the agent and it will be designed, simulated and verified — or open one you
              already built.
            </div>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                useAppStore.getState().setAgentOpen(true);
                requestAnimationFrame(() => document.getElementById("agent-composer")?.focus());
              }}
              className="flex items-center gap-1.5 rounded-ctl border border-agent/40 bg-agent-dim px-3 py-1.5 text-xs font-medium text-agent hover:bg-agent/20"
            >
              <Bot size={13} aria-hidden />
              Ask the agent
            </button>
            <button
              type="button"
              onClick={() => useAppStore.getState().setWorkspace("library")}
              className="flex items-center gap-1.5 rounded-ctl border border-hairline bg-panel px-3 py-1.5 text-xs text-secondary hover:text-primary"
            >
              <LibraryBig size={13} aria-hidden />
              Browse the Library
            </button>
          </div>
        </div>
      )}

      {/* Transport bar (episode replay) */}
      {mode === "episode" && hasEpisode && (
        <div className="absolute bottom-3 left-1/2 flex w-[min(62%,540px)] -translate-x-1/2 items-center gap-2.5 rounded-panel border border-hairline bg-panel/90 px-3 py-2 backdrop-blur">
          <Btn variant="ghost" onClick={() => engine.stepFrame(-1)} title="Step back (←)">
            <SkipBack size={13} aria-hidden />
          </Btn>
          <Btn variant="ghost" onClick={() => engine.togglePlay()} title="Play/Pause (Space)">
            {playing ? <Pause size={14} aria-hidden /> : <Play size={14} aria-hidden />}
          </Btn>
          <Btn variant="ghost" onClick={() => engine.stepFrame(1)} title="Step forward (→)">
            <SkipForward size={13} aria-hidden />
          </Btn>
          <input
            ref={scrubRef}
            type="range"
            min={0}
            max={Math.max(frameCount - 1, 0)}
            defaultValue={0}
            step={1}
            aria-label="Episode frame"
            className="vs-scrub flex-1"
            onInput={(e) => engine.seek(Number((e.target as HTMLInputElement).value))}
          />
          <span ref={frameLabelRef} className="min-w-14 text-right font-mono text-2xs text-muted" />
        </div>
      )}
      </div>

      {/* Scene strip (Simulate workspace) */}
      {showSceneStrip && scenes.length > 0 && (
        <div className="flex shrink-0 gap-1.5 overflow-x-auto border-t border-hairline bg-app px-2 py-1.5">
          {scenes.map((s) => (
            <button
              key={s.scene_id}
              type="button"
              onClick={() => {
                useViewportStore.getState().setMode("scene");
                useViewportStore.getState().setSceneXml(s.mujoco_xml);
              }}
              className={`shrink-0 rounded-ctl border px-2.5 py-1 text-left transition-colors ${
                sceneXml === s.mujoco_xml && mode === "scene"
                  ? "border-accent/50 bg-accent-dim text-accent"
                  : "border-hairline bg-panel text-secondary hover:text-primary"
              }`}
            >
              <div className="text-2xs font-semibold uppercase tracking-wide">{s.purpose}</div>
              <div className="font-mono text-2xs text-muted">
                {s.object_count ?? "?"} objects
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
