import { create } from "zustand";
import type { ViewMode } from "./engine/engine";

// HUD/UI-facing viewport state. High-frequency playback (frame index while
// playing) is pushed straight to a DOM node via engine callbacks so 30 fps
// playback never re-renders React (transient subscription pattern).

interface ViewportState {
  mode: ViewMode;
  sceneXml: string | null;
  status: string;
  statusError: boolean;
  playing: boolean;
  hasEpisode: boolean;
  frameCount: number;
  episodeMeta: string | null;
  setMode: (mode: ViewMode) => void;
  setSceneXml: (xml: string | null) => void;
  setStatus: (status: string, isError: boolean) => void;
  setPlaying: (playing: boolean) => void;
  setEpisode: (meta: string | null, frameCount: number) => void;
}

export const useViewportStore = create<ViewportState>()((set) => ({
  mode: "robot",
  sceneXml: null,
  status: "",
  statusError: false,
  playing: false,
  hasEpisode: false,
  frameCount: 0,
  episodeMeta: null,
  setMode: (mode) => set({ mode }),
  setSceneXml: (sceneXml) => set({ sceneXml }),
  setStatus: (status, statusError) => set({ status, statusError }),
  setPlaying: (playing) => set({ playing }),
  setEpisode: (episodeMeta, frameCount) =>
    set({ episodeMeta, frameCount, hasEpisode: frameCount > 0 }),
}));
