import { create } from "zustand";
import { persist } from "zustand/middleware";

export type WorkspaceId = "design" | "simulate" | "train" | "verify" | "library" | "memory";

// Workbench editor tabs — one per stage of the robot lifecycle, plus Library.
// Rendered as IDE-style tabs above the editor area (never preached elsewhere).
export const WORKSPACES: Array<{ id: WorkspaceId; label: string; shortcut: string; hint: string }> = [
  { id: "design", label: "Design", shortcut: "1", hint: "Shape the robot — 3D view, joints, parts and the build form" },
  { id: "simulate", label: "Simulate", shortcut: "2", hint: "Watch the robot act inside physics test scenes" },
  { id: "train", label: "Train", shortcut: "3", hint: "Teach the robot its task through trial and error (RL)" },
  { id: "verify", label: "Verify", shortcut: "4", hint: "Honesty gates — what is proven vs. only claimed" },
  { id: "library", label: "Library", shortcut: "5", hint: "Every robot you have built, searchable" },
  { id: "memory", label: "Memory", shortcut: "6", hint: "The verified-morphology bank — what is banked, under which gate, and whether recalling it actually helped" },
];

export type DockTab = "console" | "jobs" | "artifacts" | "data";

// Dockable side panels — each can live on either edge and is user-resizable.
export type SidePanelId = "agent" | "inspector";
export type PanelSide = "left" | "right";

export interface LogEntry {
  ts: number;
  kind: "info" | "warn" | "error" | "agent";
  message: string;
}

interface AppState {
  workspace: WorkspaceId;
  agentOpen: boolean;
  inspectorOpen: boolean;
  dockOpen: boolean;
  dockTab: DockTab;
  panelSides: Record<SidePanelId, PanelSide>;
  tourSeen: boolean; // welcome tour dismissed — never auto-shows again
  tourOpen: boolean;
  paletteOpen: boolean;
  activePackage: string | null;
  log: LogEntry[];
  setWorkspace: (ws: WorkspaceId) => void;
  setAgentOpen: (open: boolean) => void;
  setInspectorOpen: (open: boolean) => void;
  setDockOpen: (open: boolean) => void;
  setDockTab: (tab: DockTab) => void;
  movePanel: (panel: SidePanelId, side: PanelSide) => void;
  setTourOpen: (open: boolean) => void;
  setPaletteOpen: (open: boolean) => void;
  setActivePackage: (id: string | null) => void;
  appendLog: (kind: LogEntry["kind"], message: string) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      workspace: "design",
      agentOpen: true,
      inspectorOpen: true,
      dockOpen: true,
      dockTab: "console",
      panelSides: { agent: "right", inspector: "left" },
      tourSeen: false,
      tourOpen: false,
      paletteOpen: false,
      activePackage: null,
      log: [],
      setWorkspace: (workspace) => set({ workspace }),
      setAgentOpen: (agentOpen) => set({ agentOpen }),
      setInspectorOpen: (inspectorOpen) => set({ inspectorOpen }),
      setDockOpen: (dockOpen) => set({ dockOpen }),
      setDockTab: (dockTab) => set({ dockTab, dockOpen: true }),
      movePanel: (panel, side) =>
        set((s) => ({ panelSides: { ...s.panelSides, [panel]: side } })),
      // Closing the tour marks it seen forever; reopening is always manual.
      setTourOpen: (tourOpen) => set(tourOpen ? { tourOpen } : { tourOpen, tourSeen: true }),
      setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
      setActivePackage: (activePackage) => set({ activePackage }),
      appendLog: (kind, message) =>
        set((s) => ({ log: [...s.log.slice(-499), { ts: Date.now(), kind, message }] })),
    }),
    {
      name: "virturoid.studio.v3",
      partialize: (s) => ({
        workspace: s.workspace,
        agentOpen: s.agentOpen,
        inspectorOpen: s.inspectorOpen,
        dockOpen: s.dockOpen,
        panelSides: s.panelSides,
        tourSeen: s.tourSeen,
        activePackage: s.activePackage,
      }),
    },
  ),
);

export function log(kind: LogEntry["kind"], message: string): void {
  useAppStore.getState().appendLog(kind, message);
}

// ---- Selection (select → inspect → edit backbone) ----

export type SelectionKind = "package" | "link" | "joint" | "sensor" | "scene" | "claim" | "decision";

export interface Selection {
  kind: SelectionKind;
  id: string;
  context?: Record<string, unknown>;
}

interface SelectionState {
  selection: Selection | null;
  select: (sel: Selection | null) => void;
}

export const useSelectionStore = create<SelectionState>()((set) => ({
  selection: null,
  select: (selection) => set({ selection }),
}));
