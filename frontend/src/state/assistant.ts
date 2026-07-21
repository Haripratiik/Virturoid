import { create } from "zustand";

// One scrollable feed of typed entries (the Agent Rail heart).
export type FeedEntry =
  | { type: "user"; id: string; ts: number; text: string }
  | { type: "agent"; id: string; ts: number; text: string; streaming?: boolean }
  | { type: "run"; id: string; ts: number; jobId: string; goal: string }
  | { type: "recovery"; id: string; ts: number; title: string; detail: string; actions: Array<{ label: string; prompt: string }> }
  | { type: "tool"; id: string; ts: number; tool: string; args: Record<string, unknown>; ok?: boolean; result?: unknown };

export type ComposerMode = "build" | "ask";

interface AssistantState {
  feed: FeedEntry[];
  mode: ComposerMode;
  busy: boolean;
  selectedModel: string | null;
  setMode: (mode: ComposerMode) => void;
  setBusy: (busy: boolean) => void;
  setSelectedModel: (model: string | null) => void;
  push: (entry: FeedEntry) => void;
  patch: (id: string, patch: Partial<FeedEntry>) => void;
  clear: () => void;
}

let seq = 0;
export function feedId(): string {
  seq += 1;
  return `fe_${Date.now().toString(36)}_${seq}`;
}

export const useAssistantStore = create<AssistantState>()((set) => ({
  feed: [],
  mode: "build",
  busy: false,
  selectedModel: null,
  setMode: (mode) => set({ mode }),
  setBusy: (busy) => set({ busy }),
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  push: (entry) => set((s) => ({ feed: [...s.feed, entry] })),
  patch: (id, patch) =>
    set((s) => ({
      feed: s.feed.map((e) => (e.id === id ? ({ ...e, ...patch } as FeedEntry) : e)),
    })),
  clear: () => set({ feed: [] }),
}));
