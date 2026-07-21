import { registerCommands } from "./registry";
import { useAppStore, WORKSPACES } from "@/state/app";
import { engine } from "@/viewport/engine/engine";
import { useViewportStore } from "@/viewport/viewportState";
import type { ToolSpec } from "@/api/types";

// Built-in commands. The palette, keyboard shortcuts and UI buttons all route
// through these — one action bus (manual parity by construction).

export function registerBuiltins(): () => void {
  const app = () => useAppStore.getState();
  return registerCommands([
    ...WORKSPACES.map((ws) => ({
      id: `workspace.${ws.id}`,
      title: `Go to ${ws.label}`,
      keywords: `workspace ${ws.id}`,
      section: "Workspaces",
      shortcut: `Ctrl+${ws.shortcut}`,
      run: () => app().setWorkspace(ws.id),
    })),
    {
      id: "view.toggleAgent",
      title: "Toggle Agent panel",
      section: "View",
      shortcut: "Ctrl+B",
      run: () => app().setAgentOpen(!app().agentOpen),
    },
    {
      id: "view.toggleInspector",
      title: "Toggle Inspector panel",
      section: "View",
      shortcut: "Ctrl+I",
      run: () => app().setInspectorOpen(!app().inspectorOpen),
    },
    {
      id: "view.toggleDock",
      title: "Toggle console panel",
      section: "View",
      shortcut: "Ctrl+J",
      run: () => app().setDockOpen(!app().dockOpen),
    },
    {
      id: "help.tour",
      title: "Open the welcome tour",
      keywords: "beginner novice onboarding tutorial help intro",
      section: "Help",
      run: () => app().setTourOpen(true),
    },
    {
      id: "agent.focusComposer",
      title: "Focus the agent composer",
      section: "Agent",
      shortcut: "Ctrl+L",
      run: () => {
        app().setAgentOpen(true);
        requestAnimationFrame(() => document.getElementById("agent-composer")?.focus());
      },
    },
    {
      id: "viewport.frame",
      title: "Frame content in viewport",
      section: "Viewport",
      shortcut: "F",
      run: () => engine.frameContent(),
    },
    {
      id: "viewport.playPause",
      title: "Play / pause episode",
      section: "Viewport",
      shortcut: "Space",
      run: () => engine.togglePlay(),
    },
    {
      id: "viewport.modeRobot",
      title: "Viewport: Robot mode",
      section: "Viewport",
      run: () => useViewportStore.getState().setMode("robot"),
    },
    {
      id: "viewport.modeScene",
      title: "Viewport: Scene mode",
      section: "Viewport",
      run: () => useViewportStore.getState().setMode("scene"),
    },
    {
      id: "viewport.modeEpisode",
      title: "Viewport: Episode replay",
      section: "Viewport",
      run: () => useViewportStore.getState().setMode("episode"),
    },
    ...(["console", "jobs", "artifacts", "data"] as const).map((tab) => ({
      id: `dock.${tab}`,
      title: `Open ${tab} dock`,
      section: "Dock",
      run: () => app().setDockTab(tab),
    })),
  ]);
}

/** One command per backend tool, auto-registered from GET /api/tools. */
export function registerToolCommands(tools: ToolSpec[]): () => void {
  return registerCommands(
    tools.map((t) => ({
      id: `tool.${t.name}`,
      title: `Tool: ${t.name}${t.heavy ? " (slow, real physics)" : ""}`,
      keywords: t.description,
      section: "Agent tools",
      run: () => {
        requestAnimationFrame(() => {
          const el = document.getElementById("agent-composer") as HTMLTextAreaElement | null;
          if (el) {
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value")?.set;
            setter?.call(el, `/${t.name} `);
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.focus();
          }
        });
      },
    })),
  );
}
