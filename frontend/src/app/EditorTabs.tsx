import { PencilRuler, Orbit, GraduationCap, ShieldCheck, LibraryBig } from "lucide-react";
import { useAppStore, WORKSPACES, type WorkspaceId } from "@/state/app";
import { useLifecycle } from "./useLifecycle";
import { Tip } from "@/components/Explain";

// Editor tabs — the center area's views, styled like an IDE's tab strip.
// The tiny dot on a tab is live state from real artifacts (green = this stage
// has verified output, amber = needs attention), like a modified-file marker.

const TAB_ICONS: Record<WorkspaceId, typeof PencilRuler> = {
  design: PencilRuler,
  simulate: Orbit,
  train: GraduationCap,
  verify: ShieldCheck,
  library: LibraryBig,
};

export function EditorTabs() {
  const workspace = useAppStore((s) => s.workspace);
  const setWorkspace = useAppStore((s) => s.setWorkspace);
  const lifecycle = useLifecycle();

  return (
    <div role="tablist" aria-label="Editor views" className="flex h-9 shrink-0 items-end border-b border-hairline bg-app">
      {WORKSPACES.map((ws) => {
        const Icon = TAB_ICONS[ws.id];
        const active = workspace === ws.id;
        const st = lifecycle[ws.id];
        const dot = st.state === "done" ? "bg-ok" : st.state === "attention" ? "bg-warn" : null;
        return (
          <Tip key={ws.id} text={`${ws.hint} (Ctrl+${ws.shortcut})`}>
            <button
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setWorkspace(ws.id)}
              className={`relative flex h-9 items-center gap-1.5 border-r border-hairline px-3.5 text-xs transition-colors ${
                active
                  ? "bg-canvas text-primary"
                  : "bg-transparent text-muted hover:bg-panel hover:text-secondary"
              }`}
            >
              {active && <span className="absolute inset-x-0 top-0 h-[2px] bg-accent" aria-hidden />}
              <Icon size={13} strokeWidth={1.8} aria-hidden />
              {ws.label}
              {dot && <span className={`ml-0.5 h-1.5 w-1.5 rounded-full ${dot}`} title={st.hint} aria-label={st.hint} />}
            </button>
          </Tip>
        );
      })}
      <div className="flex-1 border-b-0" />
    </div>
  );
}
