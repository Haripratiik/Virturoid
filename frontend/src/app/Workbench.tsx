import { Fragment } from "react";
import { Bot, SlidersHorizontal } from "lucide-react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { useAppStore, type PanelSide, type SidePanelId } from "@/state/app";
import { PanelShell } from "@/components/PanelShell";
import { AgentPanel } from "@/panels/agent/AgentPanel";
import { Inspector } from "@/panels/inspector/Inspector";

// The workbench body: [side stack?] [editor column] [side stack?].
// Side panels (Agent, Inspector) are resizable via drag handles AND movable —
// each can dock to either edge; two panels on the same edge stack vertically.

const PANEL_META: Record<SidePanelId, { title: string; icon: React.ReactNode; el: React.ReactNode }> = {
  agent: { title: "Agent", icon: <Bot size={13} className="text-agent" aria-hidden />, el: <AgentPanel /> },
  inspector: {
    title: "Inspector",
    icon: <SlidersHorizontal size={13} className="text-muted" aria-hidden />,
    el: <Inspector />,
  },
};

function useSidePanels(side: PanelSide): SidePanelId[] {
  const agentOpen = useAppStore((s) => s.agentOpen);
  const inspectorOpen = useAppStore((s) => s.inspectorOpen);
  const sides = useAppStore((s) => s.panelSides);
  const out: SidePanelId[] = [];
  if (inspectorOpen && sides.inspector === side) out.push("inspector");
  if (agentOpen && sides.agent === side) out.push("agent");
  return out;
}

function SideStack({ side }: { side: PanelSide }) {
  const panels = useSidePanels(side);
  const close = (id: SidePanelId) =>
    id === "agent" ? useAppStore.getState().setAgentOpen(false) : useAppStore.getState().setInspectorOpen(false);
  return (
    <PanelGroup direction="vertical" autoSaveId={`vs.side.${side}.v3`}>
      {panels.map((id, i) => (
        <Fragment key={id}>
          {i > 0 && <PanelResizeHandle className="h-px bg-hairline data-[resize-handle-state=hover]:bg-accent/60" />}
          <Panel id={id} order={i + 1} minSize={18}>
            <PanelShell id={id} title={PANEL_META[id].title} icon={PANEL_META[id].icon} onClose={() => close(id)}>
              {PANEL_META[id].el}
            </PanelShell>
          </Panel>
        </Fragment>
      ))}
    </PanelGroup>
  );
}

export function WorkbenchBody({ children }: { children: React.ReactNode }) {
  const left = useSidePanels("left");
  const right = useSidePanels("right");
  return (
    <PanelGroup direction="horizontal" autoSaveId="vs.workbench.h.v3">
      {left.length > 0 && (
        <>
          <Panel id="left" order={1} defaultSize={20} minSize={14} maxSize={40}>
            <SideStack side="left" />
          </Panel>
          <PanelResizeHandle className="w-px bg-hairline data-[resize-handle-state=hover]:bg-accent/60" />
        </>
      )}
      <Panel id="center" order={2} minSize={35}>
        {children}
      </Panel>
      {right.length > 0 && (
        <>
          <PanelResizeHandle className="w-px bg-hairline data-[resize-handle-state=hover]:bg-accent/60" />
          <Panel id="right" order={3} defaultSize={22} minSize={14} maxSize={40}>
            <SideStack side="right" />
          </Panel>
        </>
      )}
    </PanelGroup>
  );
}
