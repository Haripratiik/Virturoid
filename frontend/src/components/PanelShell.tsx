import type { ReactNode } from "react";
import { X, PanelLeftClose, PanelRightClose } from "lucide-react";
import { useAppStore, type SidePanelId } from "@/state/app";
import { Tip } from "./Explain";

// Uniform chrome for dockable side panels: a slim header with the panel name,
// optional extra actions, a "move to other side" control and a close control.
// Every side panel is resizable (react-resizable-panels) AND movable (dock to
// either edge) — this is where the movable half lives.

export function PanelShell({
  id,
  title,
  icon,
  actions,
  children,
  onClose,
}: {
  id: SidePanelId;
  title: string;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  onClose: () => void;
}) {
  const side = useAppStore((s) => s.panelSides[id]);
  const movePanel = useAppStore((s) => s.movePanel);
  const other = side === "left" ? "right" : "left";
  const MoveIcon = side === "left" ? PanelRightClose : PanelLeftClose;
  return (
    <section className="flex h-full min-h-0 flex-col bg-panel" aria-label={title}>
      <header className="flex h-8 shrink-0 items-center gap-1.5 border-b border-hairline bg-app px-2.5">
        {icon}
        <span className="text-2xs font-semibold uppercase tracking-wider text-secondary">{title}</span>
        <span className="flex-1" />
        {actions}
        <Tip text={`Dock this panel on the ${other} side`}>
          <button
            type="button"
            aria-label={`Move ${title} to the ${other} side`}
            onClick={() => movePanel(id, other)}
            className="rounded-ctl p-1 text-muted hover:bg-raised hover:text-primary"
          >
            <MoveIcon size={13} aria-hidden />
          </button>
        </Tip>
        <Tip text="Hide this panel (reopen from the icon bar on the far left)">
          <button
            type="button"
            aria-label={`Close ${title}`}
            onClick={onClose}
            className="rounded-ctl p-1 text-muted hover:bg-raised hover:text-primary"
          >
            <X size={13} aria-hidden />
          </button>
        </Tip>
      </header>
      <div className="min-h-0 flex-1 overflow-hidden">{children}</div>
    </section>
  );
}
