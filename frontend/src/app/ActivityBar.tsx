import { Bot, SlidersHorizontal, TerminalSquare, Command, GraduationCap } from "lucide-react";
import { useAppStore } from "@/state/app";
import { Tip } from "@/components/Explain";

// Activity bar — the thin icon rail on the far left, exactly like an IDE.
// Icons only, tooltips on hover, pressed state = the panel is visible.
// No labels, no numbering, no lifecycle preaching.

function RailButton({
  label,
  pressed,
  onClick,
  children,
}: {
  label: string;
  pressed: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Tip text={label} side="bottom">
      <button
        type="button"
        aria-label={label}
        aria-pressed={pressed}
        onClick={onClick}
        className={`relative flex h-10 w-10 items-center justify-center transition-colors ${
          pressed ? "text-primary" : "text-muted hover:text-secondary"
        }`}
      >
        {pressed && <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-accent" aria-hidden />}
        {children}
      </button>
    </Tip>
  );
}

export function ActivityBar() {
  const agentOpen = useAppStore((s) => s.agentOpen);
  const inspectorOpen = useAppStore((s) => s.inspectorOpen);
  const dockOpen = useAppStore((s) => s.dockOpen);
  const app = useAppStore.getState;

  return (
    <nav aria-label="Panels" className="flex w-11 shrink-0 flex-col items-center border-r border-hairline bg-app py-1">
      <RailButton
        label="Agent — describe what to build, in plain language (Ctrl+B)"
        pressed={agentOpen}
        onClick={() => app().setAgentOpen(!agentOpen)}
      >
        <Bot size={18} strokeWidth={1.8} aria-hidden />
      </RailButton>
      <RailButton
        label="Inspector — properties, anatomy, parts, sensors (Ctrl+I)"
        pressed={inspectorOpen}
        onClick={() => app().setInspectorOpen(!inspectorOpen)}
      >
        <SlidersHorizontal size={17} strokeWidth={1.8} aria-hidden />
      </RailButton>
      <RailButton
        label="Console panel — logs, jobs, files, raw data (Ctrl+J)"
        pressed={dockOpen}
        onClick={() => app().setDockOpen(!dockOpen)}
      >
        <TerminalSquare size={17} strokeWidth={1.8} aria-hidden />
      </RailButton>

      <span className="flex-1" />

      <RailButton
        label="Welcome tour — a two-minute walkthrough of the studio"
        pressed={false}
        onClick={() => app().setTourOpen(true)}
      >
        <GraduationCap size={17} strokeWidth={1.8} aria-hidden />
      </RailButton>
      <RailButton
        label="All commands — search everything the studio can do (Ctrl+K)"
        pressed={false}
        onClick={() => app().setPaletteOpen(true)}
      >
        <Command size={16} strokeWidth={1.8} aria-hidden />
      </RailButton>
    </nav>
  );
}
