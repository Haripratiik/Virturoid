import { Bot, User, Wrench } from "lucide-react";
import { useAppStore, useSelectionStore } from "@/state/app";
import { useBuildSummary } from "@/api/queries";
import { fmt } from "@/components/ui";

// Fusion-style timeline: every recorded decision from the build's honest trace
// (autonomous_build_summary.json decisions[]) as a scrubbable chip. Click a chip
// → inspect that decision (what changed, success before/after).

export function Timeline() {
  const activePackage = useAppStore((s) => s.activePackage);
  const summary = useBuildSummary(activePackage);
  const selection = useSelectionStore((s) => s.selection);
  const select = useSelectionStore((s) => s.select);
  const decisions = summary.data?.decisions ?? [];

  // No decisions -> no strip. An empty timeline is clutter, not information.
  if (!activePackage || decisions.length === 0) return null;

  const selectedIdx = selection?.kind === "decision" ? Number(selection.id) : -1;
  const selectedDecision = selectedIdx >= 0 ? decisions[selectedIdx] : null;

  return (
    <div className="border-t border-hairline bg-app">
      <div className="flex items-center gap-1.5 overflow-x-auto px-3 py-1.5">
        <span className="mr-1 shrink-0 text-2xs font-semibold uppercase tracking-wider text-muted">Timeline</span>
        {decisions.map((d, i) => {
          const isAgent = d.stage !== "manual_edit";
          const active = selectedIdx === i;
          return (
            <button
              key={`${d.stage}_${i}`}
              type="button"
              title={`${d.stage}: ${d.action}`}
              onClick={() => select(active ? null : { kind: "decision", id: String(i) })}
              className={`flex shrink-0 items-center gap-1 rounded-ctl border px-2 py-0.5 text-2xs transition-colors ${
                active
                  ? "border-agent/60 bg-agent-dim text-agent"
                  : "border-hairline bg-panel text-secondary hover:border-agent/40 hover:text-primary"
              }`}
            >
              {isAgent ? <Bot size={10} className="text-agent" aria-label="agent decision" /> : <User size={10} aria-label="manual edit" />}
              <span className="font-mono">{d.stage}</span>
            </button>
          );
        })}
      </div>
      {selectedDecision && (
        <div className="flex items-start gap-3 border-t border-hairline px-3 py-2">
          <Wrench size={12} className="mt-0.5 shrink-0 text-agent" aria-hidden />
          <div className="min-w-0 flex-1">
            <div className="text-xs text-primary">{selectedDecision.action}</div>
            <div className="mt-0.5 font-mono text-2xs text-muted">{selectedDecision.detail}</div>
          </div>
          <div className="shrink-0 text-right font-mono text-2xs text-muted">
            <div>iter {selectedDecision.iteration}</div>
            <div>
              success {fmt(selectedDecision.success_before)} → <span className="text-primary">{fmt(selectedDecision.success_after)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
