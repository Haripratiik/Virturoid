import { useState } from "react";
import { Check, X, Circle, ChevronDown, ChevronRight, OctagonX, Loader2 } from "lucide-react";
import { useJobsStore, PHASES, phaseStates, phaseForStage } from "@/state/jobs";
import { cancelJob } from "@/api/jobs";
import { Pill } from "@/components/ui";
import type { PhaseId } from "@/state/jobs";
import type { JobEvent } from "@/api/types";

// A live run card: goal + five phase rows (Design → Build → Evaluate → Improve →
// Report), each expandable to its raw {stage, message, data} events. The raw
// backend stages stay honest; grouping happens here in the UI.

function PhaseIcon({ state }: { state: "pending" | "active" | "done" | "failed" }) {
  if (state === "done") return <Check size={12} className="text-agent" aria-label="done" />;
  if (state === "failed") return <X size={12} className="text-fail" aria-label="failed" />;
  if (state === "active")
    return <Loader2 size={12} className="vs-phase-active animate-spin rounded-full text-agent" aria-label="running" />;
  return <Circle size={10} className="text-muted" aria-label="pending" />;
}

function elapsedLabel(startedAt?: number | null, finishedAt?: number | null): string {
  if (!startedAt) return "";
  const end = finishedAt ?? Date.now() / 1000;
  const s = Math.max(0, Math.round(end - startedAt));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

export function RunCard({ jobId, goal }: { jobId: string; goal: string }) {
  const rec = useJobsStore((s) => s.jobs[jobId]);
  const [openPhase, setOpenPhase] = useState<PhaseId | null>(null);
  if (!rec) return null;
  const { job, events } = rec;
  const states = phaseStates(events, job.status);
  const running = job.status === "running" || job.status === "queued";
  const eventsByPhase = new Map<PhaseId, JobEvent[]>();
  for (const ev of events) {
    const p = phaseForStage(ev.stage);
    if (!eventsByPhase.has(p)) eventsByPhase.set(p, []);
    eventsByPhase.get(p)!.push(ev);
  }
  const result = (job.result ?? {}) as Record<string, unknown>;

  return (
    <div className="rounded-card border border-agent/30 bg-agent-dim/40 p-2.5">
      <div className="mb-1.5 flex items-start justify-between gap-2">
        <div className="text-xs font-medium text-primary">{goal}</div>
        <Pill
          kind={
            job.status === "succeeded" ? "ok" : job.status === "failed" ? "bad" : job.status === "cancelled" ? "warn" : "skip"
          }
          label={job.status}
        />
      </div>

      <div className="flex flex-col gap-0.5">
        {PHASES.map(({ id, label }) => {
          const phaseEvents = eventsByPhase.get(id) ?? [];
          const open = openPhase === id;
          return (
            <div key={id}>
              <button
                type="button"
                onClick={() => setOpenPhase(open ? null : id)}
                disabled={phaseEvents.length === 0}
                className="flex w-full items-center gap-2 rounded-ctl px-1.5 py-1 text-left hover:bg-raised/60 disabled:cursor-default disabled:hover:bg-transparent"
              >
                <PhaseIcon state={states[id]} />
                <span
                  className={`flex-1 text-xs ${states[id] === "pending" ? "text-muted" : states[id] === "active" ? "text-agent" : "text-secondary"}`}
                >
                  {label}
                </span>
                {phaseEvents.length > 0 && (
                  <>
                    <span className="font-mono text-2xs text-muted">{phaseEvents.length}</span>
                    {open ? (
                      <ChevronDown size={11} className="text-muted" aria-hidden />
                    ) : (
                      <ChevronRight size={11} className="text-muted" aria-hidden />
                    )}
                  </>
                )}
              </button>
              {open && phaseEvents.length > 0 && (
                <div className="ml-5 flex flex-col gap-1 border-l border-hairline py-1 pl-2.5">
                  {phaseEvents.map((ev) => (
                    <div key={ev.seq} className="font-mono text-2xs leading-relaxed text-secondary">
                      <span className="text-agent">[{ev.stage}]</span> {ev.message}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-2 flex items-center justify-between gap-2 border-t border-hairline pt-1.5">
        <span className="font-mono text-2xs text-muted">{elapsedLabel(job.started_at, job.finished_at)}</span>
        {running ? (
          <button
            type="button"
            onClick={() => void cancelJob(jobId)}
            className="inline-flex items-center gap-1 rounded-ctl px-1.5 py-0.5 text-2xs text-fail hover:bg-fail-dim"
          >
            <OctagonX size={11} aria-hidden />
            Cancel
          </button>
        ) : job.status === "succeeded" ? (
          <div className="flex flex-wrap items-center justify-end gap-1">
            {typeof result.robot_class === "string" && <Pill kind="skip" label={result.robot_class} />}
            {typeof result.final_success_rate === "number" && (
              <Pill
                kind={(result.final_success_rate as number) > 0.5 ? "ok" : "warn"}
                label={`success ${Math.round((result.final_success_rate as number) * 100)}%`}
              />
            )}
          </div>
        ) : null}
      </div>
      {job.error && <div className="mt-1.5 font-mono text-2xs text-fail">{job.error}</div>}
    </div>
  );
}
