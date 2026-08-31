import { create } from "zustand";
import type { Job, JobEvent, JobStatus } from "@/api/types";
import type { VerdictKind } from "@/components/ui";

// ---- Job verdict: ONE derivation of the chip every job surface renders ----
// The backend decides the verdict (services/job_registry.py): a build that produced no robot
// package finishes as `no_output`, never as a green `succeeded`. An honest refusal is a legitimate
// outcome, so it is deliberately NOT rendered as a failure.

const JOB_STATUS_LABEL: Record<JobStatus, string> = {
  queued: "queued",
  running: "running",
  succeeded: "succeeded",
  no_output: "no robot built",
  failed: "failed",
  cancelled: "cancelled",
};

const JOB_STATUS_KIND: Record<JobStatus, VerdictKind> = {
  queued: "skip",
  running: "skip",
  succeeded: "ok",
  no_output: "warn",
  failed: "bad",
  cancelled: "warn",
};

export function jobStatusLabel(status: Job["status"]): string {
  return JOB_STATUS_LABEL[status] ?? status;
}

export function jobStatusKind(status: Job["status"]): VerdictKind {
  return JOB_STATUS_KIND[status] ?? "skip";
}

// The backend emits ~18 raw honest stages; the UI groups them into 5 phases.
// Mapping lives in the FRONTEND so the backend event stream stays raw and honest.
export type PhaseId = "design" | "build" | "evaluate" | "improve" | "report";

export const PHASES: Array<{ id: PhaseId; label: string }> = [
  { id: "design", label: "Design" },
  { id: "build", label: "Build" },
  { id: "evaluate", label: "Evaluate" },
  { id: "improve", label: "Improve" },
  { id: "report", label: "Report" },
];

const STAGE_TO_PHASE: Record<string, PhaseId> = {
  start: "design",
  species: "design",
  task_graph: "design",
  ai_design: "design",
  recommend_parts: "design",
  build: "build",
  buildability: "build",
  memory_warm_start: "build",
  evaluate: "evaluate",
  evaluate_done: "evaluate",
  navigate: "evaluate",
  co_design: "improve",
  analyze_failures: "improve",
  keystone: "improve",
  redesign: "improve",
  train_policy: "improve",
  reward: "improve",
  done: "report",
  error: "report",
};

const PHASE_KEYWORDS: Array<[PhaseId, RegExp]> = [
  ["design", /design|species|task|part|propose/i],
  ["build", /build|compil|assembl|feasib|buildab/i],
  ["evaluate", /evaluat|navigat|physics|episode|run/i],
  ["improve", /improve|redesign|co.?design|tune|train|reward|keystone|failure/i],
  ["report", /done|report|export|summary/i],
];

export function phaseForStage(stage: string): PhaseId {
  const direct = STAGE_TO_PHASE[stage];
  if (direct) return direct;
  for (const [phase, re] of PHASE_KEYWORDS) {
    if (re.test(stage)) return phase;
  }
  return "design";
}

export type PhaseState = "pending" | "active" | "done" | "failed";

export function phaseStates(events: JobEvent[], jobStatus: Job["status"]): Record<PhaseId, PhaseState> {
  const states: Record<PhaseId, PhaseState> = {
    design: "pending",
    build: "pending",
    evaluate: "pending",
    improve: "pending",
    report: "pending",
  };
  const order: PhaseId[] = ["design", "build", "evaluate", "improve", "report"];
  let lastIdx = -1;
  let sawDone = false;
  for (const ev of events) {
    if (ev.stage === "done") sawDone = true;
    const idx = order.indexOf(phaseForStage(ev.stage));
    if (idx > lastIdx) lastIdx = idx;
  }
  for (let i = 0; i <= lastIdx; i += 1) states[order[i]] = "done";
  const finished = jobStatus === "succeeded" || jobStatus === "no_output" || sawDone;
  if (!finished && lastIdx >= 0 && (jobStatus === "running" || jobStatus === "queued")) {
    states[order[lastIdx]] = "active";
  }
  if (finished) for (const p of order) if (states[p] === "active") states[p] = "done";
  // Only a job that actually delivered gets its Report row ticked — a refused build stops where it
  // stopped, which is exactly what the unticked Build/Evaluate/Improve rows already showed.
  if (jobStatus === "succeeded") states.report = "done";
  if (jobStatus === "failed" && lastIdx >= 0) states[order[lastIdx]] = "failed";
  return states;
}

// ---- Jobs store (feeds are state, not query cache) ----

export interface JobRecord {
  job: Job;
  events: JobEvent[];
}

interface JobsState {
  jobs: Record<string, JobRecord>;
  order: string[]; // newest first
  trackJob: (job: Job) => void;
  applyUpdate: (job: Job, newEvents: JobEvent[]) => void;
  activeJobIds: () => string[];
}

export const useJobsStore = create<JobsState>()((set, get) => ({
  jobs: {},
  order: [],
  trackJob: (job) =>
    set((s) => {
      if (s.jobs[job.id]) return s;
      return {
        jobs: { ...s.jobs, [job.id]: { job, events: [] } },
        order: [job.id, ...s.order],
      };
    }),
  applyUpdate: (job, newEvents) =>
    set((s) => {
      const rec = s.jobs[job.id] ?? { job, events: [] };
      const merged = newEvents.length ? [...rec.events, ...newEvents] : rec.events;
      return {
        jobs: { ...s.jobs, [job.id]: { job, events: merged } },
        order: s.order.includes(job.id) ? s.order : [job.id, ...s.order],
      };
    }),
  activeJobIds: () =>
    get().order.filter((id) => {
      const st = get().jobs[id]?.job.status;
      return st === "queued" || st === "running";
    }),
}));
