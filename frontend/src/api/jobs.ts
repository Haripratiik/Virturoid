import { getJSON, postJSON } from "./client";
import type { Job, JobEventsResponse } from "./types";
import { useJobsStore } from "@/state/jobs";
import { log } from "@/state/app";

// Live job feeds are Zustand state (not query cache): a 1 Hz poller merges
// {job, events since N} into the jobs slice; TanStack invalidation happens
// separately when a job finishes (see AgentRail / JobsPanel).

const POLL_MS = 1000;
const pollers = new Map<string, number>();
const finishListeners = new Set<(job: Job) => void>();

export function onJobFinished(cb: (job: Job) => void): () => void {
  finishListeners.add(cb);
  return () => finishListeners.delete(cb);
}

export async function dispatchJob(kind: string, args: Record<string, unknown>): Promise<Job> {
  const { job } = await postJSON<{ job: Job }>("/api/jobs", { kind, args });
  useJobsStore.getState().trackJob(job);
  startPolling(job.id);
  log("info", `Job ${job.id} dispatched (${kind})`);
  return job;
}

export async function cancelJob(id: string): Promise<void> {
  await postJSON(`/api/jobs/${encodeURIComponent(id)}/cancel`, {});
  log("warn", `Cancel requested for job ${id}`);
}

export function startPolling(id: string): void {
  if (pollers.has(id)) return;
  const tick = async () => {
    try {
      const store = useJobsStore.getState();
      const since = store.jobs[id]?.events.length ?? 0;
      const resp = await getJSON<JobEventsResponse>(`/api/jobs/${encodeURIComponent(id)}?since=${since}`);
      store.applyUpdate(resp.job, resp.events);
      for (const ev of resp.events) {
        log(resp.job.status === "failed" ? "error" : "agent", `[${ev.stage}] ${ev.message}`);
      }
      const done = ["succeeded", "failed", "cancelled"].includes(resp.job.status);
      if (done) {
        stopPolling(id);
        for (const cb of finishListeners) cb(resp.job);
        return;
      }
    } catch {
      // transient — keep polling; the server may be busy with a heavy build step
    }
    pollers.set(id, window.setTimeout(tick, POLL_MS));
  };
  pollers.set(id, window.setTimeout(tick, 0));
}

export function stopPolling(id: string): void {
  const t = pollers.get(id);
  if (t !== undefined) window.clearTimeout(t);
  pollers.delete(id);
}

/** Resume polling for any running jobs after a reload (poll-based feeds survive restarts). */
export async function resumeRunningJobs(): Promise<void> {
  try {
    const { jobs } = await getJSON<{ jobs: Job[] }>("/api/jobs");
    const store = useJobsStore.getState();
    for (const job of jobs) {
      store.applyUpdate(job, []);
      if (job.status === "running" || job.status === "queued") startPolling(job.id);
    }
  } catch {
    // jobs API not available (older server) — the UI degrades gracefully
  }
}
