import { useEffect, useRef, useState } from "react";
import { Terminal, ListTodo, FolderTree, Database, OctagonX } from "lucide-react";
import { useAppStore } from "@/state/app";
import { useJobsStore, jobStatusLabel, jobStatusKind } from "@/state/jobs";
import { cancelJob } from "@/api/jobs";
import { useBuildSummary } from "@/api/queries";
import { tryJSON, packageUrl, getJSON } from "@/api/client";
import { EmptyState, Pill, Btn } from "@/components/ui";
import type { DockTab } from "@/state/app";

// Bottom dock: Console (structured log) · Jobs (all runs + cancel) ·
// Artifacts (per-package report browser + JSON preview) · Data (raw endpoint inspector).

const DOCK_TABS: Array<{ id: DockTab; label: string; Icon: typeof Terminal }> = [
  { id: "console", label: "Console", Icon: Terminal },
  { id: "jobs", label: "Jobs", Icon: ListTodo },
  { id: "artifacts", label: "Artifacts", Icon: FolderTree },
  { id: "data", label: "Data", Icon: Database },
];

function ConsoleView() {
  const log = useAppStore((s) => s.log);
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [log.length]);
  if (!log.length) return <EmptyState title="Console is quiet" sub="Build events, tool calls and errors land here." />;
  const color: Record<string, string> = { info: "text-secondary", warn: "text-warn", error: "text-fail", agent: "text-agent" };
  return (
    <div className="h-full overflow-y-auto px-3 py-2 font-mono text-2xs leading-relaxed">
      {log.map((e, i) => (
        <div key={`${e.ts}_${i}`} className={color[e.kind] ?? "text-secondary"}>
          <span className="mr-2 text-muted">{new Date(e.ts).toLocaleTimeString()}</span>
          {e.message}
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

function JobsView() {
  const jobs = useJobsStore((s) => s.jobs);
  const order = useJobsStore((s) => s.order);
  if (!order.length) return <EmptyState title="No jobs yet" sub="Builds and heavy tool calls run as cancellable background jobs." />;
  return (
    <div className="h-full overflow-y-auto p-2">
      <table className="w-full border-collapse">
        <thead>
          <tr className="text-left text-2xs uppercase tracking-wide text-muted">
            <th className="px-2 py-1 font-medium">Job</th>
            <th className="px-2 py-1 font-medium">Kind</th>
            <th className="px-2 py-1 font-medium">Status</th>
            <th className="px-2 py-1 font-medium">Last event</th>
            <th className="px-2 py-1 text-right font-medium" />
          </tr>
        </thead>
        <tbody>
          {order.map((id) => {
            const rec = jobs[id];
            if (!rec) return null;
            const last = rec.events[rec.events.length - 1];
            const running = rec.job.status === "running" || rec.job.status === "queued";
            return (
              <tr key={id} className="border-t border-hairline">
                <td className="px-2 py-1.5 font-mono text-2xs text-secondary">{id.slice(0, 10)}</td>
                <td className="px-2 py-1.5 text-2xs text-secondary">{rec.job.kind}</td>
                <td className="px-2 py-1.5">
                  <Pill kind={jobStatusKind(rec.job.status)} label={jobStatusLabel(rec.job.status)} />
                </td>
                <td className="max-w-96 truncate px-2 py-1.5 font-mono text-2xs text-muted">
                  {last ? `[${last.stage}] ${last.message}` : "—"}
                </td>
                <td className="px-2 py-1.5 text-right">
                  {running && (
                    <Btn variant="danger" onClick={() => void cancelJob(id)} title="Cancel job">
                      <OctagonX size={11} aria-hidden />
                      Cancel
                    </Btn>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const KNOWN_REPORTS = [
  "reports/autonomous_build_summary.json",
  "reports/product_readiness_ledger.json",
  "reports/honesty_scorecard.json",
  "reports/robot_package_contract.json",
  "reports/spec_sheet.json",
  "reports/spec_compliance.json",
  "reports/bom_sim_fidelity.json",
  "reports/package_validation_report.json",
  "reports/evaluation_run.json",
  "reports/design_validation_report.json",
  "robot/robot_genome.json",
  "robot/bill_of_materials.json",
  "simulation/scene_set.json",
  "simulation/mujoco/compiled_scene_index.json",
  "simulation/perception_config.json",
];

function ArtifactsView() {
  const activePackage = useAppStore((s) => s.activePackage);
  const summary = useBuildSummary(activePackage);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string | null>(null);

  const artifactPaths = Array.from(
    new Set([...Object.values(summary.data?.artifacts ?? {}), ...KNOWN_REPORTS]),
  ).filter((p) => p.endsWith(".json"));

  async function openArtifact(rel: string) {
    setSelected(rel);
    setContent("Loading…");
    const data = await tryJSON<unknown>(packageUrl(activePackage!, rel));
    setContent(data ? JSON.stringify(data, null, 2) : "Not present in this package.");
  }

  if (!activePackage) return <EmptyState title="No robot selected" />;
  return (
    <div className="flex h-full min-h-0">
      <div className="w-72 shrink-0 overflow-y-auto border-r border-hairline p-1.5">
        {artifactPaths.map((rel) => (
          <button
            key={rel}
            type="button"
            onClick={() => void openArtifact(rel)}
            className={`block w-full truncate rounded-ctl px-2 py-1 text-left font-mono text-2xs ${
              selected === rel ? "bg-accent-dim text-accent" : "text-secondary hover:bg-raised"
            }`}
          >
            {rel}
          </button>
        ))}
      </div>
      <pre className="min-w-0 flex-1 overflow-auto p-3 font-mono text-2xs leading-relaxed text-secondary">
        {content ?? "Select an artifact to preview its JSON."}
      </pre>
    </div>
  );
}

const DATA_ENDPOINTS = ["/api/packages", "/api/scorecard", "/api/moat", "/api/flywheel", "/api/design_brain", "/api/tools", "/api/assistant/status", "/api/jobs"];

/** Endpoints whose answer is scoped to the selected robot — the Data view must send the package or it shows
 *  a workspace-wide number under a per-robot heading. */
const PACKAGE_SCOPED = new Set(["/api/scorecard", "/api/moat"]);

function DataView() {
  const activePackage = useAppStore((s) => s.activePackage);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string | null>(null);

  async function openEndpoint(ep: string) {
    const url = PACKAGE_SCOPED.has(ep) && activePackage ? `${ep}?package=${encodeURIComponent(activePackage)}` : ep;
    setSelected(ep);
    setContent("Loading…");
    try {
      const data = await getJSON<unknown>(url);
      setContent(JSON.stringify(data, null, 2));
    } catch (e) {
      setContent(`Request failed: ${(e as Error).message}`);
    }
  }

  return (
    <div className="flex h-full min-h-0">
      <div className="w-72 shrink-0 overflow-y-auto border-r border-hairline p-1.5">
        {DATA_ENDPOINTS.map((ep) => (
          <button
            key={ep}
            type="button"
            onClick={() => void openEndpoint(ep)}
            className={`block w-full truncate rounded-ctl px-2 py-1 text-left font-mono text-2xs ${
              selected === ep ? "bg-accent-dim text-accent" : "text-secondary hover:bg-raised"
            }`}
          >
            GET {ep}
          </button>
        ))}
      </div>
      <pre className="min-w-0 flex-1 overflow-auto p-3 font-mono text-2xs leading-relaxed text-secondary">
        {content ?? "Raw inspection is first-class: pick an endpoint to see exactly what the backend says."}
      </pre>
    </div>
  );
}

export function Dock() {
  const dockTab = useAppStore((s) => s.dockTab);
  const setDockTab = useAppStore((s) => s.setDockTab);
  const activeJobs = useJobsStore((s) => s.order.filter((id) => ["running", "queued"].includes(s.jobs[id]?.job.status ?? "")).length);

  return (
    // Terminal-dark body (bg-canvas) + chrome header strip: the console region
    // must read as a different KIND of place than the side panels.
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      <div className="flex h-8 shrink-0 items-center gap-0.5 border-b border-hairline bg-app px-1.5" role="tablist" aria-label="Dock tabs">
        {DOCK_TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={dockTab === id}
            onClick={() => setDockTab(id)}
            className={`flex h-full items-center gap-1.5 border-b-2 px-2.5 text-2xs transition-colors ${
              dockTab === id ? "border-accent text-accent" : "border-transparent text-muted hover:text-secondary"
            }`}
          >
            <Icon size={12} aria-hidden />
            {label}
            {id === "jobs" && activeJobs > 0 && (
              <span className="rounded-full bg-agent-dim px-1.5 font-mono text-2xs text-agent">{activeJobs}</span>
            )}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1">
        {dockTab === "console" && <ConsoleView />}
        {dockTab === "jobs" && <JobsView />}
        {dockTab === "artifacts" && <ArtifactsView />}
        {dockTab === "data" && <DataView />}
      </div>
    </div>
  );
}
