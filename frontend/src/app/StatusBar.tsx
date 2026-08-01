import { Circle, ShieldCheck, ShieldAlert } from "lucide-react";
import { useAppStore } from "@/state/app";
import { useAssistantStore } from "@/state/assistant";
import { useJobsStore, phaseStates, PHASES } from "@/state/jobs";
import { usePackages, useAssistantStatus } from "@/api/queries";
import { useLifecycle } from "./useLifecycle";
import { Spinner } from "@/components/ui";
import { Tip } from "@/components/Explain";

// Status bar — the thin strip at the very bottom, IDE-style. Facts only:
// backend link, local AI model, running work on the left; the active robot's
// vitals and its verification state on the right. Everything explains itself
// on hover.

function JobStatus() {
  const jobs = useJobsStore((s) => s.jobs);
  const order = useJobsStore((s) => s.order);
  const setDockTab = useAppStore((s) => s.setDockTab);
  const activeId = order.find((id) => ["running", "queued"].includes(jobs[id]?.job.status ?? ""));
  if (!activeId) return null;
  const rec = jobs[activeId];
  const states = phaseStates(rec.events, rec.job.status);
  const current = PHASES.find((p) => states[p.id] === "active")?.label ?? "queued";
  return (
    <Tip text="A build is running in the background. Click to watch its progress in the Jobs panel." side="top">
      <button
        type="button"
        onClick={() => setDockTab("jobs")}
        className="flex items-center gap-1.5 px-2 font-mono text-2xs text-agent hover:bg-raised"
      >
        <Spinner size={9} />
        {rec.job.kind === "autonomous_build" ? "building" : rec.job.kind} · {current.toLowerCase()}
      </button>
    </Tip>
  );
}

export function StatusBar() {
  const activePackage = useAppStore((s) => s.activePackage);
  const packages = usePackages();
  const assistantStatus = useAssistantStatus();
  const selectedModel = useAssistantStore((s) => s.selectedModel);
  const setSelectedModel = useAssistantStore((s) => s.setSelectedModel);
  const lifecycle = useLifecycle();
  const meta = packages.data?.packages.find((p) => p.id === activePackage);
  const activeModel = selectedModel ?? assistantStatus.data?.model;
  const selectedAvailable = selectedModel
    ? (assistantStatus.data?.models ?? []).includes(selectedModel)
    : !!assistantStatus.data?.model_available;
  // Preserve the existing status shape for the rest of the surface, while making the selected per-chat model
  // the one reported in the tooltip as well as the select control.
  const status = {
    ...assistantStatus,
    data: assistantStatus.data ? { ...assistantStatus.data, model: activeModel ?? assistantStatus.data.model } : undefined,
  };
  const backendFailed = packages.isError || status.isError;
  const online = !backendFailed && !!status.data?.online && selectedAvailable;
  const gate = lifecycle.verify;

  return (
    <footer className="flex h-6 shrink-0 items-center border-t border-hairline bg-app px-1 text-2xs select-none">
      <Tip text="The Python backend that designs, simulates and verifies robots. Everything runs locally on this machine." side="top">
        <span className="flex items-center gap-1.5 px-2 text-muted">
          <Circle size={7} className={backendFailed ? "fill-fail text-fail" : "fill-ok text-ok"} aria-hidden />
          {backendFailed ? "backend unavailable" : "backend"}
        </span>
      </Tip>
      <Tip
        text={
          online
            ? `Local AI model (${status.data?.model}) is ready. It powers the Agent panel — it runs on this machine, nothing leaves your computer.`
            : "The local AI model is offline. The Agent's chat answers pause, but deterministic builds and every manual control still work."
        }
        side="top"
      >
        <label className={`flex items-center gap-1.5 px-2 font-mono ${online ? "text-muted" : "text-warn"}`}>
          <Circle size={7} className={online ? "fill-ok text-ok" : "fill-warn text-warn"} aria-hidden />
          <span>AI</span>
          {status.data?.models?.length ? (
            <select
              value={activeModel ?? ""}
              onChange={(event) => setSelectedModel(event.target.value || null)}
              aria-label="Local AI model"
              className="max-w-36 bg-transparent text-2xs text-inherit outline-none"
            >
              {status.data.models.map((model) => <option key={model} value={model}>{model}</option>)}
            </select>
          ) : (
            <span>{activeModel ?? "offline"}</span>
          )}
        </label>
      </Tip>
      <JobStatus />

      <span className="flex-1" />

      {meta && (
        <>
          <Tip text="Robot class and degrees of freedom — how many independently moving joints it has." side="top">
            <span className="px-2 font-mono text-muted">
              {meta.robot_class ?? "robot"} · {meta.dof ?? "?"} DOF
            </span>
          </Tip>
          {/* Renders the SHARED verdict word (see state/status.ts) — the footer no longer invents
              its own "verified/unverified" for the same fact the Verify tab calls EXPORT BLOCKED. */}
          <Tip text={`${gate.label ?? "Verification"} — ${gate.hint}. Open the Verify tab for the honesty gates.`} side="top">
            <button
              type="button"
              onClick={() => useAppStore.getState().setWorkspace("verify")}
              className={`flex items-center gap-1 px-2 hover:bg-raised ${gate.state === "done" ? "text-ok" : "text-warn"}`}
            >
              {gate.state === "done" ? <ShieldCheck size={11} aria-hidden /> : <ShieldAlert size={11} aria-hidden />}
              {gate.label ?? (gate.state === "done" ? "verified" : "not verified")}
            </button>
          </Tip>
        </>
      )}
    </footer>
  );
}
