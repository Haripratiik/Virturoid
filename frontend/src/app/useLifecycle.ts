import { useAppStore } from "@/state/app";
import { usePackages, useLedger } from "@/api/queries";
import { robotStatus } from "@/state/status";
import type { WorkspaceId } from "@/state/app";

// The Lifecycle Tracker's truth source: each station's state is computed from
// REAL artifacts of the active robot (never decorative). This drives both the
// Pipeline Spine dots and the Status Horizon segments.

export type StationState = "done" | "attention" | "pending" | "neutral";

export interface StationStatus {
  state: StationState;
  hint: string;
  /** Shared verdict wording, when the station has one (verify). Surfaces render this, never
   * their own private word for the same fact. */
  label?: string;
}

export function useLifecycle(): Record<WorkspaceId, StationStatus> {
  const activePackage = useAppStore((s) => s.activePackage);
  const packages = usePackages();
  const ledger = useLedger(activePackage);
  const meta = packages.data?.packages.find((p) => p.id === activePackage);
  const stages = ledger.data?.stages ?? [];
  const stage = (name: string) => stages.find((s) => s.stage === name)?.status ?? null;

  const design: StationStatus = !activePackage
    ? { state: "pending", hint: "No robot yet — describe one in the Agent panel" }
    : { state: "done", hint: `${meta?.robot_class ?? "robot"} composed` };

  const simulate: StationStatus = !activePackage
    ? { state: "pending", hint: "Needs a robot first" }
    : (meta?.scene_count ?? 0) > 0
      ? { state: "done", hint: `${meta?.scene_count} scenes compiled` }
      : { state: "neutral", hint: "No saved scenes in this package — Robot mode remains available" };

  const controller = stage("controller_exported");
  const train: StationStatus = !activePackage
    ? { state: "pending", hint: "Needs a robot first" }
    : controller === "attained"
      ? { state: "done", hint: "Controller exported" }
      : controller === "not_required"
        ? { state: "neutral", hint: "No trained controller requested" }
        : { state: "attention", hint: "Not trained yet" };

  // The verify station reports the SHARED robot status (services/package_status.py) — the same
  // object the header chip, library card and Verify tab render — so the status bar can no longer
  // say "unverified" while Verify says "EXPORT BLOCKED" and the header says "valid".
  const status = robotStatus(meta, ledger.data);
  const verify: StationStatus = !activePackage
    ? { state: "pending", hint: "Needs a robot first" }
    : status
      ? {
          state: status.kind === "ok" ? "done" : status.kind === "muted" ? "pending" : "attention",
          hint: status.detail,
          label: status.label,
        }
      : { state: "pending", hint: "No readiness ledger" };

  const library: StationStatus = {
    state: "neutral",
    hint: `${packages.data?.packages.length ?? 0} robots in the library`,
  };

  // Memory carries NO dot. Every other station's dot means "this stage produced verified output for this
  // robot"; the bank is workspace-wide and its headline is currently unfavourable, so a green dot here would
  // be decorative at best and a claim we cannot support at worst. The tab label alone.
  const memory: StationStatus = { state: "neutral", hint: "The verified-morphology bank, measured" };

  return { design, simulate, train, verify, library, memory };
}
