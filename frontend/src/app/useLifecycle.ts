import { useAppStore } from "@/state/app";
import { usePackages, useLedger } from "@/api/queries";
import type { WorkspaceId } from "@/state/app";

// The Lifecycle Tracker's truth source: each station's state is computed from
// REAL artifacts of the active robot (never decorative). This drives both the
// Pipeline Spine dots and the Status Horizon segments.

export type StationState = "done" | "attention" | "pending" | "neutral";

export interface StationStatus {
  state: StationState;
  hint: string;
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

  const verify: StationStatus = !activePackage
    ? { state: "pending", hint: "Needs a robot first" }
    : ledger.data
      ? ledger.data.safe_to_export
        ? { state: "done", hint: "Export gates clear" }
        : { state: "attention", hint: "Export blocked — open Verify" }
      : { state: "pending", hint: "No readiness ledger" };

  const library: StationStatus = {
    state: "neutral",
    hint: `${packages.data?.packages.length ?? 0} robots in the library`,
  };

  return { design, simulate, train, verify, library };
}
