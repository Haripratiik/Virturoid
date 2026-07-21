import { useQuery, useQueryClient } from "@tanstack/react-query";
import { getJSON, tryJSON, packageUrl } from "./client";
import type {
  AssistantStatus,
  BillOfMaterials,
  BuildSummary,
  DesignBrainResponse,
  FlywheelResponse,
  PackagesResponse,
  ReadinessLedger,
  RobotGenome,
  SceneIndex,
  Scorecard,
  SpecCompliance,
  SpecSheet,
  ToolSpec,
} from "./types";

export function usePackages() {
  return useQuery({
    queryKey: ["packages"],
    queryFn: () => getJSON<PackagesResponse>("/api/packages"),
    staleTime: 10_000,
  });
}

export function useAssistantStatus() {
  return useQuery({
    queryKey: ["assistant-status"],
    queryFn: () => getJSON<AssistantStatus>("/api/assistant/status"),
    refetchInterval: 15_000,
  });
}

export function useTools() {
  return useQuery({
    queryKey: ["tools"],
    queryFn: async () => (await getJSON<{ tools: ToolSpec[] }>("/api/tools")).tools,
    staleTime: Infinity,
  });
}

export function useScorecard(pkg: string | null) {
  return useQuery({
    queryKey: ["scorecard", pkg],
    queryFn: () => getJSON<Scorecard>(`/api/scorecard?package=${encodeURIComponent(pkg!)}`),
    enabled: !!pkg,
  });
}

export function useFlywheel() {
  return useQuery({
    queryKey: ["flywheel"],
    queryFn: () => getJSON<FlywheelResponse>("/api/flywheel"),
  });
}

export function useDesignBrain() {
  return useQuery({
    queryKey: ["design-brain"],
    queryFn: () => getJSON<DesignBrainResponse>("/api/design_brain"),
  });
}

// ---- Per-package artifacts (optional files; null = absent, rendered as honest empty states) ----

function usePackageArtifact<T>(pkg: string | null, rel: string) {
  return useQuery({
    queryKey: ["artifact", pkg, rel],
    queryFn: () => tryJSON<T>(packageUrl(pkg!, rel)),
    enabled: !!pkg,
    staleTime: 30_000,
  });
}

export const useGenome = (pkg: string | null) => usePackageArtifact<RobotGenome>(pkg, "robot/robot_genome.json");
export const useBom = (pkg: string | null) => usePackageArtifact<BillOfMaterials>(pkg, "robot/bill_of_materials.json");
export const useLedger = (pkg: string | null) =>
  usePackageArtifact<ReadinessLedger>(pkg, "reports/product_readiness_ledger.json");
export const useSpecSheet = (pkg: string | null) => usePackageArtifact<SpecSheet>(pkg, "reports/spec_sheet.json");
export const useBuildSummary = (pkg: string | null) =>
  usePackageArtifact<BuildSummary>(pkg, "reports/autonomous_build_summary.json");
export const useSpecCompliance = (pkg: string | null) =>
  usePackageArtifact<SpecCompliance>(pkg, "reports/spec_compliance.json");
export const useSceneIndex = (pkg: string | null) =>
  usePackageArtifact<SceneIndex>(pkg, "simulation/mujoco/compiled_scene_index.json");
export const useSim2Sim = (pkg: string | null) =>
  usePackageArtifact<Record<string, unknown>>(pkg, "reports/sim2sim_report.json");
export const useBomFidelity = (pkg: string | null) =>
  usePackageArtifact<Record<string, unknown>>(pkg, "reports/bom_sim_fidelity.json");

/** Invalidate everything a finished build could have changed. */
export function useInvalidateAfterBuild() {
  const qc = useQueryClient();
  return () => {
    void qc.invalidateQueries({ queryKey: ["packages"] });
    void qc.invalidateQueries({ queryKey: ["scorecard"] });
    void qc.invalidateQueries({ queryKey: ["flywheel"] });
    void qc.invalidateQueries({ queryKey: ["design-brain"] });
    void qc.invalidateQueries({ queryKey: ["artifact"] });
  };
}
