import { Activity, GraduationCap } from "lucide-react";
import { useAppStore } from "@/state/app";
import { useQuery } from "@tanstack/react-query";
import { tryJSON, packageUrl } from "@/api/client";
import { dispatchJob } from "@/api/jobs";
import { useAssistantStore, feedId } from "@/state/assistant";
import { EmptyState, Btn, KV, QueryState, SectionLabel } from "@/components/ui";

// Train — training configuration + curriculum for the active robot, with an
// honest empty state (no fabricated curves). Live reward charts attach to
// train_policy job events when a training-enabled build runs.

function useTrainingArtifact<T>(pkg: string | null, rel: string) {
  return useQuery({
    queryKey: ["artifact", pkg, rel],
    queryFn: () => tryJSON<T>(packageUrl(pkg!, rel)),
    enabled: !!pkg,
  });
}

export function TrainWorkspace() {
  const activePackage = useAppStore((s) => s.activePackage);
  const runConfig = useTrainingArtifact<Record<string, unknown>>(activePackage, "training/training_run_config.json");
  const curriculum = useTrainingArtifact<Record<string, unknown>>(activePackage, "training/training_curriculum.json");
  const objective = useTrainingArtifact<Record<string, unknown>>(activePackage, "training/training_objective.json");

  if (!activePackage) {
    return <EmptyState icon={<GraduationCap size={32} strokeWidth={1.3} />} title="No robot selected" sub="Training views attach to the active robot." />;
  }
  const failedArtifact = [runConfig, curriculum, objective].find((query) => query.isError);
  if (failedArtifact) {
    return <QueryState loading={false} error={failedArtifact.error} retry={() => void failedArtifact.refetch()} label="training artifacts" />;
  }
  if (runConfig.isPending || curriculum.isPending || objective.isPending) {
    return <QueryState loading error={null} label="training artifacts" />;
  }

  const stages = (curriculum.data?.stages ?? curriculum.data?.curriculum ?? []) as Array<Record<string, unknown>>;

  return (
    <div className="h-full overflow-y-auto bg-canvas p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <div className="text-base font-semibold text-primary">Training</div>
          <div className="text-2xs text-muted">
            Honest by design: no curves are shown unless a real training run produced them.
          </div>
        </div>
        <Btn
          variant="primary"
          onClick={() => {
            void dispatchJob("autonomous_build", {
              prompt: `Rebuild and train: ${activePackage}`,
              train: true,
            }).then((job) => {
              useAssistantStore.getState().push({
                type: "run",
                id: feedId(),
                ts: Date.now(),
                jobId: job.id,
                goal: `Train a controller for ${activePackage}`,
              });
            });
          }}
        >
          <Activity size={13} aria-hidden />
          Train controller (job)
        </Btn>
      </div>

      {Array.isArray(stages) && stages.length > 0 && (
        <>
          <SectionLabel>Curriculum</SectionLabel>
          <div className="mb-4 flex gap-2 overflow-x-auto px-3">
            {stages.map((s, i) => (
              <div key={i} className="min-w-44 shrink-0 rounded-card border border-hairline bg-panel p-2.5">
                <div className="font-mono text-2xs text-accent">stage {i + 1}</div>
                <pre className="mt-1 overflow-hidden font-mono text-2xs text-muted">{JSON.stringify(s, null, 1).slice(0, 260)}</pre>
              </div>
            ))}
          </div>
        </>
      )}

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="rounded-card border border-hairline bg-panel">
          <SectionLabel>Run config</SectionLabel>
          {runConfig.data ? (
            Object.entries(runConfig.data)
              .filter(([, v]) => typeof v !== "object")
              .slice(0, 14)
              .map(([k, v]) => <KV key={k} k={k} v={String(v)} />)
          ) : (
            <div className="px-3 pb-3 text-2xs text-muted">No training_run_config.json in this package.</div>
          )}
        </div>
        <div className="rounded-card border border-hairline bg-panel">
          <SectionLabel>Objective</SectionLabel>
          {objective.data ? (
            <pre className="overflow-auto px-3 pb-3 font-mono text-2xs leading-relaxed text-secondary">
              {JSON.stringify(objective.data, null, 2).slice(0, 1500)}
            </pre>
          ) : (
            <div className="px-3 pb-3 text-2xs text-muted">No training_objective.json in this package.</div>
          )}
        </div>
      </div>
    </div>
  );
}
