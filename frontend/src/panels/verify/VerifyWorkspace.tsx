import { ShieldCheck, ShieldAlert, Download } from "lucide-react";
import { useAppStore, useSelectionStore } from "@/state/app";
import { useLedger, usePackages, useScorecard, useSpecCompliance, useBomFidelity, useSim2Sim } from "@/api/queries";
import { robotStatus } from "@/state/status";
import { EmptyState, Pill, QueryState, SectionLabel, verdictKind, fmt } from "@/components/ui";
import { packageUrl } from "@/api/client";

// Verify — the honesty surface as the hero. The readiness ledger renders as a
// horizontal gate pipeline; the Export action is LITERALLY gated by it.

function GatePipeline() {
  const activePackage = useAppStore((s) => s.activePackage);
  const ledger = useLedger(activePackage);
  const packages = usePackages();
  const select = useSelectionStore((s) => s.select);
  const selection = useSelectionStore((s) => s.selection);

  if (ledger.isPending || ledger.isError) {
    return <QueryState loading={ledger.isPending} error={ledger.error} retry={() => void ledger.refetch()} label="readiness ledger" />;
  }
  if (!ledger.data) {
    return <EmptyState title="No readiness ledger" sub="This package predates the ledger, or hasn't finished building." />;
  }
  const { stages, required, safe_to_export, issues, highest_attained } = ledger.data;
  // Headline = the SHARED verdict (same object the header chip and status bar render). The Export
  // action stays gated by `safe_to_export` itself, because that is literally the export gate.
  const status = robotStatus(packages.data?.packages.find((p) => p.id === activePackage), ledger.data)!;
  const ok = status.kind === "ok";

  return (
    <div>
      <div
        className={`mx-4 mt-4 flex items-center justify-between gap-3 rounded-card border p-3 ${
          ok ? "border-ok/40 bg-ok-dim/40" : "border-fail/40 bg-fail-dim/40"
        }`}
      >
        <div className="flex items-center gap-3">
          {ok ? (
            <ShieldCheck size={22} className="text-ok" aria-hidden />
          ) : (
            <ShieldAlert size={22} className="text-fail" aria-hidden />
          )}
          <div>
            <div className={`text-base font-semibold ${ok ? "text-ok" : "text-fail"}`}>{status.label}</div>
            <div className="text-2xs text-muted">
              Highest attained: <span className="font-mono">{highest_attained ?? "—"}</span> · Export-readiness = package
              completeness + honest verification, not trained capability.
            </div>
            {status.buildable === false && (
              <div className="mt-1 text-2xs text-warn">
                Not buildable from real parts — a separate check from export-readiness. {status.notes[0] ?? ""}
              </div>
            )}
          </div>
        </div>
        <a
          href={safe_to_export && activePackage ? packageUrl(activePackage, "reports/robot_package_contract.json") : undefined}
          target="_blank"
          rel="noreferrer"
          aria-disabled={!safe_to_export}
          title={
            safe_to_export
              ? "Open the export contract"
              : `Blocked by ${stages.filter((s) => required.includes(s.stage) && s.status !== "attained").length || "unmet"} required gate(s)`
          }
          className={`inline-flex items-center gap-1.5 rounded-ctl border px-3 py-1.5 text-xs font-medium ${
            safe_to_export
              ? "border-ok/50 bg-ok-dim text-ok hover:bg-ok/25"
              : "pointer-events-none cursor-not-allowed border-hairline text-muted opacity-60"
          }`}
        >
          <Download size={13} aria-hidden />
          Export
        </a>
      </div>

      {/* Horizontal gate pipeline */}
      <div className="mx-4 mt-4 flex items-stretch gap-0 overflow-x-auto pb-1">
        {stages.map((s, i) => {
          const kind = verdictKind(s.status);
          const active = selection?.kind === "claim" && selection.id === s.stage;
          return (
            <div key={s.stage} className="flex items-center">
              <button
                type="button"
                onClick={() => select(active ? null : { kind: "claim", id: s.stage, context: { ...s } })}
                className={`flex min-w-40 flex-col gap-1 rounded-card border p-2.5 text-left transition-colors ${
                  active ? "border-accent/60 bg-accent-dim/40" : "border-hairline bg-panel hover:border-hairline-strong"
                }`}
              >
                <Pill kind={kind} label={s.status} />
                <span className="font-mono text-2xs text-primary">{s.stage}</span>
                <span className="text-2xs text-muted">{required.includes(s.stage) ? "required" : "optional"}</span>
              </button>
              {i < stages.length - 1 && <div className="h-px w-4 shrink-0 bg-hairline-strong" aria-hidden />}
            </div>
          );
        })}
      </div>

      {/* Evidence for selected gate */}
      {selection?.kind === "claim" && (
        <div className="mx-4 mt-3 rounded-card border border-hairline bg-panel p-3">
          <div className="mb-1 font-mono text-xs text-primary">{selection.id}</div>
          <div className="mb-2 text-2xs text-secondary">{String(selection.context?.detail ?? "")}</div>
          <pre className="overflow-auto rounded-ctl bg-canvas p-2 font-mono text-2xs text-muted">
            {JSON.stringify(selection.context?.evidence ?? {}, null, 2)}
          </pre>
        </div>
      )}

      {issues.length > 0 && (
        <div className="mx-4 mt-3 rounded-card border border-warn/40 bg-warn-dim/30 p-3">
          <div className="mb-1 text-xs font-semibold text-warn">Honesty violations</div>
          <ul className="list-inside list-disc text-2xs text-secondary">
            {issues.map((i) => (
              <li key={i}>{i}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ScorecardTable() {
  const activePackage = useAppStore((s) => s.activePackage);
  const scorecard = useScorecard(activePackage);
  if (scorecard.isPending || scorecard.isError) {
    return <QueryState loading={scorecard.isPending} error={scorecard.error} retry={() => void scorecard.refetch()} label="honesty scorecard" />;
  }
  if (scorecard.data?.error) {
    return <EmptyState title="Honesty scorecard unavailable" sub={scorecard.data.error} />;
  }
  if (!scorecard.data?.rows?.length) return null;
  return (
    <div className="mx-4 mt-2">
      <div className="mb-2 text-2xs text-muted">{scorecard.data.headline}</div>
      <table className="w-full border-collapse">
        <thead>
          <tr className="text-left text-2xs uppercase tracking-wide text-muted">
            <th className="py-1 pr-3 font-medium">Claim</th>
            <th className="py-1 pr-3 font-medium">Verdict</th>
            <th className="py-1 font-medium">Evidence</th>
          </tr>
        </thead>
        <tbody>
          {scorecard.data.rows.map((r) => (
            <tr key={r.claim} className="border-t border-hairline align-top">
              <td className="py-1.5 pr-3 font-mono text-2xs text-primary">{r.claim}</td>
              <td className="py-1.5 pr-3">
                <Pill kind={r.honest ? "ok" : "warn"} label={r.honest ? "honest" : "flagged"} />
              </td>
              <td className="py-1.5 font-mono text-2xs leading-relaxed text-muted">
                {typeof r.evidence === "string" ? r.evidence : JSON.stringify(r.evidence)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComplianceAndFidelity() {
  const activePackage = useAppStore((s) => s.activePackage);
  const compliance = useSpecCompliance(activePackage);
  const fidelity = useBomFidelity(activePackage);
  const sim2sim = useSim2Sim(activePackage);

  return (
    <div className="mx-4 mb-6 mt-2 grid gap-3 lg:grid-cols-3">
      <div className="rounded-card border border-hairline bg-panel p-3">
        <div className="mb-1.5 text-2xs font-semibold uppercase tracking-wider text-muted">Spec compliance</div>
        {compliance.data ? (
          <>
            <Pill kind={compliance.data.all_honored ? "ok" : "warn"} label={compliance.data.all_honored ? "all honored" : "deviations"} />
            <div className="mt-2 flex flex-col gap-1">
              {(compliance.data.constraints ?? []).map((c, i) => (
                <div key={i} className="font-mono text-2xs text-secondary">
                  {String(c.constraint ?? "constraint")}: {String(c.requested ?? "?")} → {String(c.delivered ?? "?")}{" "}
                  {c.honored ? "✓" : "✗"}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="text-2xs text-muted">No spec_compliance.json — no detailed constraints were requested.</div>
        )}
      </div>
      <div className="rounded-card border border-hairline bg-panel p-3">
        <div className="mb-1.5 text-2xs font-semibold uppercase tracking-wider text-muted">BOM ↔ sim fidelity</div>
        {fidelity.data ? (
          <>
            <div className="font-mono text-lg text-primary">{fmt(fidelity.data.mass_fidelity_ratio)}×</div>
            <div className="mt-1 text-2xs leading-relaxed text-muted">
              {Array.isArray(fidelity.data.flags) && fidelity.data.flags.length
                ? String((fidelity.data.flags as unknown[])[0])
                : "Sim mass matches BOM-grounded mass."}
            </div>
          </>
        ) : (
          <div className="text-2xs text-muted">No bom_sim_fidelity.json in this package.</div>
        )}
      </div>
      <div className="rounded-card border border-hairline bg-panel p-3">
        <div className="mb-1.5 text-2xs font-semibold uppercase tracking-wider text-muted">Sim2sim transfer</div>
        {sim2sim.data ? (
          <pre className="overflow-auto font-mono text-2xs text-secondary">{JSON.stringify(sim2sim.data, null, 2).slice(0, 600)}</pre>
        ) : (
          <div className="text-2xs text-muted">No sim2sim_report.json — cross-simulator validation hasn't run.</div>
        )}
      </div>
    </div>
  );
}

export function VerifyWorkspace() {
  const activePackage = useAppStore((s) => s.activePackage);
  if (!activePackage) {
    return <EmptyState title="No robot selected" sub="Verify shows the honesty gates for the active robot — build or select one first." />;
  }
  return (
    <div className="h-full overflow-y-auto bg-canvas">
      <GatePipeline />
      <SectionLabel>Honesty scorecard</SectionLabel>
      <ScorecardTable />
      <SectionLabel>Grounding reports</SectionLabel>
      <ComplianceAndFidelity />
    </div>
  );
}
