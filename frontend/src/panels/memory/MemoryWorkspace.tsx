import { Brain, ShieldCheck, ShieldAlert, TrendingUp, TrendingDown, Minus, Info } from "lucide-react";
import { useAppStore } from "@/state/app";
import { useMoat } from "@/api/queries";
import type { MoatRecallKind } from "@/api/types";
import { EmptyState, QueryState, pct } from "@/components/ui";
import { Tip } from "@/components/Explain";

// Memory — the verified-morphology bank, rendered as it actually is.
//
// This tab exists because the moat was the one thing the product could not show. Library reported "flywheel
// cycles" from a JSON file a demo script writes; nothing anywhere answered how many banked rows carry a
// measured error bar, where the rows came from, or whether recalling them made a robot walk further.
//
// The design rule for everything below: a number that is bad must be as easy to read as a number that is good.
// The live bank's dominant recall kind is NEGATIVE (mean -0.034 m over 2163 deploys). If this panel could only
// render wins it would be worse than no panel, so `direction` drives the arrow and the color in both
// directions, losses sit beside wins at the same weight, and the caveats the backend computes are rendered on
// the page rather than left in a docstring.

function signed(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}`;
}

function Stat({ label, value, sub, tone = "neutral", tip }: {
  label: string; value: string; sub?: string; tone?: "good" | "bad" | "neutral"; tip?: string;
}) {
  const toneCls = tone === "good" ? "text-ok" : tone === "bad" ? "text-fail" : "text-primary";
  const body = (
    <div className="rounded-card border border-hairline bg-panel px-3 py-2.5">
      <div className={`font-mono text-lg leading-tight ${toneCls}`}>{value}</div>
      <div className="text-2xs text-muted">{label}</div>
      {sub && <div className="mt-0.5 text-2xs text-muted/80">{sub}</div>}
    </div>
  );
  return tip ? <Tip text={tip} block>{body}</Tip> : body;
}

/** A labelled split of a counter, widest bucket first. Counts are shown as counts — never only as a percentage,
 *  because "1.0% gated" and "1 of 101 gated" land very differently and the second one is the true one. */
function Breakdown({ title, counts, total, highlight }: {
  title: string; counts: Record<string, number>; total: number; highlight?: (k: string) => "good" | "bad" | null;
}) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;
  return (
    <div className="rounded-card border border-hairline bg-panel p-3">
      <div className="mb-2 text-2xs font-semibold uppercase tracking-wider text-muted">{title}</div>
      <div className="flex flex-col gap-1.5">
        {entries.map(([k, n]) => {
          const tone = highlight?.(k) ?? null;
          const bar = tone === "good" ? "bg-ok" : tone === "bad" ? "bg-fail" : "bg-accent/60";
          return (
            <div key={k} className="flex items-center gap-2">
              <span className="w-40 shrink-0 truncate font-mono text-2xs text-secondary" title={k}>{k}</span>
              <span className="h-1.5 min-w-[2px] flex-1 rounded-full bg-raised">
                <span className={`block h-1.5 rounded-full ${bar}`} style={{ width: `${total ? (n / total) * 100 : 0}%` }} />
              </span>
              <span className="w-20 shrink-0 text-right font-mono text-2xs text-primary">
                {n} <span className="text-muted">/ {total}</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RecallRow({ k }: { k: MoatRecallKind }) {
  const Icon = k.direction === "helps" ? TrendingUp : k.direction === "hurts" ? TrendingDown : Minus;
  const tone = k.direction === "helps" ? "text-ok" : k.direction === "hurts" ? "text-fail" : "text-muted";
  return (
    <tr className="border-t border-hairline align-top">
      <td className="py-1.5 pr-3">
        <div className="font-mono text-2xs text-primary">{k.kind}</div>
        <div className="max-w-md text-2xs text-muted">{k.means}</div>
      </td>
      <td className="py-1.5 pr-3 text-right font-mono text-2xs text-secondary">{k.edges}</td>
      <td className={`py-1.5 pr-3 text-right font-mono text-2xs ${tone}`}>
        <span className="inline-flex items-center gap-1">
          <Icon size={11} strokeWidth={2} aria-hidden />
          {signed(k.mean_delta_m)} m
        </span>
      </td>
      <td className="py-1.5 pr-3 text-right font-mono text-2xs">
        <span className="text-ok">{k.wins}</span>
        <span className="text-muted"> / </span>
        <span className="text-fail">{k.losses}</span>
        <span className="text-muted"> / {k.ties}</span>
      </td>
      <td className="py-1.5 text-right font-mono text-2xs text-secondary">
        {k.decided_win_rate === null ? "—" : pct(k.decided_win_rate)}
      </td>
    </tr>
  );
}

export function MemoryWorkspace() {
  const activePackage = useAppStore((s) => s.activePackage);
  const moat = useMoat(activePackage);

  if (moat.isPending || moat.isError) {
    return <QueryState loading={moat.isPending} error={moat.error} retry={() => void moat.refetch()} label="the memory bank" />;
  }
  const data = moat.data;
  if (!data || data.error || !data.db_present || !data.bank) {
    return (
      <div className="h-full overflow-y-auto bg-canvas p-4">
        <EmptyState
          icon={<Brain size={22} />}
          title="No memory bank at this build root"
          sub={data?.error ?? `Nothing has been banked in ${data?.memory_dir ?? "this workspace"} yet. Verify or train a robot and its operating point lands here.`}
        />
      </div>
    );
  }

  const { bank, recall, this_build: build, notes } = data;
  const dominant = recall?.kinds.find((k) => k.kind === recall.dominant_kind) ?? recall?.kinds[0];
  const gatedTone = bank.gated_fraction >= 0.5 ? "good" : "bad";

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto bg-canvas p-4">
      {/* The one-sentence honest verdict on reuse, whichever way it points. */}
      <div className={`mb-4 flex items-start gap-2.5 rounded-card border p-3 ${
        dominant?.direction === "hurts" ? "border-fail/40 bg-fail-dim/25" :
        dominant?.direction === "helps" ? "border-ok/40 bg-ok-dim/25" : "border-hairline bg-panel"}`}>
        {dominant?.direction === "hurts"
          ? <ShieldAlert size={18} className="mt-0.5 shrink-0 text-fail" aria-hidden />
          : <ShieldCheck size={18} className="mt-0.5 shrink-0 text-ok" aria-hidden />}
        <div className="min-w-0">
          <div className="text-2xs font-semibold uppercase tracking-wider text-muted">Does recall pay off?</div>
          <div className="text-xs leading-relaxed text-primary">{recall?.headline}</div>
        </div>
      </div>

      {/* What is banked, and how sure we are of it. */}
      <div className="mb-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label={`banked ${bank.task} rows`} value={String(bank.rows)}
              sub={`${bank.bodies.distinct} distinct bodies`}
              tip="One row is one operating point (gait parameters) that made a specific body move, keyed by its morphology." />
        <Stat label="carry a measured fragility margin" value={`${bank.gated_rows} / ${bank.rows}`}
              tone={gatedTone} sub={pct(bank.gated_fraction)}
              tip="A gated row was re-measured under perturbation and still walked. Everything else records an operating point with no error bar — usable as a hint, not as evidence." />
        <Stat label="attributed to a real build" value={String(bank.by_source.real ?? 0)}
              tone={(bank.by_source.real ?? 0) > (bank.rows ?? 0) / 2 ? "good" : "bad"}
              sub={`${bank.by_source.suite ?? 0} suite-authored · ${(bank.by_source.unattributed ?? 0) + (bank.by_source.unstamped ?? 0)} unattributed`}
              tip="Stamped by scripts/bank_provenance.py. Suite-authored rows are fixtures that reached the bank; they are evidence of nothing." />
        <Stat label="largest single body's share" value={pct(bank.bodies.largest_share_fraction)}
              tone={bank.bodies.largest_share_fraction > 0.15 ? "bad" : "good"}
              sub={bank.bodies.largest_share_body ?? undefined}
              tip="Rows are not independent observations. If one body supplies a third of them, a gate that 'passed on N rows' passed on far fewer bodies." />
      </div>

      {/* What recall did for the robot currently open. */}
      <div className="mb-3 rounded-card border border-hairline bg-panel p-3">
        <div className="mb-1.5 flex items-center gap-2">
          <div className="text-2xs font-semibold uppercase tracking-wider text-muted">
            This robot {activePackage ? <span className="font-mono normal-case text-secondary">· {activePackage}</span> : null}
          </div>
        </div>
        {!activePackage ? (
          <div className="text-xs text-muted">No robot selected — pick one in the Library tab to see what memory did for it.</div>
        ) : (
          <>
            <div className="text-xs leading-relaxed text-primary">{build?.summary}</div>
            {build?.matched && build.events.length > 0 && (
              <div className="mt-2 overflow-x-auto">
                <table className="w-full min-w-[560px] text-left">
                  <thead>
                    <tr className="text-2xs uppercase tracking-wider text-muted">
                      <th className="pb-1 pr-3 font-medium">when</th>
                      <th className="pb-1 pr-3 font-medium">recalled from</th>
                      <th className="pb-1 pr-3 text-right font-medium">recalled gait</th>
                      <th className="pb-1 pr-3 text-right font-medium">own default</th>
                      <th className="pb-1 pr-3 text-right font-medium">delta</th>
                      <th className="pb-1 font-medium">kept?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {build.events.map((e, i) => (
                      <tr key={`${e.when}-${i}`} className="border-t border-hairline">
                        <td className="py-1 pr-3 font-mono text-2xs text-muted">{(e.when ?? "").replace("T", " ").slice(0, 16)}</td>
                        <td className="py-1 pr-3 font-mono text-2xs text-secondary">{e.source ?? e.kind}</td>
                        <td className="py-1 pr-3 text-right font-mono text-2xs text-secondary">
                          {e.hint_forward_m === null || e.hint_forward_m === undefined ? "—" : `${Number(e.hint_forward_m).toFixed(3)} m`}
                        </td>
                        <td className="py-1 pr-3 text-right font-mono text-2xs text-secondary">
                          {e.default_forward_m === null || e.default_forward_m === undefined ? "—" : `${Number(e.default_forward_m).toFixed(3)} m`}
                        </td>
                        <td className={`py-1 pr-3 text-right font-mono text-2xs ${
                          (e.delta_m ?? 0) > 0 ? "text-ok" : (e.delta_m ?? 0) < 0 ? "text-fail" : "text-muted"}`}>
                          {signed(e.delta_m, 3)} m
                        </td>
                        <td className="py-1 font-mono text-2xs text-secondary">{e.selected === "hint" ? "kept" : "discarded"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {(build.event_count ?? 0) > build.events.length && (
                  <div className="pt-1 text-2xs text-muted">showing the {build.events.length} most recent of {build.event_count}</div>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Every reuse kind, wins and losses side by side. */}
      <div className="mb-3 rounded-card border border-hairline bg-panel p-3">
        <div className="mb-2 text-2xs font-semibold uppercase tracking-wider text-muted">Measured reuse, by kind</div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-left">
            <thead>
              <tr className="text-2xs uppercase tracking-wider text-muted">
                <th className="pb-1 pr-3 font-medium">kind</th>
                <th className="pb-1 pr-3 text-right font-medium">deploys</th>
                <th className="pb-1 pr-3 text-right font-medium">mean delta</th>
                <th className="pb-1 pr-3 text-right font-medium">better / worse / same</th>
                <th className="pb-1 text-right font-medium">win rate*</th>
              </tr>
            </thead>
            <tbody>{recall?.kinds.map((k) => <RecallRow key={k.kind} k={k} />)}</tbody>
          </table>
        </div>
        <div className="pt-1.5 text-2xs text-muted">
          * of the deploys that changed anything. Ties are excluded from the denominator and shown separately —
          folding them in makes a flat memory look like a winning one.
        </div>
      </div>

      {/* Where the rows came from. */}
      <div className="mb-3 grid gap-2 lg:grid-cols-3">
        <Breakdown title="Fragility gate" counts={bank.by_gate} total={bank.rows}
                   highlight={(k) => (k === "fragility_v1" ? "good" : k === "ungated" ? "bad" : null)} />
        <Breakdown title="Provenance" counts={bank.by_source} total={bank.rows}
                   highlight={(k) => (k === "real" ? "good" : k === "suite" ? "bad" : null)} />
        <Breakdown title="Banked by" counts={bank.by_door} total={bank.rows}
                   highlight={(k) => (k === "unnamed" ? "bad" : null)} />
      </div>

      {/* The caveats travel with the numbers, on the page. */}
      {!!notes?.length && (
        <div className="rounded-card border border-warn/40 bg-warn-dim/20 p-3">
          <div className="mb-1.5 flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-wider text-warn">
            <Info size={12} aria-hidden /> Read these numbers with
          </div>
          <ul className="flex flex-col gap-1">
            {notes.map((n) => (
              <li key={n} className="text-2xs leading-relaxed text-secondary">— {n}</li>
            ))}
          </ul>
        </div>
      )}
      <div className="pt-2 text-2xs text-muted">Read live from {data.memory_dir}</div>
    </div>
  );
}
