import { useState } from "react";
import { Search, Brain, TrendingUp, Bot } from "lucide-react";
import { useAppStore } from "@/state/app";
import { usePackages, useFlywheel, useDesignBrain } from "@/api/queries";
import { EmptyState, Pill, QueryState, fmt, pct } from "@/components/ui";

// Library — every built robot, searchable, fed by the REAL backend
// (/api/packages + flywheel + design-brain), not hardcoded taxonomy.

function MetricCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-card border border-hairline bg-panel px-3 py-2.5">
      <span className="text-agent">{icon}</span>
      <div className="min-w-0">
        <div className="font-mono text-lg leading-tight text-primary">{value}</div>
        <div className="text-2xs text-muted">{label}</div>
        {sub && <div className="truncate text-2xs text-muted/80">{sub}</div>}
      </div>
    </div>
  );
}

export function LibraryWorkspace() {
  const packages = usePackages();
  const flywheel = useFlywheel();
  const brain = useDesignBrain();
  const activePackage = useAppStore((s) => s.activePackage);
  const setActivePackage = useAppStore((s) => s.setActivePackage);
  const setWorkspace = useAppStore((s) => s.setWorkspace);
  const [query, setQuery] = useState("");
  const [classFilter, setClassFilter] = useState("");

  if (packages.isPending || packages.isError) {
    return <QueryState loading={packages.isPending} error={packages.error} retry={() => void packages.refetch()} label="robot library" />;
  }

  const all = packages.data?.packages ?? [];
  const classes = Array.from(new Set(all.map((p) => p.robot_class).filter(Boolean))) as string[];
  const filtered = all.filter(
    (p) =>
      (!query || p.id.toLowerCase().includes(query.toLowerCase()) || (p.species ?? "").toLowerCase().includes(query.toLowerCase())) &&
      (!classFilter || p.robot_class === classFilter),
  );

  return (
    <div className="flex h-full min-h-0 flex-col overflow-y-auto bg-canvas p-4">
      {/* Flywheel + design brain headline (the moat, measured) */}
      <div className="mb-4 grid gap-2 sm:grid-cols-3">
        <MetricCard
          icon={<Bot size={18} aria-hidden />}
          label="robots in this library"
          value={String(all.length)}
          sub={packages.data?.build_root}
        />
        <MetricCard
          icon={<TrendingUp size={18} aria-hidden />}
          label="flywheel cycles"
          value={String(flywheel.data?.n_cycles ?? 0)}
          sub={flywheel.data?.headline}
        />
        <MetricCard
          icon={<Brain size={18} aria-hidden />}
          label="design-brain coverage"
          value={brain.data?.error ? "unavailable" : fmt(brain.data?.archive_coverage ?? 0)}
          sub={brain.data?.error ?? brain.data?.headline ?? `${fmt(brain.data?.provenance_edges ?? 0)} provenance edges`}
        />
      </div>
      {(flywheel.isError || brain.isError || brain.data?.error) && (
        <div className="mb-3 rounded-card border border-warn/40 bg-warn-dim/30 p-2 text-2xs text-warn">
          {brain.data?.error ?? "Some library metrics could not be loaded. Robot packages remain available; retry by refreshing this panel."}
        </div>
      )}

      {/* Search + filters */}
      <div className="mb-3 flex items-center gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-ctl border border-hairline bg-panel px-2.5 py-1.5 focus-within:border-accent/50">
          <Search size={13} className="text-muted" aria-hidden />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search robots by name or species…"
            aria-label="Search robots"
            className="w-full bg-transparent text-xs text-primary outline-none placeholder:text-muted"
          />
        </div>
        <select
          value={classFilter}
          onChange={(e) => setClassFilter(e.target.value)}
          aria-label="Filter by class"
          className="rounded-ctl border border-hairline bg-panel px-2 py-1.5 text-xs text-secondary"
        >
          <option value="">All classes</option>
          {classes.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      {/* Robot grid */}
      {filtered.length === 0 ? (
        <EmptyState
          title={all.length ? "No robots match the filter" : "No robots yet"}
          sub={all.length ? "Try a different search." : "Build your first robot from the Agent Rail — it will appear here."}
        />
      ) : (
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => {
                setActivePackage(p.id);
                setWorkspace("design");
              }}
              className={`flex flex-col gap-1.5 rounded-card border p-3 text-left transition-colors ${
                activePackage === p.id ? "border-accent/60 bg-accent-dim/30" : "border-hairline bg-panel hover:border-hairline-strong"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="truncate font-mono text-xs text-primary">{p.id}</span>
                {p.valid === true && <Pill kind="ok" label="valid" />}
                {p.valid === false && <Pill kind="bad" label="invalid" />}
                {p.valid == null && <Pill kind="muted" label="unverified" />}
              </div>
              <div className="text-2xs text-secondary">
                {p.robot_class ?? "unknown class"}
                {p.species ? ` · ${p.species}` : ""}
              </div>
              <div className="mt-auto flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-2xs text-muted">
                <span>{p.dof != null ? `${p.dof} DOF` : "— DOF"}</span>
                <span>{p.scene_count} scenes</span>
                {p.spec?.success != null && <span>success {pct(p.spec.success)}</span>}
                {p.spec?.cost_usd != null && <span>${Math.round(p.spec.cost_usd).toLocaleString()}</span>}
              </div>
              {p.honesty && (p.honesty.fidelity_flags ?? 0) > 0 && (
                <Pill kind="warn" label={`${p.honesty.fidelity_flags} honesty flag(s)`} />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
