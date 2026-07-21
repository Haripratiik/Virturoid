import type { ReactNode } from "react";
import { CheckCircle2, XCircle, AlertTriangle, MinusCircle, CircleDashed, RefreshCw } from "lucide-react";

// Shared primitives. Semantic colors are used ONLY for verdicts and always
// carry an icon + label (never color-only status).

export type VerdictKind = "ok" | "bad" | "warn" | "skip" | "muted";

const VERDICT_STYLE: Record<VerdictKind, { cls: string; Icon: typeof CheckCircle2 }> = {
  ok: { cls: "text-ok bg-ok-dim", Icon: CheckCircle2 },
  bad: { cls: "text-fail bg-fail-dim", Icon: XCircle },
  warn: { cls: "text-warn bg-warn-dim", Icon: AlertTriangle },
  skip: { cls: "text-skip bg-skip-dim", Icon: MinusCircle },
  muted: { cls: "text-muted bg-skip-dim", Icon: CircleDashed },
};

export function verdictKind(status: string | null | undefined): VerdictKind {
  const s = String(status ?? "").toLowerCase();
  if (s === "attained" || s === "true" || s === "valid" || s === "ok") return "ok";
  if (s === "not_required") return "skip";
  if (s === "scaffolded" || s === "not_run") return "warn";
  if (!s) return "muted";
  return "bad"; // placeholder, dry_run, collision, mismatch, ...
}

export function Pill({ kind, label }: { kind: VerdictKind; label: string }) {
  const { cls, Icon } = VERDICT_STYLE[kind];
  return (
    <span className={`inline-flex items-center gap-1 rounded-ctl px-1.5 py-0.5 text-2xs font-medium uppercase tracking-wide ${cls}`}>
      <Icon size={11} strokeWidth={2} aria-hidden />
      {label}
    </span>
  );
}

export function EmptyState({ icon, title, sub, action }: { icon?: ReactNode; title: string; sub?: string; action?: ReactNode }) {
  return (
    <div className="flex h-full min-h-24 flex-col items-center justify-center gap-1.5 p-6 text-center">
      {icon && <div className="text-muted">{icon}</div>}
      <div className="text-sm font-medium text-secondary">{title}</div>
      {sub && <div className="max-w-72 text-xs text-muted">{sub}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

/** Transport state for honesty-critical surfaces. Missing artifacts and a
 * failed request are intentionally rendered as different states. */
export function QueryState({ loading, error, retry, label = "data" }: {
  loading: boolean; error: unknown; retry?: () => void; label?: string;
}) {
  if (loading) return <EmptyState icon={<Spinner size={18} />} title={`Loading ${label}`} sub="Contacting the local Virturoid backend." />;
  if (!error) return null;
  const detail = error instanceof Error ? error.message : "Unknown request failure";
  return (
    <EmptyState
      icon={<AlertTriangle size={22} />}
      title={`Could not load ${label}`}
      sub={detail}
      action={retry ? <Btn variant="danger" onClick={retry}><RefreshCw size={12} />Retry</Btn> : undefined}
    />
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="px-3 pb-1 pt-3 text-2xs font-semibold uppercase tracking-wider text-muted">{children}</div>
  );
}

export function KV({ k, v, mono = true }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 px-3 py-1">
      <span className="text-xs text-muted">{k}</span>
      <span className={`text-right text-xs text-primary ${mono ? "font-mono" : ""}`}>{v ?? "—"}</span>
    </div>
  );
}

export function Btn({
  children,
  onClick,
  variant = "default",
  disabled,
  title,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "ghost" | "danger";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
}) {
  const base =
    "inline-flex items-center justify-center gap-1.5 rounded-ctl px-2.5 py-1 text-xs font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45";
  const styles: Record<string, string> = {
    default: "bg-raised text-secondary hover:text-primary border border-hairline hover:border-hairline-strong",
    primary: "bg-accent/15 text-accent border border-accent/40 hover:bg-accent/25",
    ghost: "text-secondary hover:text-primary hover:bg-raised",
    danger: "bg-fail-dim text-fail border border-fail/40 hover:bg-fail/25",
  };
  return (
    <button type={type} className={`${base} ${styles[variant]}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

export function Spinner({ size = 14 }: { size?: number }) {
  return (
    <span
      className="inline-block animate-spin rounded-full border-2 border-hairline border-t-accent"
      style={{ width: size, height: size }}
      role="status"
      aria-label="loading"
    />
  );
}

export function fmt(v: unknown, digits = 2): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(digits);
  }
  return String(v);
}

export function pct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${Math.round(v * 100)}%`;
}
