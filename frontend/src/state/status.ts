import type { PackageMeta, PackageStatus, ReadinessLedger } from "@/api/types";

// THE robot-status accessor for the whole UI.
//
// The verdict is computed ONCE on the server (services/package_status.py) and shipped on
// /api/packages; every surface — header chip, library card, inspector, status bar, Verify tab —
// renders that same object, so a customer can never see a green "VALID" next to "EXPORT BLOCKED".
// The fallback below only covers the window where the package list has not loaded yet, and it uses
// the SAME words for the same facts.

export function robotStatus(
  meta: PackageMeta | null | undefined,
  ledger?: ReadinessLedger | null,
): PackageStatus | null {
  if (meta?.status) return meta.status;
  if (!ledger) return null;
  return {
    label: ledger.safe_to_export ? "EXPORT-READY" : "EXPORT BLOCKED",
    kind: ledger.safe_to_export ? "ok" : "bad",
    detail: ledger.safe_to_export
      ? `Every required readiness gate attained a real result (highest: ${ledger.highest_attained ?? "none"}).`
      : ledger.issues?.[0] ?? "Export blocked by an unmet required readiness gate.",
    notes: [],
    contract_ok: null,
    safe_to_export: ledger.safe_to_export,
    buildable: null,
    highest_attained: ledger.highest_attained ?? null,
  };
}
