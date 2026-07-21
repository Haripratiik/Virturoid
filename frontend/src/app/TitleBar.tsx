import { useEffect, useState } from "react";
import { Minus, Square, X, ChevronDown, Hexagon } from "lucide-react";
import { useAppStore } from "@/state/app";
import { usePackages } from "@/api/queries";
import { Pill } from "@/components/ui";
import { Tip } from "@/components/Explain";

// Title bar — quiet, like a real desktop tool. Brand on the left, the active
// robot front and center (the document you're "editing"), window controls on
// the right. The whole strip is a drag region in the native window.

declare global {
  interface Window {
    pywebview?: { api?: { minimize(): void; toggle_maximize(): void; close(): void } };
  }
}

function RobotSwitcher() {
  const activePackage = useAppStore((s) => s.activePackage);
  const setActivePackage = useAppStore((s) => s.setActivePackage);
  const packages = usePackages();
  const meta = packages.data?.packages.find((p) => p.id === activePackage);
  return (
    <Tip text="The robot you are working on. Everything in the workbench — 3D view, properties, tests — follows this selection." block>
      <div className="relative flex items-center gap-2 rounded-ctl border border-hairline bg-panel px-2.5 py-1 hover:border-hairline-strong">
        <span className="font-mono text-xs text-primary">{activePackage ?? "no robot open"}</span>
        {meta && (
          <Pill
            kind={meta.valid === true ? "ok" : meta.valid === false ? "bad" : "muted"}
            label={meta.valid === true ? "valid" : meta.valid === false ? "invalid" : "unverified"}
          />
        )}
        <ChevronDown size={12} className="text-muted" aria-hidden />
        <select
          aria-label="Active robot"
          className="absolute inset-0 cursor-pointer opacity-0"
          value={activePackage ?? ""}
          onChange={(e) => setActivePackage(e.target.value || null)}
        >
          <option value="">— no robot —</option>
          {(packages.data?.packages ?? []).map((p) => (
            <option key={p.id} value={p.id}>
              {p.id}
            </option>
          ))}
        </select>
      </div>
    </Tip>
  );
}

export function TitleBar() {
  // pywebview injects window.pywebview.api ASYNCHRONOUSLY in the native window;
  // wait for the pywebviewready handshake before showing window controls.
  const [hasWindowControls, setHasWindowControls] = useState(
    typeof window !== "undefined" && !!window.pywebview?.api,
  );
  useEffect(() => {
    if (hasWindowControls) return;
    const onReady = () => setHasWindowControls(!!window.pywebview?.api);
    window.addEventListener("pywebviewready", onReady);
    return () => window.removeEventListener("pywebviewready", onReady);
  }, [hasWindowControls]);

  return (
    <header className="flex h-9 shrink-0 items-center border-b border-hairline bg-app select-none">
      <div className="pywebview-drag-region flex flex-1 items-center gap-2 pl-3">
        <Hexagon size={14} className="text-accent" aria-hidden />
        <span className="vs-display text-2xs font-semibold tracking-[0.24em] text-secondary">VIRTUROID STUDIO</span>
      </div>
      <RobotSwitcher />
      <div className="pywebview-drag-region flex flex-1 items-center justify-end">
        {hasWindowControls && (
          <div className="flex items-center">
            <button type="button" aria-label="Minimize" onClick={() => window.pywebview!.api!.minimize()} className="px-3 py-2 text-muted hover:bg-raised hover:text-primary">
              <Minus size={12} aria-hidden />
            </button>
            <button type="button" aria-label="Maximize" onClick={() => window.pywebview!.api!.toggle_maximize()} className="px-3 py-2 text-muted hover:bg-raised hover:text-primary">
              <Square size={10} aria-hidden />
            </button>
            <button type="button" aria-label="Close" onClick={() => window.pywebview!.api!.close()} className="px-3 py-2 text-muted hover:bg-fail-dim hover:text-fail">
              <X size={12} aria-hidden />
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
