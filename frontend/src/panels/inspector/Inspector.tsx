import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Cpu, GitBranch, Package2, Radar } from "lucide-react";
import { useAppStore, useSelectionStore } from "@/state/app";
import { useBom, useGenome, useSpecSheet } from "@/api/queries";
import { engine } from "@/viewport/engine/engine";
import { useViewportStore } from "@/viewport/viewportState";
import { EmptyState, KV, Pill, QueryState, SectionLabel, fmt, pct } from "@/components/ui";
import { Term, Tip } from "@/components/Explain";
import { usePackages } from "@/api/queries";
import { robotStatus } from "@/state/status";
import type { GenomeJoint } from "@/api/types";

// Inspector: select → inspect. Tabs follow the Isaac Stage→Property backbone:
// Properties / Anatomy / Parts (BOM) / Sensors. Building happens through the
// Agent panel (or /tools) — no duplicate build form here.

type Tab = "properties" | "anatomy" | "parts" | "sensors";

const TABS: Array<{ id: Tab; label: string; Icon: typeof Cpu; hint: string }> = [
  { id: "properties", label: "Properties", Icon: Cpu, hint: "The robot's vital stats — read from its real design files, not estimates." },
  { id: "anatomy", label: "Anatomy", Icon: GitBranch, hint: "The skeleton: body parts (links) connected by joints. Click any part to highlight it in the 3D view." },
  { id: "parts", label: "Parts", Icon: Package2, hint: "The bill of materials — the actual shopping list to build this robot physically, with weights and prices." },
  { id: "sensors", label: "Sensors", Icon: Radar, hint: "What the robot senses the world with, and where each sensor is mounted." },
];

function JointRow({ joint }: { joint: GenomeJoint }) {
  const selection = useSelectionStore((s) => s.selection);
  const select = useSelectionStore((s) => s.select);
  const selected = selection?.kind === "joint" && selection.id === joint.name;
  const [angle, setAngle] = useState(0);
  const viewportMode = useViewportStore((s) => s.mode);
  const viewportStatus = useViewportStore((s) => s.status); // refresh after asynchronous URDF load completes
  const lower = joint.limit?.lower ?? -3.14;
  const upper = joint.limit?.upper ?? 3.14;
  const movable = joint.joint_type === "revolute" || joint.joint_type === "continuous";
  const poseable = viewportMode === "robot" && Boolean(engine.robot?.joints?.[joint.name]);
  return (
    <div
      className={`rounded-ctl border px-2 py-1.5 ${selected ? "border-accent/50 bg-accent-dim/50" : "border-hairline bg-raised/40"}`}
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => select({ kind: "joint", id: joint.name })}
      >
        <span className="truncate font-mono text-2xs text-primary">{joint.name}</span>
        <span className="shrink-0 text-2xs text-muted">{joint.joint_type}</span>
      </button>
      {movable && (
        <div className="mt-1 flex items-center gap-2">
          <input
            type="range"
            min={lower}
            max={upper}
            step={0.01}
            value={angle}
            aria-label={`${joint.name} angle`}
            className="vs-scrub flex-1"
            disabled={!poseable}
            onChange={(e) => {
              const v = Number(e.target.value);
              setAngle(v);
              engine.setJointValue(joint.name, v); // live pose — same joints the URDF drives
            }}
          />
          <span className="w-11 text-right font-mono text-2xs text-muted">{angle.toFixed(2)}</span>
        </div>
      )}
      {movable && !poseable && (
        <div className="mt-1 text-2xs text-muted">
          {viewportMode === "robot" && viewportStatus.includes("Loading")
            ? "Loading the Robot view before joint posing is available."
            : "Switch to Robot mode to pose this URDF joint; Scene and Episode replay are read-only."}
        </div>
      )}
      <div className="mt-1 grid grid-cols-2 gap-x-3 font-mono text-2xs text-muted">
        <span>lim {fmt(joint.limit?.lower)}…{fmt(joint.limit?.upper)}</span>
        <span className="text-right">effort {fmt(joint.limit?.effort)} Nm</span>
      </div>
    </div>
  );
}

function AnatomyTree() {
  const activePackage = useAppStore((s) => s.activePackage);
  const genome = useGenome(activePackage);
  const selection = useSelectionStore((s) => s.selection);
  const select = useSelectionStore((s) => s.select);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  if (genome.isPending || genome.isError) {
    return <QueryState loading={genome.isPending} error={genome.error} retry={() => void genome.refetch()} label="robot anatomy" />;
  }

  const childrenOf = useMemo(() => {
    const map = new Map<string, GenomeJoint[]>();
    for (const j of genome.data?.joints ?? []) {
      if (!map.has(j.parent_link)) map.set(j.parent_link, []);
      map.get(j.parent_link)!.push(j);
    }
    return map;
  }, [genome.data]);

  if (!genome.data) {
    return <EmptyState title="No genome" sub="This package has no robot_genome.json — engine-built bodies expose anatomy via Episode replay instead." />;
  }

  const roots = genome.data.links.filter((l) => !genome.data!.joints.some((j) => j.child_link === l));

  function renderLink(link: string, depth: number) {
    const joints = childrenOf.get(link) ?? [];
    const isOpen = open[link] !== false;
    const selected = selection?.kind === "link" && selection.id === link;
    return (
      <div key={link}>
        <div
          className={`flex items-center gap-1 rounded-ctl py-0.5 pr-1 ${selected ? "bg-accent-dim text-accent" : "text-secondary hover:bg-raised/60"}`}
          style={{ paddingLeft: depth * 12 + 4 }}
        >
          {joints.length > 0 ? (
            <button
              type="button"
              aria-label={isOpen ? "collapse" : "expand"}
              onClick={() => setOpen((o) => ({ ...o, [link]: !isOpen }))}
              className="text-muted"
            >
              {isOpen ? <ChevronDown size={11} aria-hidden /> : <ChevronRight size={11} aria-hidden />}
            </button>
          ) : (
            <span className="w-[11px]" />
          )}
          <button
            type="button"
            className="flex-1 truncate text-left font-mono text-2xs"
            onClick={() => {
              select({ kind: "link", id: link });
              engine.highlightByName(link);
            }}
          >
            {link}
          </button>
        </div>
        {isOpen &&
          joints.map((j) => (
            <div key={j.name}>
              <button
                type="button"
                className={`flex w-full items-center gap-1.5 rounded-ctl py-0.5 pr-1 text-left ${
                  selection?.kind === "joint" && selection.id === j.name ? "bg-accent-dim text-accent" : "text-muted hover:bg-raised/60"
                }`}
                style={{ paddingLeft: (depth + 1) * 12 + 4 }}
                onClick={() => select({ kind: "joint", id: j.name })}
              >
                <GitBranch size={9} aria-hidden />
                <span className="truncate font-mono text-2xs">{j.name}</span>
                <span className="ml-auto shrink-0 text-2xs opacity-70">{j.joint_type}</span>
              </button>
              {renderLink(j.child_link, depth + 1)}
            </div>
          ))}
      </div>
    );
  }

  return <div className="p-2">{roots.map((r) => renderLink(r, 0))}</div>;
}

function PropertiesTab() {
  const activePackage = useAppStore((s) => s.activePackage);
  const packages = usePackages();
  const spec = useSpecSheet(activePackage);
  const genome = useGenome(activePackage);
  const selection = useSelectionStore((s) => s.selection);
  const meta = packages.data?.packages.find((p) => p.id === activePackage);
  const status = robotStatus(meta);

  if (!activePackage) return <EmptyState title="No robot selected" sub="Build one via the Agent Rail or pick one in Library." />;

  // Joint selection → joint properties (select → inspect → edit backbone).
  if (selection?.kind === "joint" && genome.data) {
    const joint = genome.data.joints.find((j) => j.name === selection.id);
    if (joint) {
      return (
        <div className="py-1">
          <SectionLabel>Joint · {joint.name}</SectionLabel>
          <KV k="type" v={joint.joint_type} />
          <KV k="parent" v={joint.parent_link} />
          <KV k="child" v={joint.child_link} />
          <KV k="axis" v={(joint.axis_xyz ?? []).join(", ")} />
          <KV k="limit lower" v={`${fmt(joint.limit?.lower)} rad`} />
          <KV k="limit upper" v={`${fmt(joint.limit?.upper)} rad`} />
          <KV k="velocity" v={`${fmt(joint.limit?.velocity)} rad/s`} />
          <KV k="effort" v={`${fmt(joint.limit?.effort)} Nm`} />
          <KV k="actuator" v={joint.actuator_component_id ?? "—"} />
          <div className="p-3">
            <JointRow joint={joint} />
          </div>
        </div>
      );
    }
  }

  return (
    <div className="py-1">
      <SectionLabel>Robot</SectionLabel>
      <KV k="name" v={spec.data?.name ?? activePackage} mono={false} />
      <KV k="class" v={meta?.robot_class ?? spec.data?.robot_class ?? "—"} />
      <KV k="species" v={meta?.species ?? spec.data?.species ?? "—"} />
      <KV k={<Term k="dof">DOF</Term>} v={meta?.dof ?? spec.data?.dof ?? "—"} />
      {/* The shared verdict, plus the facts UNDER it named individually — the inspector has the
          room, so "the contract parses" and "real parts cover every joint" say so out loud
          instead of collapsing into one generic word. */}
      <div className="flex flex-wrap items-center gap-1 px-3 py-1">
        {status && (
          <Tip text={status.detail} side="bottom">
            <Pill kind={status.kind} label={status.label} />
          </Tip>
        )}
        {status?.buildable === false && <Pill kind="warn" label="not buildable from real parts" />}
        {status?.contract_ok === false && <Pill kind="bad" label="contract artifacts missing" />}
      </div>
      {status?.notes.map((n) => (
        <p key={n} className="px-3 pb-1 text-2xs leading-relaxed text-muted">
          {n}
        </p>
      ))}
      <SectionLabel>Physical</SectionLabel>
      <KV k="mass" v={spec.data?.physical?.mass_kg != null ? `${spec.data.physical.mass_kg} kg` : "—"} />
      <KV k="actuators" v={spec.data?.physical?.actuators ?? "—"} />
      <KV
        k="size (L×W×H)"
        v={
          spec.data?.physical?.size_m
            ? `${fmt(spec.data.physical.size_m.length_m)} × ${fmt(spec.data.physical.size_m.width_m)} × ${fmt(spec.data.physical.size_m.height_m)} m`
            : "—"
        }
      />
      <SectionLabel>Power &amp; cost</SectionLabel>
      <KV k="power draw" v={spec.data?.power_and_cost?.est_power_draw_w != null ? `${spec.data.power_and_cost.est_power_draw_w} W` : "—"} />
      <KV
        k="parts cost"
        v={spec.data?.power_and_cost?.est_parts_cost_usd != null ? `$${spec.data.power_and_cost.est_parts_cost_usd.toLocaleString()}` : "—"}
      />
      <KV k={<Term k="torque">peak torque</Term>} v={spec.data?.actuation?.peak_joint_torque_nm != null ? `${spec.data.actuation.peak_joint_torque_nm} Nm` : "—"} />
      <SectionLabel>Performance</SectionLabel>
      <KV k="task" v={spec.data?.performance?.task ?? "—"} />
      <KV k={<Term k="successRate">success</Term>} v={pct(spec.data?.performance?.success_rate)} />
      {spec.data?.summary && <p className="px-3 py-2 text-2xs leading-relaxed text-muted">{spec.data.summary}</p>}
      {genome.data && (
        <>
          <SectionLabel>
            <Term k="joint">Joints</Term> ({genome.data.joints.length})
          </SectionLabel>
          <div className="flex flex-col gap-1.5 px-3 pb-3">
            {genome.data.joints.map((j) => (
              <JointRow key={j.name} joint={j} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function PartsTab() {
  const activePackage = useAppStore((s) => s.activePackage);
  const bom = useBom(activePackage);
  if (!activePackage) return <EmptyState title="No robot selected" />;
  if (bom.isPending || bom.isError) {
    return <QueryState loading={bom.isPending} error={bom.error} retry={() => void bom.refetch()} label="bill of materials" />;
  }
  if (!bom.data) return <EmptyState title="No BOM" sub="This package has no bill_of_materials.json." />;
  const lines = bom.data.lines ?? [];
  const totalMass = lines.reduce((a, l) => a + (l.mass_kg ?? 0), 0);
  const totalCost = lines.reduce((a, l) => a + (l.price_usd ?? 0), 0);
  return (
    <div className="py-1">
      <SectionLabel>
        <Term k="bom">Bill of materials</Term>
      </SectionLabel>
      <div className="px-3">
        <table className="w-full border-collapse">
          <thead>
            <tr className="text-left text-2xs uppercase tracking-wide text-muted">
              <th className="py-1 pr-2 font-medium">Part</th>
              <th className="py-1 pr-2 text-right font-medium">Qty</th>
              <th className="py-1 pr-2 text-right font-medium">kg</th>
              <th className="py-1 text-right font-medium">USD</th>
            </tr>
          </thead>
          <tbody>
            {lines.map((l, i) => (
              <tr key={`${l.part}_${i}`} className="border-t border-hairline align-top">
                <td className="py-1.5 pr-2">
                  <div className="text-2xs text-primary">{l.part}</div>
                  <div className="text-2xs text-muted">{l.detail ?? l.category}</div>
                </td>
                <td className="py-1.5 pr-2 text-right font-mono text-2xs text-secondary">{l.qty}</td>
                <td className="py-1.5 pr-2 text-right font-mono text-2xs text-secondary">{fmt(l.mass_kg)}</td>
                <td className="py-1.5 text-right font-mono text-2xs text-secondary">{l.price_usd?.toLocaleString() ?? "—"}</td>
              </tr>
            ))}
            <tr className="border-t border-hairline-strong">
              <td className="py-1.5 pr-2 text-2xs font-semibold text-primary">Total</td>
              <td />
              <td className="py-1.5 pr-2 text-right font-mono text-2xs font-semibold text-primary">{totalMass.toFixed(2)}</td>
              <td className="py-1.5 text-right font-mono text-2xs font-semibold text-primary">${totalCost.toLocaleString()}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {bom.data.actuator_map && (
        <>
          <SectionLabel>Actuator map</SectionLabel>
          {Object.entries(bom.data.actuator_map).map(([k, v]) => (
            <KV key={k} k={k} v={v} />
          ))}
        </>
      )}
    </div>
  );
}

function SensorsTab() {
  const activePackage = useAppStore((s) => s.activePackage);
  const genome = useGenome(activePackage);
  const sensors = genome.data?.sensors ?? [];
  if (!activePackage) return <EmptyState title="No robot selected" />;
  if (genome.isPending || genome.isError) {
    return <QueryState loading={genome.isPending} error={genome.error} retry={() => void genome.refetch()} label="sensor declaration" />;
  }
  if (!sensors.length) return <EmptyState title="No sensors" sub="The genome declares no sensor components." />;
  return (
    <div className="py-1">
      <SectionLabel>Sensors ({sensors.length})</SectionLabel>
      {sensors.map((s) => (
        <div key={s.name} className="mx-3 mb-2 rounded-ctl border border-hairline bg-raised/40 p-2">
          <div className="font-mono text-2xs text-primary">{s.name}</div>
          <div className="mt-0.5 text-2xs text-muted">component: {s.sensor_component_id ?? "—"}</div>
          <div className="text-2xs text-muted">mounted on: {s.parent_link ?? "—"}</div>
        </div>
      ))}
    </div>
  );
}

export function Inspector() {
  const [tab, setTab] = useState<Tab>("properties");
  const activePackage = useAppStore((s) => s.activePackage);
  return (
    <div className="flex h-full min-h-0 flex-col bg-panel">
      <div className="flex items-center border-b border-hairline" role="tablist" aria-label="Inspector tabs">
        {TABS.map(({ id, label, Icon, hint }) => (
          <Tip key={id} text={hint} block>
            <button
              type="button"
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`flex flex-1 items-center justify-center gap-1 border-b-2 px-1 py-2 text-2xs transition-colors ${
                tab === id ? "border-accent text-accent" : "border-transparent text-muted hover:text-secondary"
              }`}
            >
              <Icon size={12} aria-hidden />
              <span className="hidden xl:inline">{label}</span>
            </button>
          </Tip>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {tab === "properties" && <PropertiesTab />}
        {tab === "anatomy" && (activePackage ? <AnatomyTree /> : <EmptyState title="No robot selected" />)}
        {tab === "parts" && <PartsTab />}
        {tab === "sensors" && <SensorsTab />}
      </div>
    </div>
  );
}
