// API + artifact shapes, grounded in the Python server (ui_server.py) and real package JSON.

export interface PackageSpecSummary {
  summary?: string | null;
  mass_kg?: number | null;
  cost_usd?: number | null;
  power_w?: number | null;
  success?: number | null;
  task?: string | null;
}

export interface PackageHonesty {
  mass_fidelity_ratio?: number | null;
  fidelity_flags?: number;
  spec_all_honored?: boolean | null;
  spec_constraints?: number;
}

/** The ONE robot-status verdict, derived server-side in services/package_status.py. Every surface
 * renders this object; the individual facts below it (contract_ok / buildable) are reported under
 * their own names, because they mean different things and must never all read "valid". */
export interface PackageStatus {
  label: string;                    // EXPORT-READY | EXPORT BLOCKED | PACKAGE INCOMPLETE | UNVERIFIED | NO ROBOT
  kind: "ok" | "bad" | "warn" | "muted";
  detail: string;
  notes: string[];
  contract_ok: boolean | null;      // declared artifacts exist + parse (file integrity)
  safe_to_export: boolean | null;   // every REQUIRED readiness gate attained a real result
  buildable: boolean | null;        // real actuators cover every joint + structure survives loads
  highest_attained: string | null;
}

export interface PackageMeta {
  id: string;
  scene_count: number;
  has_meshes: boolean;
  status: PackageStatus;
  /** Raw package-contract fact, kept on the wire for older clients. Render `status`, not this. */
  valid: boolean | null;
  robot_class: string | null;
  species: string | null;
  dof: number | null;
  spec: PackageSpecSummary | null;
  honesty: PackageHonesty | null;
}

export interface PackagesResponse {
  build_root: string;
  packages: PackageMeta[];
}

export interface AssistantStatus {
  provider: string;
  model: string;
  online: boolean;
  models: string[];
  model_available?: boolean;
}

export interface ToolSpec {
  name: string;
  description: string;
  parameters: { type: string; required?: string[]; properties?: Record<string, unknown> };
  heavy: boolean;
}

export interface ScorecardRow {
  claim: string;
  verdict: string;
  honest: boolean;
  evidence: unknown;
}

export interface Scorecard {
  rows: ScorecardRow[];
  n_claims: number;
  n_honest?: number;
  n_flagged?: number;
  headline: string;
  error?: string;
}

export interface LedgerStage {
  stage: string;
  status: string;
  detail?: string;
  evidence?: Record<string, unknown>;
}

export interface ReadinessLedger {
  package_dir?: string;
  robot_class?: string;
  safe_to_export: boolean;
  highest_attained?: string;
  required: string[];
  enforced?: boolean;
  issues: string[];
  stages: LedgerStage[];
}

export interface FlywheelResponse {
  series: Array<Record<string, number>>;
  n_cycles: number;
  compounding: boolean;
  headline: string;
}

export interface DesignBrainResponse {
  archive_coverage?: number;
  provenance_edges?: number;
  headline?: string;
  error?: string;
  [key: string]: unknown;
}

// ---- The verified-morphology memory (/api/moat) ----
// Mirrors services/moat_panel.py. Losses and ties are first-class fields, not derived from wins, because the
// panel's whole job is to be able to render a memory that is currently NOT paying off.

export interface MoatRecallKind {
  kind: string;
  means: string;
  edges: number;
  mean_delta_m: number | null;
  wins: number;
  losses: number;
  ties: number;
  decided_win_rate: number | null;
  direction: "helps" | "hurts" | "neutral";
}

export interface MoatRecallEvent {
  when: string | null;
  kind: string;
  gene_id: string;
  region: string | null;
  delta_m: number | null;
  source?: string | null;
  selected?: string | null;
  hint_forward_m?: number | null;
  default_forward_m?: number | null;
  hint_credible?: boolean | null;
  default_credible?: boolean | null;
}

export interface MoatResponse {
  memory_dir?: string;
  db_present?: boolean;
  error?: string;
  bank?: {
    task: string;
    rows: number;
    by_task: Record<string, number>;
    by_gate: Record<string, number>;
    gated_rows: number;
    gated_fraction: number;
    by_door: Record<string, number>;
    by_source: Record<string, number>;
    by_class: Record<string, number>;
    bodies: {
      distinct: number;
      largest_share_body: string | null;
      largest_share_rows: number;
      largest_share_fraction: number;
    };
  };
  recall?: { kinds: MoatRecallKind[]; dominant_kind: string | null; headline: string };
  this_build?: {
    matched: boolean;
    gene_ids: string[];
    events: MoatRecallEvent[];
    event_count?: number;
    kept?: number;
    mean_delta_m?: number | null;
    summary: string;
  };
  notes?: string[];
}

export interface EpisodeGeom {
  type: string;
  size?: number[];
  rgba?: number[];
  mesh_uri?: string;
  mesh_scale?: number;
}

export interface EpisodeView {
  geoms: EpisodeGeom[];
  frames: number[][][];
  frame_count: number;
  task?: string;
  outcome?: Record<string, unknown>;
  error?: string;
}

export interface SceneIndexEntry {
  scene_set_id: string;
  scene_id: string;
  purpose: string;
  mujoco_xml: string;
  object_count?: number;
}

export interface SceneIndex {
  scene_count: number;
  scenes: SceneIndexEntry[];
}

export interface GenomeJointLimit {
  lower?: number;
  upper?: number;
  velocity?: number;
  effort?: number;
}

export interface GenomeJoint {
  name: string;
  joint_type: string;
  parent_link: string;
  child_link: string;
  axis_xyz?: number[];
  limit?: GenomeJointLimit;
  actuator_component_id?: string;
}

export interface GenomeSensor {
  name: string;
  sensor_component_id?: string;
  parent_link?: string;
  transform_xyz_rpy?: number[];
}

export interface RobotGenome {
  id: string;
  version?: string;
  name?: string;
  species?: string;
  morphology_template_id?: string;
  links: string[];
  joints: GenomeJoint[];
  sensors?: GenomeSensor[];
  [key: string]: unknown;
}

export interface BomLine {
  part: string;
  category: string;
  qty: number;
  unit_mass_kg?: number;
  unit_price_usd?: number;
  detail?: string;
  mass_kg?: number;
  price_usd?: number;
}

export interface BillOfMaterials {
  robot_class?: string;
  dof?: number;
  actuator_map?: Record<string, string>;
  lines: BomLine[];
  [key: string]: unknown;
}

export interface SpecSheet {
  name?: string;
  species?: string;
  robot_class?: string;
  dof?: number;
  physical?: { mass_kg?: number; actuators?: number; size_m?: Record<string, number> };
  power_and_cost?: { est_power_draw_w?: number; est_parts_cost_usd?: number };
  actuation?: { actuator_types?: string[]; peak_joint_torque_nm?: number };
  sensing?: string[];
  compute?: string[];
  performance?: { task?: string; success_rate?: number };
  summary?: string;
}

export interface BuildDecision {
  iteration: number;
  stage: string;
  action: string;
  detail: string;
  success_before: number;
  success_after: number;
}

export interface BuildSummary {
  prompt?: string;
  task_type?: string;
  selected_robot_class?: string;
  selected_species?: string;
  package_valid?: boolean;
  decisions?: BuildDecision[];
  notes?: string[];
  artifacts?: Record<string, string>;
  readiness?: {
    ready?: boolean;
    safe_to_export?: boolean;
    highest_attained?: string;
    failed_required_gates?: string[];
  };
  species_exact?: boolean;
  species_note?: string;
  compute?: Record<string, unknown>;
}

export interface SpecComplianceConstraint {
  constraint?: string;
  requested?: unknown;
  delivered?: unknown;
  honored?: boolean;
  note?: string;
  [key: string]: unknown;
}

export interface SpecCompliance {
  all_honored?: boolean;
  constraints?: SpecComplianceConstraint[];
  [key: string]: unknown;
}

// ---- Jobs (the additive /api/jobs surface) ----

/** `no_output` = the job ran honestly to completion and produced nothing that was asked for (an
 * impossible/ambiguous prompt, a rejected ungrounded design). NOT a success, NOT an error. */
export type JobStatus = "queued" | "running" | "succeeded" | "no_output" | "failed" | "cancelled";

export interface JobEvent {
  seq: number;
  ts: number;
  stage: string;
  message: string;
  data?: Record<string, unknown>;
}

export interface Job {
  id: string;
  kind: string;
  args: Record<string, unknown>;
  status: JobStatus;
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  result?: Record<string, unknown> | null;
  error?: string | null;
  n_events: number;
}

export interface JobEventsResponse {
  job: Job;
  events: JobEvent[];
}

export interface ChatResponse {
  role: string;
  content: string;
  action: string;
  build?: Record<string, unknown>;
  build_intent?: {
    prompt?: string;
    sensor?: string | null;
    payload_kg?: number | null;
    reach_m?: number | null;
    train?: boolean;
  };
  model_used?: boolean;
}
