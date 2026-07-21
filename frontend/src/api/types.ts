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

export interface PackageMeta {
  id: string;
  scene_count: number;
  has_meshes: boolean;
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

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

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
