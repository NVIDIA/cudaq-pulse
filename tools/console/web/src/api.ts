export interface ApiError {
  stage?: string;
  kind: string;
  message: string;
  line?: number | null;
  trace?: string[];
}

export interface Example {
  file: string;
  title: string;
  kernel: string;
  args: string;
}

export interface ParamSpec {
  name: string;
  label: string;
  default: number;
  min: number;
  max: number;
  step: number;
  unit: string;
}

export interface ExperimentInfo {
  id: string;
  title: string;
  plain: string;
  detail: string;
  example: string;
  params: ParamSpec[];
}

export interface Meta {
  version: string;
  targets: { id: string; name: string }[];
  passes: string[];
  default_passes: string[];
  schedules: string[];
  examples: Example[];
  experiments: ExperimentInfo[];
}

export interface KernelInfo {
  name: string;
  params: { name: string; annotation: string }[];
  schedule: string | null;
}

/** The schedule as columns; `kind` indexes `kinds`. Times in clock cycles. */
export interface Events {
  kinds: string[];
  kind: number[];
  line: number[];
  qudit: number[];
  start: number[];
  duration: number[];
}

export interface LineInfo {
  line: number;
  role: string;
  qudit: number;
}

export interface Metrics {
  trace_ms: number;
  ffi_ms: number;
  passes_ms: number;
  schedule_ms: number;
  total_ms: number;
  op_count: number;
  target: string;
}

export interface ControllerProgram {
  format: string;
  target: string;
  clock_ghz: number;
  schedule: string;
  qudits: number[];
  lines: LineInfo[];
  events: Record<string, unknown>[];
  epochs: Record<string, unknown>[];
  calls: Record<string, unknown>[];
  manifest: Record<string, unknown>;
  limits: Record<string, unknown> | null;
  event_count?: number;
}

export interface CompileResult {
  ok: boolean;
  error?: ApiError;
  kernel?: string;
  target?: string;
  passes?: string[];
  schedule_policy?: string;
  clock_ghz?: number;
  qudits?: number[];
  parameters?: string[];
  machine?: string | null;
  stages?: {
    traced: { mlir: string; metrics: Metrics };
    optimized: { mlir: string; metrics: Metrics };
    llvm?: { mlir?: string; error?: string };
  };
  events?: Events;
  events_truncated?: boolean;
  cached?: boolean;
  server_ms?: number;
  epochs?: Record<string, unknown>[];
  calls?: Record<string, unknown>[];
  controller?: ControllerProgram;
  controller_error?: ApiError;
}

export interface CompileRequest {
  source: string;
  kernel: string;
  args: string;
  target: string;
  passes: string[];
  schedule: string;
  llvm?: boolean;
}

export interface SimResult {
  ok: boolean;
  error?: ApiError;
  qudits?: number[];
  dims?: number[];
  series?: { t_ns: number; populations: number[][] }[];
  clock_ghz?: number;
  branches?: { outcomes: number[]; p: number }[];
  wall_s?: number;
}

export interface DeviceInfo {
  id: string;
  name: string;
  architecture: string;
  clock_ghz: number;
  calibrated: boolean;
  controller_backends: string[];
  couplings: [number, number][];
  qudits: {
    index: number;
    carrier: number;
    label: string;
    frequency_ghz: number;
    anharmonicity_mhz: number;
    t1_us: number | null;
    t2_star_us: number | null;
    t2_echo_us: number | null;
    thermal_population: number | null;
    readout_ghz: number;
  }[];
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  health: () => getJson<{ ok: boolean; version: string }>("/api/health"),
  meta: () => getJson<Meta>("/api/meta"),
  example: (file: string) => getJson<{ source: string }>(`/api/example?file=${encodeURIComponent(file)}`),
  device: (target: string) => getJson<DeviceInfo>(`/api/device?target=${encodeURIComponent(target)}`),
  kernels: (source: string) =>
    postJson<{ ok: boolean; kernels?: KernelInfo[]; error?: ApiError }>("/api/kernels", { source }),
  compile: (req: CompileRequest) => postJson<CompileResult>("/api/compile", req),
  simulate: (req: CompileRequest & { points: number; levels: number }) =>
    postJson<SimResult>("/api/simulate", req),
  run: <T = any>(experiment: string, params: Record<string, number>) =>
    postJson<RunResult<T>>("/api/run", { experiment, params }),
};

export interface RunResult<T> {
  ok: boolean;
  error?: ApiError | string;
  experiment?: string;
  params?: Record<string, number>;
  data?: T;
  server_ms?: number;
  cached?: boolean;
}

