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

export interface Meta {
  version: string;
  targets: { id: string; name: string }[];
  passes: string[];
  default_passes: string[];
  schedules: string[];
  examples: Example[];
}

export interface KernelInfo {
  name: string;
  params: { name: string; annotation: string }[];
  schedule: string | null;
}

export interface ScheduleEvent {
  kind: string;
  line: number;
  qudit: number;
  start: number | null;
  duration: number | null;
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
  lines: { line: number; role: string; qudit: number }[];
  events: Record<string, unknown>[];
  epochs: Record<string, unknown>[];
  calls: Record<string, unknown>[];
  manifest: Record<string, unknown>;
  limits: Record<string, unknown> | null;
  events_truncated?: boolean;
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
  events?: ScheduleEvent[];
  events_truncated?: boolean;
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
  events?: ScheduleEvent[];
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
};

/** Server-sent events of a calibration run; returns a function that stops it. */
export function stream(
  path: string,
  params: Record<string, string | number>,
  onEvent: (event: Record<string, any>) => void,
  onEnd: () => void,
): () => void {
  const query = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
  const source = new EventSource(`${path}?${query}`);
  let ended = false;
  const finish = () => {
    if (ended) return;
    ended = true;
    source.close();
    onEnd();
  };
  source.onmessage = (msg) => {
    const event = JSON.parse(msg.data);
    onEvent(event);
    if (event.type === "done" || event.type === "error") finish();
  };
  source.onerror = finish;
  return finish;
}
