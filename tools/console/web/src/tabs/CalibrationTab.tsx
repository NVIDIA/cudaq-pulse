import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { stream } from "../api";
import Panel from "../components/Panel";
import StatCard from "../components/StatCard";

const SERIES = ["#76b900", "#00d4ff", "#f5a623", "#a855f7"];

const tick = (v: number) => (Math.abs(v) >= 100 || Number.isInteger(v) ? v.toFixed(0) : String(+v.toPrecision(3)));
const axis = { stroke: "var(--muted)", fontSize: 10, tickLine: false, tickFormatter: tick };
const tooltip = {
  contentStyle: { background: "var(--panel)", border: "1px solid var(--border)", fontSize: 11 },
  labelStyle: { color: "var(--muted)" },
};
const fmt = (v: number, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : "–");

type RunState = "idle" | "running" | "done" | "error";

function useRun(path: string) {
  const [state, setState] = useState<RunState>("idle");
  const [error, setError] = useState<string | null>(null);
  const stop = useRef<(() => void) | null>(null);
  useEffect(() => () => stop.current?.(), []);
  const run = (params: Record<string, string | number>, onEvent: (e: Record<string, any>) => void) => {
    stop.current?.();
    setState("running");
    setError(null);
    stop.current = stream(
      path,
      params,
      (e) => {
        if (e.type === "error") {
          setError(`${e.kind}: ${e.message}`);
          setState("error");
        }
        onEvent(e);
      },
      () => setState((s) => (s === "error" ? s : "done")),
    );
  };
  return { state, error, run, stop: () => stop.current?.() };
}

const RunButton = ({ state, onClick, label = "Run" }: { state: RunState; onClick: () => void; label?: string }) => (
  <button
    onClick={onClick}
    disabled={state === "running"}
    className="rounded-md bg-[var(--accent)] px-3 py-1 text-[11px] font-bold text-black disabled:opacity-50"
  >
    {state === "running" ? "Running…" : state === "done" ? "Run again" : label}
  </button>
);

const Source = ({ children }: { children: ReactNode }) => (
  <span className="text-[10px] normal-case text-[var(--muted)]">{children}</span>
);

// ── Example 20: tracking a drifting device ──────────────────────────────────

const DriftPanel = () => {
  const { state, error, run } = useRun("/api/calibration/drift");
  const [points, setPoints] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [seed, setSeed] = useState(21);
  const start = () => {
    setPoints([]);
    setSummary(null);
    run({ seed, delay: 0.1 }, (e) => {
      if (e.type === "point")
        setPoints((p) => [
          ...p,
          { ...e, band: [e.offset_est_khz - e.offset_sigma_khz, e.offset_est_khz + e.offset_sigma_khz] },
        ]);
      if (e.type === "done") setSummary(e);
    });
  };
  const last = points[points.length - 1];
  return (
    <Panel
      className="shrink-0"
      title={
        <>
          Drift tracking between shots <Source>· example 20 §2 · TelegraphT1 + Ornstein–Uhlenbeck plant</Source>
        </>
      }
      actions={
        <div className="flex items-center gap-2 text-[11px]">
          seed
          <input type="number" className="w-16" value={seed} onChange={(e) => setSeed(+e.target.value)} />
          <RunButton state={state} onClick={start} />
        </div>
      }
    >
      <div className="mb-2 grid grid-cols-4 gap-2">
        <StatCard label="Shots" value={last ? last.i.toLocaleString() : "0"} delta={last ? `t = ${fmt(last.time_s, 2)} s` : "6000 planned"} />
        <StatCard label="T1 estimate" value={last ? `${fmt(last.t1_est_us)} µs` : "–"} delta={last ? `true ${fmt(last.t1_true_us)} µs` : ""} accent />
        <StatCard label="Offset estimate" value={last ? `${fmt(last.offset_est_khz)} kHz` : "–"} delta={last ? `true ${fmt(last.offset_true_khz)} kHz` : ""} accent />
        <StatCard
          label="Tracking"
          value={summary ? `${summary.t1_state_right}/${summary.blocks}` : "–"}
          delta={summary ? `T1 state right · rms ${fmt(summary.tracking_rms_khz)} kHz` : "T1 state blocks right"}
        />
      </div>
      {error && <div className="mb-2 text-xs text-[var(--danger)]">{error}</div>}
      <div className="grid grid-cols-2 gap-2">
        <div className="h-56">
          <ResponsiveContainer>
            <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="time_s" type="number" domain={[0, 6]} {...axis} unit=" s" />
              <YAxis {...axis} unit=" µs" domain={[0, "auto"]} />
              <Tooltip {...tooltip} formatter={(v: number) => fmt(v, 2)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="stepAfter" dataKey="t1_true_us" name="true T1" stroke="#6b7280" dot={false} strokeDasharray="4 3" isAnimationActive={false} />
              <Line dataKey="t1_est_us" name="T1 estimate" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="h-56">
          <ResponsiveContainer>
            <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="time_s" type="number" domain={[0, 6]} {...axis} unit=" s" />
              <YAxis {...axis} unit=" kHz" />
              <Tooltip {...tooltip} formatter={(v: any) => (Array.isArray(v) ? v.map((x) => fmt(x)).join(" – ") : fmt(v))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Area dataKey="band" name="±σ" stroke="none" fill={SERIES[1]} fillOpacity={0.15} isAnimationActive={false} />
              <Line dataKey="offset_true_khz" name="true offset" stroke="#6b7280" dot={false} strokeDasharray="4 3" isAnimationActive={false} />
              <Line dataKey="offset_est_khz" name="estimate" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Panel>
  );
};

// ── Example 21: frequency binary search ─────────────────────────────────────

const SearchPanel = () => {
  const { state, error, run } = useRun("/api/calibration/search");
  const [rows, setRows] = useState<Record<string, number>[]>([]);
  const [offsets, setOffsets] = useState<number[]>([]);
  const start = () => {
    setRows([]);
    setOffsets([]);
    run({ rounds: 15, delay: 0.05 }, (e) => {
      if (e.type === "series") setOffsets((o) => [...o, e.offset_khz]);
      if (e.type === "point")
        setRows((rs) => {
          const next = rs.slice();
          while (next.length < e.round + 1) next.push({ round: next.length });
          next[e.round] = { ...next[e.round], [`m${e.seed}`]: e.mean_khz, [`s${e.seed}`]: e.sigma_khz };
          if (!next[0][`m${e.seed}`]) next[0] = { ...next[0], round: 0, [`m${e.seed}`]: 0, [`s${e.seed}`]: 1000 };
          return next;
        });
    });
  };
  return (
    <Panel
      className="shrink-0"
      title={
        <>
          Frequency binary search, one shot, no reset <Source>· example 21 §2 · after arXiv:2501.05386</Source>
        </>
      }
      actions={<RunButton state={state} onClick={start} />}
    >
      {error && <div className="mb-2 text-xs text-[var(--danger)]">{error}</div>}
      <div className="grid grid-cols-[2fr_1fr] gap-2">
        <div className="h-60">
          <ResponsiveContainer>
            <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="round" type="number" domain={[0, 15]} {...axis} />
              <YAxis {...axis} width={64} unit=" kHz" domain={[-1200, 1200]} ticks={[-1200, -450, -150, 150, 450, 1200]} allowDataOverflow />
              <Tooltip {...tooltip} formatter={(v: number) => fmt(v)} />
              {offsets.map((o, i) => (
                <ReferenceLine key={i} y={o} stroke={SERIES[i]} strokeDasharray="4 3" />
              ))}
              {offsets.map((o, i) => (
                <Line key={i} dataKey={`m${i}`} name={`true ${o > 0 ? "+" : ""}${o} kHz`} stroke={SERIES[i]} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} connectNulls />
              ))}
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="h-60">
          <ResponsiveContainer>
            <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="round" type="number" domain={[0, 15]} {...axis} />
              <YAxis {...axis} width={64} scale="log" domain={[100, 1000]} ticks={[100, 200, 300, 500, 1000]} unit=" kHz" allowDataOverflow />
              <Tooltip {...tooltip} formatter={(v: number) => fmt(v)} />
              {offsets.map((_, i) => (
                <Line key={i} dataKey={`s${i}`} name={`σ ${i}`} stroke={SERIES[i]} dot={false} isAnimationActive={false} connectNulls />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
          <div className="text-center text-[10px] text-[var(--muted)]">posterior width σ per round</div>
        </div>
      </div>
    </Panel>
  );
};

// ── Example 21: the calibration chain on the controller ─────────────────────

const STEP_LABEL: Record<string, string> = {
  assignment_errors: "1 · readout assignment",
  ramsey_estimator: "2 · drive frequency",
  amplitude_estimator: "3–4 · π, π/2 amplitude",
  decay_delays: "5 · decay delays",
  decay_estimator: "5 · T1 (decay)",
  first_wait: "6 · first wait",
  adaptive_t1: "6 · T1 (adaptive)",
};

const ChainPanel = () => {
  const { state, error, run } = useRun("/api/calibration/chain");
  const [summary, setSummary] = useState<any>(null);
  const [steps, setSteps] = useState<any[]>([]);
  const [waiting, setWaiting] = useState(false);
  const start = () => {
    setSummary(null);
    setSteps([]);
    setWaiting(true);
    run({ delay: 0.04 }, (e) => {
      if (e.type === "summary") {
        setSummary(e);
        setWaiting(false);
      }
      if (e.type === "step") setSteps((s) => [...s, e]);
    });
  };
  const tracker = steps
    .filter((s) => s.function === "adaptive_t1")
    .map((s, k) => {
      const [shape, rate, wait] = s.answer;
      const t1 = rate / shape / 2 / 1e3;
      const sd = t1 / Math.sqrt(shape);
      return { shot: k + 1, t1, band: [Math.max(0, t1 - sd), t1 + sd], wait: wait / 2 / 1e3 };
    });
  const shownResults = summary?.results ?? [];
  const fmtVal = (v: any) => (Array.isArray(v) ? v.map((x: number) => fmt(x, 3)).join(", ") : fmt(v, 4));
  return (
    <Panel
      className="shrink-0"
      title={
        <>
          Millisecond calibration chain on the controller <Source>· example 21 §1, §3 · after arXiv:2602.11912, arXiv:2506.09576</Source>
        </>
      }
      actions={<RunButton state={state} onClick={start} label="Compile + simulate" />}
    >
      {error && <div className="mb-2 text-xs text-[var(--danger)]">{error}</div>}
      {waiting && (
        <div className="mb-2 animate-pulse text-xs text-[var(--accent-2)]">
          Compiling the chain and simulating one shot (≈15 s the first time; cached after)…
        </div>
      )}
      <div className="mb-2 grid grid-cols-5 gap-2">
        <StatCard label="Compile" value={summary ? `${fmt(summary.compile_s, 2)} s` : "–"} delta="pulse.compile(...).lower()" />
        <StatCard label="Simulate 1 shot" value={summary ? `${fmt(summary.simulate_s, 1)} s` : "–"} delta="pulse.simulate(paths='sampled')" />
        <StatCard label="Readouts" value={summary ? String(summary.readouts) : "–"} />
        <StatCard label="Classical calls" value={summary ? String(summary.calls) : "–"} delta="on the controller link" accent />
        <StatCard label="Epochs" value={summary ? String(summary.epochs) : "–"} delta="variable waits" />
      </div>
      <div className="grid grid-cols-[1fr_1fr] gap-3">
        <div>
          <table className="w-full text-xs">
            <thead className="text-[10px] tracking-wider text-[var(--muted)]">
              <tr>
                <th className="py-1 text-left">Step</th>
                <th className="text-right">Estimate</th>
                <th className="text-right">Truth</th>
              </tr>
            </thead>
            <tbody>
              {shownResults.map((r: any) => (
                <tr key={r.name} className="border-t border-[var(--border)]">
                  <td className="py-1">{r.name}</td>
                  <td className="mono text-right text-[var(--accent)]">{fmtVal(r.value)}</td>
                  <td className="mono text-right text-[var(--muted)]">{fmtVal(r.truth)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mono mt-3 max-h-40 overflow-auto rounded border border-[var(--border)] bg-[var(--bg)] p-2 text-[11px]">
            {steps.length === 0 && <div className="text-[var(--muted)]">classical calls appear here as the shot replays</div>}
            {steps
              .filter((s) => s.function !== "adaptive_t1")
              .concat(steps.filter((s) => s.function === "adaptive_t1").slice(-1))
              .map((s, i) => (
                <div key={i} className="flex gap-2">
                  <span className="w-20 text-right text-[var(--muted)]">{fmt(s.t_us, 1)} µs</span>
                  <span className="w-44 text-[var(--accent-2)]">{STEP_LABEL[s.function] ?? s.function}</span>
                  <span className="truncate">{s.answer.map((a: number) => (Math.abs(a) > 1e3 ? a.toExponential(3) : a.toFixed(4))).join(", ")}</span>
                </div>
              ))}
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer>
            <ComposedChart data={tracker} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" />
              <XAxis dataKey="shot" type="number" domain={[0, 50]} {...axis} />
              <YAxis yAxisId="t1" {...axis} unit=" µs" domain={[0, 60]} allowDataOverflow />
              <YAxis yAxisId="w" orientation="right" {...axis} unit=" µs" />
              <Tooltip {...tooltip} formatter={(v: any) => (Array.isArray(v) ? v.map((x) => fmt(x)).join(" – ") : fmt(v, 2))} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <ReferenceLine yAxisId="t1" y={18.3} stroke="#6b7280" strokeDasharray="4 3" label={{ value: "true 18.3 µs", fill: "var(--muted)", fontSize: 10 }} />
              <Area yAxisId="t1" dataKey="band" name="±σ" stroke="none" fill={SERIES[0]} fillOpacity={0.15} isAnimationActive={false} />
              <Line yAxisId="t1" dataKey="t1" name="adaptive T1 posterior" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
              <Line yAxisId="w" dataKey="wait" name="next wait" stroke={SERIES[2]} dot={false} strokeDasharray="2 2" isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      {summary && (
        <div className="mt-3 grid grid-cols-[3fr_2fr] gap-3">
          <table className="w-full text-xs">
            <thead className="text-[10px] tracking-wider text-[var(--muted)]">
              <tr>
                <th className="py-1 text-left">Estimator</th>
                <th className="text-right">Ops</th>
                <th className="text-right">State B</th>
                <th className="w-56 text-left pl-3">Compute vs budget</th>
                <th className="text-right">Calls</th>
              </tr>
            </thead>
            <tbody>
              {summary.functions.map((f: any) => (
                <tr key={f.function} className="border-t border-[var(--border)]">
                  <td className="mono py-1">{f.function}</td>
                  <td className="text-right tabular-nums">{f.ops}</td>
                  <td className="text-right tabular-nums">{f.state_bytes}</td>
                  <td className="pl-3">
                    <div className="flex items-center gap-2">
                      <div className="h-2 flex-1 rounded bg-[var(--bg)]">
                        <div
                          className="h-2 rounded"
                          style={{
                            width: `${Math.min(100, (100 * (f.compute_ns ?? 0)) / f.budget_ns)}%`,
                            background: (f.compute_ns ?? 0) > f.budget_ns ? "var(--danger)" : "var(--accent)",
                          }}
                        />
                      </div>
                      <span className="mono w-24 text-right text-[10px] text-[var(--muted)]">
                        {fmt(f.compute_ns, 0)}/{fmt(f.budget_ns, 0)} ns
                      </span>
                    </div>
                  </td>
                  <td className="text-right tabular-nums">{f.calls}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="text-xs">
            {summary.links.map((l: any) => (
              <div key={String(l.link)} className="rounded border border-[var(--border)] p-2">
                <div className="font-semibold">link “{String(l.link)}”</div>
                <div className="mt-1 grid grid-cols-2 gap-1 text-[var(--muted)]">
                  <span>calls / shot</span>
                  <span className="text-right text-[var(--text)]">{l.calls_per_shot}</span>
                  <span>peak in flight</span>
                  <span className="text-right text-[var(--text)]">
                    {l.peak_in_flight} / {l.max_in_flight}
                  </span>
                  <span>largest message</span>
                  <span className="text-right text-[var(--text)]">
                    {l.largest_message_bytes} / {l.slot_bytes} B
                  </span>
                  <span>bytes / shot</span>
                  <span className="text-right text-[var(--text)]">{l.bytes_per_shot}</span>
                </div>
              </div>
            ))}
            <div className={`mt-2 rounded p-2 ${summary.violations.length ? "bg-[var(--danger)]/15 text-[var(--danger)]" : "bg-[var(--accent)]/10 text-[var(--accent)]"}`}>
              {summary.violations.length ? summary.violations.join("; ") : "within the declared ControllerLimits"}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
};

const CalibrationTab = () => (
  <div className="flex h-full flex-col gap-3 overflow-auto pb-6 pr-1">
    <ChainPanel />
    <SearchPanel />
    <DriftPanel />
  </div>
);

export default CalibrationTab;
