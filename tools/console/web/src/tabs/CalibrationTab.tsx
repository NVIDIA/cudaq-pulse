import { memo, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";
import { api, type ExperimentInfo, type RunResult } from "../api";
import { useConsole } from "../store";
import Plot, { type Series } from "../components/Plot";
import GoButton from "../components/GoButton";
import ErrorCard from "../components/ErrorCard";
import { fmtMs, servedLabel } from "../perf";
import { subClock, useReplay, useReplayStep, window01, type Clock, type Replay } from "../replay";

const GREEN = "#76b900";
const BLUE = "#00d4ff";
const AMBER = "#f5a623";
const PURPLE = "#a855f7";
const GREY = "#9ca3af";

/** Text that follows a replay without re-rendering React. */
const Live = ({ clock, text }: { clock: Clock; text: (p: number) => string }) => {
  const ref = useRef<HTMLSpanElement>(null);
  const textRef = useRef(text);
  textRef.current = text;
  useEffect(
    () =>
      clock.subscribe((p) => {
        if (ref.current) ref.current.textContent = textRef.current(p);
      }),
    [clock],
  );
  return <span ref={ref} />;
};

const Readout = ({ label, children, accent, sub }: { label: string; children: ReactNode; accent?: boolean; sub?: ReactNode }) => (
  <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] px-3 py-2">
    <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">{label}</div>
    <div className={clsx("text-xl font-black tabular-nums", accent ? "text-[var(--accent)]" : "text-[var(--text)]")}>{children}</div>
    {sub && <div className="text-[11px] text-[var(--muted)]">{sub}</div>}
  </div>
);

const Frame = ({ children, className }: { children: ReactNode; className?: string }) => (
  <div className={clsx("relative min-h-0 rounded-lg border border-[var(--border)] bg-[var(--bg)]", className)}>{children}</div>
);

const at = (arr: ArrayLike<number>, p: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.ceil(p * arr.length) - 1))];

// ── Rabi and T1: a sweep and a fit ───────────────────────────────────────────

interface Sweep {
  x: number[];
  p1: number[];
  fit_x: number[];
  fit: number[];
  pi_amp?: number;
  contrast?: number;
  t1_us?: number;
  t1_true_us?: number;
}

/** A vertical gauge of the excited-state probability at the replay's point. */
const Gauge = ({ clock, values }: { clock: Clock; values: number[] }) => {
  const bar = useRef<HTMLDivElement>(null);
  useEffect(
    () =>
      clock.subscribe((p) => {
        if (bar.current) bar.current.style.transform = `scaleY(${at(values, p)})`;
      }),
    [clock, values],
  );
  return (
    <div className="flex h-full w-10 flex-col items-center gap-1 text-[10px] text-[var(--muted)]">
      <span>|1⟩</span>
      <div className="relative w-4 flex-1 overflow-hidden rounded-full border border-[var(--border)] bg-[var(--panel)]">
        <div ref={bar} className="absolute inset-0 origin-bottom rounded-full bg-gradient-to-t from-[var(--accent)] to-[var(--accent-2)]" style={{ transform: "scaleY(0)" }} />
      </div>
      <span>|0⟩</span>
    </div>
  );
};

const SweepView = memo(({ kind, data, replay, simple }: { kind: "rabi" | "t1"; data: Sweep; replay: Replay; simple: boolean }) => {
  const rabi = kind === "rabi";
  const series: Series[] = useMemo(
    () => [
      { x: data.fit_x, y: data.fit, color: BLUE, reveal: "end", width: 2, label: rabi ? "cosine fit" : "exponential fit" },
      { x: data.x, y: data.p1, color: GREEN, mode: "dots", reveal: "progress", label: simple ? "simulated shots" : "P(1), simulated" },
    ],
    [data, rabi, simple],
  );
  const markers = useMemo(
    () =>
      rabi
        ? [{ value: data.pi_amp!, color: AMBER, label: simple ? "flip" : "π", reveal: "end" as const }]
        : [{ value: data.t1_us!, color: AMBER, label: "T1", reveal: "end" as const }],
    [data, rabi, simple],
  );
  const done = (p: number, s: string) => (p < 1 ? "…" : s);
  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_220px] gap-3">
      <Frame className="flex gap-2 p-2">
        <div className="min-w-0 flex-1">
          <Plot
            series={series}
            xMarkers={markers}
            yDomain={[0, 1]}
            xLabel={rabi ? (simple ? "drive strength" : "drive amplitude") : simple ? "wait (µs)" : "wait before readout (µs)"}
            yLabel={simple ? "chance of 1" : "P(1)"}
            replay={replay}
            cursor
          />
        </div>
        <Gauge clock={replay} values={data.p1} />
      </Frame>
      <div className="flex flex-col gap-2">
        <Readout label={rabi ? (simple ? "drive strength" : "amplitude") : "wait"}>
          <Live clock={replay} text={(p) => (rabi ? at(data.x, p).toFixed(3) : `${at(data.x, p).toFixed(1)} µs`)} />
        </Readout>
        <Readout label={simple ? "chance of 1" : "P(1)"}>
          <Live clock={replay} text={(p) => at(data.p1, p).toFixed(3)} />
        </Readout>
        {rabi ? (
          <Readout label={simple ? "strength that flips it" : "π amplitude (fit)"} accent sub={!simple && <Live clock={replay} text={(p) => done(p, `contrast ${data.contrast!.toFixed(3)}`)} />}>
            <Live clock={replay} text={(p) => done(p, data.pi_amp!.toFixed(4))} />
          </Readout>
        ) : (
          <Readout label={simple ? "lifetime" : "T1 (fit)"} accent sub={<Live clock={replay} text={(p) => done(p, `device: ${data.t1_true_us!.toFixed(1)} µs`)} />}>
            <Live clock={replay} text={(p) => done(p, `${data.t1_us!.toFixed(1)} µs`)} />
          </Readout>
        )}
      </div>
    </div>
  );
});

// ── Drift tracking (example 20) ──────────────────────────────────────────────

interface Drift {
  time_s: number[];
  t1_est_us: number[];
  t1_true_us: number[];
  offset_est_khz: number[];
  offset_sigma_khz: number[];
  offset_true_khz: number[];
  shots: number;
  t1_state_right: number;
  blocks: number;
  tracking_rms_khz: number;
}

const DriftView = memo(({ data, replay, simple }: { data: Drift; replay: Replay; simple: boolean }) => {
  const t1: Series[] = useMemo(
    () => [
      { x: data.time_s, y: data.t1_true_us, color: GREY, mode: "step", reveal: "progress", dash: [5, 4], label: "true" },
      { x: data.time_s, y: data.t1_est_us, color: GREEN, mode: "step", reveal: "progress", width: 2, label: "tracked" },
    ],
    [data],
  );
  const offset: Series[] = useMemo(() => {
    const lo = data.offset_est_khz.map((m, i) => m - data.offset_sigma_khz[i]);
    const hi = data.offset_est_khz.map((m, i) => m + data.offset_sigma_khz[i]);
    return [
      { x: data.time_s, y: data.offset_true_khz, color: GREY, reveal: "progress", dash: [5, 4], label: "true" },
      { x: data.time_s, y: data.offset_est_khz, lo, hi, color: BLUE, reveal: "progress", width: 2, label: simple ? "tracked" : "tracked ± σ" },
    ];
  }, [data, simple]);
  const xd = useMemo<[number, number]>(() => [0, data.time_s[data.time_s.length - 1]], [data]);
  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_220px] gap-3">
      <div className="grid min-h-0 grid-rows-2 gap-2">
        <Frame className="p-2">
          <Plot series={t1} xDomain={xd} xLabel="time (s)" yLabel={simple ? "lifetime (µs)" : "T1 (µs)"} replay={replay} cursor />
        </Frame>
        <Frame className="p-2">
          <Plot series={offset} xDomain={xd} xLabel="time (s)" yLabel={simple ? "frequency error (kHz)" : "detuning (kHz)"} replay={replay} cursor />
        </Frame>
      </div>
      <div className="flex flex-col gap-2">
        <Readout label="shots">
          <Live clock={replay} text={(p) => Math.round(p * data.shots).toLocaleString()} />
        </Readout>
        <Readout label={simple ? "lifetime now" : "T1 tracked / true"}>
          <Live clock={replay} text={(p) => `${at(data.t1_est_us, p).toFixed(1)} / ${at(data.t1_true_us, p).toFixed(1)} µs`} />
        </Readout>
        <Readout label={simple ? "frequency error" : "detuning tracked / true"}>
          <Live clock={replay} text={(p) => `${at(data.offset_est_khz, p).toFixed(0)} / ${at(data.offset_true_khz, p).toFixed(0)} kHz`} />
        </Readout>
        <Readout label={simple ? "how well it kept up" : "tracking"} accent>
          <Live clock={replay} text={(p) => (p < 1 ? "…" : `${data.tracking_rms_khz.toFixed(0)} kHz rms`)} />
        </Readout>
        <div className="text-[11px] text-[var(--muted)]">
          <Live clock={replay} text={(p) => (p < 1 ? "" : `T1 state right in ${data.t1_state_right} of ${data.blocks} blocks of ${data.shots / data.blocks} shots`)} />
        </div>
      </div>
    </div>
  );
});

// ── The calibration chain (example 21) ───────────────────────────────────────

interface Chain {
  compile_s: number;
  simulate_s: number;
  readouts: number;
  calls: number;
  end_us: number;
  results: { name: string; value: number | number[]; truth: number | number[] }[];
  steps: { function: string; t_us: number; answer: number[] }[];
  tracker: { shot: number; t_us: number; t1_us: number; t1_sigma_us: number; wait_us: number; bit: number }[];
  functions: { function: string; ops: number; state_bytes: number; compute_ns: number; budget_ns: number; calls: number; on_controller: boolean }[];
  violations: string[];
}

const STAGES = [
  { id: "readout", plain: "Readout", expert: "assignment_errors", result: "readout errors", fns: ["assignment_errors"] },
  { id: "frequency", plain: "Frequency", expert: "ramsey_estimator", result: "detuning (kHz)", fns: ["ramsey_estimator"] },
  { id: "pi", plain: "Flip pulse", expert: "amplitude_estimator (π)", result: "pi amplitude", fns: ["amplitude_estimator#0"] },
  { id: "half", plain: "Half-flip pulse", expert: "amplitude_estimator (π/2)", result: "pi/2 amplitude", fns: ["amplitude_estimator#1"] },
  { id: "decay", plain: "Lifetime, first guess", expert: "decay_estimator", result: "T1, decay estimator (us)", fns: ["decay_delays", "decay_estimator"] },
  { id: "adaptive", plain: "Lifetime, adaptive", expert: "adaptive_t1 × shots", result: "T1, adaptive (us)", fns: ["first_wait", "adaptive_t1"] },
];

/** The replay runs the first stages over this share of its time and the
 * adaptive shots over the rest. */
const SPLIT = 0.35;

const fmtValue = (v: number | number[]) => (Array.isArray(v) ? v.map((x) => x.toFixed(3)).join(", ") : Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(1));

const stageOf = (steps: Chain["steps"]) => {
  let amp = 0;
  return steps.map((s) => {
    const name = s.function === "amplitude_estimator" ? `amplitude_estimator#${amp++}` : s.function;
    return STAGES.findIndex((st) => st.fns.includes(name));
  });
};

const useChain = () => {
  const replay = useReplay();
  const [run, setRun] = useState<RunResult<Chain> | null>(null);
  const [busy, setBusy] = useState(false);
  const go = async (ms: number) => {
    const clicked = performance.now();
    setBusy(true);
    const r = await api.run<Chain>("chain", {});
    setBusy(false);
    setRun(r);
    if (r.ok) replay.start(ms, clicked);
  };
  return { replay, run, busy, go };
};

const TrackerPlot = ({ data, clock, simple }: { data: Chain; clock: Clock; simple: boolean }) => {
  const truth = data.results.find((r) => r.name === "T1, adaptive (us)")?.truth as number;
  const series: Series[] = useMemo(() => {
    const x = data.tracker.map((t) => t.shot);
    return [
      {
        x,
        y: data.tracker.map((t) => t.t1_us),
        lo: data.tracker.map((t) => t.t1_us - t.t1_sigma_us),
        hi: data.tracker.map((t) => t.t1_us + t.t1_sigma_us),
        color: GREEN,
        reveal: "progress",
        width: 2,
        label: simple ? "estimate" : "posterior mean ± σ",
      },
    ];
  }, [data, simple]);
  const markers = useMemo(() => [{ value: truth, color: GREY, label: simple ? "true lifetime" : `true T1 ${truth} µs`, reveal: "always" as const }], [truth, simple]);
  return (
    <Plot
      series={series}
      yMarkers={markers}
      xDomain={[1, data.tracker.length]}
      yDomain={[0, Math.max(60, ...data.tracker.slice(3).map((t) => t.t1_us + t.t1_sigma_us))]}
      xLabel="shot"
      yLabel={simple ? "lifetime (µs)" : "T1 (µs)"}
      replay={clock}
      cursor
    />
  );
};

/** The adaptive T1 posterior shot by shot, for the compiler explorer. */
export const ChainConvergence = ({ simple }: { simple: boolean }) => {
  const { replay, run, busy, go } = useChain();
  const data = run?.ok ? run.data! : null;
  const last = data?.tracker[data.tracker.length - 1];
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex items-center gap-3">
        <GoButton onClick={() => go(5000)} busy={busy} />
        <span className="text-xs text-[var(--muted)]">
          {simple
            ? "Each shot updates the lifetime estimate on the controller and picks the next wait."
            : "adaptive_t1 runs on the controller after every shot: a gamma posterior over 1/T1 and the next wait. One simulated run, replayed."}
        </span>
      </div>
      <Frame className="flex-1 p-2">
        {data ? <TrackerPlot data={data} clock={replay} simple={simple} /> : <div className="flex h-full items-center justify-center text-xs text-[var(--muted)]">Press Go</div>}
      </Frame>
      {data && last && (
        <div className="text-xs text-[var(--muted)]">
          <Live clock={replay} text={(p) => { const t = at(data.tracker.map((x) => x.t1_us), p); const s = at(data.tracker.map((x) => x.t1_sigma_us), p); return `T1 ${t.toFixed(1)} ± ${s.toFixed(1)} µs after ${Math.max(1, Math.ceil(p * data.tracker.length))} shots`; }} />
        </div>
      )}
    </div>
  );
};

const ChainView = memo(({ data, replay, simple, live }: { data: Chain; replay: Replay; simple: boolean; live: boolean }) => {
  const stages = useMemo(() => stageOf(data.steps), [data]);
  const early = useMemo(() => stages.filter((s) => s < 5).length, [stages]);
  // Progress through the early calls, then through the adaptive shots.
  const earlyClock = useMemo(() => subClock(replay, window01(0, SPLIT)), [replay]);
  const shotClock = useMemo(() => subClock(replay, window01(SPLIT, 1)), [replay]);
  const earlyStep = useReplayStep(earlyClock, early);
  const shotStep = useReplayStep(shotClock, 20);
  const stageDone = (i: number) => (i < 5 ? earlyStep >= stages.lastIndexOf(i) + 1 : shotStep >= 20);
  const stageActive = (i: number) => !stageDone(i) && (i === 0 ? true : stageDone(i - 1));
  return (
    <div className="grid h-full min-h-0 grid-cols-[260px_minmax(0,1fr)] gap-3">
      <div className="flex min-h-0 flex-col gap-1.5 overflow-auto">
        {STAGES.map((st, i) => {
          const done = stageDone(i);
          const active = stageActive(i);
          const res = data.results.find((r) => r.name === st.result);
          return (
            <div
              key={st.id}
              className={clsx(
                "rounded-lg border px-3 py-2 transition-[border-color,background-color,opacity] duration-300",
                done ? "border-[var(--accent)]/70 bg-[var(--accent)]/10" : active ? "border-[var(--accent-2)] bg-[var(--accent-2)]/10" : "border-[var(--border)] opacity-50",
              )}
            >
              <div className="flex items-center gap-2 text-xs font-semibold">
                <span className={clsx("inline-block h-2 w-2 rounded-full", done ? "bg-[var(--accent)]" : active ? "animate-pulse bg-[var(--accent-2)]" : "bg-[var(--border)]")} />
                {simple ? st.plain : <span className="mono">{st.expert}</span>}
              </div>
              {res && (!simple || st.id !== "decay") && (
                <div className={clsx("mono mt-1 text-[11px] transition-opacity duration-300", done ? "opacity-100" : "opacity-0")}>
                  <span className="text-[var(--accent)]">{fmtValue(res.value)}</span>
                  <span className="text-[var(--muted)]"> · true {fmtValue(res.truth)}</span>
                </div>
              )}
            </div>
          );
        })}
        <div className="mt-1 text-[11px] leading-relaxed text-[var(--muted)]">
          {simple
            ? `One program: ${data.readouts.toLocaleString()} measurements, with every decision made next to the qubit between shots.`
            : `${data.readouts.toLocaleString()} readouts, ${data.calls} classical calls on the controller` +
              (live ? ` · compiled in ${data.compile_s.toFixed(2)} s, one shot simulated in ${data.simulate_s.toFixed(1)} s.` : ".")}
        </div>
      </div>
      <div className="flex min-h-0 flex-col gap-2">
        <Frame className="min-h-[180px] flex-[3] p-2">
          <TrackerPlot data={data} clock={shotClock} simple={simple} />
        </Frame>
        {simple ? (
          <div className="grid grid-cols-3 gap-2">
            {[
              { name: "detuning (kHz)", label: "frequency correction", unit: " kHz" },
              { name: "pi amplitude", label: "flip pulse strength", unit: "" },
              { name: "T1, adaptive (us)", label: "lifetime", unit: " µs" },
            ].map(({ name, label, unit }) => {
              const r = data.results.find((x) => x.name === name);
              return (
                r && (
                  <Readout key={name} label={label} accent sub={`true ${fmtValue(r.truth)}${unit}`}>
                    <Live clock={replay} text={(p) => (p < 1 ? "…" : `${fmtValue(r.value)}${unit}`)} />
                  </Readout>
                )
              );
            })}
          </div>
        ) : (
          <Frame className="flex-[2] overflow-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[var(--panel)] text-[10px] text-[var(--muted)]">
                <tr>
                  {["estimator", "ops", "state B", "compute ns", "budget ns", "calls / shot", ""].map((h) => (
                    <th key={h} className="px-2 py-1.5 text-right font-medium first:text-left">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="mono">
                {data.functions.map((f) => (
                  <tr key={f.function} className="border-t border-[var(--border)]">
                    <td className="px-2 py-1">{f.function}</td>
                    <td className="px-2 text-right">{f.ops}</td>
                    <td className="px-2 text-right">{f.state_bytes}</td>
                    <td className="px-2 text-right">{f.compute_ns.toFixed(0)}</td>
                    <td className="px-2 text-right">{f.budget_ns.toFixed(0)}</td>
                    <td className="px-2 text-right">{f.calls}</td>
                    <td className="w-28 px-2">
                      <div className="h-1.5 rounded bg-[var(--panel)]">
                        <div className="h-1.5 rounded bg-[var(--accent)]" style={{ width: `${Math.min(100, (100 * f.compute_ns) / f.budget_ns)}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data.violations.length > 0 && <div className="p-2 text-[11px] text-[var(--danger)]">{data.violations.join(" · ")}</div>}
          </Frame>
        )}
      </div>
    </div>
  );
});

// ── The tab ──────────────────────────────────────────────────────────────────

const REPLAY_MS: Record<string, number> = { rabi: 3000, t1: 3000, drift: 8000, chain: 9000 };

const Param = ({ spec, value, onChange }: { spec: ExperimentInfo["params"][number]; value: number; onChange: (v: number) => void }) => (
  <label className="flex items-center gap-2 text-xs">
    <span className="text-[var(--muted)]">{spec.label}</span>
    <input type="range" min={spec.min} max={spec.max} step={spec.step} value={value} onChange={(e) => onChange(+e.target.value)} className="w-28 accent-[var(--accent)]" />
    <span className="mono w-16 tabular-nums">
      {value}
      {spec.unit && <span className="text-[var(--muted)]"> {spec.unit}</span>}
    </span>
  </label>
);

const CalibrationTab = () => {
  const meta = useConsole((s) => s.meta);
  const simple = useConsole((s) => s.mode === "overview");
  const experiments = meta?.experiments ?? [];
  const [selected, setSelected] = useState(() => new URLSearchParams(window.location.search).get("experiment") ?? "rabi");
  const exp = experiments.find((e) => e.id === selected) ?? experiments[0];
  const [params, setParams] = useState<Record<string, Record<string, number>>>({});
  const values = useMemo(() => Object.fromEntries((exp?.params ?? []).map((p) => [p.name, params[exp.id]?.[p.name] ?? p.default])), [exp, params]);
  const replay = useReplay();
  const [run, setRun] = useState<RunResult<any> | null>(null);
  const [busy, setBusy] = useState(false);
  const [took, setTook] = useState(0);
  const [advanced, setAdvanced] = useState(false);

  useEffect(() => {
    setRun(null);
    replay.finish();
  }, [selected, replay]);

  if (!exp) return <div className="p-6 text-sm text-[var(--muted)]">Loading…</div>;

  const go = async () => {
    const clicked = performance.now();
    setBusy(true);
    try {
      const r = await api.run(exp.id, values);
      setTook(performance.now() - clicked);
      setRun(r);
      if (r.ok) replay.start(REPLAY_MS[exp.id] ?? 4000, clicked);
    } finally {
      setBusy(false);
    }
  };

  const shownParams = simple || !advanced ? exp.params.slice(0, 2) : exp.params;
  const data = run?.ok && run.experiment === exp.id ? run.data : null;

  return (
    <div className="grid h-full min-h-0 grid-cols-[230px_minmax(0,1fr)] gap-3">
      <nav className="flex flex-col gap-1.5">
        {experiments.map((e) => (
          <button
            key={e.id}
            onClick={() => setSelected(e.id)}
            className={clsx(
              "rounded-lg border px-3 py-2 text-left normal-case transition-colors",
              e.id === exp.id ? "border-[var(--accent)] bg-[var(--accent)]/10" : "border-[var(--border)] hover:border-[var(--accent)]/60",
            )}
          >
            <div className={clsx("text-sm font-bold", e.id === exp.id && "text-[var(--accent)]")}>{e.title}</div>
            <div className="mt-0.5 text-[11px] font-normal normal-case leading-snug text-[var(--muted)]">{e.plain}</div>
          </button>
        ))}
      </nav>
      <section className="flex min-h-0 flex-col gap-3 rounded-lg border border-[var(--border)] bg-[var(--panel)] p-3">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <div className="min-w-0">
            <h2 className="text-base font-black">{exp.title}</h2>
            <div className="text-[11px] text-[var(--muted)]">
              {simple ? exp.plain : exp.detail}
              {!simple && <span className="mono"> · examples/{exp.example}</span>}
            </div>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-4">
            {shownParams.map((p) => (
              <Param key={p.name} spec={p} value={values[p.name]} onChange={(v) => setParams((all) => ({ ...all, [exp.id]: { ...all[exp.id], [p.name]: v } }))} />
            ))}
            {!simple && exp.params.length > 2 && (
              <button className="text-[11px] normal-case text-[var(--muted)] hover:text-[var(--text)]" onClick={() => setAdvanced((a) => !a)}>
                {advanced ? "fewer" : "more"}
              </button>
            )}
            <GoButton onClick={go} busy={busy} />
          </div>
        </div>
        {run && !run.ok && run.error && <ErrorCard error={typeof run.error === "string" ? { kind: "Error", message: run.error } : run.error} />}
        <div className="min-h-0 flex-1">
          {!data && (
            <div className="flex h-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-[var(--border)] text-sm text-[var(--muted)]">
              <span>{busy ? "Simulating…" : "Set the parameters and press Go."}</span>
              <span className="text-[11px]">The run is simulated on this machine, then replayed.</span>
            </div>
          )}
          {data && (exp.id === "rabi" || exp.id === "t1") && <SweepView kind={exp.id} data={data} replay={replay} simple={simple} />}
          {data && exp.id === "drift" && <DriftView data={data} replay={replay} simple={simple} />}
          {data && exp.id === "chain" && <ChainView data={data} replay={replay} simple={simple} live={!run?.cached} />}
        </div>
        {run?.ok && !simple && (
          <div className="-mb-1 text-[10.5px] text-[var(--muted)]">
            {run.cached ? `${servedLabel(run, took)} · replayed` : `simulated in ${fmtMs(run.server_ms!)}`}
          </div>
        )}
      </section>
    </div>
  );
};

export default CalibrationTab;
