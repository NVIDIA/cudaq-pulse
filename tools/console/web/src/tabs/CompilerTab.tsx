import { memo, useEffect, useMemo, useRef, useState } from "react";
import Editor, { DiffEditor, type OnMount } from "@monaco-editor/react";
import clsx from "clsx";
import { api, type CompileRequest, type CompileResult, type SimResult } from "../api";
import { useConsole, LEVEL_COLORS } from "../store";
import { monaco } from "../monaco";
import ErrorCard from "../components/ErrorCard";
import Timeline from "../components/Timeline";
import Plot, { type Series } from "../components/Plot";
import GoButton from "../components/GoButton";
import { afterPaint, fmtMs, servedLabel } from "../perf";
import { useReplay } from "../replay";
import { ChainConvergence } from "./CalibrationTab";

type Stage = "optimized" | "traced" | "diff" | "schedule" | "controller" | "llvm" | "program" | "run" | "converge";

const EXPERT: { id: Stage; label: string }[] = [
  { id: "optimized", label: "IR" },
  { id: "traced", label: "Traced" },
  { id: "diff", label: "Diff" },
  { id: "schedule", label: "Schedule" },
  { id: "controller", label: "Controller" },
  { id: "llvm", label: "LLVM" },
  { id: "run", label: "Simulate" },
];
const OVERVIEW: { id: Stage; label: string }[] = [
  { id: "program", label: "Program" },
  { id: "run", label: "Simulate" },
];

const DEBOUNCE_MS = 200;

const editorOptions = {
  minimap: { enabled: false },
  fontSize: 12.5,
  fontFamily: '"JetBrains Mono", "SF Mono", Menlo, monospace',
  scrollBeyondLastLine: false,
  automaticLayout: true,
  renderLineHighlight: "line" as const,
  padding: { top: 8 },
  overviewRulerLanes: 0,
  stickyScroll: { enabled: false },
};
const readOnly = { ...editorOptions, readOnly: true, domReadOnly: true, lineNumbersMinChars: 4 };

const request = (llvm: boolean): CompileRequest => {
  const s = useConsole.getState();
  return { source: s.source, kernel: s.kernel, args: s.args, target: s.target, passes: s.passes, schedule: s.schedule, llvm };
};

let sequence = 0;

/** Compiles the current state; a reply older than the latest request is
 * dropped. `since` is when the user changed something. */
export const compileNow = async (since = performance.now(), llvm = false) => {
  const s = useConsole.getState();
  if (!s.source || !s.kernel) return;
  const seq = ++sequence;
  const sent = performance.now();
  s.set({ compiling: true });
  try {
    const [result, kernels] = await Promise.all([api.compile(request(llvm)), api.kernels(s.source)]);
    if (seq !== sequence) return;
    useConsole.getState().set({ result, kernels: kernels.ok ? kernels.kernels! : useConsole.getState().kernels, compiling: false });
    afterPaint("compile round trip", sent);
    afterPaint("edit → output on screen", since);
  } catch (e) {
    if (seq === sequence)
      useConsole.getState().set({ result: { ok: false, error: { kind: "NetworkError", message: String(e) } }, compiling: false });
  }
};

const sources = new Map<string, string>();

export const loadExample = async (file: string) => {
  const meta = useConsole.getState().meta;
  const preset = meta?.examples.find((e) => e.file === file);
  // An example opened before keeps its edits in its editor model.
  let source = monaco.editor.getModel(monaco.Uri.parse(`example:///${file}`))?.getValue() ?? sources.get(file);
  if (source === undefined) {
    source = (await api.example(file)).source;
    sources.set(file, source);
  }
  useConsole.getState().set({ exampleFile: file, source, kernel: preset?.kernel ?? "", args: preset?.args ?? "" });
  await compileNow();
};

const Chip = ({ on, children, onClick }: { on: boolean; children: React.ReactNode; onClick: () => void }) => (
  <button
    onClick={onClick}
    className={clsx(
      "rounded-full border px-2.5 py-0.5 text-[11px] normal-case transition-colors",
      on ? "border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]" : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--text)]",
    )}
  >
    {children}
  </button>
);

/** Device, schedule policy and passes: defaults that rarely change. */
const Options = () => {
  const meta = useConsole((s) => s.meta);
  const target = useConsole((s) => s.target);
  const schedule = useConsole((s) => s.schedule);
  const passes = useConsole((s) => s.passes);
  const expert = useConsole((s) => s.mode === "expert");
  const set = useConsole((s) => s.set);
  const update = (patch: Parameters<typeof set>[0]) => {
    set(patch);
    compileNow();
  };
  const toggle = (p: string) => {
    const next = passes.includes(p) ? passes.filter((x) => x !== p) : [...passes, p];
    update({ passes: (meta?.passes ?? []).filter((x) => next.includes(x)) });
  };
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-[var(--border)] bg-[var(--bg)]/40 px-3 py-2 text-xs">
      <label className="flex items-center gap-2">
        <span className="text-[var(--muted)]">Device</span>
        <select value={target} onChange={(e) => update({ target: e.target.value })}>
          {meta?.targets.map((t) => (
            <option key={t.id} value={t.id}>
              {t.id}
            </option>
          ))}
        </select>
      </label>
      <label className="flex items-center gap-2">
        <span className="text-[var(--muted)]">Schedule</span>
        <select value={schedule} onChange={(e) => update({ schedule: e.target.value })}>
          {meta?.schedules.map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>
      </label>
      {expert && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mono text-[var(--muted)]">passes=</span>
          {meta?.passes.map((p) => (
            <Chip key={p} on={passes.includes(p)} onClick={() => toggle(p)}>
              {p}
            </Chip>
          ))}
          <button className="px-1 text-[10px] text-[var(--muted)] hover:text-[var(--text)]" onClick={() => update({ passes: meta?.default_passes ?? [] })}>
            reset
          </button>
        </div>
      )}
    </div>
  );
};

let sourceEditor: Parameters<OnMount>[0] | null = null;

const jumpTo = (line: number) => {
  sourceEditor?.revealLineInCenter(line);
  sourceEditor?.setPosition({ lineNumber: line, column: 1 });
  sourceEditor?.focus();
};

const SourcePane = () => {
  const meta = useConsole((s) => s.meta);
  const exampleFile = useConsole((s) => s.exampleFile);
  const source = useConsole((s) => s.source);
  const kernels = useConsole((s) => s.kernels);
  const kernel = useConsole((s) => s.kernel);
  const args = useConsole((s) => s.args);
  const theme = useConsole((s) => (s.theme === "nvidia-light" ? "nvq-light" : "nvq-dark"));
  const result = useConsole((s) => s.result);
  const set = useConsole((s) => s.set);
  const editor = useRef<Parameters<OnMount>[0] | null>(null);
  const timer = useRef(0);
  const firstEdit = useRef<number | null>(null);

  const onMount: OnMount = (ed) => {
    editor.current = ed;
    sourceEditor = ed;
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => compileNow());
  };

  const onChange = (v?: string) => {
    const t = performance.now();
    firstEdit.current ??= t;
    set({ source: v ?? "" });
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      firstEdit.current = null;
      compileNow(t);
    }, DEBOUNCE_MS);
  };

  useEffect(() => {
    const ed = editor.current;
    const model = ed?.getModel();
    if (!ed || !model) return;
    const err = result && !result.ok ? result.error : null;
    if (err?.line && err.line <= model.getLineCount()) {
      monaco.editor.setModelMarkers(model, "nvq", [
        {
          startLineNumber: err.line,
          endLineNumber: err.line,
          startColumn: 1,
          endColumn: model.getLineMaxColumn(err.line),
          message: `${err.kind}: ${err.message.split("\n")[0]}`,
          severity: monaco.MarkerSeverity.Error,
        },
      ]);
    } else monaco.editor.setModelMarkers(model, "nvq", []);
  }, [result]);

  return (
    <section className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--panel)]">
      <div className="flex items-center gap-2 border-b border-[var(--border)] px-2 py-1.5 text-xs">
        <select value={exampleFile} onChange={(e) => loadExample(e.target.value)} className="max-w-[220px]" aria-label="Example">
          {meta?.examples.map((e) => (
            <option key={e.file} value={e.file} title={e.title}>
              {e.file.replace(".py", "").replace(/_/g, " ")}
            </option>
          ))}
        </select>
        <div className="mono flex min-w-0 flex-1 items-center">
          <select
            value={kernel}
            onChange={(e) => {
              set({ kernel: e.target.value });
              compileNow();
            }}
            className="mono border-transparent bg-transparent font-semibold text-[var(--accent)]"
            aria-label="Kernel"
          >
            {kernels.map((k) => (
              <option key={k.name} value={k.name}>
                {k.name}
              </option>
            ))}
          </select>
          <span className="text-[var(--muted)]">(</span>
          <input
            type="text"
            className="mono min-w-0 flex-1 border-transparent bg-transparent px-1"
            value={args}
            onChange={(e) => {
              set({ args: e.target.value });
              window.clearTimeout(timer.current);
              const t = performance.now();
              timer.current = window.setTimeout(() => compileNow(t), DEBOUNCE_MS);
            }}
            title={(() => {
              const k = kernels.find((x) => x.name === kernel);
              return k ? `${kernel}(${k.params.map((p) => p.name + (p.annotation ? `: ${p.annotation}` : "")).join(", ")}) · q = dev.qudit_handles` : "";
            })()}
            aria-label="Kernel arguments"
          />
          <span className="text-[var(--muted)]">)</span>
        </div>
      </div>
      <div className="min-h-0 flex-1">
        <Editor
          path={`example:///${exampleFile}`}
          defaultValue={source}
          language="python"
          theme={theme}
          onChange={onChange}
          onMount={onMount}
          options={editorOptions}
        />
      </div>
    </section>
  );
};

const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;

/** What the program does, without IR: for the overview mode. */
const ProgramView = memo(({ result }: { result: CompileResult }) => {
  const ev = result.events!;
  const clock = result.clock_ghz!;
  const count = (k: string) => {
    const i = ev.kinds.indexOf(k);
    return i < 0 ? 0 : ev.kind.reduce((n, x) => n + (x === i ? 1 : 0), 0);
  };
  let end = 0;
  for (let i = 0; i < ev.start.length; i++) end = Math.max(end, ev.start[i] + ev.duration[i]);
  const us = end / clock / 1000;
  const facts = [
    { k: "Length", v: us >= 1 ? `${us.toPrecision(3)} µs` : `${(us * 1000).toPrecision(3)} ns` },
    { k: "Qubits", v: String(result.qudits?.length ?? 0) },
    { k: "Pulses", v: count("drive").toLocaleString() },
    { k: "Measurements", v: count("readout").toLocaleString() },
    { k: "Decisions in the loop", v: (result.calls?.length ?? 0).toLocaleString() },
  ];
  return (
    <div className="flex h-full min-h-0 flex-col gap-3 p-3">
      <div className="grid grid-cols-5 gap-2">
        {facts.map((f) => (
          <div key={f.k} className="rounded-lg border border-[var(--border)] bg-[var(--bg)] px-3 py-2">
            <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">{f.k}</div>
            <div className="text-xl font-black tabular-nums text-[var(--accent)]">{f.v}</div>
          </div>
        ))}
      </div>
      <p className="text-xs leading-relaxed text-[var(--muted)]">
        The compiler turned the Python kernel into a timed program for the control electronics: {plural(count("drive"), "pulse")} and{" "}
        {plural(count("readout"), "measurement")} across {plural(result.qudits?.length ?? 0, "qubit")}
        {result.calls?.length ? `, with ${plural(result.calls.length, "decision")} computed next to the qubits while the program runs` : ""}.
      </p>
      <div className="min-h-0 flex-1 rounded border border-[var(--border)] bg-[var(--bg)] p-2">
        <Timeline events={ev} clockGhz={clock} lines={result.controller?.lines} simple />
      </div>
    </div>
  );
});

const SIM_POINTS = 61;
const simCache = new Map<string, SimResult>();
/** How each cached simulation was first served: null when simulated then. */
const simServed = new Map<string, string | null>();

/** Populations over the kernel's run, simulated once and replayed against
 * the schedule. */
const RunView = memo(({ result, simple }: { result: CompileResult; simple: boolean }) => {
  const replay = useReplay();
  const [sim, setSim] = useState<SimResult | null>(null);
  const [busy, setBusy] = useState(false);
  // How a cached simulation was served, or null for one simulated now.
  const [served, setServed] = useState<string | null>(null);
  const key = JSON.stringify(request(false));
  const ev = result.events!;
  const tooLong = ev.start.length > 4000;

  useEffect(() => {
    const found = simCache.get(key) ?? null;
    setSim(found);
    setServed(simServed.has(key) ? simServed.get(key)! : found ? servedLabel(found, null) : null);
  }, [key]);

  const go = async () => {
    const clicked = performance.now();
    let found = simCache.get(key);
    const inBrowser = found !== undefined;
    if (!found) {
      setBusy(true);
      found = await api.simulate({ ...request(false), points: SIM_POINTS, levels: 2 });
      setBusy(false);
      if (found.ok) simCache.set(key, found);
    }
    const label = inBrowser ? servedLabel(found, null) : found.cached ? servedLabel(found, performance.now() - clicked) : null;
    if (!inBrowser && found.ok) simServed.set(key, label);
    setServed(label);
    setSim(found);
    if (found.ok) replay.start(3000, clicked);
  };

  const series = useMemo<Series[]>(() => {
    if (!sim?.ok) return [];
    const t = sim.series!.map((p) => p.t_ns / 1000);
    const n = sim.qudits!.length;
    if (n === 1)
      return [0, 1].map((level) => ({
        x: t,
        y: sim.series!.map((p) => p.populations[0][level]),
        color: LEVEL_COLORS[level],
        reveal: "progress" as const,
        label: `P(${level})`,
      }));
    return sim.qudits!.map((q, i) => ({
      x: t,
      y: sim.series!.map((p) => p.populations[i][1] ?? 0),
      color: LEVEL_COLORS[i % LEVEL_COLORS.length],
      reveal: "progress" as const,
      label: simple ? `qubit ${q} excited` : `q${q} P(1)`,
    }));
  }, [sim, simple]);

  if (tooLong)
    return (
      <div className="p-6 text-sm text-[var(--muted)]">
        This program is too long to simulate step by step here ({ev.start.length.toLocaleString()} scheduled operations). Its calibration run is under
        Calibration.
      </div>
    );
  return (
    <div className="flex h-full min-h-0 flex-col gap-2 p-3">
      <div className="flex items-center gap-3">
        <GoButton onClick={go} busy={busy} />
        <span className="text-xs text-[var(--muted)]">
          {simple
            ? "Simulate the qubits through the program and replay what happens."
            : `pulse.simulate at ${SIM_POINTS} times over the schedule, then replayed.`}
        </span>
        {sim?.ok && !simple && <span className="ml-auto text-[11px] text-[var(--muted)]">{served ?? `simulated in ${fmtMs(sim.wall_s! * 1000)}`}</span>}
      </div>
      {sim && !sim.ok && sim.error && <ErrorCard error={sim.error} />}
      <div className="min-h-0 flex-[3] rounded border border-[var(--border)] bg-[var(--bg)]">
        {sim?.ok ? (
          <Plot series={series} yDomain={[0, 1]} xLabel="time (µs)" yLabel={simple ? "probability" : "population"} replay={replay} cursor />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-[var(--muted)]">{busy ? "Simulating…" : "Press Go"}</div>
        )}
      </div>
      <div className="min-h-0 flex-[2] rounded border border-[var(--border)] bg-[var(--bg)] p-2">
        <Timeline events={ev} clockGhz={result.clock_ghz!} lines={result.controller?.lines} simple={simple} replay={sim?.ok ? replay : undefined} />
      </div>
    </div>
  );
});

const ControllerSummary = ({ result }: { result: CompileResult }) => {
  const prog = result.controller!;
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--muted)]">
      <span className="mono text-[var(--text)]">{prog.format}</span>
      <span>· {prog.lines.length} lines</span>
      <span>· {(prog.event_count ?? prog.events.length).toLocaleString()} timed ops</span>
      <span>· {prog.epochs.length} epochs</span>
      <span>· {prog.calls.length} classical calls</span>
      {prog.limits && <span className="text-[var(--accent)]">· limits checked</span>}
      {(prog.event_count ?? 0) > prog.events.length && <span>· first {prog.events.length.toLocaleString()} ops shown</span>}
    </div>
  );
};

const StatusBar = ({ result, simple }: { result: CompileResult | null; simple: boolean }) => {
  const compiling = useConsole((s) => s.compiling);
  const [latency, setLatency] = useState<string>("");
  useEffect(() => {
    const id = window.setInterval(() => {
      const last = [...window.__nvqPerf].reverse().find((s) => s.name === "edit → output on screen");
      if (last) setLatency(fmtMs(last.ms));
    }, 500);
    return () => window.clearInterval(id);
  }, []);
  if (!result) return null;
  const m = result.ok ? result.stages!.optimized.metrics : null;
  return (
    <div className="flex items-center gap-3 border-t border-[var(--border)] px-3 py-1 text-[11px] text-[var(--muted)]">
      <span className={clsx("inline-block h-1.5 w-1.5 rounded-full", compiling ? "animate-pulse bg-amber-400" : result.ok ? "bg-[var(--accent)]" : "bg-[var(--danger)]")} />
      {result.ok ? (
        <>
          <span>
            {result.cached ? (result.precomputed ? "precomputed" : "unchanged · from cache") : `compiled in ${fmtMs(result.server_ms ?? m!.total_ms)}`}
          </span>
          {!simple && m && (
            <span className="mono">
              trace {fmtMs(m.trace_ms)} · mlir {fmtMs(m.ffi_ms)} · passes {fmtMs(m.passes_ms)} · schedule {fmtMs(m.schedule_ms)} · {m.op_count} ops ·{" "}
              {result.schedule_policy}
            </span>
          )}
        </>
      ) : (
        <span className="text-[var(--danger)]">{result.error?.stage ?? "error"} failed</span>
      )}
      {latency && <span className="ml-auto">edit → output {latency}</span>}
    </div>
  );
};

const OutputPane = () => {
  const result = useConsole((s) => s.result);
  const mode = useConsole((s) => s.mode);
  const theme = useConsole((s) => (s.theme === "nvidia-light" ? "nvq-light" : "nvq-dark"));
  const simple = mode === "overview";
  const [expertStage, setExpertStage] = useState<Stage>("optimized");
  const [overviewStage, setOverviewStage] = useState<Stage>("program");
  const [options, setOptions] = useState(false);
  const stage = simple ? overviewStage : expertStage;
  const setStage = (s: Stage) => {
    const t = performance.now();
    (simple ? setOverviewStage : setExpertStage)(s);
    afterPaint("output view switch", t);
  };
  const good = useRef<CompileResult | null>(null);
  if (result?.ok) good.current = result;
  const shown = good.current;
  const adaptive = !!shown?.calls?.some((c) => c.function === "adaptive_t1");
  const stages = [...(simple ? OVERVIEW : EXPERT), ...(adaptive ? [{ id: "converge" as Stage, label: "Convergence" }] : [])];
  const active = stages.some((s) => s.id === stage) ? stage : stages[0].id;
  const passes = useConsole((s) => s.passes);

  useEffect(() => {
    if (active === "llvm" && shown && !shown.stages?.llvm) compileNow(performance.now(), true);
  }, [active, shown]);

  const text = useMemo(() => {
    if (!shown) return "";
    if (active === "traced") return shown.stages!.traced.mlir;
    if (active === "optimized") return shown.stages!.optimized.mlir;
    if (active === "controller") return shown.controller ? JSON.stringify(shown.controller, null, 1) : "";
    if (active === "llvm") return shown.stages?.llvm?.mlir ?? shown.stages?.llvm?.error ?? "// lowering pulse → qop → cudm → LLVM…";
    return "";
  }, [shown, active]);
  const textStage = ["traced", "optimized", "controller", "llvm"].includes(active);

  return (
    <section className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--panel)]">
      <div className="flex items-center gap-0.5 border-b border-[var(--border)] px-2 py-1">
        {stages.map((st) => (
          <button
            key={st.id}
            onClick={() => setStage(st.id)}
            className={clsx(
              "rounded px-2.5 py-1 text-[11px] normal-case transition-colors",
              active === st.id ? "bg-[var(--accent)]/15 font-semibold text-[var(--accent)]" : "text-[var(--muted)] hover:text-[var(--text)]",
            )}
          >
            {st.label}
          </button>
        ))}
        <button
          onClick={() => setOptions((o) => !o)}
          className={clsx("mono ml-auto truncate rounded px-2 py-1 text-[10.5px] normal-case", options ? "text-[var(--accent)]" : "text-[var(--muted)] hover:text-[var(--text)]")}
          title="Device, schedule and passes"
        >
          {simple ? "options" : `passes=(${passes.join(", ")})`} ▾
        </button>
      </div>
      {options && <Options />}
      {result && !result.ok && result.error && (
        <div className="max-h-[45%] overflow-auto border-b border-[var(--border)] p-2">
          <ErrorCard error={result.error} onJump={jumpTo} />
        </div>
      )}
      <div className={clsx("relative min-h-0 flex-1 transition-opacity", result && !result.ok && "opacity-40")}>
        {!shown && <div className="p-6 text-sm text-[var(--muted)]">Compiling…</div>}
        {shown && active === "controller" && shown.controller && <ControllerSummary result={shown} />}
        {shown && active === "controller" && !shown.controller && shown.controller_error && <ErrorCard error={shown.controller_error} />}
        {shown && textStage && (
          <div className={clsx("absolute inset-x-0 bottom-0", active === "controller" && shown.controller ? "top-[29px]" : "top-0")}>
            <Editor path={`out:${active}`} value={text} language={active === "controller" ? "json" : "mlir"} theme={theme} options={readOnly} />
          </div>
        )}
        {shown && active === "diff" && (
          <DiffEditor
            language="mlir"
            original={shown.stages!.traced.mlir}
            modified={shown.stages!.optimized.mlir}
            originalModelPath="diff:traced"
            modifiedModelPath="diff:optimized"
            keepCurrentOriginalModel
            keepCurrentModifiedModel
            theme={theme}
            options={{ ...readOnly, renderSideBySide: true }}
          />
        )}
        {shown && active === "schedule" && (
          <div className="h-full p-2">
            <Timeline events={shown.events!} clockGhz={shown.clock_ghz!} lines={shown.controller?.lines} />
          </div>
        )}
        {shown && active === "program" && <ProgramView result={shown} />}
        {shown && active === "run" && <RunView result={shown} simple={simple} />}
        {shown && active === "converge" && (
          <div className="h-full p-3">
            <ChainConvergence simple={simple} />
          </div>
        )}
      </div>
      <StatusBar result={result} simple={simple} />
    </section>
  );
};

const CompilerTab = () => {
  const ready = useConsole((s) => s.source !== "");
  if (!ready) return <div className="p-6 text-sm text-[var(--muted)]">Loading…</div>;
  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,5fr)_minmax(0,7fr)] gap-2">
      <SourcePane />
      <OutputPane />
    </div>
  );
};

export default CompilerTab;
