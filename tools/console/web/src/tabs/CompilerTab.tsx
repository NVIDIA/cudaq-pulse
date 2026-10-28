import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Editor, { DiffEditor, type OnMount } from "@monaco-editor/react";
import clsx from "clsx";
import { api, type CompileResult } from "../api";
import { useConsole, KIND_COLORS } from "../store";
import { monaco } from "../monaco";
import Panel from "../components/Panel";
import ErrorCard from "../components/ErrorCard";
import ScheduleTimeline from "../components/ScheduleTimeline";

type Stage = "traced" | "optimized" | "diff" | "schedule" | "controller" | "llvm";

const STAGES: { id: Stage; label: string }[] = [
  { id: "traced", label: "Traced IR" },
  { id: "optimized", label: "After passes" },
  { id: "diff", label: "Diff" },
  { id: "schedule", label: "Schedule" },
  { id: "controller", label: "Controller program" },
  { id: "llvm", label: "LLVM IR" },
];

const fmtMs = (ms?: number) => (ms == null ? "–" : ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : `${ms.toFixed(2)} ms`);

const editorOptions = {
  minimap: { enabled: false },
  fontSize: 12.5,
  fontFamily: '"JetBrains Mono", "SF Mono", Menlo, monospace',
  scrollBeyondLastLine: false,
  automaticLayout: true,
  renderLineHighlight: "line" as const,
  padding: { top: 8 },
};

export const useCompile = () => {
  const set = useConsole((s) => s.set);
  return useCallback(
    async (llvm = false) => {
      const s = useConsole.getState();
      if (!s.source || !s.kernel) return;
      set({ compiling: true });
      try {
        const [result, kernels] = await Promise.all([
          api.compile({
            source: s.source,
            kernel: s.kernel,
            args: s.args,
            target: s.target,
            passes: s.passes,
            schedule: s.schedule,
            llvm,
          }),
          api.kernels(s.source),
        ]);
        set({ result, kernels: kernels.ok ? kernels.kernels! : s.kernels });
      } catch (e) {
        set({ result: { ok: false, error: { kind: "NetworkError", message: String(e) } } });
      } finally {
        set({ compiling: false });
      }
    },
    [set],
  );
};

export const useLoadExample = () => {
  const set = useConsole((s) => s.set);
  const compile = useCompile();
  return useCallback(
    async (file: string) => {
      const meta = useConsole.getState().meta;
      const preset = meta?.examples.find((e) => e.file === file);
      const { source } = await api.example(file);
      set({ exampleFile: file, source, kernel: preset?.kernel ?? "", args: preset?.args ?? "", result: null });
      const k = await api.kernels(source);
      if (k.ok) set({ kernels: k.kernels! });
      await compile();
    },
    [set, compile],
  );
};

const StageStrip = ({ result }: { result: CompileResult }) => {
  const o = result.stages!.optimized.metrics;
  const tracedLines = result.stages!.traced.mlir.split("\n").length;
  const optLines = result.stages!.optimized.mlir.split("\n").length;
  const items = [
    { k: "Trace", v: fmtMs(o.trace_ms), d: `${o.op_count} ops` },
    { k: "MLIR build", v: fmtMs(o.ffi_ms), d: `${tracedLines} lines` },
    { k: "Passes", v: fmtMs(o.passes_ms), d: `${optLines} lines (${optLines - tracedLines >= 0 ? "+" : ""}${optLines - tracedLines})` },
    { k: "Schedule", v: fmtMs(o.schedule_ms), d: `${result.events?.length ?? 0} events · ${result.schedule_policy}` },
    { k: "Controller", v: result.controller ? `${result.controller.lines.length} lines` : "–", d: result.controller ? `${result.controller.events.length} timed ops` : "export failed" },
  ];
  return (
    <div className="flex items-stretch gap-1">
      {items.map((it, i) => (
        <div key={it.k} className="flex items-center gap-1">
          <div className="min-w-[96px] rounded border border-[var(--border)] bg-[var(--bg)] px-2 py-1">
            <div className="text-[9px] uppercase tracking-wider text-[var(--muted)]">{it.k}</div>
            <div className="text-sm font-bold tabular-nums text-[var(--accent)]">{it.v}</div>
            <div className="text-[10px] text-[var(--muted)]">{it.d}</div>
          </div>
          {i < items.length - 1 && <span className="text-[var(--muted)]">→</span>}
        </div>
      ))}
    </div>
  );
};

const ControllerView = ({ result, theme }: { result: CompileResult; theme: string }) => {
  const prog = result.controller;
  if (!prog) return result.controller_error ? <ErrorCard error={result.controller_error} /> : null;
  const kinds: Record<string, number> = {};
  for (const e of prog.events) kinds[String(e.kind)] = (kinds[String(e.kind)] ?? 0) + 1;
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex flex-wrap gap-2 text-xs">
        <span className="rounded border border-[var(--border)] px-2 py-1 mono">{prog.format}</span>
        {prog.lines.map((l) => (
          <span key={l.line} className="rounded border border-[var(--border)] px-2 py-1">
            L{l.line} <b>{l.role}</b> q{l.qudit}
          </span>
        ))}
        {Object.entries(kinds).map(([k, n]) => (
          <span key={k} className="rounded px-2 py-1 font-semibold text-black" style={{ background: KIND_COLORS[k] ?? "#999" }}>
            {n} {k}
          </span>
        ))}
        <span className="rounded border border-[var(--border)] px-2 py-1">{prog.epochs.length} epochs</span>
        <span className="rounded border border-[var(--border)] px-2 py-1">{prog.calls.length} calls</span>
        {prog.limits && <span className="rounded border border-[var(--accent)] px-2 py-1 text-[var(--accent)]">limits checked</span>}
      </div>
      <div className="min-h-0 flex-1 overflow-hidden rounded border border-[var(--border)]">
        <Editor
          language="json"
          value={JSON.stringify(prog, null, 2)}
          theme={theme}
          options={{ ...editorOptions, readOnly: true, folding: true }}
        />
      </div>
    </div>
  );
};

const CompilerTab = () => {
  const s = useConsole();
  const compile = useCompile();
  const loadExample = useLoadExample();
  const [stage, setStage] = useState<Stage>("optimized");
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);
  const theme = s.theme === "nvidia-light" ? "nvq-light" : "nvq-dark";
  const result = s.result;

  const onMount: OnMount = (ed) => {
    editorRef.current = ed;
    ed.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => compile());
  };

  useEffect(() => {
    const ed = editorRef.current;
    const model = ed?.getModel();
    if (!ed || !model) return;
    const err = result && !result.ok ? result.error : null;
    if (err?.line) {
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
      ed.revealLineInCenter(err.line);
    } else {
      monaco.editor.setModelMarkers(model, "nvq", []);
    }
  }, [result]);

  useEffect(() => {
    if (stage === "llvm" && result?.ok && !result.stages?.llvm) compile(true);
  }, [stage, result, compile]);

  const kernelParams = useMemo(
    () => s.kernels.find((k) => k.name === s.kernel)?.params.map((p) => p.name + (p.annotation ? `: ${p.annotation}` : "")).join(", "),
    [s.kernels, s.kernel],
  );

  const togglePass = (p: string) => {
    const order = s.meta?.passes ?? [];
    const next = s.passes.includes(p) ? s.passes.filter((x) => x !== p) : [...s.passes, p];
    s.set({ passes: order.filter((x) => next.includes(x)) });
  };

  const jump = (line: number) => {
    editorRef.current?.revealLineInCenter(line);
    editorRef.current?.setPosition({ lineNumber: line, column: 1 });
    editorRef.current?.focus();
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs">
        <label className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Example</label>
        <select value={s.exampleFile} onChange={(e) => loadExample(e.target.value)} className="max-w-[260px]">
          {s.meta?.examples.map((e) => (
            <option key={e.file} value={e.file}>
              {e.file.replace(".py", "")}
            </option>
          ))}
        </select>
        <label className="ml-2 text-[10px] uppercase tracking-wider text-[var(--muted)]">Kernel</label>
        <select value={s.kernel} onChange={(e) => s.set({ kernel: e.target.value })}>
          {s.kernels.map((k) => (
            <option key={k.name} value={k.name}>
              {k.name}
            </option>
          ))}
        </select>
        <span className="mono text-[var(--muted)]">(</span>
        <input
          type="text"
          className="mono w-56"
          value={s.args}
          onChange={(e) => s.set({ args: e.target.value })}
          onKeyDown={(e) => e.key === "Enter" && compile()}
          title={kernelParams ? `${s.kernel}(${kernelParams}); q = dev.qudit_handles` : ""}
          aria-label="Kernel arguments"
        />
        <span className="mono text-[var(--muted)]">)</span>
        <label className="ml-2 text-[10px] uppercase tracking-wider text-[var(--muted)]">Device</label>
        <select value={s.target} onChange={(e) => s.set({ target: e.target.value })}>
          {s.meta?.targets.map((t) => (
            <option key={t.id} value={t.id}>
              {t.id}
            </option>
          ))}
        </select>
        <label className="ml-2 text-[10px] uppercase tracking-wider text-[var(--muted)]">Schedule</label>
        <select value={s.schedule} onChange={(e) => s.set({ schedule: e.target.value })}>
          {s.meta?.schedules.map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>
        <button
          onClick={() => compile(stage === "llvm")}
          disabled={s.compiling}
          className="ml-auto rounded-md bg-[var(--accent)] px-4 py-1.5 text-xs font-bold text-black hover:brightness-110 disabled:opacity-50"
        >
          {s.compiling ? "Compiling…" : "Compile ⌘↵"}
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-2 px-1 text-xs">
        <span className="text-[10px] uppercase tracking-wider text-[var(--muted)]">passes=</span>
        {s.meta?.passes.map((p) => (
          <button
            key={p}
            onClick={() => togglePass(p)}
            className={clsx(
              "rounded-full border px-2.5 py-0.5 text-[11px] normal-case",
              s.passes.includes(p)
                ? "border-[var(--accent)] bg-[var(--accent)]/20 text-[var(--accent)]"
                : "border-[var(--border)] text-[var(--muted)]",
            )}
          >
            {p}
          </button>
        ))}
        <button
          className="rounded border border-[var(--border)] px-2 py-0.5 text-[10px] text-[var(--muted)]"
          onClick={() => s.set({ passes: s.meta?.default_passes ?? [] })}
        >
          default
        </button>
        <button
          className="rounded border border-[var(--border)] px-2 py-0.5 text-[10px] text-[var(--muted)]"
          onClick={() => s.set({ passes: [] })}
        >
          none
        </button>
        <span className="mono ml-2 text-[var(--muted)]">({s.passes.map((p) => `"${p}"`).join(", ")})</span>
        {result?.ok && result.machine && (
          <span className="ml-auto rounded border border-[var(--accent-2)]/50 px-2 py-0.5 text-[10px] text-[var(--accent-2)]" title={result.machine}>
            module MACHINE model
          </span>
        )}
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,5fr)_minmax(0,7fr)] gap-2">
        <Panel title={<>kernel.py <span className="normal-case text-[var(--muted)]">· {s.exampleFile}</span></>} bodyClassName="p-0">
          <Editor
            language="python"
            value={s.source}
            theme={theme}
            onChange={(v) => s.set({ source: v ?? "" })}
            onMount={onMount}
            options={editorOptions}
          />
        </Panel>
        <section className="flex min-h-0 flex-col rounded-lg border border-[var(--border)] bg-[var(--panel)]">
          <div className="flex items-center gap-1 border-b border-[var(--border)] px-2 pt-2">
            {STAGES.map((st) => (
              <button
                key={st.id}
                onClick={() => setStage(st.id)}
                className={clsx(
                  "rounded-t px-3 py-1.5 text-[11px]",
                  stage === st.id ? "bg-[var(--bg)] text-[var(--accent)]" : "text-[var(--muted)] hover:text-[var(--text)]",
                )}
              >
                {st.label}
              </button>
            ))}
          </div>
          <div className="min-h-0 flex-1 p-2">
            {!result && <div className="p-6 text-sm text-[var(--muted)]">Pick an example and compile.</div>}
            {result && !result.ok && result.error && <ErrorCard error={result.error} onJump={jump} />}
            {result?.ok && (
              <div className="flex h-full min-h-0 flex-col gap-2">
                <StageStrip result={result} />
                <div className="min-h-0 flex-1 overflow-hidden rounded border border-[var(--border)]">
                  {stage === "traced" && (
                    <Editor language="mlir" value={result.stages!.traced.mlir} theme={theme} options={{ ...editorOptions, readOnly: true }} />
                  )}
                  {stage === "optimized" && (
                    <Editor language="mlir" value={result.stages!.optimized.mlir} theme={theme} options={{ ...editorOptions, readOnly: true }} />
                  )}
                  {stage === "diff" && (
                    <DiffEditor
                      language="mlir"
                      original={result.stages!.traced.mlir}
                      modified={result.stages!.optimized.mlir}
                      theme={theme}
                      keepCurrentOriginalModel
                      keepCurrentModifiedModel
                      options={{ ...editorOptions, readOnly: true, renderSideBySide: true }}
                    />
                  )}
                  {stage === "schedule" && (
                    <div className="h-full p-2">
                      <ScheduleTimeline events={result.events!} clockGhz={result.clock_ghz!} lines={result.controller?.lines} />
                    </div>
                  )}
                  {stage === "controller" && (
                    <div className="h-full p-2">
                      <ControllerView result={result} theme={theme} />
                    </div>
                  )}
                  {stage === "llvm" &&
                    (result.stages?.llvm?.mlir ? (
                      <Editor language="mlir" value={result.stages.llvm.mlir} theme={theme} options={{ ...editorOptions, readOnly: true }} />
                    ) : (
                      <div className="p-4 text-xs text-[var(--muted)]">
                        {result.stages?.llvm?.error ?? "Lowering pulse → qop → cudm → LLVM…"}
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};

export default CompilerTab;
