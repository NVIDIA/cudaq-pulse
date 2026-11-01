import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { api, type DeviceInfo } from "../api";
import { useConsole } from "../store";
import Plot, { type Series } from "../components/Plot";
import GoButton from "../components/GoButton";
import CouplingGraph from "../components/CouplingGraph";
import { FONT, MONO, cssVar, fmtTick, ticks, useCanvas } from "../components/canvas";
import { record } from "../perf";

interface Curve {
  x: number[];
  y: number[];
}

export interface QuditSweep {
  qudit: number;
  label: string;
  chevron: number[][]; // [duration][detuning], P(1)
  t1: Curve;
  t2: Curve;
  fit: { offset_khz: number; pi_ns: number; t1_us: number | null; t2_us: number | null };
  truth: { offset_khz: number; t1_us: number; t2_us: number | null };
  elapsed_s: number;
}

interface Axes {
  detunings_khz: number[];
  durations_ns: number[];
}

// Dark to the accent green to near white: the console's palette as a colormap.
const STOPS: [number, number, number][] = [
  [10, 12, 10],
  [32, 70, 12],
  [118, 185, 0],
  [232, 255, 176],
];
const LUT = (() => {
  const out = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (STOPS.length - 1);
    const k = Math.min(STOPS.length - 2, Math.floor(t));
    const f = t - k;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = STOPS[k][c] + f * (STOPS[k + 1][c] - STOPS[k][c]);
  }
  return out;
})();

/** A chevron as a bitmap, one pixel per point, made once per qudit and
 * scaled when drawn. Rows run from the longest pulse (top) down. */
function chevronBitmap(grid: number[][]) {
  const h = grid.length;
  const w = grid[0].length;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const img = new ImageData(w, h);
  for (let r = 0; r < h; r++)
    for (let j = 0; j < w; j++) {
      const v = Math.round(Math.max(0, Math.min(1, grid[h - 1 - r][j])) * 255);
      const o = (r * w + j) * 4;
      img.data[o] = LUT[v * 3];
      img.data[o + 1] = LUT[v * 3 + 1];
      img.data[o + 2] = LUT[v * 3 + 2];
      img.data[o + 3] = 255;
    }
  c.getContext("2d")!.putImageData(img, 0, 0);
  return c;
}

const fmt = (v: number | null | undefined, d = 1) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));

const CELL_MIN_W = 250;
const CELL_H = 132;
const GAP = 8;
const FADE_MS = 350;

/** Every qudit's sweeps in one canvas: only the rows in view are drawn, so
 * scrolling a large device costs the same as a small one. */
const QuditGrid = ({
  count,
  labels,
  results,
  bitmaps,
  arrived,
  version,
  selected,
  onSelect,
  simple,
}: {
  count: number;
  labels: string[];
  results: (QuditSweep | undefined)[];
  bitmaps: (HTMLCanvasElement | undefined)[];
  arrived: number[];
  version: number;
  selected: number;
  onSelect: (i: number) => void;
  simple: boolean;
}) => {
  const scroller = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const cols = Math.max(1, Math.floor((width + GAP) / (CELL_MIN_W + GAP)));
  const cellW = (width - GAP * (cols - 1)) / cols;
  const rows = Math.ceil(count / cols);
  const total = rows * (CELL_H + GAP);
  const fading = useRef(0);

  const curve = (ctx: CanvasRenderingContext2D, c: Curve, x: number, y: number, w: number, h: number, color: string) => {
    const x1 = c.x[c.x.length - 1] || 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    c.y.forEach((v, i) => ctx.lineTo(x + (c.x[i] / x1) * w, y + h - Math.max(0, Math.min(1, v)) * h));
    ctx.stroke();
  };

  const { ref, request } = useCanvas(({ ctx, w, h }) => {
    const t0 = performance.now();
    const top = scroller.current?.scrollTop ?? 0;
    const first = Math.max(0, Math.floor(top / (CELL_H + GAP)));
    const last = Math.min(rows - 1, Math.floor((top + h) / (CELL_H + GAP)));
    const border = cssVar("--border");
    const panel = cssVar("--bg");
    const text = cssVar("--text");
    const muted = cssVar("--muted");
    const accent = cssVar("--accent");
    const accent2 = cssVar("--accent-2");
    const now = performance.now();
    let stillFading = false;
    for (let r = first; r <= last; r++)
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        if (i >= count) break;
        const x = c * (cellW + GAP);
        const y = r * (CELL_H + GAP) - top;
        const res = results[i];
        ctx.fillStyle = panel;
        ctx.strokeStyle = i === selected ? accent : border;
        ctx.lineWidth = i === selected ? 2 : 1;
        ctx.beginPath();
        ctx.roundRect(x + 0.5, y + 0.5, cellW - 1, CELL_H - 1, 8);
        ctx.fill();
        ctx.stroke();
        ctx.font = "bold 12px Inter, -apple-system, sans-serif";
        ctx.textBaseline = "top";
        ctx.textAlign = "left";
        ctx.fillStyle = res ? text : muted;
        ctx.fillText(`Qudit ${i}`, x + 10, y + 9);
        ctx.font = FONT;
        ctx.fillStyle = muted;
        ctx.fillText(labels[i] ?? "", x + 70, y + 10);
        if (!res) {
          ctx.fillText("waiting", x + cellW - 52, y + 10);
          ctx.fillStyle = "rgba(127,127,127,0.08)";
          ctx.fillRect(x + 10, y + 30, cellW * 0.42, CELL_H - 40);
          continue;
        }
        const age = now - (arrived[i] ?? 0);
        const alpha = Math.min(1, age / FADE_MS);
        if (alpha < 1) stillFading = true;
        ctx.globalAlpha = alpha;
        const hw = cellW * 0.42;
        const bmp = bitmaps[i];
        if (bmp) {
          ctx.imageSmoothingEnabled = true;
          ctx.drawImage(bmp, x + 10, y + 30, hw, CELL_H - 40);
        }
        const px = x + 20 + hw;
        const pw = cellW - hw - 30;
        const ph = (CELL_H - 58) / 2;
        curve(ctx, res.t1, px, y + 30, pw, ph, accent);
        curve(ctx, res.t2, px, y + 30 + ph + 14, pw, ph, accent2);
        ctx.font = MONO;
        ctx.textAlign = "right";
        ctx.fillStyle = accent;
        // Each label sits in its curve's empty corner: a decay leaves the top
        // right free, a rise to one half the bottom right.
        ctx.fillText(`${simple ? "life" : "T1"} ${fmt(res.fit.t1_us)} µs`, x + cellW - 10, y + 30);
        ctx.fillStyle = accent2;
        ctx.fillText(`${simple ? "coh." : "T2e"} ${fmt(res.fit.t2_us)} µs`, x + cellW - 10, y + 30 + 2 * ph + 3);
        ctx.fillStyle = text;
        ctx.fillText(`${res.fit.offset_khz >= 0 ? "+" : ""}${fmt(res.fit.offset_khz, 0)} kHz`, x + cellW - 10, y + 10);
        ctx.globalAlpha = 1;
      }
    record("sweep grid draw", performance.now() - t0);
    if (stillFading) {
      cancelAnimationFrame(fading.current);
      fading.current = requestAnimationFrame(request);
    }
  });

  useEffect(() => request(), [request, version, selected, cols, count, simple]);

  useEffect(() => {
    const el = scroller.current!;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    let last = 0;
    const onScroll = () => {
      const t = performance.now();
      request();
      if (t - last > 0) requestAnimationFrame(() => record("sweep grid scroll → frame", performance.now() - t));
      last = t;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      ro.disconnect();
      el.removeEventListener("scroll", onScroll);
    };
  }, [request]);

  const onClick = (e: React.MouseEvent) => {
    const el = scroller.current!;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top + el.scrollTop;
    const c = Math.floor(x / (cellW + GAP));
    const r = Math.floor(y / (CELL_H + GAP));
    const i = r * cols + c;
    if (i < count && c < cols) onSelect(i);
  };

  return (
    <div ref={scroller} className="relative min-h-0 flex-1 cursor-pointer overflow-auto" onClick={onClick}>
      <div className="sticky top-0 h-full w-full">
        <canvas ref={ref} className="absolute inset-0 h-full w-full" />
      </div>
      <div style={{ height: Math.max(0, total - (scroller.current?.clientHeight ?? 0)) }} />
    </div>
  );
};

/** A chevron with axes: detuning across, pulse length up. */
const Chevron = ({ bitmap, axes, offset, truth, simple }: { bitmap?: HTMLCanvasElement; axes: Axes; offset?: number; truth?: number; simple: boolean }) => {
  const { ref, request } = useCanvas(({ ctx, w, h }) => {
    const L = 44;
    const B = 30;
    const iw = w - L - 10;
    const ih = h - B - 8;
    const d0 = axes.detunings_khz[0];
    const d1 = axes.detunings_khz[axes.detunings_khz.length - 1];
    const t1 = axes.durations_ns[axes.durations_ns.length - 1];
    const X = (v: number) => L + ((v - d0) / (d1 - d0)) * iw;
    ctx.font = FONT;
    ctx.fillStyle = cssVar("--muted");
    if (bitmap) ctx.drawImage(bitmap, L, 8, iw, ih);
    else {
      ctx.fillStyle = "rgba(127,127,127,0.08)";
      ctx.fillRect(L, 8, iw, ih);
      ctx.fillStyle = cssVar("--muted");
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of ticks(d0 / 1000, d1 / 1000, 6)) ctx.fillText(fmtTick(v), X(v * 1000), 8 + ih + 5);
    ctx.fillText(simple ? "frequency offset (MHz)" : "drive detuning (MHz)", L + iw / 2, h - 12);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of ticks(0, t1, 4)) ctx.fillText(fmtTick(v), L - 6, 8 + ih - (v / t1) * ih);
    ctx.save();
    ctx.translate(10, 8 + ih / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText(simple ? "pulse length (ns)" : "pulse (ns)", 0, 0);
    ctx.restore();
    const line = (v: number | undefined, color: string, dash: number[]) => {
      if (v == null) return;
      ctx.strokeStyle = color;
      ctx.setLineDash(dash);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(X(v), 8);
      ctx.lineTo(X(v), 8 + ih);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(truth, "#9ca3af", [3, 4]);
    line(offset, cssVar("--accent-2"), []);
  });
  useEffect(() => request(), [request, bitmap, offset, truth, simple]);
  return (
    <div className="relative h-full min-h-0 w-full">
      <canvas ref={ref} className="absolute inset-0 h-full w-full" />
    </div>
  );
};

const decay = (c: Curve, tau: number | null, floor: number, rising: boolean): Series[] => {
  const out: Series[] = [{ x: c.x, y: c.y, color: rising ? "#00d4ff" : "#76b900", mode: "dots", label: "simulated" }];
  if (tau) {
    const y0 = c.y[0];
    const xs = Array.from({ length: 80 }, (_, i) => (i / 79) * c.x[c.x.length - 1]);
    out.push({ x: xs, y: xs.map((x) => floor + (y0 - floor) * Math.exp(-x / tau)), color: "#f5a623", width: 1.5, label: "fit" });
  }
  return out;
};

const Detail = ({ res, bitmap, axes, simple }: { res?: QuditSweep; bitmap?: HTMLCanvasElement; axes: Axes | null; simple: boolean }) => {
  const t1 = useMemo(() => (res ? decay(res.t1, res.fit.t1_us, 0, false) : []), [res]);
  const t2 = useMemo(() => (res ? decay(res.t2, res.fit.t2_us, 0.5, true) : []), [res]);
  const rows = res
    ? [
        { k: simple ? "Frequency error" : "Detuning", v: `${fmt(res.fit.offset_khz, 0)} kHz`, t: `${fmt(res.truth.offset_khz, 0)} kHz` },
        { k: simple ? "Flip time" : "π pulse", v: `${fmt(res.fit.pi_ns, 0)} ns`, t: "" },
        { k: simple ? "Lifetime" : "T1", v: `${fmt(res.fit.t1_us)} µs`, t: `${fmt(res.truth.t1_us)} µs` },
        { k: simple ? "Coherence" : "T2 echo", v: `${fmt(res.fit.t2_us)} µs`, t: `${fmt(res.truth.t2_us)} µs` },
      ]
    : [];
  if (!axes)
    return <div className="flex flex-1 items-center justify-center text-xs text-[var(--muted)]">Press Sweep all qudits; results stream in qudit by qudit.</div>;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="min-h-[150px] flex-[1.3] rounded-lg border border-[var(--border)] bg-[var(--bg)] p-1">
        <Chevron bitmap={bitmap} axes={axes} offset={res?.fit.offset_khz} truth={res?.truth.offset_khz} simple={simple} />
      </div>
      <div className="grid min-h-[110px] flex-1 grid-cols-2 gap-2">
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-1">
          {res && <Plot series={t1} yDomain={[0, 1]} xLabel="wait (µs)" yLabel={simple ? "chance of 1" : "P(1)"} legend={false} />}
        </div>
        <div className="rounded-lg border border-[var(--border)] bg-[var(--bg)] p-1">
          {res && <Plot series={t2} yDomain={[0, 1]} xLabel={simple ? "wait (µs)" : "echo delay (µs)"} yLabel={simple ? "chance of 1" : "P(1)"} legend={false} />}
        </div>
      </div>
      <table className="w-full text-xs">
        <thead className="text-[10px] text-[var(--muted)]">
          <tr>
            <th className="py-1 text-left font-medium" />
            <th className="py-1 text-right font-medium">found</th>
            <th className="py-1 text-right font-medium">simulated device</th>
          </tr>
        </thead>
        <tbody className="mono">
          {rows.map((r) => (
            <tr key={r.k} className="border-t border-[var(--border)]">
              <td className="py-0.5 font-sans text-[var(--muted)]">{r.k}</td>
              <td className="text-right text-[var(--accent)]">{r.v}</td>
              <td className="text-right text-[var(--muted)]">{r.t}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const SweepsTab = () => {
  const meta = useConsole((s) => s.meta);
  const simple = useConsole((s) => s.mode === "overview");
  const targets = (meta?.targets ?? []).filter((t) => t.id.startsWith("transmon"));
  const [target, setTarget] = useState("transmon_17q");
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [axes, setAxes] = useState<Axes | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const results = useRef<(QuditSweep | undefined)[]>([]);
  const bitmaps = useRef<(HTMLCanvasElement | undefined)[]>([]);
  const arrived = useRef<number[]>([]);
  const [version, setVersion] = useState(0);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);
  // "precomputed" or "cached" when the sweeps replay a finished run.
  const [served, setServed] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState(0);
  const [seed, setSeed] = useState(3);
  const source = useRef<EventSource | null>(null);

  useEffect(() => {
    api.device(target).then((d) => {
      setInfo(d);
      setLabels(d.qudits.map((q) => q.label));
    });
  }, [target]);
  useEffect(() => () => source.current?.close(), []);

  const done = results.current.filter(Boolean).length;
  const count = labels.length;

  const go = (draw = seed) => {
    source.current?.close();
    const clicked = performance.now();
    let first = true;
    results.current = [];
    bitmaps.current = [];
    arrived.current = [];
    setError(null);
    setElapsed(null);
    setServed(null);
    setRunning(true);
    setVersion((v) => v + 1);
    const es = new EventSource(`/api/sweeps?target=${encodeURIComponent(target)}&seed=${draw}`);
    source.current = es;
    es.onmessage = (m) => {
      const e = JSON.parse(m.data);
      if (e.type === "start") {
        setAxes({ detunings_khz: e.detunings_khz, durations_ns: e.durations_ns });
        setLabels(e.qudits.map((q: { label: string }) => q.label));
      } else if (e.type === "qudit") {
        results.current[e.qudit] = e;
        bitmaps.current[e.qudit] = chevronBitmap(e.chevron);
        arrived.current[e.qudit] = performance.now();
        if (first) {
          record("go → first qudit on screen", performance.now() - clicked);
          first = false;
        }
        // A replayed run's elapsed_s is the original run's; show the time on screen.
        if (e.cached) setServed(e.precomputed ? "precomputed" : "cached");
        setElapsed(e.cached ? (performance.now() - clicked) / 1000 : e.elapsed_s);
        setVersion((v) => v + 1);
      } else if (e.type === "done" || e.type === "error") {
        if (e.type === "error") setError(e.message);
        else setElapsed(e.cached ? (performance.now() - clicked) / 1000 : e.elapsed_s);
        record("go → all qudits", performance.now() - clicked);
        setRunning(false);
        es.close();
      }
    };
    es.onerror = () => {
      setRunning(false);
      es.close();
    };
  };

  const doneSet = useMemo(() => new Set(results.current.map((r, i) => (r ? i : -1)).filter((i) => i >= 0)), [version]);
  const summary = useMemo(() => {
    const rs = results.current.filter(Boolean) as QuditSweep[];
    if (!rs.length) return null;
    const df = Math.sqrt(rs.reduce((s, r) => s + (r.fit.offset_khz - r.truth.offset_khz) ** 2, 0) / rs.length);
    const t1 = rs.reduce((s, r) => s + Math.abs((r.fit.t1_us ?? 0) / r.truth.t1_us - 1), 0) / rs.length;
    return { df, t1 };
  }, [version]);

  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_400px] gap-3">
      <section className="flex min-h-0 flex-col gap-3 rounded-lg border border-[var(--border)] bg-[var(--panel)] p-3">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <div className="min-w-0">
            <h2 className="text-base font-black">Device sweeps</h2>
            <div className="text-[11px] text-[var(--muted)]">
              {simple
                ? "Calibrate every qudit on the device: find its frequency, how long a flip takes, and how long it keeps its state."
                : "Per qudit: a detuning × pulse-length chevron, T1 and Hahn-echo T2, each simulated with cudaq-pulse on a device whose qudits sit off their nominal frequencies."}
            </div>
          </div>
          <div className="ml-auto flex items-center gap-3">
            {!simple && targets.length > 1 && (
              <select value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Device">
                {targets.map((t) => (
                  <option key={t.id}>{t.id}</option>
                ))}
              </select>
            )}
            {!simple && (
              <button
                className="text-[11px] normal-case text-[var(--muted)] hover:text-[var(--text)]"
                title="Draw new frequency offsets for every qudit and sweep again"
                disabled={running}
                onClick={() => {
                  setSeed(seed + 1);
                  go(seed + 1);
                }}
              >
                new device
              </button>
            )}
            <GoButton onClick={() => go()} busy={running} label="Sweep all qudits" />
          </div>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-[var(--muted)]">
          <div className="h-1 flex-1 overflow-hidden rounded-full bg-[var(--bg)]">
            <div className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-300" style={{ width: `${count ? (100 * done) / count : 0}%` }} />
          </div>
          <span className="mono tabular-nums">
            {done} / {count} qudits{served && ` · ${served}`}{elapsed != null && ` · ${elapsed.toFixed(1)} s`}
          </span>
        </div>
        {error && <div className="rounded border border-[var(--danger)]/50 bg-[var(--danger)]/10 p-2 text-xs text-[var(--danger)]">{error}</div>}
        <QuditGrid
          count={count}
          labels={labels}
          results={results.current}
          bitmaps={bitmaps.current}
          arrived={arrived.current}
          version={version}
          selected={selected}
          onSelect={setSelected}
          simple={simple}
        />
      </section>
      <aside className="flex min-h-0 flex-col gap-3 rounded-lg border border-[var(--border)] bg-[var(--panel)] p-3">
        <div className="flex items-baseline gap-2">
          <h2 className="text-sm font-black">Qudit {selected}</h2>
          <span className="text-[11px] text-[var(--muted)]">{labels[selected]}</span>
          {summary && (
            <span className="ml-auto text-[10.5px] text-[var(--muted)]">
              device: frequencies within {summary.df.toFixed(0)} kHz rms · T1 within {(summary.t1 * 100).toFixed(1)}%
            </span>
          )}
        </div>
        <Detail res={results.current[selected]} bitmap={bitmaps.current[selected]} axes={axes} simple={simple} />
        {info && (
          <div className="h-[170px] shrink-0">
            <CouplingGraph info={info} used={doneSet} />
          </div>
        )}
      </aside>
    </div>
  );
};

export default SweepsTab;
