import { useEffect, useMemo, useRef, useState } from "react";
import { FONT, MONO, cssVar, ticks, useCanvas } from "./canvas";
import type { Events, LineInfo } from "../api";
import { KIND_COLORS } from "../store";
import { record } from "../perf";
import type { Clock } from "../replay";

interface Row {
  line: number;
  label: string;
  idx: Int32Array; // events of the row, by start
  start: Float64Array;
  end: Float64Array;
  maxDur: number;
}

interface TimelineProps {
  events: Events;
  clockGhz: number;
  lines?: LineInfo[];
  /** Plain row names ("Qubit 0 · control") instead of line numbers. */
  simple?: boolean;
  /** Moves a cursor across the schedule as the replay plays. */
  replay?: Clock;
}

const LABEL_W = 132;
const AXIS_H = 22;

function buildRows(events: Events, lines: LineInfo[] | undefined, simple: boolean): Row[] {
  const byLine = new Map<number, number[]>();
  for (let i = 0; i < events.line.length; i++) {
    const l = events.line[i];
    let a = byLine.get(l);
    if (!a) byLine.set(l, (a = []));
    a.push(i);
  }
  const readout = events.kinds.indexOf("readout");
  return [...byLine.keys()]
    .sort((a, b) => a - b)
    .map((line) => {
      const ids = byLine.get(line)!.sort((a, b) => events.start[a] - events.start[b]);
      const start = new Float64Array(ids.length);
      const end = new Float64Array(ids.length);
      let maxDur = 0;
      ids.forEach((e, k) => {
        start[k] = events.start[e];
        end[k] = events.start[e] + events.duration[e];
        maxDur = Math.max(maxDur, events.duration[e]);
      });
      const info = lines?.find((l) => l.line === line);
      const qudit = info?.qudit ?? events.qudit[ids[0]];
      const role = info?.role ?? (ids.some((e) => events.kind[e] === readout) ? "readout" : "drive");
      const label = simple
        ? `Qubit ${qudit} · ${role === "readout" ? "measure" : role === "drive" ? "control" : role}`
        : `L${line} ${role} q${qudit}`;
      return { line, label, idx: Int32Array.from(ids), start, end, maxDur };
    });
}

/** First index of `a` (sorted) not below `v`. */
function lowerBound(a: Float64Array, v: number) {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

const fmtTime = (ns: number) => (Math.abs(ns) >= 1000 ? `${+(ns / 1000).toPrecision(4)} µs` : `${+ns.toPrecision(4)} ns`);

/** The schedule, one row per line, on a canvas: wheel to zoom, drag to pan,
 * double-click to fit. Only events in view are drawn. */
const Timeline = ({ events, clockGhz, lines, simple = false, replay }: TimelineProps) => {
  const rows = useMemo(() => buildRows(events, lines, simple), [events, lines, simple]);
  const end = useMemo(() => {
    let m = 0;
    for (const r of rows) for (let k = 0; k < r.end.length; k++) m = Math.max(m, r.end[k]);
    return Math.max(m, 1);
  }, [rows]);
  const view = useRef<[number, number]>([0, end * 1.02]);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const kindColors = useMemo(() => events.kinds.map((k) => KIND_COLORS[k] ?? "#999"), [events.kinds]);

  const rowH = (h: number) => Math.max(16, Math.min(34, (h - AXIS_H) / Math.max(rows.length, 1)));

  const { ref, request } = useCanvas(({ ctx, w, h }) => {
    const t0 = performance.now();
    const [v0, v1] = view.current;
    const iw = w - LABEL_W;
    const X = (c: number) => LABEL_W + ((c - v0) / (v1 - v0)) * iw;
    const rh = rowH(h);
    const muted = cssVar("--muted");
    ctx.font = FONT;
    ctx.textBaseline = "middle";

    ctx.strokeStyle = cssVar("--grid");
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    const ns0 = v0 / clockGhz;
    const ns1 = v1 / clockGhz;
    for (const t of ticks(ns0, ns1, Math.max(2, Math.round(iw / 110)))) {
      const x = Math.round(X(t * clockGhz)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, AXIS_H);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.fillText(fmtTime(t), x, AXIS_H / 2);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(LABEL_W, AXIS_H, iw, h - AXIS_H);
    ctx.clip();
    rows.forEach((r, ri) => {
      const y = AXIS_H + ri * rh;
      ctx.fillStyle = ri % 2 ? "rgba(127,127,127,0.04)" : "transparent";
      ctx.fillRect(LABEL_W, y, iw, rh);
      const bar = Math.max(6, rh - 8);
      let k = lowerBound(r.start, v0 - r.maxDur);
      let lastPx = -1;
      for (; k < r.start.length && r.start[k] <= v1; k++) {
        if (r.end[k] < v0) continue;
        const e = r.idx[k];
        const xa = X(r.start[k]);
        const xb = Math.max(xa + 1, X(r.end[k]));
        // Sub-pixel events that land on a drawn pixel add nothing.
        if (xb <= lastPx + 0.5 && xb - xa < 1.5) continue;
        lastPx = xb;
        ctx.fillStyle = kindColors[events.kind[e]];
        const wait = events.kinds[events.kind[e]] === "wait";
        ctx.globalAlpha = wait ? 0.35 : 0.9;
        const hh = wait ? bar * 0.4 : bar;
        ctx.fillRect(xa, y + (rh - hh) / 2, xb - xa, hh);
      }
      ctx.globalAlpha = 1;
    });
    if (replay && replay.p < 1) {
      const x = X(replay.p * end);
      ctx.strokeStyle = cssVar("--text");
      ctx.globalAlpha = 0.6;
      ctx.beginPath();
      ctx.moveTo(x, AXIS_H);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    ctx.font = simple ? FONT : MONO;
    ctx.textAlign = "left";
    rows.forEach((r, ri) => {
      ctx.fillStyle = muted;
      ctx.fillText(r.label, 8, AXIS_H + ri * rh + rh / 2);
    });
    record("timeline draw", performance.now() - t0);
  });

  useEffect(() => {
    view.current = [0, end * 1.02];
    request();
  }, [end, request]);
  useEffect(() => request(), [rows, request]);
  useEffect(() => replay?.subscribe(request), [replay, request]);

  useEffect(() => {
    const canvas = ref.current!;
    const toCycles = (px: number) => {
      const [v0, v1] = view.current;
      return v0 + ((px - LABEL_W) / (canvas.clientWidth - LABEL_W)) * (v1 - v0);
    };
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const t = performance.now();
      const [v0, v1] = view.current;
      const span = v1 - v0;
      if (Math.abs(ev.deltaX) > Math.abs(ev.deltaY) || ev.shiftKey) {
        const d = ((ev.shiftKey ? ev.deltaY : ev.deltaX) / canvas.clientWidth) * span;
        view.current = [v0 + d, v1 + d];
      } else {
        const at = toCycles(ev.offsetX);
        const f = Math.exp(ev.deltaY * 0.002);
        const next = Math.max(span * f, 4);
        view.current = [at - (at - v0) * (next / span), at + (v1 - at) * (next / span)];
      }
      request();
      requestAnimationFrame(() => record("timeline zoom/pan → frame", performance.now() - t));
    };
    let drag: { x: number; v: [number, number] } | null = null;
    const onDown = (ev: PointerEvent) => {
      drag = { x: ev.clientX, v: [...view.current] as [number, number] };
      canvas.setPointerCapture(ev.pointerId);
    };
    const onMove = (ev: PointerEvent) => {
      if (drag) {
        const t = performance.now();
        const span = drag.v[1] - drag.v[0];
        const d = ((drag.x - ev.clientX) / (canvas.clientWidth - LABEL_W)) * span;
        view.current = [drag.v[0] + d, drag.v[1] + d];
        request();
        requestAnimationFrame(() => record("timeline zoom/pan → frame", performance.now() - t));
        setHover(null);
        return;
      }
      const rh = rowH(canvas.clientHeight);
      const ri = Math.floor((ev.offsetY - AXIS_H) / rh);
      const r = rows[ri];
      if (!r || ev.offsetX < LABEL_W) return setHover(null);
      const c = toCycles(ev.offsetX);
      const tol = (2 / (canvas.clientWidth - LABEL_W)) * (view.current[1] - view.current[0]);
      let k = lowerBound(r.start, c - r.maxDur - tol);
      let hit = -1;
      for (; k < r.start.length && r.start[k] <= c + tol; k++) if (r.end[k] >= c - tol) hit = r.idx[k];
      if (hit < 0) return setHover(null);
      const ns = events.start[hit] / clockGhz;
      setHover({
        x: ev.offsetX,
        y: ev.offsetY,
        text: `${events.kinds[events.kind[hit]]} · ${fmtTime(ns)} → ${fmtTime(ns + events.duration[hit] / clockGhz)} · ${events.duration[hit]} cycles`,
      });
    };
    const onUp = () => (drag = null);
    const onDbl = () => {
      view.current = [0, end * 1.02];
      request();
    };
    const onLeave = () => setHover(null);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("dblclick", onDbl);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("dblclick", onDbl);
      canvas.removeEventListener("pointerleave", onLeave);
    };
  }, [ref, rows, end, events, clockGhz, request]);

  const kinds = events.kinds.filter((_, i) => events.kind.includes(i));
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative min-h-0 flex-1 cursor-grab active:cursor-grabbing">
        <canvas ref={ref} className="absolute inset-0 h-full w-full touch-none" />
        {hover && (
          <div
            className="mono pointer-events-none absolute z-10 whitespace-nowrap rounded border border-[var(--border)] bg-[var(--panel)] px-2 py-1 text-[11px] shadow-lg"
            style={{ left: Math.min(hover.x + 12, 9999), top: hover.y + 12 }}
          >
            {hover.text}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 px-2 pt-1 text-[10.5px] text-[var(--muted)]">
        {kinds.map((k) => (
          <span key={k} className="flex items-center gap-1">
            <i className="inline-block h-2 w-2.5 rounded-sm" style={{ background: KIND_COLORS[k] ?? "#999" }} />
            {simple ? ({ drive: "pulse", readout: "measurement", submit: "decision sent", answer: "decision back" } as Record<string, string>)[k] ?? k : k}
          </span>
        ))}
        <span className="ml-auto">scroll to zoom · drag to pan · double-click to fit</span>
      </div>
    </div>
  );
};

export default Timeline;
