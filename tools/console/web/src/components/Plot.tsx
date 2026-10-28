import { useEffect } from "react";
import { FONT, cssVar, fmtTick, ticks, useCanvas } from "./canvas";
import type { Clock } from "../replay";

/** When a series shows during a replay: drawn up to the replay's x
 * (`progress`), faded in at the end (`end`), or always. */
export type Reveal = "progress" | "end" | "always";

export interface Series {
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  color: string;
  mode?: "line" | "dots" | "step";
  width?: number;
  dash?: number[];
  /** A shaded band, e.g. mean ± sigma. */
  lo?: ArrayLike<number>;
  hi?: ArrayLike<number>;
  reveal?: Reveal;
  label?: string;
}

export interface Marker {
  value: number;
  color: string;
  label?: string;
  reveal?: Reveal;
}

interface PlotProps {
  series: Series[];
  xLabel?: string;
  yLabel?: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  xMarkers?: Marker[];
  yMarkers?: Marker[];
  replay?: Clock;
  /** Draw the replay cursor (a vertical line at the replay's x). */
  cursor?: boolean;
  legend?: boolean;
}

const PAD = { l: 44, r: 12, t: 10, b: 30 };

function extent(series: Series[], key: "x" | "y"): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    const arrs = key === "y" ? [s.y, s.lo, s.hi] : [s.x];
    for (const a of arrs) {
      if (!a) continue;
      for (let i = 0; i < a.length; i++) {
        const v = a[i];
        if (Number.isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  if (key === "y") {
    const pad = (hi - lo) * 0.06;
    return [lo - pad, hi + pad];
  }
  return [lo, hi];
}

const alphaFor = (reveal: Reveal | undefined, p: number) =>
  reveal === "end" ? Math.max(0, Math.min(1, (p - 0.92) / 0.08)) : 1;

/** A line, dot and band plot on a canvas, redrawn on each replay frame. */
const Plot = ({ series, xLabel, yLabel, xDomain, yDomain, xMarkers, yMarkers, replay, cursor, legend = true }: PlotProps) => {
  const { ref, request } = useCanvas(({ ctx, w, h }) => {
    const p = replay?.p ?? 1;
    const [x0, x1] = xDomain ?? extent(series, "x");
    const [y0, y1] = yDomain ?? extent(series, "y");
    const iw = w - PAD.l - PAD.r;
    const ih = h - PAD.t - PAD.b;
    const X = (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * iw;
    const Y = (v: number) => PAD.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
    const muted = cssVar("--muted");
    const grid = cssVar("--grid");
    const xCut = x0 + p * (x1 - x0);

    ctx.font = FONT;
    ctx.lineWidth = 1;
    ctx.strokeStyle = grid;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of ticks(y0, y1, Math.max(2, Math.round(ih / 45)))) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(PAD.l + iw, y);
      ctx.stroke();
      ctx.fillText(fmtTick(v), PAD.l - 6, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of ticks(x0, x1, Math.max(2, Math.round(iw / 80)))) {
      const x = Math.round(X(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, PAD.t);
      ctx.lineTo(x, PAD.t + ih);
      ctx.stroke();
      ctx.fillText(fmtTick(v), x, PAD.t + ih + 5);
    }
    if (xLabel) ctx.fillText(xLabel, PAD.l + iw / 2, h - 13);
    if (yLabel) {
      ctx.save();
      ctx.translate(11, PAD.t + ih / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = "middle";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.l, PAD.t, iw, ih);
    ctx.clip();
    const marker = (m: Marker, vertical: boolean) => {
      const a = alphaFor(m.reveal, p);
      if (a <= 0) return;
      ctx.globalAlpha = a;
      ctx.strokeStyle = m.color;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      if (vertical) {
        ctx.moveTo(X(m.value), PAD.t);
        ctx.lineTo(X(m.value), PAD.t + ih);
      } else {
        ctx.moveTo(PAD.l, Y(m.value));
        ctx.lineTo(PAD.l + iw, Y(m.value));
      }
      ctx.stroke();
      ctx.setLineDash([]);
      if (m.label) {
        ctx.fillStyle = m.color;
        ctx.textAlign = vertical ? "left" : "right";
        ctx.textBaseline = "top";
        if (vertical) ctx.fillText(m.label, X(m.value) + 4, PAD.t + 2);
        else ctx.fillText(m.label, PAD.l + iw - 4, Y(m.value) + 3);
      }
      ctx.globalAlpha = 1;
    };
    yMarkers?.forEach((m) => marker(m, false));
    xMarkers?.forEach((m) => marker(m, true));

    for (const s of series) {
      const a = alphaFor(s.reveal, p);
      if (a <= 0) continue;
      const cut = s.reveal === "progress" ? xCut : Infinity;
      const n = s.x.length;
      // The last index shown, and how far towards the next one the cut is.
      let last = -1;
      while (last + 1 < n && s.x[last + 1] <= cut) last++;
      const frac =
        last >= 0 && last + 1 < n && cut !== Infinity ? (cut - s.x[last]) / (s.x[last + 1] - s.x[last] || 1) : 0;
      if (last < 0) continue;
      ctx.globalAlpha = a;
      if (s.lo && s.hi) {
        ctx.fillStyle = s.color;
        ctx.globalAlpha = a * 0.16;
        ctx.beginPath();
        for (let i = 0; i <= last; i++) ctx.lineTo(X(s.x[i]), Y(s.hi[i]));
        for (let i = last; i >= 0; i--) ctx.lineTo(X(s.x[i]), Y(s.lo[i]));
        ctx.closePath();
        ctx.fill();
        ctx.globalAlpha = a;
      }
      ctx.strokeStyle = s.color;
      ctx.fillStyle = s.color;
      ctx.lineWidth = s.width ?? 1.75;
      ctx.setLineDash(s.dash ?? []);
      if (s.mode === "dots") {
        const r = n > 80 ? 2 : 3;
        for (let i = 0; i <= last; i++) {
          ctx.beginPath();
          ctx.arc(X(s.x[i]), Y(s.y[i]), r, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        ctx.beginPath();
        for (let i = 0; i <= last; i++) {
          if (s.mode === "step" && i > 0) ctx.lineTo(X(s.x[i]), Y(s.y[i - 1]));
          ctx.lineTo(X(s.x[i]), Y(s.y[i]));
        }
        if (frac > 0) {
          const xe = s.x[last] + frac * (s.x[last + 1] - s.x[last]);
          const ye = s.mode === "step" ? s.y[last] : s.y[last] + frac * (s.y[last + 1] - s.y[last]);
          ctx.lineTo(X(xe), Y(ye));
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.globalAlpha = 1;
    }
    if (cursor && replay && p < 1) {
      ctx.strokeStyle = cssVar("--text");
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.moveTo(X(xCut), PAD.t);
      ctx.lineTo(X(xCut), PAD.t + ih);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    if (legend) {
      const named = series.filter((s) => s.label && alphaFor(s.reveal, p) > 0);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      let x = PAD.l + 8;
      for (const s of named) {
        ctx.fillStyle = s.color;
        ctx.fillRect(x, PAD.t + 8, 10, 3);
        ctx.fillStyle = muted;
        ctx.fillText(s.label!, x + 14, PAD.t + 9.5);
        x += 24 + ctx.measureText(s.label!).width;
      }
    }
  });

  useEffect(() => {
    request();
    return replay?.subscribe(request);
  }, [replay, request, series, xDomain, yDomain, xMarkers, yMarkers]);

  return (
    <div className="relative h-full min-h-0 w-full">
      <canvas ref={ref} className="absolute inset-0 h-full w-full" />
    </div>
  );
};

export default Plot;
