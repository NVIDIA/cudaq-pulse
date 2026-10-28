import { useEffect, useRef } from "react";

const cache = new Map<string, string>();

/** A theme colour, read from the CSS variables once per theme. */
export function cssVar(name: string) {
  let v = cache.get(name);
  if (v === undefined) {
    v = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
    cache.set(name, v);
  }
  return v;
}

const themeListeners = new Set<() => void>();

export function themeChanged() {
  cache.clear();
  themeListeners.forEach((l) => l());
}

export interface Surface {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
}

/** A canvas that fills its parent at the device pixel ratio. `draw` runs at
 * most once per animation frame, after `request()` or a resize. */
export function useCanvas(draw: (s: Surface) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const frame = useRef(0);
  const size = useRef({ w: 0, h: 0 });

  const request = useRef(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      const canvas = ref.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx || size.current.w === 0) return;
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.current.w, size.current.h);
      drawRef.current({ ctx, w: size.current.w, h: size.current.h });
    });
  }).current;

  useEffect(() => {
    const canvas = ref.current!;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      const dpr = window.devicePixelRatio || 1;
      size.current = { w: width, h: height };
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      request();
    });
    ro.observe(canvas.parentElement!);
    themeListeners.add(request);
    return () => {
      ro.disconnect();
      themeListeners.delete(request);
      cancelAnimationFrame(frame.current);
    };
  }, [request]);

  return { ref, request };
}

/** About `count` round tick values over [lo, hi]. */
export function ticks(lo: number, hi: number, count = 5) {
  const span = hi - lo || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

export const fmtTick = (v: number) => {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(0);
  return String(+v.toPrecision(3));
};

export const FONT = '11px "Inter", -apple-system, "Segoe UI", sans-serif';
export const MONO = '10.5px "JetBrains Mono", "SF Mono", Menlo, monospace';
