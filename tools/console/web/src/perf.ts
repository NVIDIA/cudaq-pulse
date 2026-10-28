/** Interaction latencies, kept in memory: shown in the status bar and read by
 * the latency script through `window.__nvqPerf`. Nothing leaves the page. */

export interface Sample {
  name: string;
  ms: number;
  at: number;
}

const samples: Sample[] = [];
const listeners = new Set<(s: Sample) => void>();

declare global {
  interface Window {
    __nvqPerf: Sample[];
  }
}
window.__nvqPerf = samples;

export function record(name: string, ms: number) {
  const s = { name, ms, at: performance.now() };
  samples.push(s);
  if (samples.length > 2000) samples.splice(0, 1000);
  listeners.forEach((l) => l(s));
}

export function onSample(l: (s: Sample) => void) {
  listeners.add(l);
  return () => void listeners.delete(l);
}

/** Records the time since `t0` once the next frame is on screen. */
export function afterPaint(name: string, t0: number) {
  requestAnimationFrame(() => setTimeout(() => record(name, performance.now() - t0), 0));
}

export const fmtMs = (ms: number) => (ms < 1 ? `${(ms * 1000).toFixed(0)} µs` : ms < 100 ? `${ms.toFixed(1)} ms` : `${ms.toFixed(0)} ms`);

/** "precomputed · 23 ms" or "cached · 23 ms" for a cached result, with the
 * time it took to arrive now: a cached result's own run time is from
 * another run. */
export const servedLabel = (r: { precomputed?: boolean }, ms: number | null) =>
  `${r.precomputed ? "precomputed" : "cached"}${ms == null ? "" : ` · ${fmtMs(ms)}`}`;
