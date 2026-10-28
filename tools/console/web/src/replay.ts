import { useEffect, useRef, useState } from "react";
import { record } from "./perf";

type Listener = (p: number) => void;

/** What a plot needs of a replay: its progress and a frame callback. */
export interface Clock {
  readonly p: number;
  subscribe(l: Listener): () => void;
}

/** A clock that runs over part of another one: `f` maps the parent's
 * progress to this one's. */
export function subClock(parent: Clock, f: (p: number) => number): Clock {
  return {
    get p() {
      return f(parent.p);
    },
    subscribe: (l) => parent.subscribe((p) => l(f(p))),
  };
}

export const window01 = (a: number, b: number) => (p: number) => Math.max(0, Math.min(1, (p - a) / (b - a)));

/** One clock for a replayed run: every plot, timeline cursor and readout of
 * the run draws from the same progress (0 to 1) in one animation frame. */
export class Replay implements Clock {
  p = 1;
  private t0 = 0;
  private ms = 1;
  private raf = 0;
  private clickedAt: number | null = null;
  private listeners = new Set<Listener>();
  private ends = new Set<() => void>();

  get running() {
    return this.raf !== 0;
  }

  subscribe(l: Listener) {
    this.listeners.add(l);
    l(this.p);
    return () => void this.listeners.delete(l);
  }

  onEnd(l: () => void) {
    this.ends.add(l);
    return () => void this.ends.delete(l);
  }

  /** Replays over `ms`; `clickedAt` is when Go was pressed, for the
   * Go-to-first-frame latency. */
  start(ms: number, clickedAt?: number) {
    cancelAnimationFrame(this.raf);
    this.ms = ms;
    this.t0 = performance.now();
    this.clickedAt = clickedAt ?? null;
    this.p = 0;
    const tick = (now: number) => {
      this.p = Math.min(1, (now - this.t0) / this.ms);
      this.listeners.forEach((l) => l(this.p));
      if (this.clickedAt != null) {
        record("go → first frame", performance.now() - this.clickedAt);
        this.clickedAt = null;
      }
      if (this.p < 1) this.raf = requestAnimationFrame(tick);
      else {
        this.raf = 0;
        this.ends.forEach((l) => l());
      }
    };
    this.raf = requestAnimationFrame(tick);
  }

  finish() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.clickedAt = null;
    this.p = 1;
    this.listeners.forEach((l) => l(1));
    this.ends.forEach((l) => l());
  }
}

export function useReplay() {
  const ref = useRef<Replay>();
  if (!ref.current) ref.current = new Replay();
  return ref.current;
}

/** Re-renders when the replay's progress crosses into a new step (of
 * `steps`), not every frame. */
export function useReplayStep(replay: Clock, steps: number) {
  const [step, setStep] = useState(steps);
  useEffect(() => replay.subscribe((p) => setStep(Math.min(steps, Math.floor(p * steps + 1e-9)))), [replay, steps]);
  return step;
}

export function useReplayRunning(replay: Replay) {
  const [running, setRunning] = useState(replay.running);
  useEffect(() => {
    let last = replay.running;
    const off = replay.subscribe(() => {
      if (replay.running !== last) setRunning((last = replay.running));
    });
    const offEnd = replay.onEnd(() => setRunning((last = false)));
    return () => {
      off();
      offEnd();
    };
  }, [replay]);
  return running;
}
