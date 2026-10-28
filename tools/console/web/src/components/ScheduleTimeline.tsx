import { useMemo, useRef, useState, useLayoutEffect } from "react";
import type { ScheduleEvent } from "../api";
import { KIND_COLORS } from "../store";

interface Props {
  events: ScheduleEvent[];
  clockGhz: number;
  lines?: { line: number; role: string; qudit: number }[];
  height?: number;
  cursorNs?: number | null;
}

const ROW = 30;
const LABEL = 110;

const niceStep = (span: number, target: number) => {
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const n = raw / mag;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
};

/** One row per drive or readout line (and per classical link), each
 * operation a bar at its scheduled cycles. */
const ScheduleTimeline = ({ events, clockGhz, lines, cursorNs }: Props) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [zoom, setZoom] = useState(1);
  const [hover, setHover] = useState<ScheduleEvent | null>(null);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const { rows, end } = useMemo(() => {
    const keyOf = (e: ScheduleEvent) =>
      e.kind === "submit" || e.kind === "answer" ? `link ${e.line}` : `line ${e.line}`;
    const role = new Map(lines?.map((l) => [l.line, l]) ?? []);
    const rowMap = new Map<string, { key: string; label: string; sub: string; events: ScheduleEvent[] }>();
    let end = 1;
    for (const e of events) {
      if (e.start == null || e.duration == null) continue;
      end = Math.max(end, e.start + e.duration);
      const key = keyOf(e);
      if (!rowMap.has(key)) {
        const info = role.get(e.line);
        const isLink = key.startsWith("link");
        rowMap.set(key, {
          key,
          label: isLink ? `link ${e.line}` : `L${e.line} ${info?.role ?? (e.kind === "readout" ? "readout" : "drive")}`,
          sub: isLink ? "classical" : `q${info?.qudit ?? e.qudit}`,
          events: [],
        });
      }
      rowMap.get(key)!.events.push(e);
    }
    const rows = [...rowMap.values()].sort((a, b) => {
      const la = a.key.startsWith("link") ? 1 : 0;
      const lb = b.key.startsWith("link") ? 1 : 0;
      if (la !== lb) return la - lb;
      return parseInt(a.key.split(" ")[1]) - parseInt(b.key.split(" ")[1]);
    });
    return { rows, end };
  }, [events, lines]);

  const plotW = Math.max(200, (width - LABEL - 16) * zoom);
  const x = (c: number) => LABEL + (c / end) * plotW;
  const endNs = end / clockGhz;
  const step = niceStep(endNs, Math.max(4, (8 * plotW) / 900));
  const ticks: number[] = [];
  for (let t = 0; t <= endNs + 1e-9; t += step) ticks.push(t);
  const h = rows.length * ROW + 34;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-2 flex items-center gap-3 text-[11px] text-[var(--muted)]">
        {Object.entries(KIND_COLORS)
          .filter(([k]) => events.some((e) => e.kind === k))
          .map(([k, c]) => (
            <span key={k} className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: c }} /> {k}
            </span>
          ))}
        <span className="ml-auto tabular-nums">
          {events.length} ops · {end} cycles · {endNs.toLocaleString(undefined, { maximumFractionDigits: 1 })} ns
        </span>
        <div className="flex items-center gap-1">
          <button className="rounded border border-[var(--border)] px-2" onClick={() => setZoom((z) => Math.max(1, z / 2))}>
            −
          </button>
          <span className="w-8 text-center tabular-nums">{zoom}×</span>
          <button className="rounded border border-[var(--border)] px-2" onClick={() => setZoom((z) => Math.min(256, z * 2))}>
            +
          </button>
        </div>
      </div>
      <div ref={wrapRef} className="relative min-h-0 flex-1 overflow-auto">
        <svg width={plotW + LABEL + 16} height={h} className="block">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t * clockGhz)} x2={x(t * clockGhz)} y1={0} y2={h - 18} stroke="var(--grid)" />
              <text x={x(t * clockGhz)} y={h - 4} fontSize={10} fill="var(--muted)" textAnchor="middle">
                {t >= 1000 ? `${(t / 1000).toLocaleString(undefined, { maximumFractionDigits: 2 })} µs` : `${+t.toFixed(1)} ns`}
              </text>
            </g>
          ))}
          {rows.map((row, i) => (
            <g key={row.key} transform={`translate(0, ${i * ROW + 4})`}>
              <rect x={LABEL} y={0} width={plotW} height={ROW - 6} fill="var(--bg)" opacity={0.5} />
              <text x={6} y={11} fontSize={11} fontWeight={600} fill="var(--text)">
                {row.label}
              </text>
              <text x={6} y={22} fontSize={9} fill="var(--muted)">
                {row.sub}
              </text>
              {row.events.map((e, j) => {
                const w = Math.max(1.5, x(e.start! + e.duration!) - x(e.start!));
                const isWait = e.kind === "wait";
                return (
                  <rect
                    key={j}
                    x={x(e.start!)}
                    y={isWait ? 9 : 2}
                    width={e.duration === 0 ? 2 : w}
                    height={isWait ? 6 : ROW - 10}
                    rx={2}
                    fill={KIND_COLORS[e.kind] ?? "#999"}
                    opacity={isWait ? 0.45 : 0.9}
                    onMouseEnter={() => setHover(e)}
                    onMouseLeave={() => setHover(null)}
                  />
                );
              })}
            </g>
          ))}
          {cursorNs != null && (
            <line
              x1={x(cursorNs * clockGhz)}
              x2={x(cursorNs * clockGhz)}
              y1={0}
              y2={h - 18}
              stroke="var(--accent-2)"
              strokeWidth={1.5}
            />
          )}
        </svg>
      </div>
      <div className="mono mt-1 h-4 text-[11px] text-[var(--muted)]">
        {hover &&
          `${hover.kind} · line ${hover.line} · q${hover.qudit} · start ${hover.start} cyc (${(hover.start! / clockGhz).toFixed(1)} ns) · ${hover.duration} cyc`}
      </div>
    </div>
  );
};

export default ScheduleTimeline;
