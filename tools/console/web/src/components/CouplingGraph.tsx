import { useMemo } from "react";
import type { DeviceInfo } from "../api";

/** Qudits and couplings: a spring layout for transmons, a grid or chain for
 * atoms. */
const CouplingGraph = ({ info, used }: { info: DeviceInfo; used: Set<number> }) => {
  const n = info.qudits.length;
  const size = 340;
  const pos = useMemo(() => {
    if (info.architecture !== "transmon") {
      const cols = info.id.includes("square") ? Math.ceil(Math.sqrt(n)) : n;
      const rows = Math.ceil(n / cols);
      return info.qudits.map((_, i) => ({
        x: 40 + ((i % cols) * (size - 80)) / Math.max(1, cols - 1),
        y: rows === 1 ? size / 2 : 40 + (Math.floor(i / cols) * (size - 80)) / Math.max(1, rows - 1),
      }));
    }
    // A deterministic spring layout from a circle: couplings pull, every
    // pair pushes.
    const p = info.qudits.map((_, i) => {
      const a = (2 * Math.PI * i) / n;
      return { x: Math.cos(a), y: Math.sin(a) };
    });
    const k = 1.6 / Math.sqrt(n);
    for (let it = 0; it < 400; it++) {
      const t = 0.1 * (1 - it / 400) + 0.005;
      const d = p.map(() => ({ x: 0, y: 0 }));
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
          const dx = p[i].x - p[j].x;
          const dy = p[i].y - p[j].y;
          const r2 = Math.max(dx * dx + dy * dy, 1e-4);
          const f = (k * k) / r2;
          d[i].x += dx * f;
          d[i].y += dy * f;
          d[j].x -= dx * f;
          d[j].y -= dy * f;
        }
      for (const [a, b] of info.couplings) {
        const dx = p[a].x - p[b].x;
        const dy = p[a].y - p[b].y;
        const r = Math.max(Math.hypot(dx, dy), 1e-3);
        const f = r / k;
        d[a].x -= dx * f;
        d[a].y -= dy * f;
        d[b].x += dx * f;
        d[b].y += dy * f;
      }
      for (let i = 0; i < n; i++) {
        const m = Math.hypot(d[i].x, d[i].y) || 1;
        p[i].x += (d[i].x / m) * Math.min(m, t);
        p[i].y += (d[i].y / m) * Math.min(m, t);
      }
    }
    const xs = p.map((q) => q.x);
    const ys = p.map((q) => q.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const span = Math.max(x1 - x0, y1 - y0) || 1;
    return p.map((q) => ({
      x: 34 + ((q.x - x0) / span) * (size - 68) + ((size - 68) * (1 - (x1 - x0) / span)) / 2,
      y: 34 + ((q.y - y0) / span) * (size - 68) + ((size - 68) * (1 - (y1 - y0) / span)) / 2,
    }));
  }, [info, n]);
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="h-full w-full">
      {info.couplings.map(([a, b], i) => (
        <line key={i} x1={pos[a].x} y1={pos[a].y} x2={pos[b].x} y2={pos[b].y} stroke="var(--border)" strokeWidth={1.5} />
      ))}
      {info.qudits.map((q, i) => (
        <g key={i}>
          <circle
            cx={pos[i].x}
            cy={pos[i].y}
            r={15}
            fill={used.has(i) ? "var(--accent)" : "var(--bg)"}
            stroke={used.has(i) ? "var(--accent)" : "var(--accent-2)"}
            strokeWidth={1.5}
          />
          <text x={pos[i].x} y={pos[i].y + 3.5} textAnchor="middle" fontSize={9} fontWeight={700} fill={used.has(i) ? "#000" : "var(--text)"}>
            {i}
          </text>
          <text x={pos[i].x} y={pos[i].y + 27} textAnchor="middle" fontSize={8} fill="var(--muted)">
            {q.label}
          </text>
        </g>
      ))}
    </svg>
  );
};

export default CouplingGraph;
