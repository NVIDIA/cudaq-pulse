import { useEffect, useMemo, useState } from "react";
import { api, type DeviceInfo } from "../api";
import { useConsole } from "../store";
import Panel from "../components/Panel";
import StatCard from "../components/StatCard";

const fmt = (v: number | null | undefined, d = 2) => (v == null ? "–" : v.toFixed(d));

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

const DeviceTab = () => {
  const { meta, target: compilerTarget, result } = useConsole();
  const [target, setTarget] = useState(compilerTarget);
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  useEffect(() => {
    api.device(target).then(setInfo).catch(() => setInfo(null));
  }, [target]);
  const lines = result?.ok && result.target === target ? result.controller?.lines ?? [] : [];
  const used = new Set(result?.ok && result.target === target ? result.qudits ?? [] : []);

  if (!info) return <div className="p-6 text-sm text-[var(--muted)]">Loading device…</div>;
  const t1s = info.qudits.map((q) => q.t1_us ?? 0);
  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs">
        <label className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Device</label>
        <select value={target} onChange={(e) => setTarget(e.target.value)}>
          {meta?.targets.map((t) => (
            <option key={t.id}>{t.id}</option>
          ))}
        </select>
        <span className="mono text-[var(--muted)]">pulse.get_device(pulse.Target(&quot;{target}&quot;))</span>
      </div>
      <div className="grid grid-cols-6 gap-2">
        <StatCard label="Architecture" value={info.architecture} accent />
        <StatCard label="Qudits" value={String(info.qudits.length)} />
        <StatCard label="Couplings" value={String(info.couplings.length)} />
        <StatCard label="Clock" value={`${info.clock_ghz} GHz`} delta={`${(1 / info.clock_ghz).toFixed(2)} ns / cycle`} />
        <StatCard label="T1 range" value={`${Math.min(...t1s).toFixed(0)}–${Math.max(...t1s).toFixed(0)} µs`} />
        <StatCard
          label="Calibration"
          value={info.calibrated ? "shipped" : "none"}
          delta={info.controller_backends.length ? `backends: ${info.controller_backends.join(", ")}` : "no controller backends installed"}
        />
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,2fr)_minmax(0,1fr)] gap-2">
        <Panel title="Qudits" bodyClassName="overflow-auto p-0">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-[var(--panel)] text-[10px] tracking-wider text-[var(--muted)]">
              <tr>
                {["#", "label", "f01 GHz", "α MHz", "T1 µs", "T2* µs", "T2e µs", "thermal", "readout GHz"].map((h) => (
                  <th key={h} className="px-3 py-2 text-right first:text-left">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {info.qudits.map((q) => (
                <tr key={q.index} className={`border-t border-[var(--border)] ${used.has(q.index) ? "bg-[var(--accent)]/10" : ""}`}>
                  <td className="px-3 py-1 font-bold">{q.index}</td>
                  <td className="px-3 text-right">{q.label}</td>
                  <td className="mono px-3 text-right">{fmt(q.frequency_ghz, 4)}</td>
                  <td className="mono px-3 text-right">{fmt(q.anharmonicity_mhz, 0)}</td>
                  <td className="mono px-3 text-right">{fmt(q.t1_us, 1)}</td>
                  <td className="mono px-3 text-right">{fmt(q.t2_star_us, 1)}</td>
                  <td className="mono px-3 text-right">{fmt(q.t2_echo_us, 1)}</td>
                  <td className="mono px-3 text-right">{fmt(q.thermal_population, 3)}</td>
                  <td className="mono px-3 text-right">{fmt(q.readout_ghz, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <div className="flex min-h-0 flex-col gap-2">
          <Panel title="Coupling map" className="flex-1">
            <CouplingGraph info={info} used={used} />
          </Panel>
          <Panel title="Lines of the last compiled kernel">
            {lines.length === 0 ? (
              <div className="text-xs text-[var(--muted)]">compile a kernel on this device in the compiler explorer</div>
            ) : (
              <div className="flex flex-wrap gap-1.5 text-xs">
                {lines.map((l) => (
                  <span key={l.line} className="rounded border border-[var(--border)] px-2 py-1">
                    L{l.line} <b className={l.role === "readout" ? "text-[var(--accent-2)]" : "text-[var(--accent)]"}>{l.role}</b> → q{l.qudit}
                  </span>
                ))}
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
};

export default DeviceTab;
