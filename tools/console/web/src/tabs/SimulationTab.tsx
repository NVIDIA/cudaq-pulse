import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type SimResult } from "../api";
import { useConsole, LEVEL_COLORS } from "../store";
import Panel from "../components/Panel";
import ErrorCard from "../components/ErrorCard";
import ScheduleTimeline from "../components/ScheduleTimeline";

const SimulationTab = () => {
  const s = useConsole();
  const [points, setPoints] = useState(61);
  const [levels, setLevels] = useState(2);
  const [busy, setBusy] = useState(false);
  const [sim, setSim] = useState<SimResult | null>(null);
  const [cursor, setCursor] = useState<number | null>(null);

  const run = async () => {
    setBusy(true);
    try {
      setSim(
        await api.simulate({
          source: s.source,
          kernel: s.kernel,
          args: s.args,
          target: s.target,
          passes: s.passes,
          schedule: s.schedule,
          points,
          levels,
        }),
      );
    } catch (e) {
      setSim({ ok: false, error: { kind: "NetworkError", message: String(e) } });
    } finally {
      setBusy(false);
    }
  };

  const series = sim?.series ?? [];
  const data = series.map((p) => {
    const row: Record<string, number> = { t: p.t_ns };
    p.populations.forEach((pops, qi) => pops.forEach((v, l) => (row[`q${qi}_${l}`] = v)));
    return row;
  });

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-[var(--border)] bg-[var(--panel)] px-3 py-2 text-xs">
        <span className="text-[var(--muted)]">Simulates the compiler explorer’s kernel with the NumPy reference integrator:</span>
        <span className="mono text-[var(--accent)]">
          pulse.simulate({s.kernel}({s.args}), t_end=t)
        </span>
        <label className="ml-auto text-[10px] uppercase tracking-wider text-[var(--muted)]">Samples</label>
        <input type="number" className="w-16" value={points} min={2} max={121} onChange={(e) => setPoints(+e.target.value)} />
        <label className="text-[10px] uppercase tracking-wider text-[var(--muted)]">Levels</label>
        <select value={levels} onChange={(e) => setLevels(+e.target.value)}>
          {[2, 3].map((l) => (
            <option key={l}>{l}</option>
          ))}
        </select>
        <button
          onClick={run}
          disabled={busy || !s.kernel}
          className="rounded-md bg-[var(--accent)] px-4 py-1.5 text-xs font-bold text-black disabled:opacity-50"
        >
          {busy ? "Simulating…" : "Simulate"}
        </button>
      </div>
      {sim && !sim.ok && sim.error && <ErrorCard error={sim.error} />}
      {!sim && (
        <div className="p-6 text-sm text-[var(--muted)]">
          Pick a kernel in the compiler explorer (e.g. <span className="mono">01_single_qubit_rabi · rabi</span>), then simulate it here.
        </div>
      )}
      {sim?.ok && (
        <div className="grid min-h-0 flex-1 grid-rows-[3fr_2fr] gap-2">
          <Panel
            title={
              <>
                Level populations over time{" "}
                <span className="normal-case text-[var(--muted)]">
                  · qudits {sim.qudits!.join(", ")} · {sim.series!.length} simulations in {sim.wall_s!.toFixed(2)} s
                </span>
              </>
            }
          >
            <div className="grid h-full gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(sim.dims!.length, 3)}, minmax(0, 1fr))` }}>
              {sim.dims!.map((dim, qi) => (
                <div key={qi} className="flex min-h-0 flex-col">
                  <div className="text-[11px] font-semibold">qudit {sim.qudits![qi]}</div>
                  <div className="min-h-0 flex-1">
                    <ResponsiveContainer>
                      <LineChart
                        data={data}
                        margin={{ top: 6, right: 10, bottom: 0, left: -10 }}
                        onMouseMove={(e: any) => setCursor(e?.activeLabel ?? null)}
                        onMouseLeave={() => setCursor(null)}
                      >
                        <CartesianGrid stroke="var(--grid)" />
                        <XAxis dataKey="t" type="number" stroke="var(--muted)" fontSize={10} unit=" ns" domain={[0, "dataMax"]} />
                        <YAxis domain={[0, 1]} stroke="var(--muted)" fontSize={10} />
                        <Tooltip
                          contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", fontSize: 11 }}
                          formatter={(v: number) => v.toFixed(4)}
                          labelFormatter={(t: number) => `${t.toFixed(1)} ns`}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        {Array.from({ length: dim }, (_, l) => (
                          <Line
                            key={l}
                            dataKey={`q${qi}_${l}`}
                            name={`|${l}⟩`}
                            stroke={LEVEL_COLORS[l]}
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </Panel>
          <div className="grid min-h-0 grid-cols-[3fr_1fr] gap-2">
            <Panel title="Schedule">
              <ScheduleTimeline events={sim.events!} clockGhz={sim.clock_ghz!} cursorNs={cursor} />
            </Panel>
            <Panel title="Measurement branches" bodyClassName="overflow-auto">
              {sim.branches!.length === 0 || (sim.branches!.length === 1 && sim.branches![0].outcomes.length === 0) ? (
                <div className="text-xs text-[var(--muted)]">no measurements</div>
              ) : (
                <table className="w-full text-xs">
                  <tbody>
                    {sim.branches!.map((b, i) => (
                      <tr key={i} className="border-t border-[var(--border)]">
                        <td className="mono py-1">{b.outcomes.join(" ")}</td>
                        <td className="w-24">
                          <div className="h-2 rounded bg-[var(--accent)]" style={{ width: `${b.p * 100}%` }} />
                        </td>
                        <td className="mono text-right">{b.p.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationTab;
