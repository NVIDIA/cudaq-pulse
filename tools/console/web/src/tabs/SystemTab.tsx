import { useEffect, useState } from "react";
import clsx from "clsx";
import { api, type DeviceInfo } from "../api";
import { useConsole } from "../store";
import CouplingGraph from "../components/CouplingGraph";

interface Node {
  id: string;
  x: number;
  y: number;
  title: string;
  sub: string;
  zone: "host" | "control" | "quantum";
}

const W = 1000;
const H = 330;
const NW = 150;
const NH = 58;

const NODES: Node[] = [
  { id: "kernel", x: 90, y: 80, title: "Python kernel", sub: "@pulse.kernel", zone: "host" },
  { id: "compiler", x: 290, y: 80, title: "cudaq-pulse", sub: "trace · passes · schedule", zone: "host" },
  { id: "program", x: 490, y: 80, title: "Controller program", sub: "timed pulses + functions", zone: "host" },
  { id: "controller", x: 490, y: 240, title: "Control electronics", sub: "plays it on its clock", zone: "control" },
  { id: "decide", x: 290, y: 240, title: "On-controller decision", sub: "estimators between shots", zone: "control" },
  { id: "qubits", x: 730, y: 240, title: "Qubits", sub: "", zone: "quantum" },
  { id: "host", x: 900, y: 80, title: "GPU host", sub: "analysis · larger models", zone: "host" },
];

const pos = (id: string) => NODES.find((n) => n.id === id)!;

interface Edge {
  id: string;
  path: string;
  /** Label text and where it sits. */
  label?: [string, number, number];
  link?: boolean;
}

const EDGES: Edge[] = [
  { id: "kernel-compiler", path: `M ${90 + NW / 2} 80 L ${290 - NW / 2} 80` },
  { id: "compiler-program", path: `M ${290 + NW / 2} 80 L ${490 - NW / 2} 80` },
  { id: "program-controller", path: `M 490 ${80 + NH / 2} L 490 ${240 - NH / 2}`, label: ["load", 500, 165] },
  { id: "controller-qubits", path: `M ${490 + NW / 2} 230 L ${730 - NW / 2} 230`, label: ["pulses", 610, 222] },
  { id: "qubits-controller", path: `M ${730 - NW / 2} 250 L ${490 + NW / 2} 250`, label: ["readout", 610, 266] },
  { id: "controller-decide", path: `M ${490 - NW / 2} 230 L ${290 + NW / 2} 230` },
  { id: "decide-controller", path: `M ${290 + NW / 2} 250 L ${490 - NW / 2} 250`, label: ["next pulse", 390, 290] },
  { id: "controller-host", path: `M ${490 + NW / 2} 222 C 760 150, 820 140, 900 ${80 + NH / 2}`, label: ["NVQLink", 790, 158], link: true },
];

interface Step {
  nodes: string[];
  edges: string[];
  plain: string;
  expert: string;
}

const STEPS: Step[] = [
  {
    nodes: ["kernel"],
    edges: [],
    plain: "The experiment is written as an ordinary Python function: which pulses, on which qubit, and what to do with each measurement.",
    expert: "A @pulse.kernel: drives, waits, readouts and @pulse.classical functions, traced from Python.",
  },
  {
    nodes: ["compiler"],
    edges: ["kernel-compiler"],
    plain: "cudaq-pulse turns it into an exact plan: every pulse gets a start time, down to a single clock tick.",
    expert: "Traced to the pulse dialect, optimized by passes (verify, virtual_z, fusion), scheduled per line.",
  },
  {
    nodes: ["program"],
    edges: ["compiler-program"],
    plain: "The plan becomes a program the control electronics can run on their own, without waiting on a computer.",
    expert: "Lowered and exported as a controller program: timed events, epochs and the classical functions with their latency budgets.",
  },
  {
    nodes: ["controller"],
    edges: ["program-controller"],
    plain: "The program is loaded into the control electronics next to the qubits.",
    expert: "The controller holds the program; resources are checked against its limits before it runs.",
  },
  {
    nodes: ["controller", "qubits"],
    edges: ["controller-qubits"],
    plain: "Microwave pulses drive the qubits.",
    expert: "Drive lines play shaped pulses at the scheduled cycles.",
  },
  {
    nodes: ["qubits", "controller"],
    edges: ["qubits-controller"],
    plain: "Each qubit is measured and the signal is turned into a 0 or a 1.",
    expert: "Readout lines return bits to the controller.",
  },
  {
    nodes: ["decide", "controller"],
    edges: ["controller-decide", "decide-controller"],
    plain: "Small estimators run right there, update what they know and choose the next pulse, between one shot and the next.",
    expert: "Classical functions run on the controller within their latency budget (e.g. adaptive_t1 < 2 µs) and feed the next shot.",
  },
  {
    nodes: ["host"],
    edges: ["controller-host"],
    plain: "Results go over NVQLink to the GPU host for analysis and larger models.",
    expert: "Results and estimates go over NVQLink to the host.",
  },
];

const ZONES = [
  { zone: "host", x: 10, y: 22, w: 980, h: 118, label: "host" },
  { zone: "control", x: 200, y: 180, w: 370, h: 130, label: "next to the qubits" },
];

const Diagram = ({ step, simple }: { step: number; simple: boolean }) => {
  const s = STEPS[step];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
        </marker>
      </defs>
      {ZONES.map((z) => (
        <g key={z.zone}>
          <rect x={z.x} y={z.y} width={z.w} height={z.h} rx={10} fill="none" stroke="var(--border)" strokeDasharray="3 5" />
          <text x={z.x + 10} y={z.y + 14} fontSize={10} fill="var(--muted)" letterSpacing={1}>
            {z.label.toUpperCase()}
          </text>
        </g>
      ))}
      {EDGES.map((e) => {
        const on = s.edges.includes(e.id);
        return (
          <g key={e.id} className="transition-opacity duration-500" style={{ color: on ? "var(--accent)" : "var(--border)", opacity: on ? 1 : 0.7 }}>
            <path d={e.path} fill="none" stroke="currentColor" strokeWidth={on ? 2.5 : 1.5} strokeDasharray={e.link ? "6 5" : undefined} markerEnd="url(#arrow)" />
            {e.label && (
              <text x={e.label[1]} y={e.label[2]} fontSize={10.5} textAnchor={e.id === "program-controller" ? "start" : "middle"} fill={on ? "var(--accent)" : "var(--muted)"}>
                {e.label[0]}
              </text>
            )}
            {on && (
              <circle key={`${e.id}-${step}`} r={5} fill="var(--accent-2)" style={{ filter: "drop-shadow(0 0 6px var(--accent-2))" }}>
                <animateMotion dur="1.6s" repeatCount="indefinite" path={e.path} />
              </circle>
            )}
          </g>
        );
      })}
      {NODES.map((n) => {
        const on = s.nodes.includes(n.id);
        const sub = n.id === "compiler" && simple ? "the compiler" : n.sub;
        return (
          <g key={n.id} transform={`translate(${n.x - NW / 2} ${n.y - NH / 2})`} className="transition-opacity duration-500" style={{ opacity: on ? 1 : 0.55 }}>
            <rect
              width={NW}
              height={NH}
              rx={8}
              fill={on ? "color-mix(in srgb, var(--accent) 14%, var(--panel))" : "var(--panel)"}
              stroke={on ? "var(--accent)" : n.zone === "quantum" ? "var(--accent-2)" : "var(--border)"}
              strokeWidth={on ? 2 : 1.2}
              style={{ filter: on ? "drop-shadow(0 0 10px var(--glow-green))" : undefined, transition: "fill 400ms, stroke 400ms" }}
            />
            <text x={NW / 2} y={sub ? 24 : 34} textAnchor="middle" fontSize={13} fontWeight={700} fill="var(--text)">
              {n.title}
            </text>
            {sub && (
              <text x={NW / 2} y={42} textAnchor="middle" fontSize={10} fill="var(--muted)" className={!simple && n.id === "kernel" ? "mono" : undefined}>
                {sub}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

const STEP_MS = 3500;

const DataFlow = () => {
  const simple = useConsole((s) => s.mode === "overview");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => setStep((s) => (s + 1) % STEPS.length), STEP_MS);
    return () => window.clearInterval(id);
  }, [playing]);
  const go = (d: number) => {
    setPlaying(false);
    setStep((s) => (s + d + STEPS.length) % STEPS.length);
  };
  return (
    <section className="flex min-h-0 flex-col rounded-lg border border-[var(--border)] bg-[var(--panel)]">
      <div className="flex items-center gap-3 border-b border-[var(--border)] px-3 py-2">
        <h2 className="text-sm font-black">How a run flows through the system</h2>
        <div className="ml-auto flex items-center gap-2">
          <button className="rounded px-2 py-1 text-xs text-[var(--muted)] hover:text-[var(--text)]" onClick={() => go(-1)} aria-label="Previous step">
            ◀
          </button>
          <button
            onClick={() => setPlaying((p) => !p)}
            className="flex items-center gap-2 rounded-md bg-[var(--accent)] px-4 py-1 text-xs font-black text-black hover:brightness-110"
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button className="rounded px-2 py-1 text-xs text-[var(--muted)] hover:text-[var(--text)]" onClick={() => go(1)} aria-label="Next step">
            ▶
          </button>
        </div>
      </div>
      <div className="min-h-0 flex-1 px-3 pt-2">
        <Diagram step={step} simple={simple} />
      </div>
      <div className="flex items-center gap-4 border-t border-[var(--border)] px-4 py-3">
        <div className="flex gap-1.5">
          {STEPS.map((_, i) => (
            <button
              key={i}
              onClick={() => {
                setPlaying(false);
                setStep(i);
              }}
              aria-label={`Step ${i + 1}`}
              className={clsx("h-2 rounded-full transition-all duration-300", i === step ? "w-6 bg-[var(--accent)]" : "w-2 bg-[var(--border)]")}
            />
          ))}
        </div>
        <p key={step} className="animate-[fadein_400ms_ease-out] text-sm leading-snug">
          <span className="mr-2 font-black text-[var(--accent)]">{step + 1}</span>
          {simple ? STEPS[step].plain : STEPS[step].expert}
        </p>
      </div>
    </section>
  );
};

const fmt = (v: number | null | undefined, d = 2) => (v == null ? "–" : v.toFixed(d));

const Device = () => {
  const meta = useConsole((s) => s.meta);
  const simple = useConsole((s) => s.mode === "overview");
  const compiled = useConsole((s) => s.result);
  const [target, setTarget] = useState(useConsole.getState().target);
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  useEffect(() => {
    api.device(target).then(setInfo).catch(() => setInfo(null));
  }, [target]);
  const used = new Set(compiled?.ok && compiled.target === target ? compiled.qudits ?? [] : []);
  if (!info) return <section className="rounded-lg border border-[var(--border)] bg-[var(--panel)] p-4 text-sm text-[var(--muted)]">Loading device…</section>;
  const t1s = info.qudits.map((q) => q.t1_us ?? 0);
  const facts = [
    { k: simple ? "Qubits" : "Qudits", v: String(info.qudits.length) },
    { k: "Couplings", v: String(info.couplings.length) },
    { k: "Clock", v: `${info.clock_ghz} GHz` },
    { k: simple ? "Lifetimes" : "T1", v: `${Math.min(...t1s).toFixed(0)}–${Math.max(...t1s).toFixed(0)} µs` },
  ];
  return (
    <section className="grid min-h-0 grid-cols-[minmax(0,300px)_minmax(0,1fr)] gap-3 rounded-lg border border-[var(--border)] bg-[var(--panel)] p-3">
      <div className="flex min-h-0 flex-col gap-2">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-black">Device</h2>
          <select className="ml-auto" value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Device">
            {meta?.targets.map((t) => (
              <option key={t.id} value={t.id}>
                {t.id}
              </option>
            ))}
          </select>
        </div>
        <div className="text-[11px] text-[var(--muted)]">
          {info.architecture === "transmon" ? "Superconducting qubits" : "Neutral atoms"}, simulated · {info.name}
        </div>
        <div className="grid grid-cols-2 gap-2">
          {facts.map((f) => (
            <div key={f.k} className="rounded-lg border border-[var(--border)] bg-[var(--bg)] px-3 py-1.5">
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted)]">{f.k}</div>
              <div className="text-lg font-black tabular-nums">{f.v}</div>
            </div>
          ))}
        </div>
        {used.size > 0 && <div className="text-[11px] text-[var(--accent)]">Highlighted: the qubits of the last compiled kernel</div>}
      </div>
      <div className={clsx("grid min-h-0 gap-3", simple ? "grid-cols-1" : "grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]")}>
        <div className="min-h-0">
          <CouplingGraph info={info} used={used} />
        </div>
        {!simple && (
          <div className="min-h-0 overflow-auto rounded border border-[var(--border)]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[var(--panel)] text-[10px] text-[var(--muted)]">
                <tr>
                  {["#", "label", "f01 GHz", "α MHz", "T1 µs", "T2* µs", "thermal"].map((h) => (
                    <th key={h} className="px-2 py-1.5 text-right font-medium first:text-left">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="mono">
                {info.qudits.map((q) => (
                  <tr key={q.index} className={clsx("border-t border-[var(--border)]", used.has(q.index) && "bg-[var(--accent)]/10")}>
                    <td className="px-2 py-0.5 font-bold">{q.index}</td>
                    <td className="px-2 text-right">{q.label}</td>
                    <td className="px-2 text-right">{fmt(q.frequency_ghz, 4)}</td>
                    <td className="px-2 text-right">{fmt(q.anharmonicity_mhz, 0)}</td>
                    <td className="px-2 text-right">{fmt(q.t1_us, 1)}</td>
                    <td className="px-2 text-right">{fmt(q.t2_star_us, 1)}</td>
                    <td className="px-2 text-right">{fmt(q.thermal_population, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
};

const SystemTab = () => (
  <div className="grid h-full min-h-0 grid-rows-[minmax(0,1.15fr)_minmax(0,1fr)] gap-3">
    <DataFlow />
    <Device />
  </div>
);

export default SystemTab;
