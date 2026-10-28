import { lazy, Suspense, useEffect, useState } from "react";
import clsx from "clsx";
import "./monaco";
import { api } from "./api";
import { useConsole, type Theme } from "./store";
import CompilerTab, { useLoadExample } from "./tabs/CompilerTab";

const CalibrationTab = lazy(() => import("./tabs/CalibrationTab"));
const SimulationTab = lazy(() => import("./tabs/SimulationTab"));
const DeviceTab = lazy(() => import("./tabs/DeviceTab"));

const TABS = [
  { id: "compiler", label: "Compiler Explorer", hint: "kernel → IR → passes → schedule → controller program" },
  { id: "calibration", label: "Calibration", hint: "examples 20 and 21, live" },
  { id: "simulation", label: "Simulation", hint: "populations over time" },
  { id: "device", label: "Device", hint: "qudits, couplings, lines" },
] as const;
type TabId = (typeof TABS)[number]["id"];

const tabFromHash = (): TabId => {
  const h = window.location.hash.replace("#", "");
  return (TABS.find((t) => t.id === h)?.id ?? "compiler") as TabId;
};

const App = () => {
  const { meta, connected, theme, set } = useConsole();
  const [tab, setTab] = useState<TabId>(tabFromHash());
  const loadExample = useLoadExample();

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    api
      .meta()
      .then((m) => {
        set({ meta: m, connected: true, passes: m.default_passes });
        const initial = new URLSearchParams(window.location.search).get("example") ?? useConsole.getState().exampleFile;
        return loadExample(initial);
      })
      .catch(() => set({ connected: false }));
    const id = window.setInterval(() => {
      api
        .health()
        .then(() => set({ connected: true }))
        .catch(() => set({ connected: false }));
    }, 5000);
    return () => window.clearInterval(id);
  }, [set, loadExample]);

  const go = (id: TabId) => {
    window.location.hash = id;
    setTab(id);
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-6 border-b border-[var(--border)] bg-[var(--panel)]/70 px-5 py-2.5">
        <div className="flex items-baseline gap-2">
          <span className="text-lg font-black tracking-tight text-[var(--accent)]">NVQ</span>
          <span className="text-lg font-black tracking-tight">Console</span>
          <span className="text-[11px] text-[var(--muted)]">for cudaq-pulse {meta?.version ?? ""}</span>
        </div>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => go(t.id)}
              title={t.hint}
              className={clsx(
                "rounded-md px-3 py-1.5 text-xs",
                tab === t.id
                  ? "bg-[var(--accent)] font-bold text-black"
                  : "text-[var(--muted)] hover:text-[var(--text)]",
              )}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-[11px]">
          <span className="flex items-center gap-1.5">
            <span
              className={clsx(
                "inline-block h-2 w-2 rounded-full",
                connected ? "bg-emerald-400" : connected === false ? "bg-rose-500" : "bg-amber-400",
              )}
            />
            <span className="text-[var(--muted)]">
              {connected ? "python · localhost" : connected === false ? "server offline" : "connecting"}
            </span>
          </span>
          <select
            value={theme}
            onChange={(e) => set({ theme: e.target.value as Theme })}
            aria-label="Theme"
          >
            <option value="nvidia">Dark</option>
            <option value="nvidia-light">Light</option>
          </select>
        </div>
      </header>
      <main className="min-h-0 flex-1 p-3">
        {connected === false && (
          <div className="mb-2 rounded border border-[var(--danger)]/50 bg-[var(--danger)]/10 p-2 text-xs text-[var(--danger)]">
            The Python server is not reachable. Start it with <span className="mono">python tools/console/run.py</span>.
          </div>
        )}
        <Suspense fallback={<div className="p-6 text-sm text-[var(--muted)]">Loading…</div>}>
          <div className={clsx("h-full", tab !== "compiler" && "hidden")}>
            <CompilerTab />
          </div>
          {tab === "calibration" && <CalibrationTab />}
          {tab === "simulation" && <SimulationTab />}
          {tab === "device" && <DeviceTab />}
        </Suspense>
      </main>
    </div>
  );
};

export default App;
