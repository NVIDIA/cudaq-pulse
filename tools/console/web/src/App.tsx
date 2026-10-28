import { useEffect, useState } from "react";
import clsx from "clsx";
import "./monaco";
import { api } from "./api";
import { useConsole, type Mode } from "./store";
import { themeChanged } from "./components/canvas";
import { afterPaint } from "./perf";
import CompilerTab, { loadExample } from "./tabs/CompilerTab";
import CalibrationTab from "./tabs/CalibrationTab";
import SystemTab from "./tabs/SystemTab";
import SweepsTab from "./tabs/SweepsTab";

const TABS = [
  { id: "compiler", label: "Compiler Explorer" },
  { id: "system", label: "System" },
  { id: "calibration", label: "Calibration" },
  { id: "sweeps", label: "Sweeps" },
] as const;
type TabId = (typeof TABS)[number]["id"];

const tabFromHash = (): TabId => {
  const h = window.location.hash.replace("#", "");
  return (TABS.find((t) => t.id === h)?.id ?? "compiler") as TabId;
};

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "overview", label: "Overview", hint: "No pulse-level detail" },
  { id: "expert", label: "Expert", hint: "IR, schedules, controller programs" },
];

const App = () => {
  const version = useConsole((s) => s.meta?.version);
  const connected = useConsole((s) => s.connected);
  const theme = useConsole((s) => s.theme);
  const mode = useConsole((s) => s.mode);
  const set = useConsole((s) => s.set);
  const [tab, setTab] = useState<TabId>(tabFromHash());
  const [visited, setVisited] = useState<Set<TabId>>(() => new Set([tabFromHash()]));

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    themeChanged();
  }, [theme]);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => setVisited((v) => (v.has(tab) ? v : new Set(v).add(tab))), [tab]);

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
  }, [set]);

  const go = (id: TabId) => {
    afterPaint("tab switch", performance.now());
    history.replaceState(null, "", `${window.location.search}#${id}`);
    setTab(id);
  };

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-6 border-b border-[var(--border)] px-4 py-2">
        <div className="flex items-baseline gap-1.5" title={version ? `cudaq-pulse ${version}` : undefined}>
          <span className="text-base font-black tracking-tight text-[var(--accent)]">NVQ</span>
          <span className="text-base font-black tracking-tight">Console</span>
        </div>
        <nav className="flex gap-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => go(t.id)}
              className={clsx(
                "relative rounded-md px-3 py-1.5 text-xs transition-colors",
                tab === t.id ? "font-bold text-[var(--text)]" : "text-[var(--muted)] hover:text-[var(--text)]",
              )}
            >
              {t.label}
              <span
                className={clsx(
                  "absolute inset-x-3 -bottom-[9px] h-0.5 rounded-full bg-[var(--accent)] transition-transform duration-300",
                  tab === t.id ? "scale-x-100" : "scale-x-0",
                )}
              />
            </button>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-[11px]">
          <div className="flex rounded-md border border-[var(--border)] p-0.5" role="radiogroup" aria-label="Detail">
            {MODES.map((m) => (
              <button
                key={m.id}
                role="radio"
                aria-checked={mode === m.id}
                title={m.hint}
                onClick={() => set({ mode: m.id })}
                className={clsx(
                  "rounded px-2.5 py-0.5 normal-case transition-colors",
                  mode === m.id ? "bg-[var(--accent)] font-bold text-black" : "text-[var(--muted)] hover:text-[var(--text)]",
                )}
              >
                {m.label}
              </button>
            ))}
          </div>
          <button
            onClick={() => set({ theme: theme === "nvidia" ? "nvidia-light" : "nvidia" })}
            className="rounded px-1.5 py-0.5 text-[var(--muted)] hover:text-[var(--text)]"
            aria-label="Toggle light and dark theme"
            title="Light / dark"
          >
            {theme === "nvidia" ? "☾" : "☀"}
          </button>
          <span
            className={clsx("inline-block h-2 w-2 rounded-full", connected ? "bg-[var(--accent)]" : connected === false ? "bg-[var(--danger)]" : "bg-amber-400")}
            title={connected ? "Python server on this machine" : connected === false ? "Server offline" : "Connecting"}
          />
        </div>
      </header>
      <main className="min-h-0 flex-1 p-3">
        {connected === false && (
          <div className="mb-2 rounded border border-[var(--danger)]/50 bg-[var(--danger)]/10 p-2 text-xs text-[var(--danger)]">
            The Python server is not reachable. Start it with <span className="mono">python tools/console/run.py</span>.
          </div>
        )}
        {TABS.map(
          (t) =>
            visited.has(t.id) && (
              <div key={t.id} className={clsx("h-full", tab !== t.id && "hidden")}>
                {t.id === "compiler" && <CompilerTab />}
                {t.id === "system" && <SystemTab />}
                {t.id === "calibration" && <CalibrationTab />}
                {t.id === "sweeps" && <SweepsTab />}
              </div>
            ),
        )}
      </main>
    </div>
  );
};

export default App;
