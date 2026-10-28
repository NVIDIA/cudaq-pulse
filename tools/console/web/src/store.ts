import { create } from "zustand";
import type { CompileResult, KernelInfo, Meta } from "./api";

export type Theme = "nvidia" | "nvidia-light";
/** Overview hides pulse-level detail (IR, controller programs, line
 * numbers) for audiences that do not need it. */
export type Mode = "overview" | "expert";

interface ConsoleState {
  meta: Meta | null;
  connected: boolean | null;
  theme: Theme;
  mode: Mode;
  exampleFile: string;
  source: string;
  kernels: KernelInfo[];
  kernel: string;
  args: string;
  target: string;
  passes: string[];
  schedule: string;
  result: CompileResult | null;
  compiling: boolean;
  set: (patch: Partial<ConsoleState>) => void;
}

const saved = (key: string, fallback: string) => localStorage.getItem(`nvqconsole.${key}`) ?? fallback;

export const useConsole = create<ConsoleState>((set) => ({
  meta: null,
  connected: null,
  theme: saved("theme", "nvidia") as Theme,
  mode: saved("mode", "expert") as Mode,
  exampleFile: "09_compilation_pipeline.py",
  source: "",
  kernels: [],
  kernel: "",
  args: "",
  target: "transmon_krinner_17q",
  passes: [],
  schedule: "alap",
  result: null,
  compiling: false,
  set: (patch) => {
    if (patch.theme) localStorage.setItem("nvqconsole.theme", patch.theme);
    if (patch.mode) localStorage.setItem("nvqconsole.mode", patch.mode);
    set(patch);
  },
}));

export const KIND_COLORS: Record<string, string> = {
  drive: "#76b900",
  readout: "#00d4ff",
  wait: "#6b7280",
  submit: "#a855f7",
  answer: "#f5a623",
  sync: "#ec4899",
};

export const LEVEL_COLORS = ["#76b900", "#00d4ff", "#f5a623", "#a855f7", "#ec4899", "#f43f5e"];
