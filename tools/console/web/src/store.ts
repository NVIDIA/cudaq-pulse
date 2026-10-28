import { create } from "zustand";
import type { CompileResult, KernelInfo, Meta } from "./api";

export type Theme = "nvidia" | "nvidia-light";

interface ConsoleState {
  meta: Meta | null;
  connected: boolean | null;
  theme: Theme;
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

export const useConsole = create<ConsoleState>((set) => ({
  meta: null,
  connected: null,
  theme: "nvidia",
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
  set: (patch) => set(patch),
}));

export const KIND_COLORS: Record<string, string> = {
  drive: "#76b900",
  readout: "#00d4ff",
  wait: "#6b7280",
  submit: "#a855f7",
  answer: "#f5a623",
  sync: "#ec4899",
};

export const LEVEL_COLORS = ["#76b900", "#00d4ff", "#f5a623", "#a855f7"];
