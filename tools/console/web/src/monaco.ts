import * as monaco from "monaco-editor";
import { loader } from "@monaco-editor/react";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";

// Bundle Monaco with the app instead of fetching it from a CDN, so the
// console works offline.
self.MonacoEnvironment = {
  getWorker(_: unknown, label: string) {
    return label === "json" ? new jsonWorker() : new editorWorker();
  },
};
loader.config({ monaco });

monaco.languages.register({ id: "mlir" });
monaco.languages.setMonarchTokensProvider("mlir", {
  tokenizer: {
    root: [
      [/\/\/.*$/, "comment"],
      [/"[^"]*"/, "string"],
      [/%[\w.$-]+/, "variable"],
      [/@[\w.$-]+/, "type.identifier"],
      [/![\w.<>,: ]+/, "type"],
      [/\b(pulse|classical|qop|cudm|llvm)\.[\w.]+/, "keyword"],
      [/\b(func|arith|scf|cf|memref|math)\.[\w.]+/, "keyword.flow"],
      [/\b(module|attributes|return|true|false)\b/, "keyword"],
      [/\b[a-z_]+(?= =)/, "attribute.name"],
      [/-?\d+(\.\d+)?(e[+-]?\d+)?/, "number"],
    ],
  },
});

monaco.editor.defineTheme("nvq-dark", {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "keyword", foreground: "76b900", fontStyle: "bold" },
    { token: "keyword.flow", foreground: "9ccc65" },
    { token: "variable", foreground: "00d4ff" },
    { token: "type", foreground: "c084fc" },
    { token: "type.identifier", foreground: "f5a623" },
    { token: "attribute.name", foreground: "a0a0a0" },
    { token: "number", foreground: "fbbf24" },
  ],
  colors: {
    "editor.background": "#0d0d0d",
    "editor.lineHighlightBackground": "#1a1a1a",
    "editorLineNumber.foreground": "#4b5563",
  },
});

monaco.editor.defineTheme("nvq-light", {
  base: "vs",
  inherit: true,
  rules: [
    { token: "keyword", foreground: "3d6000", fontStyle: "bold" },
    { token: "variable", foreground: "0369a1" },
    { token: "type", foreground: "7c3aed" },
    { token: "type.identifier", foreground: "b45309" },
    { token: "number", foreground: "b45309" },
  ],
  colors: { "editor.background": "#ffffff" },
});

export { monaco };
