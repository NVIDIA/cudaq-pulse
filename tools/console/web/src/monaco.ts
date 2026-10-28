// The editor API with the few contributions the console uses, and one
// language: a fraction of the full Monaco bundle.
import * as monaco from "monaco-editor/esm/vs/editor/editor.api";
import "monaco-editor/esm/vs/editor/browser/coreCommands";
import "monaco-editor/esm/vs/editor/browser/widget/codeEditor/codeEditorWidget";
import "monaco-editor/esm/vs/editor/browser/widget/diffEditor/diffEditor.contribution";
import "monaco-editor/esm/vs/editor/contrib/bracketMatching/browser/bracketMatching";
import "monaco-editor/esm/vs/editor/contrib/clipboard/browser/clipboard";
import "monaco-editor/esm/vs/editor/contrib/comment/browser/comment";
import "monaco-editor/esm/vs/editor/contrib/cursorUndo/browser/cursorUndo";
import "monaco-editor/esm/vs/editor/contrib/find/browser/findController";
import "monaco-editor/esm/vs/editor/contrib/folding/browser/folding";
import "monaco-editor/esm/vs/editor/contrib/hover/browser/hoverContribution";
import "monaco-editor/esm/vs/editor/contrib/indentation/browser/indentation";
import "monaco-editor/esm/vs/editor/contrib/linesOperations/browser/linesOperations";
import "monaco-editor/esm/vs/editor/contrib/multicursor/browser/multicursor";
import "monaco-editor/esm/vs/editor/contrib/wordHighlighter/browser/wordHighlighter";
import "monaco-editor/esm/vs/editor/contrib/wordOperations/browser/wordOperations";
import "monaco-editor/esm/vs/editor/contrib/readOnlyMessage/browser/contribution";
import "monaco-editor/esm/vs/editor/common/standaloneStrings";
import "monaco-editor/esm/vs/base/browser/ui/codicons/codiconStyles";
import "monaco-editor/esm/vs/basic-languages/python/python.contribution";
import { loader } from "@monaco-editor/react";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";

// Bundled with the app instead of fetched from a CDN, so the console works
// offline.
self.MonacoEnvironment = { getWorker: () => new editorWorker() };
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

monaco.languages.register({ id: "json" });
monaco.languages.setMonarchTokensProvider("json", {
  tokenizer: {
    root: [
      [/"[^"]*"(?=\s*:)/, "attribute.name"],
      [/"[^"]*"/, "string"],
      [/-?\d+(\.\d+)?([eE][+-]?\d+)?/, "number"],
      [/\b(true|false|null)\b/, "keyword"],
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
