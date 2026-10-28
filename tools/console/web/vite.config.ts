import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the app with hot reload and forwards /api to the
// Python server (`python tools/console/run.py`).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  build: {
    target: "es2020",
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: {
        manualChunks: {
          "vendor-react": ["react", "react-dom"],
          "vendor-charts": ["recharts"],
          "vendor-monaco": ["monaco-editor"],
        },
      },
    },
  },
});
