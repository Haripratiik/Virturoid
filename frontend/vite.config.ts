import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The Python server (ui_server.py) serves the built app at /studio/ and proxies
// nothing itself; in dev, Vite proxies API + package-file routes to it.
const PY_SERVER = process.env.VIRTUROID_SERVER ?? "http://127.0.0.1:8765";

export default defineConfig({
  base: "/studio/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: PY_SERVER, changeOrigin: true },
      "/package": { target: PY_SERVER, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
});
