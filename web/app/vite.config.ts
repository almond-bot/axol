import { defineConfig } from "vite"
import react from "@vitejs/plugin-react-swc"
import basicSsl from "@vitejs/plugin-basic-ssl"
import tailwindcss from "@tailwindcss/vite"
import { fileURLToPath, URL } from "node:url"

// The control panel talks to `axol serve` (default :8090). In dev we proxy
// /api (REST + WebSocket logs) there so the app can use same-origin URLs in
// both dev and the production bundle served by the Python backend.
const SUPERVISOR = process.env.AXOL_SERVE_URL ?? "http://localhost:8090"

export default defineConfig({
  plugins: [react(), basicSsl(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    allowedHosts: ["sp-mbp.local"],
    host: true,
    proxy: {
      "/api": {
        target: SUPERVISOR,
        changeOrigin: true,
        ws: true,
        secure: false,
      },
    },
  },
})
