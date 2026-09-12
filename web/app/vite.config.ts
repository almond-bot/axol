import { defineConfig } from "vite"
import react from "@vitejs/plugin-react-swc"
import basicSsl from "@vitejs/plugin-basic-ssl"
import tailwindcss from "@tailwindcss/vite"
import { execSync } from "node:child_process"
import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { fileURLToPath, URL } from "node:url"

// The control panel talks to `axol serve` (default :8001). In dev we proxy
// /api (REST + WebSocket logs) there so the app can use same-origin URLs in
// both dev and the production bundle served by the Python backend.
const SUPERVISOR = process.env.AXOL_SERVE_URL ?? "http://localhost:8001"

/**
 * Git commit this bundle is built from, baked in so the control panel can
 * compare it against the backend's commit (/api/info) and warn on a mismatch.
 * Vercel builds read the env var (the clone there can be shallow/absent);
 * local builds ask git. Null when neither source is available.
 *
 * A dirty working tree gets a "-dirty.<hash>" suffix over the uncommitted
 * changes, so a dist built from one dirty state still mismatches a backend
 * running another. The scheme (status + diff, sha256, 8 hex chars) must stay
 * identical to `installed_commit()` in almond_axol/serve/update.py.
 */
function buildCommit(): string | null {
  const fromEnv = process.env.VERCEL_GIT_COMMIT_SHA
  if (fromEnv) return fromEnv
  // Run from the repo root — the backend computes its identity there, and
  // status/diff output must be byte-identical for the hashes to agree.
  const git = {
    encoding: "buffer",
    maxBuffer: 256 * 1024 * 1024,
    cwd: fileURLToPath(new URL("../..", import.meta.url)),
  } as const
  try {
    const head = execSync("git rev-parse HEAD", git).toString("utf8").trim()
    if (!head) return null
    const status = execSync("git status --porcelain", git)
    if (!status.toString("utf8").trim()) return head
    const diff = execSync("git diff HEAD", git)
    const digest = createHash("sha256").update(status).update(diff).digest("hex")
    return `${head}-dirty.${digest.slice(0, 8)}`
  } catch {
    return null
  }
}

/** Release version from the repo-root pyproject.toml (the backend's version source). */
function buildVersion(): string | null {
  try {
    const pyproject = readFileSync(
      fileURLToPath(new URL("../../pyproject.toml", import.meta.url)),
      "utf8"
    )
    return /^version\s*=\s*"([^"]+)"/m.exec(pyproject)?.[1] ?? null
  } catch {
    return null
  }
}

export default defineConfig({
  plugins: [react(), basicSsl(), tailwindcss()],
  define: {
    __AXOL_BUILD_COMMIT__: JSON.stringify(buildCommit()),
    __AXOL_BUILD_VERSION__: JSON.stringify(buildVersion()),
  },
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
