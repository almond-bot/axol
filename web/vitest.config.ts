import path from "node:path"
import { fileURLToPath } from "node:url"

import { defineConfig } from "vitest/config"

const root = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(root, "app/src"),
      // CI runs the tests before any build, so the workspace package's
      // dist/ entry points don't exist yet.
      "@almond/axol-vr-client": path.resolve(root, "packages/axol-vr-client/src/index.ts"),
    },
  },
  define: {
    __AXOL_BUILD_COMMIT__: JSON.stringify("test-ui-commit"),
    __AXOL_BUILD_VERSION__: JSON.stringify("1.2.0"),
  },
  test: {
    environment: "jsdom",
    include: ["**/*.test.{ts,tsx}"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json-summary"],
      include: [
        "app/src/lib/{camera-spec,headset,supervisor,telemetry,utils,version}.ts",
        "packages/axol-vr-client/src/serverUrl.ts",
      ],
      thresholds: {
        statements: 75,
        branches: 75,
        functions: 75,
        lines: 75,
      },
    },
  },
})
