import assert from "node:assert/strict"
import test from "node:test"

import { isAbsolutePagePath, normalizeServerPages, serverPageUrl } from "../src/lib/server-pages.ts"

test("only absolute, non-API paths count as backend pages", () => {
  assert.equal(isAbsolutePagePath("/datasets"), true)
  assert.equal(isAbsolutePagePath("/tools/x?tab=1"), true)
  assert.equal(isAbsolutePagePath("datasets"), false)
  assert.equal(isAbsolutePagePath("//evil.example/x"), false)
  assert.equal(isAbsolutePagePath("/api/datasets"), false)
})

test("the /api/info pages field is validated and de-duplicated", () => {
  assert.deepEqual(normalizeServerPages(undefined), [])
  assert.deepEqual(normalizeServerPages("nope"), [])
  assert.deepEqual(
    normalizeServerPages([
      { label: " Datasets ", path: "/datasets", description: " Browse recorded datasets " },
      { label: "Dup", path: "/datasets" },
      { label: "", path: "/blank" },
      { label: "Relative", path: "x" },
      { label: "API", path: "/api/x" },
      null,
      { label: "Plain", path: "/plain", description: 3 },
    ]),
    [
      { label: "Datasets", path: "/datasets", description: "Browse recorded datasets" },
      { label: "Plain", path: "/plain" },
    ]
  )
})

test("page hrefs stay relative on the backend's own origin and absolute elsewhere", () => {
  assert.equal(serverPageUrl("", "/datasets"), "/datasets")
  assert.equal(
    serverPageUrl("https://192.168.1.20:8001", "/datasets"),
    "https://192.168.1.20:8001/datasets"
  )
  assert.equal(
    serverPageUrl("https://station.local:8001/", "/datasets"),
    "https://station.local:8001/datasets"
  )
})
