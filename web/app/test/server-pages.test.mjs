import assert from "node:assert/strict"
import test from "node:test"

import { isAbsolutePagePath, normalizeServerPages, serverPageUrl } from "../src/lib/server-pages.ts"

test("only absolute, non-API paths count as backend pages", () => {
  assert.equal(isAbsolutePagePath("/finetune"), true)
  assert.equal(isAbsolutePagePath("/tools/x?tab=1"), true)
  assert.equal(isAbsolutePagePath("finetune"), false)
  assert.equal(isAbsolutePagePath("//evil.example/x"), false)
  assert.equal(isAbsolutePagePath("/api/finetuning"), false)
})

test("the /api/info pages field is validated and de-duplicated", () => {
  assert.deepEqual(normalizeServerPages(undefined), [])
  assert.deepEqual(normalizeServerPages("nope"), [])
  assert.deepEqual(
    normalizeServerPages([
      { label: " Finetuning ", path: "/finetune", description: " Train on Pi " },
      { label: "Dup", path: "/finetune" },
      { label: "", path: "/blank" },
      { label: "Relative", path: "x" },
      { label: "API", path: "/api/x" },
      null,
      { label: "Plain", path: "/plain", description: 3 },
    ]),
    [
      { label: "Finetuning", path: "/finetune", description: "Train on Pi" },
      { label: "Plain", path: "/plain" },
    ]
  )
})

test("page hrefs stay relative on the backend's own origin and absolute elsewhere", () => {
  assert.equal(serverPageUrl("", "/finetune"), "/finetune")
  assert.equal(
    serverPageUrl("https://192.168.1.20:8001", "/finetune"),
    "https://192.168.1.20:8001/finetune"
  )
  assert.equal(
    serverPageUrl("https://station.local:8001/", "/finetune"),
    "https://station.local:8001/finetune"
  )
})
