import { describe, expect, it } from "vitest"

import { versionMismatch } from "./version"

describe("version mismatch", () => {
  it("accepts identical commits", () => {
    expect(
      versionMismatch({
        hostname: "x",
        lanIp: "x",
        viewerPort: 1,
        vrPort: 2,
        commit: "test-ui-commit",
      })
    ).toBeNull()
  })

  it("allows a hosted release on the same version", () => {
    expect(
      versionMismatch({
        hostname: "x",
        lanIp: "x",
        viewerPort: 1,
        vrPort: 2,
        commit: "release-commit",
        version: "1.2.0",
        releaseInstall: true,
      })
    ).toBeNull()
  })

  it("reports an older backend release", () => {
    const mismatch = versionMismatch({
      hostname: "x",
      lanIp: "x",
      viewerPort: 1,
      vrPort: 2,
      commit: "old-commit",
      version: "1.1.9",
      releaseInstall: true,
    })
    expect(mismatch?.serverOlder).toBe(true)
    expect(mismatch?.serverDev).toBe(false)
  })
})
