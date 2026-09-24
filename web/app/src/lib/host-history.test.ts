import { beforeEach, describe, expect, it } from "vitest"

import { HOST_HISTORY_STORAGE, loadHostHistory, recordHost } from "./host-history"

describe("host history", () => {
  beforeEach(() => localStorage.clear())

  it("starts empty", () => {
    expect(loadHostHistory()).toEqual([])
  })

  it("keeps the last five hosts, newest first", () => {
    for (const h of ["a", "b", "c", "d", "e", "f"]) recordHost(h)
    expect(loadHostHistory()).toEqual(["f", "e", "d", "c", "b"])
  })

  it("moves a reconnected host to the front without duplicating it", () => {
    recordHost("10.0.0.1")
    recordHost("robot.local")
    expect(recordHost(" ROBOT.local ")).toEqual(["ROBOT.local", "10.0.0.1"])
  })

  it("ignores blank hosts", () => {
    recordHost("10.0.0.1")
    expect(recordHost("  ")).toEqual(["10.0.0.1"])
  })

  it("tolerates corrupt storage", () => {
    localStorage.setItem(HOST_HISTORY_STORAGE, "{not json")
    expect(loadHostHistory()).toEqual([])
    localStorage.setItem(HOST_HISTORY_STORAGE, JSON.stringify(["ok", 3, ""]))
    expect(loadHostHistory()).toEqual(["ok"])
  })
})
