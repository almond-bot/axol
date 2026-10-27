import { describe, expect, it } from "vitest"

import {
  OPERATION_CATALOG,
  catalogEntry,
  catalogMeta,
  launcherEntries,
  operationStatus,
  quickStartOperations,
} from "./operation-catalog"
import { OPERATIONS, type OperationMeta } from "./supervisor"

const host = (ids: string[]): OperationMeta[] =>
  ids.map((id) => OPERATIONS.find((o) => o.id === id) ?? { ...OPERATIONS[0], id, label: id })

const ids = (entries: { id: string }[]) => entries.map((e) => e.id)

describe("launcherEntries", () => {
  it("shows every operation before a host is connected", () => {
    const entries = launcherEntries({ hostOperations: [], connected: false, mantis: false })
    expect(ids(entries)).toEqual(ids(OPERATION_CATALOG))
  })

  it("keeps Axol's operations a host lacks, and adds ones the app doesn't know", () => {
    const entries = launcherEntries({
      hostOperations: host(["teleop", "calibrate-wrist"]),
      connected: true,
      mantis: false,
    })
    expect(ids(entries)).toEqual([...ids(OPERATION_CATALOG), "calibrate-wrist"])
    expect(entries.at(-1)?.group).toBe("more")
  })

  it("shows only what the host runs on Mantis", () => {
    const entries = launcherEntries({
      hostOperations: host(["teleop", "collect-data"]),
      connected: true,
      mantis: true,
    })
    expect(ids(entries)).toEqual(["teleop", "collect-data"])
  })
})

describe("operationStatus", () => {
  const entry = catalogEntry("collect-data")
  const meta = host(["collect-data"])[0]
  const base = {
    entry,
    hostMeta: meta,
    available: true,
    connected: true,
    running: false,
    otherRunning: false,
    needsMount: false,
    hasCameras: true,
  }

  it("lists what the operation needs when nothing is in the way", () => {
    expect(operationStatus(base)).toEqual({ tone: "ready", label: "VR headset · Cameras" })
  })

  it("puts a run ahead of everything, then the connection", () => {
    expect(operationStatus({ ...base, running: true, connected: false }).label).toBe("Running")
    expect(operationStatus({ ...base, connected: false, needsMount: true }).label).toBe(
      "Connect to start"
    )
  })

  it.each([
    [{ hostMeta: null }, "Not on this robot"],
    [{ available: false }, "Not available"],
    [{ otherRunning: true }, "Another operation is running"],
    [{ needsMount: true, hasCameras: false }, "Choose a mount"],
    [{ hasCameras: false }, "Assign cameras"],
  ])("says what blocks it: %j", (patch, label) => {
    expect(operationStatus({ ...base, ...patch }).label).toBe(label)
  })
})

describe("catalogMeta", () => {
  it("describes a known operation the host didn't report, and nothing else", () => {
    expect(catalogMeta("waypoints")?.label).toBe("Waypoints")
    expect(catalogMeta("collect-data")?.requiresCameras).toBe(true)
    expect(catalogMeta("calibrate-wrist")).toBeNull()
  })
})

describe("quickStartOperations", () => {
  it("offers recent operations first, then the usual ones, once each", () => {
    const picked = quickStartOperations(OPERATION_CATALOG, ["replay-dataset", "teleop"])
    expect(ids(picked)).toEqual(["replay-dataset", "teleop", "collect-data"])
  })

  it("skips operations that aren't shown", () => {
    const shown = OPERATION_CATALOG.filter((e) => e.id !== "teleop")
    expect(ids(quickStartOperations(shown, ["teleop"]))).toEqual([
      "collect-data",
      "run-policy",
      "gravity-comp",
    ])
  })
})
