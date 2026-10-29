import { describe, expect, it } from "vitest"

import { motorFaultLabels, safetyView } from "./safety-view"
import type { JellyStatus, RobotStatus, SafetyBoardStatus, SafetyLinkStatus } from "./supervisor"

const board = (over: Partial<SafetyBoardStatus> = {}): SafetyBoardStatus => ({
  estopOk: true,
  latched: true,
  outputOn: true,
  softwareEnable: true,
  remoteStartActive: false,
  heartbeatOn: false,
  supplyOk: true,
  supplyVolts: 26.4,
  outputVolts: 26.0,
  stopCause: "none",
  firmware: "0.2",
  faults: {
    relay: false,
    k2Stuck: false,
    k2NoClose: false,
    k1PoleB: false,
    remoteStartFailed: false,
    heartbeatLost: false,
  },
  ...over,
})

const link = (
  status: SafetyBoardStatus | null,
  over: Partial<SafetyLinkStatus> = {}
): SafetyLinkStatus => ({
  channel: "can_alm_axol_c",
  connected: true,
  paused: false,
  detected: true,
  reachable: true,
  ageSeconds: 0.1,
  status,
  error: null,
  ...over,
})

const off = { outputOn: false, latched: false }

describe("safetyView", () => {
  it("hides on a Jelly without a safety board", () => {
    expect(safetyView(null).visible).toBe(false)
    expect(safetyView(link(null, { detected: false })).visible).toBe(false)
  })

  it("offers the stop while power is on", () => {
    const v = safetyView(link(board()))
    expect(v).toMatchObject({ visible: true, tone: "on", canStop: true, canStart: false })
  })

  it("names a pressed E-Stop and blocks the restore", () => {
    const v = safetyView(link(board({ ...off, estopOk: false })))
    expect(v).toMatchObject({ tone: "estop", label: "E-Stop pressed", canStart: false })
    expect(v.startHint).toBe("Release the E-Stop first.")
  })

  it("offers a restore after a software stop, with the cause", () => {
    const v = safetyView(link(board({ ...off, softwareEnable: false, stopCause: "heartbeat" })))
    expect(v).toMatchObject({ tone: "off", label: "Power cut", canStart: true })
    expect(v.detail).toContain("stopped responding")
  })

  it("blocks the restore while the board updates its firmware", () => {
    const v = safetyView(
      link(board({ ...off, softwareEnable: false, stopCause: "update", updateLocked: true }))
    )
    expect(v).toMatchObject({ label: "Updating firmware", canStart: false, canStop: false })
    expect(v.startHint).toBe("Wait for the update to finish.")
  })

  it("notes a supply dip while power is off", () => {
    const faults = { ...board().faults, supplyDip: true }
    expect(safetyView(link(board({ ...off, faults }))).notes).toContain(
      "Jelly's 24 V supply dipped, which can cut power on its own."
    )
    expect(safetyView(link(board({ faults }))).notes).toEqual([])
  })

  it("offers a restore when the latch is open", () => {
    expect(safetyView(link(board(off)))).toMatchObject({ label: "Power off", canStart: true })
  })

  it("blocks the restore on a flat battery", () => {
    const v = safetyView(link(board({ ...off, supplyOk: false, supplyVolts: 16.2 })))
    expect(v.canStart).toBe(false)
    expect(v.detail).toContain("16.2 V")
  })

  it("puts a relay fault ahead of everything, keeping the stop while powered", () => {
    const faults = { ...board().faults, relay: true, k2Stuck: true }
    const v = safetyView(link(board({ softwareEnable: false, faults })))
    expect(v).toMatchObject({ tone: "fault", canStop: true, canStart: false })
  })

  it("says when the board stops answering, and keeps trying the stop", () => {
    const v = safetyView(link(board(), { reachable: false }))
    expect(v).toMatchObject({ visible: true, tone: "unknown", canStop: true, canStart: false })
    expect(v.detail).toContain("Use the E-Stop")
  })

  it("notes a restore that did not latch", () => {
    const faults = { ...board().faults, remoteStartFailed: true }
    expect(safetyView(link(board({ ...off, faults }))).notes).toContain(
      "The last restore did not switch power on."
    )
  })
})

describe("motorFaultLabels", () => {
  const robot = (over: Partial<RobotStatus> = {}) =>
    ({
      state: "connected",
      connected: true,
      profile: "axol",
      faults: [{ arm: "left", joint: "WRIST_2", problem: "over current", temperature: 40 }],
      ...over,
    }) as RobotStatus
  const jelly = {
    wheels: {
      connected: true,
      motors: [
        { name: "front_left", id: 1, reachable: false, status: null },
        { name: "front_right", id: 2, reachable: true, status: "OK" },
      ],
    },
  } as unknown as JellyStatus

  it("lists arm and wheel faults", () => {
    expect(motorFaultLabels(robot(), jelly)).toEqual([
      "L wrist 2: over current",
      "front left wheel: unreachable",
    ])
  })

  it("leaves out Mantis grippers, which are not on Axol's supply", () => {
    expect(motorFaultLabels(robot({ profile: "mantis" }), null)).toEqual([])
  })

  it("is empty when nothing is faulted", () => {
    expect(motorFaultLabels(robot({ faults: [] }), null)).toEqual([])
  })
})
