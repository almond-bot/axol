import { describe, expect, it } from "vitest"

import { operationToFollow } from "./session-follow"
import type { SessionInfo, SessionStatus } from "./supervisor"

const session = (id: string, command: string, status: SessionStatus): SessionInfo => ({
  id,
  command,
  args: {},
  status,
  exitCode: null,
  error: null,
  startedAt: 0,
  pid: null,
})

const operations = new Set(["teleop", "collect-data"])

describe("following a run started from another browser", () => {
  it("picks up a live operation the panel is not showing", () => {
    const run = session("b", "collect-data", "starting")
    expect(operationToFollow([session("a", "teleop", "exited"), run], operations, "a")).toBe(run)
    expect(operationToFollow([run], operations, null)).toBe(run)
  })

  it("leaves the panel alone when it already shows that run", () => {
    expect(operationToFollow([session("b", "teleop", "running")], operations, "b")).toBeNull()
  })

  it("ignores finished runs and setup commands", () => {
    expect(operationToFollow([session("a", "teleop", "exited")], operations, null)).toBeNull()
    expect(operationToFollow([session("c", "can.setup", "running")], operations, null)).toBeNull()
  })

  it("follows a run that is stopping, so its teardown is shown", () => {
    const run = session("b", "teleop", "stopping")
    expect(operationToFollow([run], operations, null)).toBe(run)
  })
})
