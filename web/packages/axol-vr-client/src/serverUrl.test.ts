import { describe, expect, it } from "vitest"

import { axolHttpsOrigin, axolWsUrl, resolveAuthority } from "./serverUrl"

describe("server URL helpers", () => {
  it.each([
    ["robot.local", "robot.local:8000"],
    [" 192.168.1.20 ", "192.168.1.20:8000"],
    ["robot.local:9000", "robot.local:9000"],
    ["https://robot.example/path", "robot.example:443"],
    ["wss://robot.example/ws", "robot.example:443"],
  ])("resolves %s", (input, expected) => {
    expect(resolveAuthority(input, 8000)).toBe(expected)
  })

  it("builds matching secure HTTP and websocket endpoints", () => {
    expect(axolWsUrl("robot.local", 8000)).toBe("wss://robot.local:8000/ws")
    expect(axolHttpsOrigin("robot.local", 8000)).toBe("https://robot.local:8000")
  })
})
