import { afterEach, describe, expect, it, vi } from "vitest"
import {
  CAMERA_STREAMS,
  cameraLabel,
  cameraNoun,
  cameraSocketPath,
  desktopHold,
  driveKeyOf,
  isCameraStream,
  isTextEntry,
  keyAxes,
  parseCameraChoice,
  parseCameraFrame,
  parseCameraInfo,
  phoneActivity,
  phoneConnectHint,
  setMarionetteStop,
  type DriveKey,
  type MarionetteArm,
  type MarionettePhone,
} from "./marionette"

const NO_CONTROLS = {
  pad: [0, 0, 0] as [number, number, number],
  turn: 0 as const,
  lift: 0 as const,
}

describe("driveKeyOf", () => {
  it("maps keys by their place on the keyboard", () => {
    expect(driveKeyOf({ code: "KeyW", key: "w" })).toBe("w")
    expect(driveKeyOf({ code: "KeyX", key: "X" })).toBe("x")
    // AZERTY: the key where W sits on QWERTY types "z" but still drives forward.
    expect(driveKeyOf({ code: "KeyW", key: "z" })).toBe("w")
  })

  it("ignores every other key", () => {
    expect(driveKeyOf({ code: "KeyR", key: "r" })).toBeNull()
    expect(driveKeyOf({ code: "Escape", key: "Escape" })).toBeNull()
    expect(driveKeyOf({ code: "ArrowUp", key: "ArrowUp" })).toBeNull()
  })

  it("falls back to the key name when there is no code", () => {
    expect(driveKeyOf({ key: "Q" })).toBe("q")
    expect(driveKeyOf({ key: "p" })).toBeNull()
    expect(driveKeyOf({})).toBeNull()
  })
})

describe("keyAxes", () => {
  it("maps each key to its axis and sign", () => {
    const cases: [DriveKey, [number, number, number], number][] = [
      ["w", [1, 0, 0], 0],
      ["s", [-1, 0, 0], 0],
      ["a", [0, 1, 0], 0],
      ["d", [0, -1, 0], 0],
      ["q", [0, 0, 1], 0],
      ["e", [0, 0, -1], 0],
      ["z", [0, 0, 0], 1],
      ["x", [0, 0, 0], -1],
    ]
    for (const [key, drive, lift] of cases) expect(keyAxes([key])).toEqual({ drive, lift })
  })

  it("cancels opposing keys", () => {
    expect(keyAxes(["w", "s", "a", "d", "q", "e", "z", "x"])).toEqual({
      drive: [0, 0, 0],
      lift: 0,
    })
    expect(keyAxes(["w", "s", "a"])).toEqual({ drive: [0, 1, 0], lift: 0 })
  })
})

describe("desktopHold", () => {
  it("is idle with nothing held", () => {
    expect(desktopHold([], NO_CONTROLS, 0.5)).toEqual({ drive: [0, 0, 0], lift: 0 })
  })

  it("scales driving and turning by the speed, but not the lift", () => {
    expect(desktopHold(["w", "q", "z"], NO_CONTROLS, 0.5)).toEqual({
      drive: [0.5, 0, 0.5],
      lift: 1,
    })
    expect(desktopHold(["d", "e", "x"], NO_CONTROLS, 0.25)).toEqual({
      drive: [0, -0.25, -0.25],
      lift: -1,
    })
  })

  it("drives a diagonal no faster than straight ahead, like the pad", () => {
    const [vx, vy] = desktopHold(["w", "a"], NO_CONTROLS, 0.8).drive
    expect(Math.hypot(vx, vy)).toBeCloseTo(0.8)
    expect(vx).toBeCloseTo(vy)
  })

  it("clamps the speed to 0..1", () => {
    expect(desktopHold(["w"], NO_CONTROLS, 3).drive).toEqual([1, 0, 0])
    expect(desktopHold(["w", "q"], NO_CONTROLS, -1).drive).toEqual([0, 0, 0])
  })

  it("adds the on-screen controls to the keys, capped at the speed", () => {
    const pad: [number, number, number] = [0.5, 0, 0]
    expect(desktopHold(["w"], { pad, turn: 0, lift: 0 }, 0.5).drive).toEqual([0.5, 0, 0])
    const [vx, vy] = desktopHold(["a"], { pad, turn: 0, lift: 0 }, 0.5).drive
    expect(Math.hypot(vx, vy)).toBeCloseTo(0.5)
    expect(desktopHold([], { pad, turn: 1, lift: -1 }, 0.5)).toEqual({
      drive: [0.5, 0, 0.5],
      lift: -1,
    })
  })

  it("cancels a key against the opposite on-screen button", () => {
    expect(desktopHold(["q", "z"], { pad: [0, 0, 0], turn: -1, lift: -1 }, 1)).toEqual({
      drive: [0, 0, 0],
      lift: 0,
    })
    expect(desktopHold(["q"], { pad: [0, 0, 0], turn: 1, lift: 0 }, 1).drive[2]).toBe(1)
  })
})

describe("isTextEntry", () => {
  it("is true for text fields, text areas and selects", () => {
    const text = document.createElement("input")
    const number = document.createElement("input")
    number.type = "number"
    expect(isTextEntry(text)).toBe(true)
    expect(isTextEntry(number)).toBe(true)
    expect(isTextEntry(document.createElement("textarea"))).toBe(true)
    expect(isTextEntry(document.createElement("select"))).toBe(true)
  })

  it("is false for sliders, buttons and the page", () => {
    const range = document.createElement("input")
    range.type = "range"
    expect(isTextEntry(range)).toBe(false)
    expect(isTextEntry(document.createElement("button"))).toBe(false)
    expect(isTextEntry(document.body)).toBe(false)
    expect(isTextEntry(window)).toBe(false)
    expect(isTextEntry(null)).toBe(false)
  })
})

describe("camera streams", () => {
  it("names each stream", () => {
    expect(CAMERA_STREAMS.map((s) => [s.id, s.label])).toEqual([
      ["head", "Head"],
      ["left", "Left wrist"],
      ["right", "Right wrist"],
    ])
    expect(cameraLabel("left")).toBe("Left wrist")
    expect(cameraNoun("head")).toBe("head camera")
    expect(cameraNoun("right")).toBe("right wrist camera")
    expect(isCameraStream("head")).toBe(true)
    expect(isCameraStream("wrists")).toBe(false)
  })

  it("shows every camera unless a valid choice was stored", () => {
    expect(parseCameraChoice(null)).toEqual(["head", "left", "right"])
    expect(parseCameraChoice("not json")).toEqual(["head", "left", "right"])
    expect(parseCameraChoice('{"head":true}')).toEqual(["head", "left", "right"])
    expect(parseCameraChoice('["right","bogus","head"]')).toEqual(["head", "right"])
    expect(parseCameraChoice("[]")).toEqual([])
  })

  it("asks the socket for the chosen streams in a fixed order", () => {
    expect(cameraSocketPath(["right", "head"])).toBe(
      "/api/marionette/cameras?streams=head,right&fps=10"
    )
    expect(cameraSocketPath(["head", "left", "right"], 5)).toBe(
      "/api/marionette/cameras?streams=head,left,right&fps=5"
    )
  })
})

describe("parseCameraFrame", () => {
  const frame = (id: string, body: number[]) => {
    const head = new TextEncoder().encode(`${id}\n`)
    const bytes = new Uint8Array(head.length + body.length)
    bytes.set(head)
    bytes.set(body, head.length)
    return bytes.buffer
  }

  it("splits the stream id from the JPEG at the first newline", () => {
    // The JPEG itself may hold newline bytes.
    const parsed = parseCameraFrame(frame("left", [0xff, 0xd8, 0x0a, 0xff, 0xd9]))
    expect(parsed?.stream).toBe("left")
    expect(Array.from(parsed?.jpeg ?? [])).toEqual([0xff, 0xd8, 0x0a, 0xff, 0xd9])
  })

  it("rejects a message with no newline, no id or no image", () => {
    expect(parseCameraFrame(new Uint8Array([0xff, 0xd8]).buffer)).toBeNull()
    expect(parseCameraFrame(frame("", [0xff]))).toBeNull()
    expect(parseCameraFrame(frame("head", []))).toBeNull()
    expect(parseCameraFrame(new ArrayBuffer(0))).toBeNull()
  })
})

describe("parseCameraInfo", () => {
  it("reads the available streams and the error", () => {
    expect(
      parseCameraInfo('{"type":"cameras","available":["head","right","x"],"error":null}')
    ).toEqual({ available: ["head", "right"], error: null })
    expect(
      parseCameraInfo('{"type":"cameras","available":[],"error":"The head camera is unplugged."}')
    ).toEqual({ available: [], error: "The head camera is unplugged." })
  })

  it("ignores other messages", () => {
    expect(parseCameraInfo('{"type":"state"}')).toBeNull()
    expect(parseCameraInfo("not json")).toBeNull()
  })
})

describe("phoneActivity", () => {
  const phone: MarionettePhone = {
    id: "phone-1",
    name: "Phone 1",
    color: "#4ade80",
    connected: true,
    tracking: "normal",
    source: "arkit",
    calibration: "captured",
    arm: "left",
    view: "all",
    scale: 1,
    speed: 0.5,
    deadman: false,
    stop: false,
    engaged: false,
    driving: false,
    poseAge: 0.02,
  }
  const arm = (owner: string | null, ownerName: string | null): MarionetteArm => ({
    owner,
    ownerName,
    engaged: false,
    limited: false,
    grip: 0,
    reason: null,
    resetting: false,
  })

  it("says what the phone is doing, most important first", () => {
    expect(phoneActivity({ ...phone, connected: false, engaged: true })).toBe("Offline")
    expect(phoneActivity({ ...phone, stop: true, engaged: true })).toBe("Stop is on")
    expect(phoneActivity({ ...phone, engaged: true, driving: true })).toBe("Moving the left arm")
    expect(phoneActivity({ ...phone, arm: "right", engaged: true })).toBe("Moving the right arm")
    expect(phoneActivity({ ...phone, driving: true })).toBe("Driving the base")
    expect(phoneActivity({ ...phone, calibration: "capturing" })).toBe("Calibrating")
    expect(phoneActivity(phone)).toBe("Idle")
  })

  it("names the phone that has its arm", () => {
    const arms = { left: arm("phone-2", "Phone 2"), right: arm(null, null) }
    expect(phoneActivity(phone, arms)).toBe("Phone 2 has the left arm")
    expect(phoneActivity({ ...phone, id: "phone-2" }, arms)).toBe("Idle")
  })
})

describe("phoneConnectHint", () => {
  it("names a reachable gateway address as typed", () => {
    expect(phoneConnectHint("10.1.10.155:8787")).toBe(
      "In the Marionette app, connect to 10.1.10.155:8787."
    )
    expect(phoneConnectHint(" ws://gateway.local:8787 ")).toBe(
      "In the Marionette app, connect to ws://gateway.local:8787."
    )
  })

  it("points at the gateway's computer when the address is local to Axol", () => {
    const local = "In the Marionette app, connect to port 8787 on the computer running the gateway."
    expect(phoneConnectHint("localhost:8787")).toBe(local)
    expect(phoneConnectHint("")).toBe(local)
    expect(phoneConnectHint("127.0.0.1:8787")).toBe(local)
    expect(phoneConnectHint("[::1]:8787")).toBe(local)
    expect(phoneConnectHint("localhost")).toBe(
      "In the Marionette app, connect to the computer running the gateway."
    )
    expect(phoneConnectHint("localhost:8787", "127.0.0.1")).toBe(local)
  })

  it("names Axol's address when the gateway runs on Axol", () => {
    expect(phoneConnectHint("localhost:8787", "10.1.10.220")).toBe(
      "In the Marionette app, connect to 10.1.10.220:8787."
    )
    expect(phoneConnectHint("", "axol.local")).toBe(
      "In the Marionette app, connect to axol.local:8787."
    )
    expect(phoneConnectHint("localhost", "[fd7a::5]")).toBe(
      "In the Marionette app, connect to [fd7a::5]."
    )
    // A gateway elsewhere is named as typed, wherever Axol is.
    expect(phoneConnectHint("10.1.10.155:8787", "10.1.10.220")).toBe(
      "In the Marionette app, connect to 10.1.10.155:8787."
    )
  })
})

describe("setMarionetteStop", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("posts the desktop Stop and surfaces server errors", async () => {
    const calls: [string, RequestInit | undefined][] = []
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push([url, init])
        if (calls.length > 1)
          return new Response(JSON.stringify({ error: "Marionette isn't running." }), {
            status: 409,
          })
        return new Response(JSON.stringify({ on: true, by: "Desktop" }), { status: 200 })
      })
    )
    expect(await setMarionetteStop(true)).toEqual({ on: true, by: "Desktop" })
    expect(calls[0][0]).toBe("/api/marionette/stop")
    expect(calls[0][1]?.method).toBe("POST")
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({ on: true })
    await expect(setMarionetteStop(false)).rejects.toThrow("Marionette isn't running.")
  })
})
