import { afterEach, describe, expect, it, vi } from "vitest"
import { Quaternion, Vector3 } from "three"
import {
  cancelJogMove,
  combineHold,
  describePlanError,
  executeJogPlan,
  holdMessage,
  IDLE_HOLD,
  jointValuesFor,
  liftJointValue,
  padCommand,
  planJogMove,
  PreviewQueue,
  withArmJog,
  type JogPose,
  type JogState,
} from "./jog"
import { nudgePose } from "./jog-pose"

describe("padCommand", () => {
  it("maps screen up to forward and screen left to +y", () => {
    expect(padCommand(0, -100, 100)).toEqual([1, -0, 0])
    const [vx, vy] = padCommand(-100, 0, 100)
    expect(vx).toBeCloseTo(0)
    expect(vy).toBeCloseTo(1)
  })

  it("clips to the unit circle and applies the speed limit", () => {
    const [vx, vy] = padCommand(-300, -300, 100, 0.5)
    expect(Math.hypot(vx, vy)).toBeCloseTo(0.5)
    expect(vx).toBeCloseTo(vy)
  })

  it("is zero near the centre and for a degenerate pad", () => {
    expect(padCommand(3, 3, 100)).toEqual([0, 0, 0])
    expect(padCommand(50, 50, 0)).toEqual([0, 0, 0])
  })
})

describe("combineHold", () => {
  it("carries the pad, a scaled turn and the lift direction", () => {
    expect(combineHold([0.4, -0.2, 0], 1, -1, 0.5)).toEqual({ drive: [0.4, -0.2, 0.5], lift: -1 })
    expect(combineHold([0, 0, 0], 0, 0)).toEqual({ drive: [0, 0, 0], lift: 0 })
  })
})

describe("withArmJog", () => {
  it("adds one arm's twist, frame and speed to the held command", () => {
    const hold = withArmJog(
      combineHold([0.2, 0, 0], 0, 1),
      "right",
      [0, 0, 1, 0, 0, -1],
      "gripper",
      0.5
    )
    expect(hold).toEqual({
      drive: [0.2, 0, 0],
      lift: 1,
      arms: { right: [0, 0, 1, 0, 0, -1] },
      armFrame: "gripper",
      armSpeed: 0.5,
    })
  })

  it("clips the twist and speed and fills missing axes", () => {
    const hold = withArmJog(IDLE_HOLD, "left", [3, -3, Number.NaN], "torso", 7)
    expect(hold.arms?.left).toEqual([1, -1, 0, 0, 0, 0])
    expect(hold.armSpeed).toBe(1)
    expect(withArmJog(IDLE_HOLD, "left", [], "torso", -1).armSpeed).toBe(0)
  })

  it("goes out on the hold message, and a plain hold stays as it was", () => {
    const hold = withArmJog(IDLE_HOLD, "left", [0, 1, 0, 0, 0, 0], "torso", 0.3)
    expect(holdMessage(hold)).toEqual({
      type: "hold",
      drive: [0, 0, 0],
      lift: 0,
      arms: { left: [0, 1, 0, 0, 0, 0] },
      armFrame: "torso",
      armSpeed: 0.3,
    })
    expect(holdMessage(IDLE_HOLD)).toEqual({ type: "hold", drive: [0, 0, 0], lift: 0 })
  })
})

describe("PreviewQueue", () => {
  const pose = (x: number): JogPose => ({ position: [x, 0, 0], quaternion: [0, 0, 0, 1] })

  it("keeps one request on the wire and sends only the latest target next", () => {
    const queue = new PreviewQueue(1000)
    expect(queue.next(0)).toBeNull()
    queue.request("left", pose(1))
    expect(queue.next(0)).toMatchObject({ type: "preview", seq: 1, position: [1, 0, 0] })
    queue.request("left", pose(2))
    queue.request("left", pose(3))
    expect(queue.next(10)).toBeNull() // still waiting for seq 1
    expect(queue.answer(1)).toBe(true)
    expect(queue.next(20)).toMatchObject({ seq: 3, position: [3, 0, 0] })
    expect(queue.next(30)).toBeNull()
  })

  it("never shows an older answer over a newer one, nor a failed one", () => {
    const queue = new PreviewQueue(1000)
    queue.request("left", pose(1))
    queue.next(0)
    queue.request("left", pose(2))
    expect(queue.answer(1, true)).toBe(false)
    queue.next(1)
    expect(queue.answer(2)).toBe(true)
    expect(queue.answer(1)).toBe(false)
  })

  it("gives up on a lost answer after the timeout, and on a dropped link", () => {
    const queue = new PreviewQueue(100)
    queue.request("right", pose(1))
    queue.next(0)
    queue.request("right", pose(2))
    expect(queue.next(50)).toBeNull()
    expect(queue.next(150)).toMatchObject({ seq: 2, arm: "right" })
    queue.request("right", pose(3))
    queue.reconnected()
    expect(queue.next(160)).toMatchObject({ seq: 3 })
  })
})

describe("liftJointValue", () => {
  it("maps the raised fraction onto the URDF lift joint (0 = fully raised)", () => {
    expect(liftJointValue(1, 0.34)).toBe(0)
    expect(liftJointValue(0, 0.34)).toBeCloseTo(0.34)
    expect(liftJointValue(0.25, 0.34)).toBeCloseTo(0.255)
    expect(liftJointValue(null, 0.34)).toBe(0)
    expect(liftJointValue(2, 0.34)).toBe(0)
  })
})

describe("jointValuesFor", () => {
  const state: JogState = {
    running: true,
    arms: { left: [1, 2, 3, 4, 5, 6, 7], right: [0, 0, 0, 0, 0, 0, 0.5] },
    lift: { raised: 0.5, velocity: 0, moving: false },
    base: {
      x: 1,
      y: 2,
      yaw: 0.3,
      command: [0, 0, 0],
      velocity: [0, 0, 0],
      moving: false,
      estimated: false,
    },
  }

  it("names every URDF joint the view needs", () => {
    const values = jointValuesFor(state, 0.34)
    expect(values.left_s1_0).toBe(1)
    expect(values.left_w2_0).toBe(7)
    expect(values.right_w2_0).toBe(0.5)
    expect(values.lift).toBeCloseTo(0.17)
    expect(values.lift_2).toBe(values.lift)
    expect([values.base_x, values.base_y, values.base_yaw]).toEqual([1, 2, 0.3])
  })

  it("overlays a plan on its arm only", () => {
    const values = jointValuesFor(state, 0.34, { arm: "right", joints: [9, 9, 9, 9, 9, 9, 9] })
    expect(values.right_s1_0).toBe(9)
    expect(values.left_s1_0).toBe(1)
  })

  it("is empty but safe before the session reports anything", () => {
    expect(jointValuesFor(null, 0.34)).toEqual({ lift: 0, lift_2: 0 })
  })
})

describe("describePlanError", () => {
  it("reports millimetres and degrees", () => {
    const { mm, degrees } = describePlanError({
      positionError: 0.0123,
      orientationError: Math.PI / 18,
    })
    expect(mm).toBeCloseTo(12.3)
    expect(degrees).toBeCloseTo(10)
  })
})

describe("nudgePose", () => {
  const pose = {
    position: [0.1, 0.2, 0.3] as [number, number, number],
    quaternion: [0, 0, 0, 1] as [number, number, number, number],
  }

  it("moves along a torso axis without turning", () => {
    const next = nudgePose(pose, "move", 2, 0.01)
    expect(next.position[2]).toBeCloseTo(0.31)
    expect(next.quaternion).toEqual([0, 0, 0, 1])
    expect(pose.position[2]).toBe(0.3)
  })

  it("turns about the gripper's own axis", () => {
    const yawed = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), Math.PI / 2)
    const start = {
      ...pose,
      quaternion: [yawed.x, yawed.y, yawed.z, yawed.w] as [number, number, number, number],
    }
    const next = nudgePose(start, "turn", 0, Math.PI / 2)
    const q = new Quaternion(...next.quaternion)
    // The gripper's x axis now points along torso +y; rolling about it
    // must leave that axis where it is.
    const x = new Vector3(1, 0, 0).applyQuaternion(q)
    expect(x.y).toBeCloseTo(1)
    const z = new Vector3(0, 0, 1).applyQuaternion(q)
    expect(z.x).toBeCloseTo(1)
    expect(next.position).toEqual(pose.position)
  })
})

describe("jog API client", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("posts plans, executes and cancels, and surfaces server errors", async () => {
    const calls: [string, RequestInit | undefined][] = []
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push([url, init])
        if (url.endsWith("/api/jog/execute"))
          return new Response(
            JSON.stringify({ error: "That plan is out of date. Plan the move again." }),
            {
              status: 409,
            }
          )
        return new Response(JSON.stringify({ id: 4, stopped: true }), { status: 200 })
      })
    )
    const plan = await planJogMove("left", { position: [0.1, 0.2, 0.3], quaternion: [0, 0, 0, 1] })
    expect(plan.id).toBe(4)
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({
      arm: "left",
      position: [0.1, 0.2, 0.3],
      quaternion: [0, 0, 0, 1],
    })
    await expect(executeJogPlan(4)).rejects.toThrow("That plan is out of date")
    expect((await cancelJogMove()).stopped).toBe(true)
  })
})
