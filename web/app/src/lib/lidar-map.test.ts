import { describe, expect, it } from "vitest"
import {
  isMapChunk,
  LidarMap,
  mapStatusLine,
  parseMapChunk,
  parseMapPose,
  parseMapStatus,
  parseMapTrail,
} from "./lidar-map"

/** The server's map chunk: "LDM1" | epoch | flags | n | int16 cm xyz. */
function chunk(epoch: number, replace: boolean, points: [number, number, number][]) {
  const buf = new ArrayBuffer(16 + points.length * 6)
  const view = new DataView(buf)
  view.setUint32(0, 0x314d444c, true)
  view.setUint32(4, epoch, true)
  view.setUint32(8, replace ? 1 : 0, true)
  view.setUint32(12, points.length, true)
  points.forEach((p, i) =>
    p.forEach((v, k) => view.setInt16(16 + i * 6 + k * 2, Math.round(v * 100), true))
  )
  return buf
}

const pose = (epoch: number, x: number, y: number) => ({
  type: "pose",
  epoch,
  m: [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, 0],
  state: "mapping",
  match: 0.9,
})

describe("lidar map", () => {
  it("decodes map chunks in metres", () => {
    const buf = chunk(3, true, [
      [1.5, -2, 0.25],
      [-100, 0.01, 3],
    ])
    expect(isMapChunk(buf)).toBe(true)
    const c = parseMapChunk(buf)!
    expect(c.epoch).toBe(3)
    expect(c.replace).toBe(true)
    expect(Array.from(c.positions)).toEqual([1.5, -2, 0.25, -100, 0.01, 3].map(Math.fround))
    expect(parseMapChunk(buf.slice(0, 20))).toBeNull()
    expect(isMapChunk(new ArrayBuffer(12))).toBe(false)
  })

  it("adds to the map it has and starts over on a new one", () => {
    const map = new LidarMap()
    let ticks = 0
    map.subscribe(() => ticks++)
    // An update for a map it never got is ignored.
    map.apply(parseMapChunk(chunk(1, false, [[0, 0, 0]]))!)
    expect(map.count).toBe(0)
    map.apply(parseMapChunk(chunk(1, true, [[1, 0, 0.5]]))!)
    const generation = map.generation
    map.apply(parseMapChunk(chunk(1, false, [[2, 0, 1]]))!)
    expect(map.count).toBe(2)
    expect(map.generation).toBe(generation)
    expect(Array.from(map.positions.subarray(0, 6))).toEqual([1, 0, 0.5, 2, 0, 1])
    expect(map.colors.subarray(0, 6).every((c) => c >= 0 && c <= 1)).toBe(true)
    // Updates for the old map after a reset are dropped.
    map.apply(parseMapChunk(chunk(2, true, []))!)
    map.apply(parseMapChunk(chunk(1, false, [[9, 9, 9]]))!)
    expect(map.count).toBe(0)
    expect(map.epoch).toBe(2)
    expect(ticks).toBe(3)
  })

  it("grows its buffers past their first size", () => {
    const map = new LidarMap()
    const many = Array.from(
      { length: 70000 },
      (_, i) => [(i % 3000) / 100, 0, 0] as [number, number, number]
    )
    map.apply(parseMapChunk(chunk(1, true, many))!)
    map.apply(parseMapChunk(chunk(1, false, many.slice(0, 1000)))!)
    expect(map.count).toBe(71000)
    expect(map.positions.length).toBeGreaterThanOrEqual(71000 * 3)
    expect(map.positions[69999 * 3]).toBeCloseTo(9.99)
  })

  it("follows the robot and extends the trail as it moves", () => {
    const map = new LidarMap()
    map.apply(parseMapChunk(chunk(4, true, []))!)
    map.setTrail(4, parseMapTrail({ type: "trail", epoch: 4, points: [[0, 0, 0]] })!.points)
    map.setPose(parseMapPose(pose(4, 0.05, 0))!)
    expect(map.trail.length).toBe(3)
    map.setPose(parseMapPose(pose(4, 0.5, 0))!)
    expect(map.trail).toEqual([0, 0, 0, 0.5, 0, 0])
    expect(map.pose?.m[3]).toBe(0.5)
    // A pose for another map is ignored.
    map.setPose(parseMapPose(pose(3, 9, 9))!)
    expect(map.pose?.m[3]).toBe(0.5)
    expect(parseMapPose({ type: "pose", m: [1, 2] })).toBeNull()
  })

  it("says what the map is doing", () => {
    const status = parseMapStatus({
      source: null,
      epoch: 1,
      error: null,
      state: "mapping",
      points: 245000,
      distance: 12.34,
      full: false,
    })!
    expect(mapStatusLine(status)).toEqual({
      text: "Mapping, 12.3 m driven, 245k points",
      tone: "ok",
    })
    expect(mapStatusLine({ ...status, state: "lost", source: "office" }).tone).toBe("bad")
    expect(mapStatusLine({ ...status, error: "There is no saved map named x." })).toEqual({
      text: "There is no saved map named x.",
      tone: "bad",
    })
    expect(mapStatusLine(null).tone).toBe("wait")
    expect(parseMapStatus({ state: "weird" })!.state).toBe("starting")
    expect(parseMapStatus(null)).toBeNull()
  })
})
