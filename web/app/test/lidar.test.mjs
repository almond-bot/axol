import assert from "node:assert/strict"
import test from "node:test"

import {
  LidarCloud,
  lidarStatusLine,
  parseLidarFrame,
  parseLidarStatus,
  sanitizeLidarSettings,
} from "../src/lib/lidar.ts"

/** The server's binary frame: "LDR1" | seq | n | int16 mm xyz | u8 reflectivity. */
function frameBytes(seq, points) {
  const n = points.length
  const buf = new ArrayBuffer(12 + n * 7)
  const view = new DataView(buf)
  view.setUint32(0, 0x3152444c, true)
  view.setUint32(4, seq, true)
  view.setUint32(8, n, true)
  points.forEach(([x, y, z, r], i) => {
    view.setInt16(12 + i * 6, Math.round(x * 1000), true)
    view.setInt16(14 + i * 6, Math.round(y * 1000), true)
    view.setInt16(16 + i * 6, Math.round(z * 1000), true)
    view.setUint8(12 + n * 6 + i, r)
  })
  return buf
}

const SETTINGS = { history: 1, pointSize: 1, hideAbove: 2, colorBy: "height" }

test("decodes a frame in metres", () => {
  const frame = parseLidarFrame(
    frameBytes(7, [
      [1.5, -2, 0.25, 30],
      [-31, 0.001, 3, 255],
    ]),
    100
  )
  assert.equal(frame.seq, 7)
  assert.equal(frame.count, 2)
  assert.deepEqual(
    Array.from(frame.positions, (v) => Math.round(v * 1000) / 1000),
    [1.5, -2, 0.25, -31, 0.001, 3]
  )
  assert.deepEqual(Array.from(frame.reflectivity), [30, 255])
  assert.equal(frame.receivedAt, 100)
})

test("rejects other messages and short frames", () => {
  assert.equal(parseLidarFrame(new ArrayBuffer(4)), null)
  const wrong = frameBytes(1, [[0, 0, 0, 0]])
  new DataView(wrong).setUint32(0, 0, true)
  assert.equal(parseLidarFrame(wrong), null)
  assert.equal(parseLidarFrame(frameBytes(1, [[0, 0, 0, 0]]).slice(0, 15)), null)
})

test("merges the history window and hides points above the cut", () => {
  const cloud = new LidarCloud()
  let ticks = 0
  cloud.subscribe(() => ticks++)
  cloud.add(parseLidarFrame(frameBytes(1, [[1, 0, 0.5, 0]]), 0))
  cloud.add(parseLidarFrame(frameBytes(2, [[2, 0, 0.5, 0]]), 600))
  cloud.add(
    parseLidarFrame(
      frameBytes(3, [
        [3, 0, 0.5, 0],
        [3, 0, 2.5, 0],
      ]),
      1200
    )
  )
  assert.equal(ticks, 3)
  let out = cloud.fill(SETTINGS, new Float32Array(0), new Float32Array(0))
  // The first frame is older than 1 s; the 2.5 m point is above the cut.
  assert.equal(out.count, 2)
  assert.deepEqual([out.positions[0], out.positions[3]], [2, 3])
  // A short history still shows the newest frame.
  out = cloud.fill({ ...SETTINGS, history: 0.1 }, out.positions, out.colors)
  assert.equal(out.count, 1)
  assert.equal(out.positions[0], 3)
  // Colors are valid RGB.
  assert.ok(Array.from(out.colors.slice(0, 3)).every((c) => c >= 0 && c <= 1))
  cloud.clear()
  assert.equal(cloud.fill(SETTINGS, out.positions, out.colors).count, 0)
})

test("drops frames older than it keeps", () => {
  const cloud = new LidarCloud()
  cloud.keep = 1
  for (let i = 0; i < 30; i++) cloud.add(parseLidarFrame(frameBytes(i, [[i, 0, 0, 0]]), i * 100))
  assert.ok(cloud.size <= 11)
})

test("reads the server status", () => {
  const status = parseLidarStatus({
    type: "status",
    state: "streaming",
    error: null,
    pointsPerSecond: 199500,
    mount: { x: 0.1, y: 0, z: 0.96, roll: 0.2, pitch: -7.7, yaw: 0 },
  })
  assert.equal(status.state, "streaming")
  assert.equal(status.mount.z, 0.96)
  assert.deepEqual(lidarStatusLine(status), { text: "Live, 200k points/s", tone: "ok" })
  assert.equal(parseLidarStatus({ type: "other" }), null)
  const failed = parseLidarStatus({ type: "status", state: "error", error: "No data", mount: {} })
  assert.equal(failed.mount, null)
  assert.deepEqual(lidarStatusLine(failed), { text: "No data", tone: "bad" })
})

test("keeps saved settings in range", () => {
  assert.deepEqual(
    sanitizeLidarSettings({ history: 99, pointSize: -1, hideAbove: "x", colorBy: "nope" }),
    { history: 10, pointSize: 0.3, hideAbove: 2.2, colorBy: "height" }
  )
})
