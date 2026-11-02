import { describe, expect, it } from "vitest"

import { POSE_SEND_QUEUE_LIMIT_BYTES, sendPose, sinkHasRoom } from "./poseBackpressure"

/** A transport whose network takes `drainBytes` per frame, like a link that
 *  carries less than the pose stream needs. */
class SlowSink {
  bufferedAmount = 0
  sent: string[] = []
  constructor(private readonly drainBytes: number) {}
  send(data: string) {
    this.sent.push(data)
    this.bufferedAmount += data.length
  }
  tick() {
    this.bufferedAmount = Math.max(0, this.bufferedAmount - this.drainBytes)
  }
}

const FRAME = "x".repeat(1000)

describe("sinkHasRoom", () => {
  it("allows a few frames of queue and no more", () => {
    expect(sinkHasRoom({ send: () => {}, bufferedAmount: 0 })).toBe(true)
    expect(sinkHasRoom({ send: () => {}, bufferedAmount: POSE_SEND_QUEUE_LIMIT_BYTES })).toBe(true)
    expect(sinkHasRoom({ send: () => {}, bufferedAmount: POSE_SEND_QUEUE_LIMIT_BYTES + 1 })).toBe(
      false
    )
  })

  it("sends on a sink that doesn't report its queue", () => {
    expect(sinkHasRoom({ send: () => {} })).toBe(true)
  })
})

describe("sendPose", () => {
  it("keeps the queue bounded on a link slower than the stream", () => {
    // The link carries 90% of the stream. Sending every frame, the queue (and
    // so the control delay) would grow by ~100 bytes per frame forever.
    const sink = new SlowSink(900)
    let worst = 0
    for (let i = 0; i < 10_000; i++) {
      sendPose(sink, FRAME)
      worst = Math.max(worst, sink.bufferedAmount)
      sink.tick()
    }
    expect(worst).toBeLessThanOrEqual(POSE_SEND_QUEUE_LIMIT_BYTES + FRAME.length)
    // Most frames still go out: only the excess is skipped.
    expect(sink.sent.length).toBeGreaterThan(8_500)
  })

  it("sends every frame on a healthy link", () => {
    const sink = new SlowSink(2000)
    for (let i = 0; i < 1000; i++) {
      expect(sendPose(sink, FRAME)).toBe(true)
      sink.tick()
    }
    expect(sink.sent).toHaveLength(1000)
  })

  it("always sends a frame that must not be lost", () => {
    // A reset is a one-frame pulse; the next frame won't repeat it.
    const sink = new SlowSink(0)
    sink.bufferedAmount = 1_000_000
    expect(sendPose(sink, FRAME)).toBe(false)
    expect(sendPose(sink, FRAME, true)).toBe(true)
    expect(sink.sent).toHaveLength(1)
  })

  it("does nothing without a sink", () => {
    expect(sendPose(null, FRAME, true)).toBe(false)
  })
})
