/**
 * Send-side backpressure for the pose stream.
 *
 * The headset produces a pose every XR frame (72-90 Hz, about 1 KB each).
 * Neither transport drops a message on its own: a WebSocket and an
 * RTCDataChannel both hold everything passed to `send()` in an unbounded
 * buffer until the network takes it. When the link carries less than the
 * stream needs (a long, lossy path where SCTP/TCP congestion control backs
 * off), that buffer grows for the rest of the session and the robot follows
 * further and further behind the operator.
 *
 * Poses are newest-wins, so a frame that can't leave promptly is worthless:
 * skip it while the transport still has more than a few frames queued, and
 * send the next one instead. The server's interpolator already plays through
 * gaps in the stream.
 */

// Pose sinks (WebSocket and RTCDataChannel) both expose `.send(string)` and
// the bytes still waiting to go out as `bufferedAmount`.
export type PoseSink = { send: (data: string) => void; readonly bufferedAmount?: number }

// A few frames of slack: enough that ordinary scheduling jitter never skips a
// frame on a healthy link, small enough that a backlog adds at most ~50 ms.
export const POSE_SEND_QUEUE_LIMIT_BYTES = 4096

/** True when `sink` has no more than `limitBytes` still waiting to be sent. */
export function sinkHasRoom(sink: PoseSink, limitBytes = POSE_SEND_QUEUE_LIMIT_BYTES): boolean {
  const queued = sink.bufferedAmount
  // A sink that doesn't report its queue can't be judged: send.
  return typeof queued !== "number" || queued <= limitBytes
}

/**
 * Send `payload` on `sink` unless it is backed up. `mustSend` overrides the
 * check for a frame carrying a one-frame event (the reset pulse), which the
 * next frame would not repeat. Returns whether the frame was sent.
 */
export function sendPose(sink: PoseSink | null, payload: string, mustSend = false): boolean {
  if (sink == null) return false
  if (!mustSend && !sinkHasRoom(sink)) return false
  sink.send(payload)
  return true
}
