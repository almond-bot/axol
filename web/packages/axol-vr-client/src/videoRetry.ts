type MutableCurrent<T> = { current: T }

/**
 * Terminal peer states that need an immediate renegotiation. Only remotely
 * caused failures land here; "disconnected" is deliberately excluded — per the
 * WebRTC spec it is transient and usually self-heals (a TURN/Funnel hiccup or a
 * Wi-Fi roam flaps connected → disconnected → connected within a few seconds),
 * so it gets a grace period via `videoDisconnectRearmDelay` instead.
 */
export function videoPeerNeedsRetry(state: string): boolean {
  return state === "failed" || state === "closed"
}

/**
 * How long to keep waiting before re-arming a peer that entered "disconnected"
 * at `disconnectedAtMs`. `null` means no rearm is pending — the state has
 * recovered (or moved to a terminal state handled by `videoPeerNeedsRetry`);
 * `0` means the grace period has elapsed and the peer should be re-armed now;
 * a positive value is the remaining wait in milliseconds.
 */
export function videoDisconnectRearmDelay(
  state: string,
  disconnectedAtMs: number,
  nowMs: number,
  graceMs: number
): number | null {
  if (state !== "disconnected") return null
  return Math.max(0, graceMs - (nowMs - disconnectedAtMs))
}

/**
 * Keep requesting until an offer arrives, but never before `notBeforeMs`: a
 * rearm caused by a negotiation failure pushes the next request out by a
 * backoff so a fast deterministic failure (rejected SDP, immediate close) can't
 * become a hot re-request loop against the server.
 */
export function videoRequestIsDue(
  offerSeen: boolean,
  lastRequestAtMs: number,
  nowMs: number,
  retryMs: number,
  notBeforeMs = 0
): boolean {
  return !offerSeen && nowMs - lastRequestAtMs >= retryMs && nowMs >= notBeforeMs
}

/**
 * Retire one peer only if it still belongs to the current signaling socket.
 *
 * Clearing the peer reference is the single-flight claim: a second terminal
 * callback for the same peer, or a delayed callback after socket replacement,
 * cannot schedule another retry or tear down its replacement.
 */
export function retireCurrentVideoPeer<Socket, Peer>(
  socketRef: MutableCurrent<Socket | null>,
  peerRef: MutableCurrent<Peer | null>,
  socket: Socket,
  peer: Peer
): boolean {
  if (socketRef.current !== socket || peerRef.current !== peer) return false
  peerRef.current = null
  return true
}
