type MutableCurrent<T> = { current: T }

/**
 * Terminal peer states that need an immediate renegotiation. "disconnected" is
 * transient per the WebRTC spec (TURN hiccups and Wi-Fi roams self-heal within
 * a few seconds) and is handled with a grace period via
 * `controlDisconnectRearmDelay` instead of an immediate teardown.
 */
export function controlPeerNeedsRetry(state: string): boolean {
  return state === "failed" || state === "closed"
}

/**
 * How long to keep waiting before re-arming a peer that entered "disconnected"
 * at `disconnectedAtMs`. `null` means no rearm is pending (recovered, or a
 * terminal state handled by `controlPeerNeedsRetry`); `0` means the grace
 * period has elapsed; a positive value is the remaining wait in milliseconds.
 */
export function controlDisconnectRearmDelay(
  state: string,
  disconnectedAtMs: number,
  nowMs: number,
  graceMs: number
): number | null {
  if (state !== "disconnected") return null
  return Math.max(0, graceMs - (nowMs - disconnectedAtMs))
}

/**
 * Keep requesting until an offer arrives; a lost/failed server task is not an
 * ack. `notBeforeMs` holds the request back after a negotiation failure so a
 * fast deterministic failure doesn't turn into a per-tick re-request loop.
 */
export function controlRequestIsDue(
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
 * Clearing the reference before calling `close()` prevents Chromium's terminal
 * callback from re-arming the peer a second time during intentional teardown.
 */
export function retireCurrentControlPeer<Socket, Peer>(
  socketRef: MutableCurrent<Socket | null>,
  peerRef: MutableCurrent<Peer | null>,
  socket: Socket,
  peer: Peer
): boolean {
  if (socketRef.current !== socket || peerRef.current !== peer) return false
  peerRef.current = null
  return true
}
