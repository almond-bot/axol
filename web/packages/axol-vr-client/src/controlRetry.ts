type MutableCurrent<T> = { current: T }

export function controlPeerNeedsRetry(state: string): boolean {
  return state === "disconnected" || state === "failed" || state === "closed"
}

/** Keep requesting until an offer arrives; a lost/failed server task is not an ack. */
export function controlRequestIsDue(
  offerSeen: boolean,
  lastRequestAtMs: number,
  nowMs: number,
  retryMs: number
): boolean {
  return !offerSeen && nowMs - lastRequestAtMs >= retryMs
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
